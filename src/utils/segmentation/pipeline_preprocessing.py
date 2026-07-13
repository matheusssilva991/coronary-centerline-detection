"""Carregamento, pré-processamento e vesselness reutilizáveis no pipeline.

Este módulo concentra as etapas que transformam um caso ImageCAS bruto em
volumes prontos para detecção de aorta/óstios e segmentação arterial.
"""

import hashlib
import json
from typing import Any, Dict

import cv2
import numpy as np

from ..processing.frangi import (
    get_vesselness,
    load_vesselness_cache,
    save_vesselness_cache,
)
from ..processing.preprocessing import (
    build_lcc_image_from_mask,
    downscale_image_ndi,
    downscale_image_opencv,
    largest_connected_component,
    threshold_image_with_offset,
)
from .lower_threshold import resolve_lower_threshold
from ..utils.nifti_io import load_raw_img_and_label
from .fuzzy_threshold import (
    fuzzy_threshold_from_config,
    get_thresholding_config,
    normalize_threshold_mode,
)


def _json_safe_cache_value(value: Any) -> Any:
    """Converte valores comuns para montar uma assinatura estável de cache."""
    if isinstance(value, dict):
        return {
            str(key): _json_safe_cache_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_cache_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe_cache_value(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def _vesselness_cache_signature(
    image: Any,
    vesselness_config: Dict[str, Any],
    use_gpu: bool,
    spacing: Any,
) -> str:
    """Gera hash curto para separar caches por parâmetros e resolução."""
    payload = {
        "version": 2,
        "shape": list(np.asarray(image).shape),
        "dtype": str(np.asarray(image).dtype),
        "use_gpu": bool(use_gpu),
        "spacing": _json_safe_cache_value(spacing),
        "vesselness_config": _json_safe_cache_value(vesselness_config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha1(encoded).hexdigest()[:12]


def load_and_preprocess_image(
    img_id: str, base_path: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Carrega imagem/label, reduz resolução e monta a maior componente conectada."""
    # Carrega o volume CCTA e a máscara de referência do ImageCAS.
    nii_img, nii_label = load_raw_img_and_label(
        f"{base_path}/{img_id}.img.nii.gz", f"{base_path}/{img_id}.label.nii.gz"
    )
    spacing = nii_img.header.get_zooms()
    img = np.array(nii_img.get_fdata(), dtype=np.float32)
    label = np.array(nii_label.get_fdata()).astype(np.uint8)

    # Define o método de downsampling usado para imagem e label.
    downscale_factors = config["DOWNSCALE_FACTORS"]
    use_opencv = config["DOWNSCALE_METHOD"] == "opencv"
    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos4": cv2.INTER_LANCZOS4,
    }
    opencv_interpolation = interpolation_map.get(
        config["OPENCV_INTERPOLATION"], cv2.INTER_LINEAR
    )

    thresholding_config = get_thresholding_config(config)
    threshold_mode = normalize_threshold_mode(thresholding_config.get("method"))

    preprocessing_details = {
        "threshold_mode": threshold_mode,
    }

    if threshold_mode == "normal":
        # Caminho histórico: threshold HU superior por percentil + LCC.
        if use_opencv:
            down_image = downscale_image_opencv(
                img,
                downscale_factors,
                interpolation=opencv_interpolation,
            )
        else:
            down_image = downscale_image_ndi(img, downscale_factors, order=3)

        min_threshold, lower_threshold_details = resolve_lower_threshold(
            down_image,
            config,
        )
        thresh_vals = (
            float(min_threshold),
            float(np.percentile(down_image, config["MAX_THRESHOLD_PERCENTILE"])),
        )
        thresh_image, thresh_mask, offset = threshold_image_with_offset(
            down_image,
            *thresh_vals,
        )

        lcc_image = np.zeros_like(thresh_image, dtype=thresh_image.dtype)
        lcc_mask = np.zeros_like(thresh_mask, dtype=bool)
        for z_idx in range(thresh_image.shape[2]):
            lcc_slice, lcc_mask_slice = largest_connected_component(
                thresh_image[:, :, z_idx],
                thresh_mask[:, :, z_idx],
            )
            lcc_image[:, :, z_idx] = lcc_slice
            lcc_mask[:, :, z_idx] = lcc_mask_slice

        lcc_image = (lcc_image - offset).astype(np.float32)
        preprocessing_details.update(
            {
                **lower_threshold_details,
                "min_threshold": float(thresh_vals[0]),
                "max_threshold": float(thresh_vals[1]),
                "threshold_voxels": int(np.sum(thresh_mask)),
                "lcc_voxels": int(np.sum(lcc_mask)),
            }
        )
    else:
        # Fuzzy threshold: mantém apenas voxels cuja classe dominante é objeto.
        if use_opencv:
            down_image = downscale_image_opencv(
                img,
                downscale_factors,
                interpolation=opencv_interpolation,
            )
        else:
            down_image = downscale_image_ndi(img, downscale_factors, order=3)
        fuzzy_config = dict(config)
        fuzzy_threshold_config = dict(thresholding_config.get("fuzzy", {}))
        fuzzy_lower_config = dict(config.get("LOWER_THRESHOLD", {}))
        fuzzy_lower_config["percentile"] = float(
            fuzzy_threshold_config.get(
                "lower_percentile",
                fuzzy_lower_config.get("percentile", 10.5),
            )
        )
        fuzzy_config["LOWER_THRESHOLD"] = fuzzy_lower_config
        min_threshold, lower_threshold_details = resolve_lower_threshold(
            down_image,
            fuzzy_config,
        )
        fuzzy_config["MIN_THRESHOLD"] = float(min_threshold)
        fuzzy_mask, fuzzy_details = fuzzy_threshold_from_config(down_image, fuzzy_config)
        lcc_image, lcc_mask = build_lcc_image_from_mask(
            down_image,
            fuzzy_mask,
            offset=abs(float(min_threshold)),
            per_slice=True,
        )
        preprocessing_details.update(
            {
                **lower_threshold_details,
                "min_threshold": float(min_threshold),
                "max_threshold": None,
                "threshold_voxels": int(np.sum(fuzzy_mask)),
                "lcc_voxels": int(np.sum(lcc_mask)),
                **fuzzy_details,
            }
        )
    # O label usa interpolação de vizinho mais próximo para preservar classes.
    label = downscale_image_ndi(label, downscale_factors, order=0)

    dx, dy, dz = (
        spacing[0] * downscale_factors[0],
        spacing[1] * downscale_factors[1],
        spacing[2] * downscale_factors[2],
    )

    return {
        "lcc_image": lcc_image,
        "label": label,
        "preprocessing_details": preprocessing_details,
        "spacing": spacing,
        "scaled_spacing": (dx, dy, dz),
        "downscale_factors": downscale_factors,
    }


def get_or_compute_vesselness(
    img_id: str,
    image: Any,
    cache_dir: str,
    vesselness_config: Dict[str, Any],
    load_cache: bool = False,
    save_cache: bool = False,
    use_gpu: bool = False,
    spacing: Any = None,
) -> Any:
    """Carrega ou calcula o mapa de vasos (Frangi) para um volume 3D."""
    cache_signature = _vesselness_cache_signature(
        image,
        vesselness_config,
        use_gpu,
        spacing,
    )
    cache_dir = f"{cache_dir}/{cache_signature}"

    # Reutiliza cache quando disponível para evitar recalcular Frangi.
    if load_cache:
        cache = load_vesselness_cache(img_id, cache_dir=cache_dir)
        if cache is not None:
            return cache

    # Monta os parâmetros do Frangi validado para aorta e artérias.
    vesselness_kwargs = {
        "sigmas": vesselness_config["sigmas"],
        "black_ridges": vesselness_config.get("black_ridges", False),
        "alpha": vesselness_config["alpha"],
        "beta": vesselness_config["beta"],
        "gamma": vesselness_config["gamma"],
        "normalization": vesselness_config.get("normalization", "none"),
        "smooth_sigma": vesselness_config.get("smooth_sigma", 0.0),
        "gpu": bool(use_gpu),
    }
    # Calcula o mapa de vasos usado na detecção dos óstios ou segmentação arterial.
    vesselness = get_vesselness(image, **vesselness_kwargs)
    if save_cache:
        save_vesselness_cache(vesselness, img_id, cache_dir=cache_dir)
    return vesselness
