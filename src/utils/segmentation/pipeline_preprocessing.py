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
    get_modified_vesselness,
    get_vesselness,
    load_vesselness_cache,
    save_vesselness_cache,
)
from ..processing.preprocessing import (
    downscale_image_ndi,
    run_core_preprocessing_pipeline,
)
from ..utils.nifti_io import load_raw_img_and_label


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

    # Aplica threshold HU, downsampling e maior componente conectada.
    _, _, lcc_image, _ = run_core_preprocessing_pipeline(
        img,
        downscale_factors=downscale_factors,
        lcc_per_slice=config.get("LCC_PER_SLICE", True),
        min_threshold=config.get("MIN_THRESHOLD", -300),
        max_threshold_percentile=config["MAX_THRESHOLD_PERCENTILE"],
        use_opencv=use_opencv,
        opencv_interpolation=opencv_interpolation,
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
    method = vesselness_config.get("method", "normal")
    vesselness_fn = {
        "normal": get_vesselness,
        "modified": get_modified_vesselness,
    }.get(method)
    if vesselness_fn is None:
        raise ValueError(
            f"Método de vesselness inválido: {method}. Use 'normal' ou 'modified'."
        )

    if method != "normal":
        cache_dir = f"{cache_dir}_{method}"
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

    # Monta os parâmetros compatíveis com Frangi normal ou modificado.
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
    if method == "modified":
        modified_config = dict(vesselness_config.get("modified", {}))
        # Algumas variações do Frangi modificado usam espaçamento físico.
        if not modified_config.pop("use_spacing", True):
            spacing = None
        vesselness_kwargs.update(modified_config)
        vesselness_kwargs["spacing"] = spacing

    # Calcula o mapa de vasos usado na detecção dos óstios ou segmentação arterial.
    vesselness = vesselness_fn(image, **vesselness_kwargs)
    if save_cache:
        save_vesselness_cache(vesselness, img_id, cache_dir=cache_dir)
    return vesselness
