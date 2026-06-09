"""Carregamento, pré-processamento e vesselness reutilizáveis no pipeline."""

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


def load_and_preprocess_image(
    img_id: str, base_path: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Carrega imagem/label e executa o pré-processamento básico."""
    nii_img, nii_label = load_raw_img_and_label(
        f"{base_path}/{img_id}.img.nii.gz", f"{base_path}/{img_id}.label.nii.gz"
    )
    spacing = nii_img.header.get_zooms()
    img = np.array(nii_img.get_fdata(), dtype=np.float32)
    label = np.array(nii_label.get_fdata()).astype(np.uint8)

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

    _, _, lcc_image, _ = run_core_preprocessing_pipeline(
        img,
        downscale_factors=downscale_factors,
        lcc_per_slice=config.get("LCC_PER_SLICE", True),
        min_threshold=config.get("MIN_THRESHOLD", -300),
        max_threshold_percentile=config["MAX_THRESHOLD_PERCENTILE"],
        use_opencv=use_opencv,
        opencv_interpolation=opencv_interpolation,
    )
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
    """Carrega ou calcula vesselness para um volume 3D."""
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

    if load_cache:
        cache = load_vesselness_cache(img_id, cache_dir=cache_dir)
        if cache is not None:
            return cache

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
        if not modified_config.pop("use_spacing", True):
            spacing = None
        vesselness_kwargs.update(modified_config)
        vesselness_kwargs["spacing"] = spacing

    vesselness = vesselness_fn(image, **vesselness_kwargs)
    if save_cache:
        save_vesselness_cache(vesselness, img_id, cache_dir=cache_dir)
    return vesselness
