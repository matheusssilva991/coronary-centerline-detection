"""Carregamento, pré-processamento e vesselness reutilizáveis no pipeline."""

from typing import Any, Dict

import cv2
import numpy as np

from ..processing.frangi import (
    get_vesselness,
    load_vesselness_cache,
    save_vesselness_cache,
)
from ..processing.preprocessing import (
    downscale_image_ndi,
    run_core_preprocessing_pipeline,
)
from ..utils import load_raw_img_and_label


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
        config["OPENCV_INTERPOLATION"], cv2.INTER_AREA
    )

    _, _, lcc_image, _ = run_core_preprocessing_pipeline(
        img,
        downscale_factors=downscale_factors,
        lcc_per_slice=True,
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
) -> Any:
    """Carrega ou calcula vesselness para um volume 3D."""
    if load_cache:
        cache = load_vesselness_cache(img_id, cache_dir=cache_dir)
        if cache is not None:
            return cache

    vesselness = get_vesselness(
        image,
        sigmas=vesselness_config["sigmas"],
        black_ridges=False,
        alpha=vesselness_config["alpha"],
        beta=vesselness_config["beta"],
        gamma=vesselness_config["gamma"],
        normalization="none",
    )
    if save_cache:
        save_vesselness_cache(vesselness, img_id, cache_dir=cache_dir)
    return vesselness

