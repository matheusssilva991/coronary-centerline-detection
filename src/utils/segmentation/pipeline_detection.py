"""Etapas de detecção/segmentação da aorta e avaliação dos óstios."""

from typing import Any, Dict, List, Sequence
from pathlib import Path

import numpy as np

from .aorta_localization import detect_aorta_circles
from .aorta_segmentation import level_set_segmentation, remove_leaks_morphology
from .ostia_detection import check_ostium_intersection, find_ostia
from ..cache_utils import load_json_cache, load_npy_cache, save_json_cache, save_npy_cache
from ..processing.binary_operations import keep_largest_component


def get_or_detect_aorta_circles(
    img_id: str,
    lcc_image: Any,
    downscale_factors: Sequence[int],
    scaled_spacing: Sequence[float],
    circle_config: Dict[str, Any],
    base_save_path: str,
    load_cache: bool = False,
    save_cache: bool = False,
    use_gpu: bool = False,
) -> List[Dict[str, Any]]:
    """Carrega ou detecta círculos da aorta."""
    json_path = (
        Path(base_save_path) / "detected_circles" / f"{img_id}_detected_circles.json"
    )

    cached_circles = load_json_cache(json_path, enabled=load_cache)
    if cached_circles is not None:
        return cached_circles

    dx, dy, _ = scaled_spacing
    radii_start = circle_config["radii_start_px"]
    radii_end = circle_config["radii_end_px"]
    radius_step = circle_config.get("radius_step_px", 1)
    hough_radii = np.arange(radii_start, radii_end, radius_step)
    pixel_spacing = (dx + dy) / 2.0

    detected_circles = detect_aorta_circles(
        lcc_image,
        hough_radii,
        pixel_spacing,
        tol_radius_mm=circle_config["tol_radius_mm"],
        tol_distance_mm=circle_config["tol_distance_mm"],
        quadrant_offset=tuple(circle_config["quadrant_offset"]),
        max_slice_miss_threshold=circle_config["max_slice_miss_threshold"],
        neighbor_distance_threshold=circle_config["neighbor_distance_threshold"],
        total_num_peaks_initial=circle_config["total_num_peaks_initial"],
        total_num_peaks=circle_config["total_num_peaks"],
        canny_sigma=circle_config["canny_sigma"],
        use_local_roi=circle_config.get("use_local_roi", True),
        local_roi_padding=circle_config.get("local_roi_padding", 20),
        use_gpu=bool(use_gpu),
    )
    save_json_cache(detected_circles, json_path, enabled=save_cache)

    return detected_circles


def get_or_segment_aorta(
    img_id: str,
    lcc_image: Any,
    detected_circles: List[Dict[str, Any]],
    level_set_config: Dict[str, Any],
    base_save_path: str,
    load_cache: bool = False,
    save_cache: bool = False,
    use_gpu: bool = False,
) -> Any:
    """Carrega ou segmenta a aorta com level set + pós-processamento."""
    mask_path = Path(base_save_path) / "segmented_aorta" / f"{img_id}_mask_aorta.npy"

    cached_mask = load_npy_cache(mask_path, enabled=load_cache)
    if cached_mask is not None:
        return cached_mask

    mask_refined = level_set_segmentation(
        lcc_image,
        detected_circles,
        radius_reduction_factor=level_set_config["radius_reduction_factor"],
        num_iter=level_set_config["num_iter"],
        balloon=level_set_config["balloon"],
        smoothing=level_set_config["smoothing"],
        use_gpu=bool(use_gpu),
    )
    aorta_mask = remove_leaks_morphology(
        mask_refined,
        radius=level_set_config["leak_removal_radius"],
        use_gpu=bool(use_gpu),
    )
    aorta_mask = keep_largest_component(aorta_mask, gpu=bool(use_gpu))
    aorta_mask = aorta_mask.astype(np.uint8)

    save_npy_cache(aorta_mask, mask_path, enabled=save_cache)

    return aorta_mask


def detect_and_evaluate_ostia(
    aorta_mask: Any,
    vesselness_ostios: Any,
    label: Any,
    scaled_spacing: Sequence[float],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Detecta os óstios e avalia correção/tolerância."""
    dx, dy, dz = scaled_spacing
    ostia_config = config["OSTIA_DETECTION"]
    ostia_left, ostia_right = find_ostia(
        aorta_mask,
        vesselness_ostios,
        spacing=(dy, dx, dz),
        top_n=ostia_config["top_n"],
        max_z_diff_mm=ostia_config["max_z_diff_mm"],
        lower_fraction=ostia_config["lower_fraction"],
        min_center_distance_factor=ostia_config["min_center_distance_factor"],
        min_lateral_factor=ostia_config["min_lateral_factor"],
        erosion_radius=ostia_config["erosion_radius"],
        use_gpu=config.get("USE_GPU", False),
    )

    label_artery = (label == 1).astype(np.uint8)
    left_info = check_ostium_intersection(
        ostia_left, label_artery, spacing=(dy, dx, dz), ostium_name="Óstio esquerdo"
    )
    right_info = check_ostium_intersection(
        ostia_right, label_artery, spacing=(dy, dx, dz), ostium_name="Óstio direito"
    )

    tolerable = config["OSTIA_VALIDATION"]["distance_threshold_mm"]
    both_correct = left_info["intersects"] and right_info["intersects"]
    both_tolerable_inclusive = (
        left_info["intersects"] or left_info["physical_dist"] <= tolerable
    ) and (right_info["intersects"] or right_info["physical_dist"] <= tolerable)

    return {
        "ostia_left": ostia_left,
        "ostia_right": ostia_right,
        "label_artery": label_artery,
        "left_info": left_info,
        "right_info": right_info,
        "both_correct": both_correct,
        "both_tolerable": both_tolerable_inclusive and (not both_correct),
    }
