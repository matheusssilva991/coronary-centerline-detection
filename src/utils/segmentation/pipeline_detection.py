"""Etapas de detecção/segmentação da aorta e avaliação dos óstios."""

import json
import os
from typing import Any, Dict, List, Sequence

import numpy as np

from .aorta_localization import detect_aorta_circles
from .aorta_segmentation import level_set_segmentation, remove_leaks_morphology
from .ostia_detection import check_ostium_intersection, find_ostia
from ..processing.binary_operations import keep_largest_component
from ..utils import save_npy_array


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
    saved_dir_circles = f"{base_save_path}/detected_circles"
    json_path = os.path.join(saved_dir_circles, f"{img_id}_detected_circles.json")

    if os.path.exists(json_path) and load_cache:
        with open(json_path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)

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
    if save_cache:
        os.makedirs(saved_dir_circles, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as file_handle:
            json.dump(detected_circles, file_handle, indent=4)

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
    saved_dir_aorta = f"{base_save_path}/segmented_aorta"
    mask_path = os.path.join(saved_dir_aorta, f"{img_id}_mask_aorta.npy")

    if os.path.exists(mask_path) and load_cache:
        return np.load(mask_path)

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
        mask_refined, radius=level_set_config["leak_removal_radius"]
    )
    aorta_mask = keep_largest_component(aorta_mask)
    aorta_mask = aorta_mask.astype(np.uint8)

    if save_cache:
        os.makedirs(saved_dir_aorta, exist_ok=True)
        save_npy_array(aorta_mask, mask_path)

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

