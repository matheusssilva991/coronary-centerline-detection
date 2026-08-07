"""Etapas de detecção/segmentação da aorta e avaliação dos óstios."""

from typing import Any, Dict, List, Sequence

import numpy as np

from .aorta_localization import detect_aorta_circles
from .aorta_segmentation import (
    correct_anomalous_aorta_slices,
    level_set_segmentation,
    remove_leaks_morphology,
    restrict_mask_to_circle_trajectory,
)
from .ostia_detection import check_ostium_intersection, find_ostia
from ..processing.binary_operations import keep_largest_component


def locate_aorta_circles(
    lcc_image: Any,
    downscale_factors: Sequence[int],
    scaled_spacing: Sequence[float],
    circle_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Detecta círculos da aorta sempre na CPU."""
    # Constrói os raios da Hough em pixels já compatíveis com a resolução atual.
    dx, dy, _ = scaled_spacing
    radii_start = circle_config["radii_start_px"]
    radii_end = circle_config["radii_end_px"]
    radius_step = circle_config.get("radius_step_px", 1)
    hough_radii = np.arange(radii_start, radii_end, radius_step)
    pixel_spacing = (dx + dy) / 2.0

    # Localiza a aorta fatia a fatia por candidatos circulares.
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
        interpolate_missed_circles=circle_config.get(
            "interpolate_missed_circles", True
        ),
        early_track_recovery=circle_config.get("early_track_recovery", True),
        early_recovery_search_slices=circle_config.get(
            "early_recovery_search_slices", 8
        ),
        early_recovery_min_circles=circle_config.get("early_recovery_min_circles", 10),
        early_recovery_require_min_circles=circle_config.get(
            "early_recovery_require_min_circles", False
        ),
    )
    return detected_circles


def segment_aorta(
    lcc_image: Any,
    detected_circles: List[Dict[str, Any]],
    level_set_config: Dict[str, Any],
    use_gpu: bool = False,
) -> Any:
    """Segmenta a aorta com level set e pós-processamento."""
    # Segmenta a aorta usando os círculos como inicialização do level set.
    mask_refined = level_set_segmentation(
        lcc_image,
        detected_circles,
        radius_reduction_factor=level_set_config["radius_reduction_factor"],
        num_iter=level_set_config["num_iter"],
        balloon=level_set_config["balloon"],
        smoothing=level_set_config["smoothing"],
        threshold=level_set_config.get("threshold", "auto"),
        roi_margin=level_set_config.get("roi_margin", 10),
        use_roi=level_set_config.get("use_roi", True),
        alpha=level_set_config.get("alpha", 1000),
        sigma=level_set_config.get("sigma", 2),
        use_gpu=False,
    )
    # Remove vazamentos e mantém apenas o maior componente da aorta.
    aorta_mask = remove_leaks_morphology(
        mask_refined,
        radius=level_set_config["leak_removal_radius"],
        use_gpu=False,
    )
    trajectory_radius_factor = level_set_config.get("trajectory_radius_factor")
    if trajectory_radius_factor is not None:
        # Limita vazamentos do level set ao tubo acompanhado pelos círculos.
        aorta_mask = restrict_mask_to_circle_trajectory(
            aorta_mask,
            detected_circles,
            radius_factor=float(trajectory_radius_factor),
        )
    area_ratio_threshold = level_set_config.get("trajectory_area_ratio_threshold")
    if area_ratio_threshold is not None:
        # Corrige somente fatias com área incompatível com o círculo rastreado.
        aorta_mask = correct_anomalous_aorta_slices(
            aorta_mask,
            detected_circles,
            area_ratio_threshold=float(area_ratio_threshold),
            radius_factor=float(
                level_set_config.get("trajectory_correction_radius_factor", 1.75)
            ),
        )
    aorta_mask = keep_largest_component(aorta_mask, gpu=False)
    aorta_mask = aorta_mask.astype(np.uint8)

    return aorta_mask


def detect_and_evaluate_ostia(
    aorta_mask: Any,
    vesselness_ostios: Any,
    label: Any,
    scaled_spacing: Sequence[float],
    config: Dict[str, Any],
    detected_circles: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Detecta os óstios e avalia correção/tolerância contra o label."""
    dx, dy, dz = scaled_spacing
    ostia_config = config["OSTIA_DETECTION"]
    # Seleciona os dois óstios na superfície inferior da aorta usando vesselness.
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
        surface_mode=ostia_config.get("surface_mode", "erosion"),
        surface_thickness_mm=ostia_config.get("surface_thickness_mm", 2.0),
        candidate_score_mode=ostia_config.get("candidate_score_mode", "voxel"),
        candidate_score_radius=ostia_config.get("candidate_score_radius", 2),
        candidate_local_percentile=ostia_config.get("candidate_local_percentile", 90.0),
        candidate_point_weight=ostia_config.get("candidate_point_weight", 0.7),
        candidate_suppression_radius_mm=ostia_config.get(
            "candidate_suppression_radius_mm", 0.0
        ),
        pair_selection_mode=ostia_config.get("pair_selection_mode", "greedy"),
        joint_pair_top_k=ostia_config.get("joint_pair_top_k", 100),
        bilateral_top_k_per_side=ostia_config.get("bilateral_top_k_per_side", 50),
        detected_circles=detected_circles,
        pair_distance_mode=ostia_config.get("pair_distance_mode", "voxel_xyz"),
    )

    # Extrai apenas a classe arterial do label para validar os óstios.
    label_artery = (label == 1).astype(np.uint8)
    left_coords = tuple(int(value) for value in ostia_left)
    right_coords = (
        tuple(int(value) for value in ostia_right) if ostia_right is not None else None
    )
    # Mede se cada óstio intersecta a artéria ou fica dentro da tolerância em mm.
    left_info = check_ostium_intersection(
        left_coords, label_artery, spacing=(dy, dx, dz), ostium_name="Óstio esquerdo"
    )
    right_info = check_ostium_intersection(
        right_coords, label_artery, spacing=(dy, dx, dz), ostium_name="Óstio direito"
    )

    # Consolida o status dos óstios em critérios estrito e tolerável.
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
