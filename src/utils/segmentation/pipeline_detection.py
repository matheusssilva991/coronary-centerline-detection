"""Etapas de detecção/segmentação da aorta e avaliação dos óstios."""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from .aorta_localization import detect_aorta_circles, filter_aorta_circle_trajectory
from .aorta_segmentation import (
    calculate_circle_mask_metrics,
    calculate_slice_area_jump_p95,
    prepare_level_set_evolution,
    remove_leaks_morphology,
    restrict_mask_to_circle_trajectory,
)
from .ostia_detection import check_ostium_intersection, find_ostia
from ..processing.binary_operations import keep_largest_component


@dataclass(frozen=True)
class AortaSegmentationResult:
    """Máscara final da aorta acompanhada dos diagnósticos do level set."""

    mask: np.ndarray
    diagnostics: Dict[str, Any]


def locate_aorta_circles(
    lcc_image: Any,
    downscale_factors: Sequence[int],
    scaled_spacing: Sequence[float],
    circle_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Detecta círculos da aorta sempre na CPU."""
    del downscale_factors  # Os raios já chegam escalados na configuração efetiva.
    dx, dy, _ = scaled_spacing
    hough_radii = np.arange(
        circle_config["radii_start_px"],
        circle_config["radii_end_px"],
        circle_config.get("radius_step_px", 1),
    )
    pixel_spacing = (dx + dy) / 2.0

    # Localiza a aorta fatia a fatia por candidatos circulares.
    return detect_aorta_circles(
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
    )


def filter_located_aorta_circles(
    detected_circles: Sequence[Dict[str, Any]],
    scaled_spacing: Sequence[float],
    image_slice_count: int,
    circle_config: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Aplica o filtro robusto opcional à trajetória localizada pela Hough."""
    dx, dy, _ = scaled_spacing
    pixel_spacing = (float(dx) + float(dy)) / 2.0
    return filter_aorta_circle_trajectory(
        detected_circles,
        pixel_spacing,
        image_slice_count,
        circle_config.get("trajectory_filter", {}),
    )


def _postprocess_aorta_mask(
    mask_refined: Any,
    detected_circles: Sequence[Dict[str, Any]],
    level_set_config: Dict[str, Any],
) -> np.ndarray:
    """Aplica abertura, envelope opcional e maior componente conectada."""
    aorta_mask = remove_leaks_morphology(
        mask_refined,
        radius=level_set_config["leak_removal_radius"],
        use_gpu=False,
    )
    trajectory_radius_factor = level_set_config.get("trajectory_radius_factor")
    if trajectory_radius_factor is not None:
        # Limita vazamentos ao tubo acompanhado pelos círculos da aorta.
        aorta_mask = restrict_mask_to_circle_trajectory(
            aorta_mask,
            detected_circles,
            radius_factor=float(trajectory_radius_factor),
            axial_margin_slices=int(
                level_set_config.get("trajectory_axial_margin_slices", 0)
            ),
        )
    aorta_mask = keep_largest_component(aorta_mask, gpu=False)
    return np.asarray(aorta_mask, dtype=np.uint8)


def _fixed_level_set_result(
    lcc_image: Any,
    detected_circles: Sequence[Dict[str, Any]],
    level_set_config: Dict[str, Any],
) -> AortaSegmentationResult:
    """Executa o level set com o número configurado de iterações."""
    num_iter = int(level_set_config["num_iter"])
    if detected_circles:
        context = prepare_level_set_evolution(
            lcc_image,
            detected_circles,
            radius_reduction_factor=level_set_config["radius_reduction_factor"],
            roi_margin=level_set_config.get("roi_margin", 10),
            use_roi=level_set_config.get("use_roi", True),
            alpha=level_set_config.get("alpha", 1000),
            sigma=level_set_config.get("sigma", 2),
            use_gpu=False,
        )
        initial_voxel_count = int(np.count_nonzero(context.current_mask))
        mask_refined = context.evolve(
            num_iter,
            balloon=level_set_config["balloon"],
            smoothing=level_set_config["smoothing"],
            threshold=level_set_config.get("threshold", "auto"),
        )
    else:
        initial_voxel_count = 0
        mask_refined = np.zeros_like(lcc_image, dtype=np.uint8)

    # Registra o volume bruto antes de aplicar a correção morfológica.
    raw_voxel_count = int(np.count_nonzero(mask_refined))
    aorta_mask = _postprocess_aorta_mask(
        mask_refined,
        detected_circles,
        level_set_config,
    )
    circle_metrics = calculate_circle_mask_metrics(aorta_mask, detected_circles)
    slice_area_jump_p95 = calculate_slice_area_jump_p95(aorta_mask)
    image_voxel_count = int(mask_refined.size)
    return AortaSegmentationResult(
        mask=aorta_mask,
        diagnostics={
            "aorta_level_set_initial_voxel_count": initial_voxel_count,
            "aorta_level_set_raw_voxel_count": raw_voxel_count,
            "aorta_level_set_initial_volume_fraction": (
                initial_voxel_count / image_voxel_count
            ),
            "aorta_level_set_raw_volume_fraction": (
                raw_voxel_count / image_voxel_count
            ),
            "aorta_level_set_iterations_used": num_iter,
            "aorta_level_set_circle_fill_q25": circle_metrics["circle_fill_q25"],
            "aorta_level_set_circle_area_ratio_p90": circle_metrics[
                "circle_area_ratio_p90"
            ],
            "aorta_slice_area_jump_p95": slice_area_jump_p95,
        },
    )


def segment_aorta_with_diagnostics(
    lcc_image: Any,
    detected_circles: List[Dict[str, Any]],
    level_set_config: Dict[str, Any],
    use_gpu: bool = False,
) -> AortaSegmentationResult:
    """Segmenta a aorta e retorna métricas da evolução fixa."""
    del use_gpu  # MorphGAC e pós-processamento permanecem na CPU.
    return _fixed_level_set_result(lcc_image, detected_circles, level_set_config)


def segment_aorta(
    lcc_image: Any,
    detected_circles: List[Dict[str, Any]],
    level_set_config: Dict[str, Any],
    use_gpu: bool = False,
) -> Any:
    """Segmenta a aorta preservando a API que retorna apenas a máscara."""
    return segment_aorta_with_diagnostics(
        lcc_image,
        detected_circles,
        level_set_config,
        use_gpu=use_gpu,
    ).mask


def detect_and_evaluate_ostia(
    aorta_mask: Any,
    vesselness_ostios: Any,
    label: Any,
    scaled_spacing: Sequence[float],
    config: Dict[str, Any],
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
        pair_distance_mode=ostia_config.get("pair_distance_mode", "voxel_xyz"),
    )

    label_artery = (label == 1).astype(np.uint8)
    left_coords = tuple(int(value) for value in ostia_left)
    right_coords = (
        tuple(int(value) for value in ostia_right) if ostia_right is not None else None
    )
    left_info = check_ostium_intersection(
        left_coords,
        label_artery,
        spacing=(dy, dx, dz),
        ostium_name="Óstio esquerdo",
    )
    right_info = check_ostium_intersection(
        right_coords,
        label_artery,
        spacing=(dy, dx, dz),
        ostium_name="Óstio direito",
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
