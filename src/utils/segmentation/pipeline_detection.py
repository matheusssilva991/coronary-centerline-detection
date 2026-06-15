"""Etapas de detecção/segmentação da aorta e avaliação dos óstios."""

from typing import Any, Dict, List, Sequence
from pathlib import Path

import numpy as np

from .aorta_localization import detect_aorta_circles
from .aorta_segmentation import level_set_segmentation, remove_leaks_morphology
from .ostia_detection import check_ostium_intersection, find_ostia
from ..processing.binary_operations import keep_largest_component
from ..project.cache import load_json_cache, load_npy_cache, save_json_cache, save_npy_cache


def get_or_detect_aorta_circles(
    img_id: str,
    lcc_image: Any,
    downscale_factors: Sequence[int],
    scaled_spacing: Sequence[float],
    circle_config: Dict[str, Any],
    base_save_path: str,
    load_cache: bool = False,
    save_cache: bool = False,
) -> List[Dict[str, Any]]:
    """Carrega ou detecta círculos da aorta sempre na CPU."""
    json_path = (
        Path(base_save_path)
        / "detected_circles_cpu"
        / f"{img_id}_detected_circles.json"
    )

    # Reutiliza círculos salvos quando a execução está com cache habilitado.
    cached_circles = load_json_cache(json_path, enabled=load_cache)
    if cached_circles is not None:
        return cached_circles

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
        out_of_tolerance_as_miss=circle_config.get(
            "out_of_tolerance_as_miss", False
        ),
        candidate_selection_strategy=circle_config.get(
            "candidate_selection_strategy", "closest"
        ),
        candidate_score_accum_weight=circle_config.get(
            "candidate_score_accum_weight", 1.0
        ),
        candidate_score_distance_weight=circle_config.get(
            "candidate_score_distance_weight", 1.0
        ),
        candidate_score_radius_weight=circle_config.get(
            "candidate_score_radius_weight", 1.0
        ),
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

    # Reutiliza a máscara da aorta quando disponível.
    cached_mask = load_npy_cache(mask_path, enabled=load_cache)
    if cached_mask is not None:
        return cached_mask

    # Segmenta a aorta usando os círculos como inicialização do level set.
    mask_refined = level_set_segmentation(
        lcc_image,
        detected_circles,
        radius_reduction_factor=level_set_config["radius_reduction_factor"],
        num_iter=level_set_config["num_iter"],
        balloon=level_set_config["balloon"],
        smoothing=level_set_config["smoothing"],
        use_gpu=False,
    )
    # Remove vazamentos e mantém apenas o maior componente da aorta.
    aorta_mask = remove_leaks_morphology(
        mask_refined,
        radius=level_set_config["leak_removal_radius"],
        use_gpu=False,
    )
    aorta_mask = keep_largest_component(aorta_mask, gpu=False)
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
    )

    # Extrai apenas a classe arterial do label para validar os óstios.
    label_artery = (label == 1).astype(np.uint8)
    # Mede se cada óstio intersecta a artéria ou fica dentro da tolerância em mm.
    left_info = check_ostium_intersection(
        ostia_left, label_artery, spacing=(dy, dx, dz), ostium_name="Óstio esquerdo"
    )
    right_info = check_ostium_intersection(
        ostia_right, label_artery, spacing=(dy, dx, dz), ostium_name="Óstio direito"
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
