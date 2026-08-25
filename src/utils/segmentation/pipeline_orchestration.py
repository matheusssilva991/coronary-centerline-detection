"""Orquestração do pipeline de segmentação coronária."""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..project.results import (
    duration_breakdown,
    get_batch_result_file,
    load_batch_timing_records,
    save_batch_timing_record,
    save_results,
    summarize_batch_timing_records,
)
from .pipeline_arteries import segment_arteries_from_ostia
from .pipeline_detection import (
    detect_and_evaluate_ostia,
    filter_located_aorta_circles,
    locate_aorta_circles,
    segment_aorta_with_diagnostics,
)
from .pipeline_preprocessing import compute_vesselness, load_and_preprocess_image
from .pipeline_visuals import save_segmentation_visual

logger = logging.getLogger(__name__)


IMAGE_RESULT_DEFAULTS = {
    "ostia_left": None,
    "ostia_right": None,
    "artery_voxels": None,
    "artery_voxels_before_morphology": None,
    "artery_voxels_after_morphology": None,
    "dice_artery": None,
    "dice_artery_before_morphology": None,
    "dice_artery_after_morphology": None,
    "dice_artery_morphology_delta": None,
    "ostia_found": False,
    "ostia_status": "not_evaluated",
    "segmentation_attempted": False,
    "proceeded_with_bad_ostia": False,
    "skip_reason": None,
    "ostia_error": None,
    "ostia_surface_mode": None,
    "ostia_surface_thickness_mm": None,
    "ostia_candidate_score_mode": None,
    "ostia_pair_selection_mode": None,
    "left_intersects": False,
    "right_intersects": False,
    "left_dist_voxels": None,
    "right_dist_voxels": None,
    "left_dist_mm": None,
    "right_dist_mm": None,
    "both_correct": False,
    "both_tolerable": False,
    "threshold_mode": None,
    "fuzzy_mask_strategy": None,
    "min_threshold": None,
    "max_threshold": None,
    "lower_threshold_method": None,
    "lower_threshold_percentile": None,
    "threshold_voxels": None,
    "lcc_voxels": None,
    "image_slice_count": None,
    "image_voxels": None,
    "aorta_circle_count": None,
    "aorta_detected_circle_count": None,
    "aorta_interpolated_circle_count": None,
    "aorta_circle_first_slice": None,
    "aorta_circle_last_slice": None,
    "aorta_circle_coverage": None,
    "aorta_circle_radius_min_px": None,
    "aorta_circle_radius_max_px": None,
    "aorta_circle_radius_mean_px": None,
    "aorta_circle_radius_median_px": None,
    "aorta_circle_radius_std_px": None,
    "aorta_circle_radius_p10_px": None,
    "aorta_circle_radius_p90_px": None,
    "aorta_circle_radius_min_mm": None,
    "aorta_circle_radius_max_mm": None,
    "aorta_circle_radius_mean_mm": None,
    "aorta_circle_radius_median_mm": None,
    "aorta_circle_radius_std_mm": None,
    "aorta_circle_radius_p10_mm": None,
    "aorta_circle_radius_p90_mm": None,
    "aorta_detected_circle_radius_median_mm": None,
    "aorta_interpolated_circle_radius_median_mm": None,
    "aorta_circle_radius_first_mm": None,
    "aorta_circle_radius_last_mm": None,
    "aorta_circle_radius_max_step_change_mm": None,
    "aorta_circle_radius_p90_step_change_mm": None,
    "aorta_circle_mean_hough_accumulator": None,
    "aorta_circle_lower_radius_bound_fraction": None,
    "aorta_circle_upper_radius_bound_fraction": None,
    "aorta_recovered_initialization": False,
    "aorta_circle_filter_method": "none",
    "aorta_circle_filter_applied": False,
    "aorta_circle_original_count": None,
    "aorta_circle_used_count": None,
    "aorta_circle_filter_interpolated_count": 0,
    "aorta_circle_filter_trimmed_tail_count": 0,
    "aorta_circle_filter_trim_start_slice": None,
    "aorta_circle_filter_original_coverage": None,
    "aorta_circle_filter_used_coverage": None,
    "aorta_circle_filter_reason": None,
    "aorta_circle_filter_fallback_enabled": False,
    "aorta_circle_filter_accepted": False,
    "aorta_circle_filter_rejected": False,
    "aorta_circle_filter_rejection_reason": None,
    "aorta_circle_filter_candidate_controller_state": None,
    "aorta_circle_filter_fallback_controller_state": None,
    "aorta_circle_filter_candidate_mask_voxel_count": None,
    "aorta_mask_voxels": None,
    "aorta_segmented_slice_count": None,
    "aorta_voxels_per_segmented_slice": None,
    "aorta_volume_fraction": None,
    "aorta_level_set_mode": None,
    "aorta_level_set_iterations_used": None,
    "aorta_level_set_stop_reason": None,
    "aorta_level_set_checkpoint_count": None,
    "aorta_level_set_rolled_back": False,
    "aorta_level_set_mask_change_fraction": None,
    "aorta_level_set_voxels_per_segmented_slice": None,
    "aorta_level_set_circle_fill_q25": None,
    "aorta_level_set_circle_area_ratio_p90": None,
    "aorta_level_set_leak_suspected": False,
    "aorta_level_set_localization_suspected": False,
    "aorta_level_set_leak_signal_count": 0,
    "aorta_level_set_trigger_iteration": None,
    "aorta_level_set_trigger_volume_fraction": None,
    "aorta_level_set_trigger_relative_growth": None,
    "aorta_level_set_trigger_mask_change_fraction": None,
    "aorta_level_set_trigger_circle_fill_q25": None,
    "aorta_level_set_trigger_circle_area_ratio_p90": None,
    "aorta_level_set_correction_applied": False,
    "aorta_level_set_correction_method": None,
    "aorta_level_set_refinement_applied": False,
    "aorta_level_set_refinement_accepted": False,
    "aorta_level_set_refinement_iterations": None,
    "aorta_level_set_refinement_balloon": None,
    "aorta_level_set_refinement_smoothing": None,
    "aorta_level_set_refinement_transition_mode": None,
    "aorta_level_set_refinement_anomaly_margin_slices": None,
    "aorta_level_set_refinement_volume_loss_fraction": None,
    "aorta_level_set_slice_area_jump_p95_before": None,
    "aorta_level_set_slice_area_jump_p95_after": None,
    "aorta_level_set_refinement_rejection_reason": None,
    "aorta_level_set_controller_state": None,
    "aorta_level_set_profile_used": None,
    "aorta_level_set_rollback_iteration": None,
    "aorta_level_set_circle_confidence_signal_count": 0,
    "aorta_level_set_alternative_attempted": False,
    "aorta_level_set_alternative_accepted": False,
    "aorta_level_set_conservative_attempted": False,
    "aorta_level_set_conservative_accepted": False,
    "aorta_level_set_permissive_attempted": False,
    "aorta_level_set_permissive_accepted": False,
    "aorta_level_set_nominal_volume_fraction": None,
    "aorta_level_set_nominal_circle_fill_q25": None,
    "aorta_level_set_nominal_circle_area_ratio_p90": None,
    "aorta_level_set_final_volume_fraction": None,
    "aorta_level_set_decision_reason": None,
}


def _new_image_result(img_id):
    """Cria o registro completo com os defaults de uma imagem."""
    return {"IMG_ID": img_id, **IMAGE_RESULT_DEFAULTS}


def _preprocessing_result_fields(
    preprocessing_details,
    image_slice_count,
    image_voxel_count=None,
):
    """Converte detalhes do pré-processamento nos campos persistidos."""
    detail_keys = (
        "threshold_mode",
        "fuzzy_mask_strategy",
        "min_threshold",
        "max_threshold",
        "lower_threshold_method",
        "lower_threshold_percentile",
        "threshold_voxels",
        "lcc_voxels",
    )
    fields = {key: preprocessing_details.get(key) for key in detail_keys}
    fields["image_slice_count"] = int(image_slice_count)
    fields["image_voxels"] = (
        int(image_voxel_count) if image_voxel_count is not None else None
    )
    return fields


def _describe_circle_radii(values, unit):
    """Calcula estatísticas robustas para uma sequência de raios."""
    prefix = "aorta_circle_radius_"
    suffix = f"_{unit}"
    keys = ("min", "max", "mean", "median", "std", "p10", "p90")
    if not values:
        return {f"{prefix}{key}{suffix}": None for key in keys}

    radii = np.asarray(values, dtype=float)
    return {
        f"{prefix}min{suffix}": float(np.min(radii)),
        f"{prefix}max{suffix}": float(np.max(radii)),
        f"{prefix}mean{suffix}": float(np.mean(radii)),
        f"{prefix}median{suffix}": float(np.median(radii)),
        f"{prefix}std{suffix}": float(np.std(radii)),
        f"{prefix}p10{suffix}": float(np.percentile(radii, 10)),
        f"{prefix}p90{suffix}": float(np.percentile(radii, 90)),
    }


def _median_or_none(values):
    """Retorna a mediana como float ou ``None`` para coleção vazia."""
    return float(np.median(values)) if values else None


def _mean_or_none(values):
    """Retorna a média como float ou ``None`` para coleção vazia."""
    return float(np.mean(values)) if values else None


def summarize_aorta_circles(
    detected_circles,
    image_slice_count,
    scaled_spacing=None,
    circle_config=None,
):
    """Resume cobertura, raios e continuidade do rastreamento da aorta."""
    circle_slices = [
        int(circle["slice_index"])
        for circle in detected_circles
        if circle.get("slice_index") is not None
    ]
    interpolated_count = sum(
        bool(circle.get("interpolated", False)) for circle in detected_circles
    )
    circle_count = len(detected_circles)
    valid_circles = [
        circle
        for circle in detected_circles
        if circle.get("radius") is not None
        and math.isfinite(float(circle["radius"]))
    ]
    radii_px = [float(circle["radius"]) for circle in valid_circles]

    # Converte os raios para a unidade física usada nas comparações entre resoluções.
    pixel_spacing = None
    if scaled_spacing is not None and len(scaled_spacing) >= 2:
        candidate_spacing = (float(scaled_spacing[0]) + float(scaled_spacing[1])) / 2
        if math.isfinite(candidate_spacing) and candidate_spacing > 0:
            pixel_spacing = candidate_spacing
    radii_mm = (
        [radius * pixel_spacing for radius in radii_px]
        if pixel_spacing is not None
        else []
    )

    detected_valid = [
        circle
        for circle in valid_circles
        if not bool(circle.get("interpolated", False))
    ]
    interpolated_valid = [
        circle for circle in valid_circles if bool(circle.get("interpolated", False))
    ]
    detected_radii_px = [float(circle["radius"]) for circle in detected_valid]
    detected_radii_mm = (
        [radius * pixel_spacing for radius in detected_radii_px]
        if pixel_spacing is not None
        else []
    )
    interpolated_radii_mm = (
        [float(circle["radius"]) * pixel_spacing for circle in interpolated_valid]
        if pixel_spacing is not None
        else []
    )

    # Mede mudanças de raio por fatia, inclusive quando há lacunas no rastreamento.
    ordered_radii = sorted(
        (
            int(circle["slice_index"]),
            float(circle["radius"]) * pixel_spacing,
        )
        for circle in valid_circles
        if pixel_spacing is not None and circle.get("slice_index") is not None
    )
    step_changes_mm = []
    for (previous_z, previous_radius), (current_z, current_radius) in zip(
        ordered_radii,
        ordered_radii[1:],
    ):
        slice_distance = max(abs(current_z - previous_z), 1)
        step_changes_mm.append(abs(current_radius - previous_radius) / slice_distance)

    # Saturação nos extremos indica que o intervalo da Hough pode estar truncado.
    lower_bound_fraction = None
    upper_bound_fraction = None
    if circle_config and detected_radii_px:
        radius_step = float(circle_config.get("radius_step_px", 1))
        lower_bound = float(circle_config["radii_start_px"])
        hough_radii = np.arange(
            lower_bound,
            float(circle_config["radii_end_px"]),
            radius_step,
        )
        if hough_radii.size:
            upper_bound = float(hough_radii[-1])
            detected_array = np.asarray(detected_radii_px)
            lower_bound_fraction = float(np.mean(np.isclose(detected_array, lower_bound)))
            upper_bound_fraction = float(np.mean(np.isclose(detected_array, upper_bound)))

    accumulators = [
        float(circle["accum"])
        for circle in detected_valid
        if circle.get("accum") is not None
        and math.isfinite(float(circle["accum"]))
    ]
    summary = {
        "aorta_circle_count": circle_count,
        "aorta_detected_circle_count": circle_count - interpolated_count,
        "aorta_interpolated_circle_count": interpolated_count,
        "aorta_circle_first_slice": min(circle_slices) if circle_slices else None,
        "aorta_circle_last_slice": max(circle_slices) if circle_slices else None,
        "aorta_circle_coverage": (
            circle_count / image_slice_count if image_slice_count else None
        ),
        "aorta_recovered_initialization": any(
            bool(circle.get("recovered_initialization", False))
            for circle in detected_circles
        ),
        "aorta_detected_circle_radius_median_mm": _median_or_none(
            detected_radii_mm
        ),
        "aorta_interpolated_circle_radius_median_mm": _median_or_none(
            interpolated_radii_mm
        ),
        "aorta_circle_radius_first_mm": (
            ordered_radii[0][1] if ordered_radii else None
        ),
        "aorta_circle_radius_last_mm": (
            ordered_radii[-1][1] if ordered_radii else None
        ),
        "aorta_circle_radius_max_step_change_mm": (
            max(step_changes_mm) if step_changes_mm else None
        ),
        "aorta_circle_radius_p90_step_change_mm": (
            float(np.percentile(step_changes_mm, 90)) if step_changes_mm else None
        ),
        "aorta_circle_mean_hough_accumulator": _mean_or_none(accumulators),
        "aorta_circle_lower_radius_bound_fraction": lower_bound_fraction,
        "aorta_circle_upper_radius_bound_fraction": upper_bound_fraction,
    }
    summary.update(_describe_circle_radii(radii_px, "px"))
    summary.update(_describe_circle_radii(radii_mm, "mm"))
    return summary


def summarize_aorta_volume(aorta_mask, image_voxel_count):
    """Calcula a ocupação total e por fatia da máscara final da aorta."""
    aorta_mask_voxels = int(aorta_mask.sum())
    segmented_slice_count = int(aorta_mask.any(axis=(0, 1)).sum())
    return {
        "aorta_mask_voxels": aorta_mask_voxels,
        "aorta_segmented_slice_count": segmented_slice_count,
        "aorta_voxels_per_segmented_slice": (
            aorta_mask_voxels / segmented_slice_count
            if segmented_slice_count
            else None
        ),
        "aorta_volume_fraction": (
            aorta_mask_voxels / image_voxel_count if image_voxel_count else None
        ),
    }


def _segment_aorta_with_circle_filter_fallback(
    lcc_image,
    original_circles,
    filtered_circles,
    filter_diagnostics,
    image_slice_count,
    scaled_spacing,
    config,
):
    """Segmenta a aorta e rejeita opcionalmente uma trajetória filtrada ruim."""
    circle_config = config["CIRCLE_DETECTION"]
    level_set_config = config["LEVEL_SET"]
    trajectory_filter = circle_config.get("trajectory_filter", {})
    fallback_enabled = bool(
        trajectory_filter.get("reject_oversegmented_result", False)
    )

    # Primeiro avalia normalmente a trajetória resultante do filtro robusto.
    filtered_summary = summarize_aorta_circles(
        filtered_circles,
        image_slice_count,
        scaled_spacing,
        circle_config,
    )
    candidate = segment_aorta_with_diagnostics(
        lcc_image,
        filtered_circles,
        level_set_config,
        use_gpu=config.get("USE_GPU", False),
        circle_summary=filtered_summary,
    )
    candidate_state = candidate.diagnostics.get("aorta_level_set_controller_state")
    filter_applied = bool(filter_diagnostics.get("aorta_circle_filter_applied"))
    reject_candidate = (
        fallback_enabled and filter_applied and candidate_state == "oversegmented"
    )
    filter_diagnostics.update(
        {
            "aorta_circle_filter_fallback_enabled": fallback_enabled,
            "aorta_circle_filter_accepted": filter_applied and not reject_candidate,
            "aorta_circle_filter_rejected": reject_candidate,
            "aorta_circle_filter_rejection_reason": (
                "filtered_mask_oversegmented" if reject_candidate else None
            ),
            "aorta_circle_filter_candidate_controller_state": candidate_state,
            "aorta_circle_filter_fallback_controller_state": None,
            "aorta_circle_filter_candidate_mask_voxel_count": int(
                np.asarray(candidate.mask).sum()
            ),
        }
    )
    if not reject_candidate:
        return filtered_circles, candidate, filter_diagnostics

    # O fallback repete apenas o level set; Hough e pré-processamento são reutilizados.
    original_summary = summarize_aorta_circles(
        original_circles,
        image_slice_count,
        scaled_spacing,
        circle_config,
    )
    fallback = segment_aorta_with_diagnostics(
        lcc_image,
        original_circles,
        level_set_config,
        use_gpu=config.get("USE_GPU", False),
        circle_summary=original_summary,
    )
    filter_diagnostics.update(
        {
            "aorta_circle_used_count": len(original_circles),
            "aorta_circle_filter_used_coverage": (
                len(original_circles) / image_slice_count
                if image_slice_count
                else None
            ),
            "aorta_circle_filter_fallback_controller_state": fallback.diagnostics.get(
                "aorta_level_set_controller_state"
            ),
        }
    )
    return original_circles, fallback, filter_diagnostics


def _circle_result_fields(
    detected_circles,
    image_slice_count,
    scaled_spacing=None,
    circle_config=None,
):
    """Compatibilidade interna para o antigo nome do resumo de círculos."""
    return summarize_aorta_circles(
        detected_circles,
        image_slice_count,
        scaled_spacing,
        circle_config,
    )


def _ostia_result_fields(ostia_eval):
    """Converte a avaliação dos óstios nos campos persistidos."""
    both_correct = bool(ostia_eval["both_correct"])
    both_tolerable = bool(ostia_eval["both_tolerable"])
    if both_correct:
        status = "both_correct"
    elif both_tolerable:
        status = "both_tolerable"
    else:
        status = "found_but_wrong"

    return {
        "ostia_left": (
            tuple(map(int, ostia_eval["ostia_left"]))
            if ostia_eval["ostia_left"] is not None
            else None
        ),
        "ostia_right": (
            tuple(map(int, ostia_eval["ostia_right"]))
            if ostia_eval["ostia_right"] is not None
            else None
        ),
        "ostia_found": True,
        "left_intersects": ostia_eval["left_info"]["intersects"],
        "right_intersects": ostia_eval["right_info"]["intersects"],
        "left_dist_voxels": ostia_eval["left_info"]["euclidean_dist"],
        "right_dist_voxels": ostia_eval["right_info"]["euclidean_dist"],
        "left_dist_mm": ostia_eval["left_info"]["physical_dist"],
        "right_dist_mm": ostia_eval["right_info"]["physical_dist"],
        "both_correct": both_correct,
        "both_tolerable": both_tolerable,
        "ostia_status": status,
        "proceeded_with_bad_ostia": not (both_correct or both_tolerable),
    }


def process_image(img_id, config, base_path, visual_output_dir=None):
    """Processa uma imagem completa e retorna o dicionário de resultados.

    Fluxo por imagem:
    1. carrega e pré-processa o volume;
    2. calcula vesselness para detecção dos óstios;
    3. detecta círculos e segmenta a aorta;
    4. seleciona/avalia os óstios;
    5. segmenta as artérias a partir dos óstios.
    """
    result = _new_image_result(img_id)
    result["aorta_level_set_mode"] = config.get("LEVEL_SET", {}).get(
        "iteration_mode", "fixed"
    )
    ostia_config = config.get("OSTIA_DETECTION", {})
    result.update(
        {
            "ostia_surface_mode": ostia_config.get("surface_mode", "erosion"),
            "ostia_surface_thickness_mm": ostia_config.get(
                "surface_thickness_mm", 2.0
            ),
            "ostia_candidate_score_mode": ostia_config.get(
                "candidate_score_mode", "voxel"
            ),
            "ostia_pair_selection_mode": ostia_config.get(
                "pair_selection_mode", "greedy"
            ),
        }
    )

    try:
        # Carrega imagem/label e gera o volume pré-processado (LCC).
        image_data = load_and_preprocess_image(img_id, base_path, config)
        lcc_image = image_data["lcc_image"]
        label = image_data["label"]
        scaled_spacing = image_data["scaled_spacing"]
        preprocessing_details = image_data.get("preprocessing_details", {})
        downscale_factors = image_data["downscale_factors"]

        image_data = None
        result.update(
            _preprocessing_result_fields(
                preprocessing_details,
                lcc_image.shape[2],
                lcc_image.size,
            )
        )

        # Calcula o mapa de vasos usado para selecionar candidatos de óstios.
        vesselness_ostios = compute_vesselness(
            lcc_image,
            vesselness_config=config["VESSELNESS_AORTA"],
            use_gpu=config.get("USE_GPU", False),
        )

        # Localiza a aorta por círculos em fatias consecutivas.
        detected_circles = locate_aorta_circles(
            lcc_image,
            downscale_factors,
            scaled_spacing,
            config["CIRCLE_DETECTION"],
        )
        original_detected_circles = [dict(circle) for circle in detected_circles]
        circle_summary = summarize_aorta_circles(
            detected_circles,
            result["image_slice_count"],
            scaled_spacing,
            config["CIRCLE_DETECTION"],
        )
        result.update(circle_summary)

        # Filtra somente a trajetória consumida pelas etapas seguintes. O
        # resumo acima continua descrevendo a saída original do detector.
        detected_circles, circle_filter_diagnostics = filter_located_aorta_circles(
            detected_circles,
            scaled_spacing,
            result["image_slice_count"],
            config["CIRCLE_DETECTION"],
        )
        # Segmenta a aorta e volta aos círculos originais quando o filtro
        # opcional ainda resultar em uma máscara sobresegmentada.
        detected_circles, aorta_segmentation, circle_filter_diagnostics = (
            _segment_aorta_with_circle_filter_fallback(
                lcc_image,
                original_detected_circles,
                detected_circles,
                circle_filter_diagnostics,
                result["image_slice_count"],
                scaled_spacing,
                config,
            )
        )
        result.update(circle_filter_diagnostics)
        aorta_mask = aorta_segmentation.mask
        result.update(aorta_segmentation.diagnostics)
        # Relaciona a máscara da aorta ao volume processado completo.
        result.update(
            summarize_aorta_volume(
                aorta_mask,
                result["image_voxels"],
            )
        )

        try:
            # Seleciona os óstios e valida contra o label arterial.
            ostia_eval = detect_and_evaluate_ostia(
                aorta_mask,
                vesselness_ostios,
                label,
                scaled_spacing,
                config,
            )
        except ValueError as ostia_exc:
            result["ostia_status"] = "not_found"
            result["ostia_error"] = str(ostia_exc)
            result["skip_reason"] = "ostia_not_found"
            result["dice_artery"] = 0.0
            if visual_output_dir is not None:
                save_segmentation_visual(
                    visual_output_dir,
                    img_id,
                    aorta_mask=aorta_mask,
                    ostia_left=None,
                    ostia_right=None,
                    artery_mask=None,
                    label_artery=(label == 1).astype("uint8"),
                    spacing=scaled_spacing,
                )
            return result

        result.update(_ostia_result_fields(ostia_eval))

        # Segmenta as artérias mesmo quando os óstios são apenas toleráveis.
        result["segmentation_attempted"] = True
        artery_metrics = segment_arteries_from_ostia(
            lcc_image,
            ostia_eval["label_artery"],
            ostia_eval["ostia_left"],
            ostia_eval["ostia_right"],
            config,
        )
        if visual_output_dir is not None:
            save_segmentation_visual(
                visual_output_dir,
                img_id,
                aorta_mask=aorta_mask,
                ostia_left=ostia_eval["ostia_left"],
                ostia_right=ostia_eval["ostia_right"],
                artery_mask=artery_metrics.get("artery_mask"),
                label_artery=ostia_eval["label_artery"],
                spacing=scaled_spacing,
            )

        del aorta_mask
        artery_metrics.pop("artery_mask", None)
        artery_metrics.pop("raw_artery_mask", None)
        result.update(artery_metrics)

    except Exception as exc:
        result["error"] = str(exc)

    return result


def _resolve_batch_plan(ids, config, resume_from_batch):
    """Calcula quantidade, tamanho e índice inicial dos lotes."""
    num_batches = config.get("NUM_BATCHES") or 5
    if num_batches <= 0:
        num_batches = 5
    num_batches = min(num_batches, len(ids))

    if resume_from_batch < 0:
        raise ValueError("resume_from_batch não pode ser negativo.")
    if resume_from_batch > num_batches:
        raise ValueError(
            f"resume_from_batch={resume_from_batch} é maior que o total de "
            f"lotes ({num_batches})."
        )

    batch_size = max(1, math.ceil(len(ids) / num_batches))
    start_batch_index = resume_from_batch - 1 if resume_from_batch > 0 else 0
    return num_batches, batch_size, start_batch_index


def _load_previous_batches(output_dir, split_name, start_batch_index):
    """Carrega lotes anteriores necessários para uma retomada consistente."""
    all_results = []
    batches_processed = []
    missing_batches = []

    # Só lotes anteriores ao ponto de retomada entram como resultados preservados.
    for batch_index in range(start_batch_index):
        batch_number = batch_index + 1
        found_path = get_batch_result_file(output_dir, split_name, batch_number)
        if found_path is None:
            missing_batches.append(batch_number)
            continue

        batch_data = pd.read_csv(found_path).to_dict("records")
        all_results.extend(batch_data)
        batches_processed.append(batch_number)
        logger.info(
            "✓ Lote %s carregado (%s registros) (arquivo: %s)",
            batch_number,
            len(batch_data),
            found_path.name,
        )

    # Impede consolidar uma execução com uma lacuna silenciosa entre os lotes.
    if missing_batches:
        missing_list = ", ".join(str(batch) for batch in missing_batches)
        raise FileNotFoundError(
            f"Não foi possível retomar o split '{split_name}': "
            f"faltam os arquivos dos lotes {missing_list}. "
        )
    return all_results, batches_processed


def _process_and_save_batch(
    batch_ids,
    batch_number,
    num_batches,
    split_name,
    config,
    base_path,
    output_dir,
    visual_output_dir=None,
):
    """Processa um lote e persiste resultados e duração imediatamente."""
    batch_started_at = datetime.now().isoformat(timespec="seconds")
    batch_start_time = time.time()
    logger.info(
        "Processando lote %s/%s (%s imagens)",
        batch_number,
        num_batches,
        len(batch_ids),
    )

    # Processa todas as imagens antes de persistir o lote de forma atômica.
    batch_results = [
        process_image(img_id, config, base_path, visual_output_dir=visual_output_dir)
        for img_id in tqdm(
            batch_ids,
            desc=f"Lote {batch_number}/{num_batches}",
            leave=False,
        )
    ]
    batch_output_path = save_results(
        batch_results,
        f"{split_name}_lote_{batch_number}",
        output_dir,
        config=config,
    )

    # O manifest separado permite recompor o tempo após queda ou retomada.
    duration = duration_breakdown(time.time() - batch_start_time)
    timing_record = {
        "split_name": split_name,
        "batch_number": batch_number,
        "total_batches": num_batches,
        "num_images": len(batch_ids),
        "first_img_id": batch_ids[0] if batch_ids else None,
        "last_img_id": batch_ids[-1] if batch_ids else None,
        "result_file": Path(batch_output_path).name,
        "started_at": batch_started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": duration["seconds"],
        "duration_minutes": duration["minutes"],
        "duration_hours": duration["hours"],
    }
    manifest_path = save_batch_timing_record(output_dir, split_name, timing_record)
    logger.info("Lote %s salvo: %s", batch_number, batch_output_path)
    logger.info(
        "Tempo do lote %s: %.1fs (%.2fmin, %.3fh). Manifest: %s",
        batch_number,
        duration["seconds"],
        duration["minutes"],
        duration["hours"],
        manifest_path,
    )
    return batch_results


def run_pipeline(
    ids,
    split_name,
    config,
    base_path,
    output_dir=None,
    resume_from_batch=0,
    visual_output_dir=None,
):
    """Processa imagens em lotes com uma config runtime já escalada para a resolução."""
    start_time = time.time()

    if not ids:
        raise ValueError(f"Nenhuma imagem encontrada para o split '{split_name}'.")

    if output_dir is None:
        raise ValueError("output_dir é obrigatório no modo batch")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_batches, batch_size, start_batch_index = _resolve_batch_plan(
        ids,
        config,
        resume_from_batch,
    )

    # Em retomadas, recupera resultados anteriores antes de processar o lote solicitado.
    if resume_from_batch > 0:
        logger.info("Retomando a partir do lote %s...", resume_from_batch)
        all_results, batches_processed = _load_previous_batches(
            output_dir,
            split_name,
            start_batch_index,
        )
    else:
        all_results, batches_processed = [], []

    for batch_num in range(start_batch_index, num_batches):
        # Define o intervalo de IDs pertencente ao lote atual.
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(ids))
        batch_ids = ids[start_idx:end_idx]
        batch_number = batch_num + 1
        batch_results = _process_and_save_batch(
            batch_ids,
            batch_number,
            num_batches,
            split_name,
            config,
            base_path,
            output_dir,
            visual_output_dir,
        )

        all_results.extend(batch_results)
        batches_processed.append(batch_number)

    # Consolida tempos persistidos, incluindo lotes executados em processos anteriores.
    execution_time = time.time() - start_time
    batch_timings = load_batch_timing_records(output_dir, split_name)
    batch_timing_summary = summarize_batch_timing_records(
        batch_timings,
        expected_batches=list(range(1, num_batches + 1)),
    )
    result = {
        "details": all_results,
        "execution_time": execution_time,
        "execution_time_breakdown": duration_breakdown(execution_time),
        "batches_processed": batches_processed,
        "batch_timings": batch_timings,
        "batch_timing_summary": batch_timing_summary,
        "is_batched": True,
    }

    return result
