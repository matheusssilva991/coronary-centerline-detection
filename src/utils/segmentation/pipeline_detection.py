"""Etapas de detecção/segmentação da aorta e avaliação dos óstios."""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from .aorta_localization import (
    detect_aorta_circles,
    filter_aorta_circle_trajectory,
)
from .aorta_segmentation import (
    calculate_circle_mask_metrics,
    calculate_mask_change_fraction,
    calculate_slice_area_jump_p95,
    iter_level_set_checkpoints,
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


@dataclass(frozen=True)
class _AdaptiveCheckpoint:
    """Estado necessário para comparar checkpoints do level set adaptativo."""

    iterations: int
    mask: np.ndarray
    voxel_count: int
    voxels_per_segmented_slice: float
    volume_fraction: float
    relative_growth: float | None
    mask_change_fraction: float | None
    circle_fill_q25: float | None
    circle_area_ratio_p90: float | None
    leak_signal: bool
    raw_voxel_count: int | None = None


@dataclass(frozen=True)
class _AdaptiveAlternative:
    """Resultado de uma evolução alternativa iniciada por rollback."""

    checkpoint: _AdaptiveCheckpoint | None
    profile: str
    start_iteration: int | None
    attempted: bool
    accepted: bool
    decision_reason: str


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
    )
    return detected_circles


def filter_located_aorta_circles(
    detected_circles: Sequence[Dict[str, Any]],
    scaled_spacing: Sequence[float],
    image_slice_count: int,
    circle_config: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Aplica o filtro experimental à trajetória localizada pela Hough."""
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
    """Aplica o pós-processamento histórico a uma máscara do level set."""
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
    """Executa o comportamento histórico com número fixo de iterações."""
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
    raw_voxel_count = int(np.count_nonzero(mask_refined))
    aorta_mask = _postprocess_aorta_mask(
        mask_refined,
        detected_circles,
        level_set_config,
    )
    circle_metrics = calculate_circle_mask_metrics(aorta_mask, detected_circles)
    slice_area_jump_p95 = calculate_slice_area_jump_p95(aorta_mask)
    return AortaSegmentationResult(
        mask=aorta_mask,
        diagnostics={
            "aorta_level_set_mode": "fixed",
            "aorta_level_set_initial_voxel_count": initial_voxel_count,
            "aorta_level_set_raw_voxel_count": raw_voxel_count,
            "aorta_level_set_initial_volume_fraction": (
                initial_voxel_count / mask_refined.size
            ),
            "aorta_level_set_raw_volume_fraction": (
                raw_voxel_count / mask_refined.size
            ),
            "aorta_level_set_iterations_used": num_iter,
            "aorta_level_set_stop_reason": "fixed_complete",
            "aorta_level_set_checkpoint_count": 1,
            "aorta_level_set_rolled_back": False,
            "aorta_level_set_mask_change_fraction": None,
            "aorta_level_set_circle_fill_q25": circle_metrics["circle_fill_q25"],
            "aorta_level_set_circle_area_ratio_p90": circle_metrics[
                "circle_area_ratio_p90"
            ],
            "aorta_level_set_leak_suspected": False,
            "aorta_level_set_localization_suspected": False,
            "aorta_level_set_localization_leak_override_triggered": False,
            "aorta_level_set_leak_signal_count": 0,
            "aorta_level_set_trigger_iteration": None,
            "aorta_level_set_trigger_volume_fraction": None,
            "aorta_level_set_trigger_relative_growth": None,
            "aorta_level_set_trigger_mask_change_fraction": None,
            "aorta_level_set_trigger_circle_fill_q25": None,
            "aorta_level_set_trigger_circle_area_ratio_p90": None,
            "aorta_level_set_correction_applied": False,
            "aorta_level_set_correction_method": "none",
            "aorta_level_set_slice_area_jump_p95_before": slice_area_jump_p95,
            "aorta_level_set_slice_area_jump_p95_after": slice_area_jump_p95,
            "aorta_level_set_controller_state": "fixed",
            "aorta_level_set_profile_used": "nominal",
            "aorta_level_set_rollback_iteration": None,
            "aorta_level_set_circle_confidence_signal_count": 0,
            "aorta_level_set_alternative_attempted": False,
            "aorta_level_set_alternative_accepted": False,
            "aorta_level_set_conservative_attempted": False,
            "aorta_level_set_conservative_accepted": False,
            "aorta_level_set_permissive_attempted": False,
            "aorta_level_set_permissive_accepted": False,
            "aorta_level_set_nominal_volume_fraction": int(aorta_mask.sum())
            / aorta_mask.size,
            "aorta_level_set_nominal_circle_fill_q25": circle_metrics[
                "circle_fill_q25"
            ],
            "aorta_level_set_nominal_circle_area_ratio_p90": circle_metrics[
                "circle_area_ratio_p90"
            ],
            "aorta_level_set_final_volume_fraction": int(aorta_mask.sum())
            / aorta_mask.size,
            "aorta_level_set_decision_reason": "fixed_complete",
        },
    )


def _adaptive_checkpoint_iterations(
    minimum: int,
    nominal: int,
    interval: int,
) -> List[int]:
    """Monta checkpoints de monitoramento até a iteração nominal."""
    if minimum <= 0 or interval <= 0:
        raise ValueError("min_iter e check_interval devem ser positivos")
    if minimum > nominal:
        raise ValueError("Esperado min_iter <= num_iter")

    checkpoints = set(range(minimum, nominal + 1, interval))
    checkpoints.add(nominal)
    return sorted(checkpoints)


def _finite_float(value: Any) -> float | None:
    """Normaliza métricas opcionais e descarta NaN/inf antes das decisões."""
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _build_adaptive_checkpoint(
    iterations: int,
    raw_mask: Any,
    previous: _AdaptiveCheckpoint | None,
    detected_circles: Sequence[Dict[str, Any]],
    level_set_config: Dict[str, Any],
    adaptive: Dict[str, Any],
) -> _AdaptiveCheckpoint:
    """Pós-processa um snapshot e calcula as métricas usadas pelo controlador."""
    raw_voxel_count = int(np.count_nonzero(raw_mask))
    mask = _postprocess_aorta_mask(raw_mask, detected_circles, level_set_config)
    voxel_count = int(mask.sum())
    segmented_slice_count = int(np.count_nonzero(mask.sum(axis=(0, 1))))
    voxels_per_segmented_slice = voxel_count / max(segmented_slice_count, 1)
    circle_metrics = calculate_circle_mask_metrics(mask, detected_circles)
    area_ratio = _finite_float(circle_metrics["circle_area_ratio_p90"])
    relative_growth = (
        (voxel_count - previous.voxel_count) / max(previous.voxel_count, 1)
        if previous is not None
        else None
    )
    volume_fraction = voxel_count / mask.size
    leak_signal = bool(
        previous is not None
        and area_ratio is not None
        and area_ratio > float(adaptive["oversegmented_area_ratio_p90"])
    )
    return _AdaptiveCheckpoint(
        iterations=iterations,
        mask=mask,
        voxel_count=voxel_count,
        voxels_per_segmented_slice=voxels_per_segmented_slice,
        volume_fraction=volume_fraction,
        relative_growth=relative_growth,
        mask_change_fraction=(
            calculate_mask_change_fraction(previous.mask, mask)
            if previous is not None
            else None
        ),
        circle_fill_q25=_finite_float(circle_metrics["circle_fill_q25"]),
        circle_area_ratio_p90=area_ratio,
        leak_signal=leak_signal,
        raw_voxel_count=raw_voxel_count,
    )


def _circle_confidence_signal_count(
    circle_summary: Dict[str, Any] | None,
    adaptive: Dict[str, Any],
) -> int:
    """Conta sinais de que a trajetória de círculos não representa a aorta."""
    if not circle_summary:
        return 0
    checks = (
        (
            "aorta_circle_radius_median_mm",
            lambda value: value < float(adaptive["localization_min_radius_median_mm"]),
        ),
        (
            "aorta_circle_radius_max_step_change_mm",
            lambda value: value > float(adaptive["localization_max_radius_step_mm"]),
        ),
        (
            "aorta_circle_radius_p90_step_change_mm",
            lambda value: value > float(adaptive["localization_max_radius_p90_step_mm"]),
        ),
        (
            "aorta_circle_mean_hough_accumulator",
            lambda value: value < float(adaptive["localization_min_hough_accumulator"]),
        ),
        (
            "aorta_circle_lower_radius_bound_fraction",
            lambda value: value > float(adaptive["localization_max_lower_bound_fraction"]),
        ),
    )
    count = 0
    for key, predicate in checks:
        value = _finite_float(circle_summary.get(key))
        count += int(value is not None and predicate(value))
    return count


def _classify_adaptive_state(
    checkpoint: _AdaptiveCheckpoint,
    circle_signal_count: int,
    adaptive: Dict[str, Any],
) -> str:
    """Classifica o checkpoint nominal em um dos quatro estados do controlador."""
    fill = checkpoint.circle_fill_q25
    area = checkpoint.circle_area_ratio_p90
    if (
        fill is None
        or fill < float(adaptive["localization_min_circle_fill_q25"])
        or circle_signal_count >= int(adaptive["localization_signal_threshold"])
    ):
        return "localization_suspected"

    if area is not None and area > float(adaptive["oversegmented_area_ratio_p90"]):
        return "oversegmented"
    return "adequate"


def _should_override_localization_for_leak(
    checkpoint: _AdaptiveCheckpoint,
    controller_state: str,
    adaptive: Dict[str, Any],
) -> bool:
    """Permite diagnosticar vazamento mesmo quando os círculos são suspeitos."""
    override = adaptive.get("localization_leak_override", {})
    area = checkpoint.circle_area_ratio_p90
    fill = checkpoint.circle_fill_q25
    return bool(
        controller_state == "localization_suspected"
        and override.get("enabled", False)
        and area is not None
        and area > float(override["min_area_ratio_p90"])
        and fill is not None
        and fill >= float(override["min_circle_fill_q25"])
        and checkpoint.volume_fraction
        >= float(override["min_volume_fraction"])
    )


def _is_adequate_checkpoint(
    checkpoint: _AdaptiveCheckpoint,
    adaptive: Dict[str, Any],
) -> bool:
    """Verifica se um checkpoint está dentro da faixa segura para parada em 26."""
    fill = checkpoint.circle_fill_q25
    area = checkpoint.circle_area_ratio_p90
    return bool(
        fill is not None
        and area is not None
        and fill >= float(adaptive["adequate_min_circle_fill_q25"])
        and float(adaptive["adequate_min_circle_area_ratio_p90"])
        <= area
        <= float(adaptive["adequate_max_circle_area_ratio_p90"])
    )


def _run_alternative_evolution(
    lcc_image: Any,
    detected_circles: Sequence[Dict[str, Any]],
    level_set_config: Dict[str, Any],
    start: _AdaptiveCheckpoint,
    target_iterations: int,
    profile_name: str,
    profile: Dict[str, Any],
) -> _AdaptiveCheckpoint:
    """Reinicia no rollback e evolui com o perfil conservador ou permissivo."""
    context = prepare_level_set_evolution(
        lcc_image,
        detected_circles,
        radius_reduction_factor=level_set_config["radius_reduction_factor"],
        roi_margin=level_set_config.get("roi_margin", 10),
        use_roi=level_set_config.get("use_roi", True),
        alpha=float(profile["alpha"]),
        sigma=level_set_config.get("sigma", 2),
        use_gpu=False,
        reset_curvature_cycle=True,
    )
    context.reset_from_full_mask(start.mask)
    threshold = float(np.percentile(context.gimage, float(profile["threshold_percentile"])))
    raw_mask = context.evolve(
        max(target_iterations - start.iterations, 0),
        smoothing=int(profile["smoothing"]),
        balloon=float(profile["balloon"]),
        threshold=threshold,
    )
    checkpoint = _build_adaptive_checkpoint(
        target_iterations,
        raw_mask,
        start,
        detected_circles,
        level_set_config,
        level_set_config["adaptive"],
    )
    # O nome é usado nos diagnósticos pelo chamador; evita um parâmetro silencioso.
    if profile_name != "conservative":
        raise ValueError(f"Perfil adaptativo desconhecido: {profile_name}")
    return checkpoint


def _segmented_slice_count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(mask).any(axis=(0, 1))))


def _area_ratio_improvement_fraction(
    nominal: _AdaptiveCheckpoint,
    candidate: _AdaptiveCheckpoint,
) -> float | None:
    """Calcula quanto o candidato reduziu R_P90 em relacao ao nominal."""
    nominal_ratio = nominal.circle_area_ratio_p90
    candidate_ratio = candidate.circle_area_ratio_p90
    if nominal_ratio is None or candidate_ratio is None or nominal_ratio <= 0:
        return None
    return float((nominal_ratio - candidate_ratio) / nominal_ratio)


def _accept_conservative_candidate(
    nominal: _AdaptiveCheckpoint,
    candidate: _AdaptiveCheckpoint,
    adaptive: Dict[str, Any],
) -> tuple[bool, str]:
    """Aceita somente reduções que preservem preenchimento e continuidade."""
    if candidate.circle_area_ratio_p90 is None or nominal.circle_area_ratio_p90 is None:
        return False, "missing_circle_metrics"
    if candidate.circle_area_ratio_p90 >= nominal.circle_area_ratio_p90:
        return False, "area_not_reduced"
    ratio_improvement = _area_ratio_improvement_fraction(nominal, candidate)
    min_improvement = float(
        adaptive.get("min_area_ratio_improvement_fraction", 0.0)
    )
    if ratio_improvement is None or ratio_improvement < min_improvement:
        return False, "area_reduction_below_minimum"
    if candidate.volume_fraction >= nominal.volume_fraction:
        return False, "volume_not_reduced"
    if candidate.circle_fill_q25 is None or nominal.circle_fill_q25 is None:
        return False, "missing_circle_fill"
    if candidate.circle_fill_q25 < nominal.circle_fill_q25 - float(adaptive["max_fill_loss"]):
        return False, "circle_fill_loss"
    if _segmented_slice_count(candidate.mask) != _segmented_slice_count(nominal.mask):
        return False, "segmented_slice_change"
    nominal_jump = calculate_slice_area_jump_p95(nominal.mask)
    candidate_jump = calculate_slice_area_jump_p95(candidate.mask)
    if candidate_jump > nominal_jump * (1 + float(adaptive["max_axial_jump_increase_fraction"])) + 1e-12:
        return False, "axial_jump_increased"
    return True, "accepted"


def _adaptive_diagnostics(
    selected: _AdaptiveCheckpoint | None,
    nominal: _AdaptiveCheckpoint | None,
    *,
    checkpoint_count: int,
    controller_state: str,
    stop_reason: str,
    profile_used: str,
    rollback_iteration: int | None,
    circle_signal_count: int,
    alternative: _AdaptiveAlternative | None,
    localization_leak_override_triggered: bool = False,
    initial_voxel_count: int | None = None,
    image_voxel_count: int | None = None,
) -> Dict[str, Any]:
    """Converte a decisão do controlador em campos persistíveis no CSV."""
    attempted_profile = alternative.profile if alternative and alternative.attempted else None
    accepted = bool(alternative and alternative.accepted)
    candidate = alternative.checkpoint if alternative and alternative.attempted else None
    candidate_ratio_improvement = (
        _area_ratio_improvement_fraction(nominal, candidate)
        if nominal is not None and candidate is not None
        else None
    )
    jump_before = calculate_slice_area_jump_p95(nominal.mask) if nominal is not None else None
    jump_after = calculate_slice_area_jump_p95(selected.mask) if selected is not None else None
    raw_voxel_count = None
    if selected is not None:
        raw_voxel_count = (
            selected.raw_voxel_count
            if selected.raw_voxel_count is not None
            else selected.voxel_count
        )
    return {
        "aorta_level_set_mode": "adaptive",
        "aorta_level_set_initial_voxel_count": initial_voxel_count,
        "aorta_level_set_raw_voxel_count": raw_voxel_count,
        "aorta_level_set_initial_volume_fraction": (
            initial_voxel_count / image_voxel_count
            if initial_voxel_count is not None and image_voxel_count
            else None
        ),
        "aorta_level_set_raw_volume_fraction": (
            raw_voxel_count / image_voxel_count
            if raw_voxel_count is not None and image_voxel_count
            else None
        ),
        "aorta_level_set_iterations_used": selected.iterations if selected else 0,
        "aorta_level_set_stop_reason": stop_reason,
        "aorta_level_set_checkpoint_count": checkpoint_count,
        "aorta_level_set_rolled_back": accepted and rollback_iteration is not None,
        "aorta_level_set_mask_change_fraction": selected.mask_change_fraction if selected else None,
        "aorta_level_set_voxels_per_segmented_slice": (
            selected.voxels_per_segmented_slice if selected else None
        ),
        "aorta_level_set_circle_fill_q25": selected.circle_fill_q25 if selected else None,
        "aorta_level_set_circle_area_ratio_p90": selected.circle_area_ratio_p90 if selected else None,
        "aorta_level_set_leak_suspected": (
            controller_state == "oversegmented"
            or localization_leak_override_triggered
        ),
        "aorta_level_set_localization_suspected": controller_state == "localization_suspected",
        "aorta_level_set_localization_leak_override_triggered": (
            localization_leak_override_triggered
        ),
        "aorta_level_set_leak_signal_count": int(
            controller_state == "oversegmented"
            or localization_leak_override_triggered
        ),
        "aorta_level_set_trigger_iteration": nominal.iterations if nominal else None,
        "aorta_level_set_trigger_volume_fraction": nominal.volume_fraction if nominal else None,
        "aorta_level_set_trigger_relative_growth": nominal.relative_growth if nominal else None,
        "aorta_level_set_trigger_mask_change_fraction": nominal.mask_change_fraction if nominal else None,
        "aorta_level_set_trigger_circle_fill_q25": nominal.circle_fill_q25 if nominal else None,
        "aorta_level_set_trigger_circle_area_ratio_p90": nominal.circle_area_ratio_p90 if nominal else None,
        "aorta_level_set_correction_applied": accepted,
        "aorta_level_set_correction_method": attempted_profile or "none",
        "aorta_level_set_controller_state": controller_state,
        "aorta_level_set_profile_used": profile_used,
        "aorta_level_set_rollback_iteration": rollback_iteration,
        "aorta_level_set_circle_confidence_signal_count": circle_signal_count,
        "aorta_level_set_alternative_attempted": bool(alternative and alternative.attempted),
        "aorta_level_set_alternative_accepted": accepted,
        "aorta_level_set_conservative_attempted": attempted_profile == "conservative",
        "aorta_level_set_conservative_accepted": attempted_profile == "conservative" and accepted,
        "aorta_level_set_permissive_attempted": attempted_profile == "permissive",
        "aorta_level_set_permissive_accepted": attempted_profile == "permissive" and accepted,
        "aorta_level_set_nominal_volume_fraction": nominal.volume_fraction if nominal else None,
        "aorta_level_set_nominal_circle_fill_q25": nominal.circle_fill_q25 if nominal else None,
        "aorta_level_set_nominal_circle_area_ratio_p90": nominal.circle_area_ratio_p90 if nominal else None,
        "aorta_level_set_candidate_voxel_count": (
            candidate.voxel_count if candidate is not None else None
        ),
        "aorta_level_set_candidate_volume_fraction": (
            candidate.volume_fraction if candidate is not None else None
        ),
        "aorta_level_set_candidate_circle_fill_q25": (
            candidate.circle_fill_q25 if candidate is not None else None
        ),
        "aorta_level_set_candidate_circle_area_ratio_p90": (
            candidate.circle_area_ratio_p90 if candidate is not None else None
        ),
        "aorta_level_set_candidate_area_ratio_improvement_fraction": (
            candidate_ratio_improvement
        ),
        "aorta_level_set_final_volume_fraction": selected.volume_fraction if selected else None,
        "aorta_level_set_decision_reason": alternative.decision_reason if alternative else stop_reason,
        "aorta_level_set_slice_area_jump_p95_before": jump_before,
        "aorta_level_set_slice_area_jump_p95_after": jump_after,
    }


def _adaptive_level_set_result(
    lcc_image: Any,
    detected_circles: Sequence[Dict[str, Any]],
    level_set_config: Dict[str, Any],
    circle_summary: Dict[str, Any] | None = None,
) -> AortaSegmentationResult:
    """Controla checkpoints e tenta nova evolução apenas fora da faixa adequada."""
    adaptive = level_set_config["adaptive"]
    nominal_iter = int(level_set_config["num_iter"])
    checkpoint_iterations = _adaptive_checkpoint_iterations(
        int(adaptive["min_iter"]), nominal_iter, int(adaptive["check_interval"])
    )
    if not detected_circles:
        empty = np.zeros_like(lcc_image, dtype=np.uint8)
        return AortaSegmentationResult(
            empty,
            _adaptive_diagnostics(
                None, None, checkpoint_count=0,
                controller_state="localization_suspected",
                stop_reason="localization_suspected", profile_used="nominal",
                rollback_iteration=None, circle_signal_count=0, alternative=None,
                initial_voxel_count=0, image_voxel_count=empty.size,
            ),
        )

    context = prepare_level_set_evolution(
        lcc_image,
        detected_circles,
        radius_reduction_factor=level_set_config["radius_reduction_factor"],
        roi_margin=level_set_config.get("roi_margin", 10),
        use_roi=level_set_config.get("use_roi", True),
        alpha=level_set_config.get("alpha", 1000),
        sigma=level_set_config.get("sigma", 2),
        use_gpu=False,
        reset_curvature_cycle=True,
    )
    initial_voxel_count = int(np.count_nonzero(context.current_mask))
    image_voxel_count = int(np.prod(lcc_image.shape))
    raw_checkpoints = iter_level_set_checkpoints(
        lcc_image,
        detected_circles,
        checkpoint_iterations,
        radius_reduction_factor=level_set_config["radius_reduction_factor"],
        balloon=level_set_config["balloon"],
        smoothing=level_set_config["smoothing"],
        threshold=level_set_config.get("threshold", "auto"),
        roi_margin=level_set_config.get("roi_margin", 10),
        use_roi=level_set_config.get("use_roi", True),
        alpha=level_set_config.get("alpha", 1000),
        sigma=level_set_config.get("sigma", 2),
        use_gpu=False,
        context=context,
    )

    results: list[_AdaptiveCheckpoint] = []
    stable_adequate_count = 0
    circle_signals = _circle_confidence_signal_count(circle_summary, adaptive)
    for iterations, raw_mask in raw_checkpoints:
        checkpoint = _build_adaptive_checkpoint(
            iterations,
            raw_mask,
            results[-1] if results else None,
            detected_circles,
            level_set_config,
            adaptive,
        )
        results.append(checkpoint)
        is_stable = (
            checkpoint.mask_change_fraction is not None
            and checkpoint.mask_change_fraction <= float(adaptive["convergence_tolerance"])
            and _is_adequate_checkpoint(checkpoint, adaptive)
            and circle_signals < int(adaptive["localization_signal_threshold"])
        )
        stable_adequate_count = stable_adequate_count + 1 if is_stable else 0
        if (
            iterations >= int(adaptive["early_stop_iteration"])
            and stable_adequate_count >= int(adaptive["convergence_patience"])
        ):
            return AortaSegmentationResult(
                checkpoint.mask,
                _adaptive_diagnostics(
                    checkpoint, checkpoint, checkpoint_count=len(results),
                    controller_state="adequate", stop_reason="early_stable",
                    profile_used="nominal", rollback_iteration=None,
                    circle_signal_count=circle_signals, alternative=None,
                    initial_voxel_count=initial_voxel_count,
                    image_voxel_count=image_voxel_count,
                ),
            )

    nominal = results[-1]
    state = _classify_adaptive_state(nominal, circle_signals, adaptive)
    localization_leak_override = _should_override_localization_for_leak(
        nominal,
        state,
        adaptive,
    )
    if state == "adequate" or (
        state == "localization_suspected" and not localization_leak_override
    ):
        reason = "nominal_complete" if state == "adequate" else "localization_suspected"
        return AortaSegmentationResult(
            nominal.mask,
            _adaptive_diagnostics(
                nominal, nominal, checkpoint_count=len(results),
                controller_state=state, stop_reason=reason, profile_used="nominal",
                rollback_iteration=None, circle_signal_count=circle_signals,
                alternative=None,
                initial_voxel_count=initial_voxel_count,
                image_voxel_count=image_voxel_count,
            ),
        )

    safe = [
        item
        for item in results[:-1]
        if _classify_adaptive_state(item, 0, adaptive) != "oversegmented"
    ]
    start = safe[-1] if safe else results[0]
    profile_name = "conservative"
    target_iterations = nominal_iter

    candidate = _run_alternative_evolution(
        lcc_image, detected_circles, level_set_config, start,
        target_iterations, profile_name, adaptive[profile_name],
    )
    accepted, decision_reason = _accept_conservative_candidate(
        nominal,
        candidate,
        adaptive,
    )
    alternative = _AdaptiveAlternative(
        # Mantem as metricas do candidato para diagnosticar tambem as rejeicoes.
        checkpoint=candidate,
        profile=profile_name,
        start_iteration=start.iterations,
        attempted=True,
        accepted=accepted,
        decision_reason=decision_reason,
    )
    selected = candidate if accepted else nominal
    stop_reason = f"{profile_name}_accepted" if accepted else f"{profile_name}_rejected"
    return AortaSegmentationResult(
        selected.mask,
        _adaptive_diagnostics(
            selected, nominal, checkpoint_count=len(results),
            controller_state=state, stop_reason=stop_reason,
            profile_used=profile_name if accepted else "nominal",
            rollback_iteration=start.iterations,
            circle_signal_count=circle_signals, alternative=alternative,
            localization_leak_override_triggered=localization_leak_override,
            initial_voxel_count=initial_voxel_count,
            image_voxel_count=image_voxel_count,
        ),
    )


def segment_aorta_with_diagnostics(
    lcc_image: Any,
    detected_circles: List[Dict[str, Any]],
    level_set_config: Dict[str, Any],
    use_gpu: bool = False,
    circle_summary: Dict[str, Any] | None = None,
) -> AortaSegmentationResult:
    """Segmenta a aorta e retorna métricas do controle de iterações."""
    del use_gpu  # O MorphGAC e o pós-processamento permanecem na CPU.
    mode = str(level_set_config.get("iteration_mode", "fixed")).lower()
    if mode == "fixed":
        return _fixed_level_set_result(lcc_image, detected_circles, level_set_config)
    if mode == "adaptive":
        return _adaptive_level_set_result(
            lcc_image,
            detected_circles,
            level_set_config,
            circle_summary=circle_summary,
        )
    raise ValueError("LEVEL_SET.iteration_mode deve ser 'fixed' ou 'adaptive'")


def segment_aorta(
    lcc_image: Any,
    detected_circles: List[Dict[str, Any]],
    level_set_config: Dict[str, Any],
    use_gpu: bool = False,
) -> Any:
    """Segmenta a aorta preservando a API histórica que retorna apenas a máscara."""
    result = segment_aorta_with_diagnostics(
        lcc_image,
        detected_circles,
        level_set_config,
        use_gpu=use_gpu,
    )
    return result.mask


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
