"""Métricas e validação do fallback da trajetória da aorta.

O módulo contém apenas a correção ativa guiada pela máscara nominal. A
localização dos círculos e a execução do level set permanecem nos módulos de
detecção e segmentação.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .aorta_segmentation import (
    calculate_circle_mask_metrics,
    calculate_circle_mask_profile,
    calculate_slice_area_jump_p95,
)


def find_mask_guided_tail_start(
    mask: np.ndarray,
    circles: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> int | None:
    """Localiza uma cauda persistente com excesso de área na máscara.

    Os círculos são avaliados na ordem do rastreamento da Hough. O retorno é
    o índice do primeiro círculo da cauda; ``None`` indica que o excesso de
    área não foi persistente ou que o corte violaria os limites de segurança.
    """
    ordered = sorted(
        circles,
        key=lambda circle: int(circle["slice_index"]),
        reverse=True,
    )
    if not ordered:
        return None

    profile = calculate_circle_mask_profile(mask, ordered)
    ratio_by_slice = {
        int(item["slice_index"]): float(item["circle_area_ratio"]) for item in profile
    }
    persistence_window = max(3, int(config.get("persistence_window", 5)))
    persistence_required = min(
        persistence_window,
        max(2, int(config.get("persistence_required", 4))),
    )
    min_tail_circles = max(
        persistence_window,
        int(config.get("min_tail_circles", 8)),
    )
    min_remaining = max(1, int(config.get("min_remaining_circles", 30)))
    search_start = max(
        min_remaining,
        int(
            round(len(ordered) * float(config.get("tail_search_start_fraction", 0.35)))
        ),
    )
    search_stop = len(ordered) - min_tail_circles + 1
    ratio_threshold = float(config.get("slice_area_ratio_threshold", 2.5))
    max_trim_fraction = float(config.get("max_tail_trim_fraction", 0.4))

    for index in range(search_start, search_stop):
        window = ordered[index : index + persistence_window]
        high_ratio_count = sum(
            ratio_by_slice.get(int(circle["slice_index"]), -np.inf) > ratio_threshold
            for circle in window
        )
        if high_ratio_count < persistence_required:
            continue

        trim_fraction = (len(ordered) - index) / len(ordered)
        if trim_fraction <= max_trim_fraction:
            return index
        return None
    return None


def evaluate_mask_guided_tail_candidate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[str]:
    """Valida a máscara gerada após o corte guiado pela máscara nominal."""
    baseline_ratio = _finite_float(baseline.get("circle_area_ratio_p90"))
    candidate_ratio = _finite_float(candidate.get("circle_area_ratio_p90"))
    baseline_fill = _finite_float(baseline.get("circle_fill_q25"))
    candidate_fill = _finite_float(candidate.get("circle_fill_q25"))
    baseline_voxels = int(baseline.get("voxel_count") or 0)
    candidate_voxels = int(candidate.get("voxel_count") or 0)

    reasons: list[str] = []
    min_ratio = float(config.get("min_area_ratio_p90", 2.5))
    if baseline_ratio is None or baseline_ratio < min_ratio:
        reasons.append("candidate_area_ratio_below_trigger")
    if baseline_ratio is None or candidate_ratio is None:
        reasons.append("missing_area_ratio")
    else:
        improvement = (baseline_ratio - candidate_ratio) / max(baseline_ratio, 1e-8)
        if improvement < float(config.get("min_ratio_improvement", 0.1)):
            reasons.append("area_ratio_improvement_too_small")
        if candidate_ratio > float(config.get("slice_area_ratio_threshold", 2.5)):
            reasons.append("retry_area_ratio_exceeded")
    if (
        baseline_fill is None
        or candidate_fill is None
        or candidate_fill < baseline_fill - float(config.get("max_fill_loss", 0.015))
    ):
        reasons.append("circle_fill_loss_exceeded")
    if candidate_voxels <= 0 or candidate_voxels >= baseline_voxels:
        reasons.append("volume_not_reduced")
    return reasons


def _finite_float(value: Any) -> float | None:
    """Converte um valor opcional em número finito."""
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def calculate_aorta_candidate_metrics(
    mask: np.ndarray,
    circles: Sequence[Mapping[str, Any]],
) -> dict[str, float | int | None]:
    """Resume uma máscara para comparar baseline e tentativa corrigida."""
    binary_mask = np.asarray(mask, dtype=bool)
    circle_metrics = calculate_circle_mask_metrics(binary_mask, circles)
    voxel_count = int(binary_mask.sum())
    segmented_slice_count = int(binary_mask.any(axis=(0, 1)).sum())
    return {
        "voxel_count": voxel_count,
        "volume_fraction": voxel_count / binary_mask.size,
        "segmented_slice_count": segmented_slice_count,
        "circle_fill_q25": circle_metrics["circle_fill_q25"],
        "circle_area_ratio_p90": circle_metrics["circle_area_ratio_p90"],
        "slice_area_jump_p95": calculate_slice_area_jump_p95(binary_mask),
    }
