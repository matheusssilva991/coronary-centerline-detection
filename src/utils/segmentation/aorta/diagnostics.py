"""Calcula diagnósticos de círculos e volume da aorta."""

import math
from typing import Any, Mapping, Sequence
import numpy as np
from numpy.typing import NDArray


def _describe_circle_radii(
    values: Sequence[float], unit: str
) -> dict[str, float | None]:
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


def _median_or_none(values: Sequence[float]) -> float | None:
    """Retorna a mediana como float ou ``None`` para coleção vazia."""
    return float(np.median(values)) if values else None


def _mean_or_none(values: Sequence[float]) -> float | None:
    """Retorna a média como float ou ``None`` para coleção vazia."""
    return float(np.mean(values)) if values else None


def summarize_aorta_circles(
    detected_circles: Sequence[Mapping[str, Any]],
    image_slice_count: int,
    scaled_spacing: Sequence[float] | None = None,
    circle_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
        if circle.get("radius") is not None and math.isfinite(float(circle["radius"]))
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
            lower_bound_fraction = float(
                np.mean(np.isclose(detected_array, lower_bound))
            )
            upper_bound_fraction = float(
                np.mean(np.isclose(detected_array, upper_bound))
            )

    accumulators = [
        float(circle["accum"])
        for circle in detected_valid
        if circle.get("accum") is not None and math.isfinite(float(circle["accum"]))
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
        "aorta_detected_circle_radius_median_mm": _median_or_none(detected_radii_mm),
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


def summarize_aorta_volume(
    aorta_mask: NDArray[Any], image_voxel_count: int
) -> dict[str, Any]:
    """Calcula a ocupação total e por fatia da máscara final da aorta."""
    aorta_mask_voxels = int(aorta_mask.sum())
    segmented_slice_count = int(aorta_mask.any(axis=(0, 1)).sum())
    return {
        "aorta_mask_voxels": aorta_mask_voxels,
        "aorta_segmented_slice_count": segmented_slice_count,
        "aorta_voxels_per_segmented_slice": (
            aorta_mask_voxels / segmented_slice_count if segmented_slice_count else None
        ),
        "aorta_volume_fraction": (
            aorta_mask_voxels / image_voxel_count if image_voxel_count else None
        ),
    }
