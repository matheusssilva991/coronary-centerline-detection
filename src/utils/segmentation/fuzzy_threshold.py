"""Threshold fuzzy reutilizável para experimentos de segmentação."""

from __future__ import annotations

import numpy as np

from ..processing import largest_connected_component


def fuzzy_trapezoid_threshold(
    volume: np.ndarray,
    min_hu: float,
    max_hu: float,
    margin_hu: float,
) -> np.ndarray:
    """Retorna a pertinência fuzzy para uma faixa HU trapezoidal."""
    if margin_hu <= 0:
        return ((volume >= min_hu) & (volume <= max_hu)).astype(np.float32)

    rising = (volume - (min_hu - margin_hu)) / margin_hu
    falling = ((max_hu + margin_hu) - volume) / margin_hu
    membership = np.minimum(rising, falling)
    return np.clip(membership, 0.0, 1.0).astype(np.float32)


def build_lcc_image_from_mask(
    volume: np.ndarray,
    mask: np.ndarray,
    offset: float,
    per_slice: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Replica o LCC do pipeline usando uma máscara externa."""
    thresholded = np.zeros_like(volume, dtype=np.float32)
    thresholded[mask] = volume[mask] + offset

    if per_slice:
        lcc_image = np.zeros_like(thresholded, dtype=np.float32)
        lcc_mask = np.zeros_like(mask, dtype=bool)
        for z_idx in range(thresholded.shape[2]):
            lcc_slice, lcc_mask_slice = largest_connected_component(
                thresholded[:, :, z_idx], mask[:, :, z_idx]
            )
            lcc_image[:, :, z_idx] = lcc_slice
            lcc_mask[:, :, z_idx] = lcc_mask_slice
    else:
        lcc_image, lcc_mask = largest_connected_component(thresholded, mask)

    return (lcc_image - offset).astype(np.float32), lcc_mask.astype(bool)
