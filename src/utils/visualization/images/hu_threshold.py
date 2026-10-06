"""Mostra intervalos HU em MIPs e cortes axiais com escala física comum."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from utils.project.analysis.aorta_hu_analysis import (
    hu_interval_mask,
)


def hu_axial_mip(
    image: NDArray[Any], interval: tuple[float, float] | None = None
) -> NDArray[np.float32]:
    """Projeta HU selecionados antes do máximo, sem preencher rejeitados com zero.

    Raios sem HU válidos recebem NaN. O cálculo em blocos evita criar outra
    cópia completa do CT para cada intervalo.
    """
    if image.ndim != 3 or any(size == 0 for size in image.shape):
        raise ValueError("O MIP exige um volume 3D não vazio.")
    if interval is not None:
        hu_interval_mask(np.empty(0), interval)
    projected = np.full(image.shape[:2], -np.inf, dtype=np.float32)
    for start in range(0, image.shape[2], 32):
        slab = image[:, :, start : start + 32]
        selected = (
            np.isfinite(slab) if interval is None else hu_interval_mask(slab, interval)
        )
        maximum = np.max(np.where(selected, slab, -np.inf), axis=2)
        np.maximum(projected, maximum, out=projected)
    projected[~np.isfinite(projected)] = np.nan
    return projected


def reference_axial_slices(mask: NDArray[Any]) -> tuple[int, int, int]:
    """Seleciona 25%, 50% e 75% da extensão axial de uma região não vazia."""
    if mask.ndim != 3:
        raise ValueError("A região de referência deve ser 3D.")
    occupied = np.flatnonzero(np.any(mask, axis=(0, 1)))
    if not occupied.size:
        raise ValueError("A região de referência está vazia.")
    first, last = int(occupied[0]), int(occupied[-1])
    indices = [
        int(round(first + fraction * (last - first))) for fraction in (0.25, 0.5, 0.75)
    ]
    return indices[0], indices[1], indices[2]
