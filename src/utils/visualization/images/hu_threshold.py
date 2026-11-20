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
    """Calcula a MIP axial apenas dos voxels dentro de um intervalo HU opcional.

    O cálculo é feito em blocos ao longo do eixo axial para evitar alocar outra
    cópia completa do volume, preenchendo com NaN os raios sem valores válidos.

    Args:
        image (NDArray[Any]): Volume 3D de entrada em unidades Hounsfield (HU),
            com eixo axial na última dimensão ``(Y, X, Z)``.
        interval (tuple[float, float] | None): Limites inclusivos
            ``(mínimo, máximo)`` em HU para filtrar voxels antes do máximo;
            ``None`` considera todos os valores finitos.

    Returns:
        NDArray[np.float32]: Projeção 2D máxima ``(Y, X)`` em HU, com ``NaN``
        nos raios sem voxels selecionados.
    """
    if image.ndim != 3 or any(size == 0 for size in image.shape):
        raise ValueError("O MIP exige um volume 3D não vazio.")
    if interval is not None:
        hu_interval_mask(np.empty(0), interval)
    # Processa o eixo axial em blocos de 32 fatias para evitar cópia integral do CT.
    projected = np.full(image.shape[:2], -np.inf, dtype=np.float32)
    for start in range(0, image.shape[2], 32):
        slab = image[:, :, start : start + 32]
        selected = (
            np.isfinite(slab) if interval is None else hu_interval_mask(slab, interval)
        )
        maximum = np.max(np.where(selected, slab, -np.inf), axis=2)
        np.maximum(projected, maximum, out=projected)
    # Converte para NaN os raios sem nenhum voxel válido ao longo do eixo axial.
    projected[~np.isfinite(projected)] = np.nan
    return projected


def reference_axial_slices(mask: NDArray[Any]) -> tuple[int, int, int]:
    """Seleciona os índices axiais a 25%, 50% e 75% de uma máscara 3D não vazia.

    Args:
        mask (NDArray[Any]): Máscara booleana ou binária 3D ``(Y, X, Z)`` que
            delimita a região anatômica de referência.

    Returns:
        tuple[int, int, int]: Índices das fatias axiais correspondentes a 25%,
        50% e 75% da extensão ocupada ao longo do eixo Z.
    """
    if mask.ndim != 3:
        raise ValueError("A região de referência deve ser 3D.")
    occupied = np.flatnonzero(np.any(mask, axis=(0, 1)))
    if not occupied.size:
        raise ValueError("A região de referência está vazia.")
    # Interpola os índices correspondentes a 25%, 50% e 75% do intervalo ocupado.
    first, last = int(occupied[0]), int(occupied[-1])
    indices = [
        int(round(first + fraction * (last - first))) for fraction in (0.25, 0.5, 0.75)
    ]
    return indices[0], indices[1], indices[2]
