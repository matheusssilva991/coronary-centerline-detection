"""Funções auxiliares para extração de regiões de interesse em volumes 3D."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import find_objects


def mask_bounding_box_slices(
    mask: NDArray[Any],
    spacing: Sequence[float],
    margin_mm: float,
) -> tuple[slice, slice, slice]:
    """Calcula o recorte 3D da máscara com uma margem física por eixo.

    Args:
        mask (NDArray[Any]): Máscara binária 3D não vazia em que a ROI será calculada.
        spacing (Sequence[float]): Espaçamento físico em ``mm`` alinhado aos 3 eixos
            de ``mask``.
        margin_mm (float): Margem física não negativa (em ``mm``) adicionada em cada
            direção.

    Returns:
        tuple[slice, slice, slice]: Tupla com os três slices ``(s0, s1, s2)``
            limitados ao shape de ``mask``.
    """
    # Valida a dimensionalidade da máscara e obtém a bounding box sem margem.
    mask_array = np.asarray(mask)
    if mask_array.ndim != 3:
        raise ValueError("A máscara deve ser tridimensional.")
    bounds = find_objects(mask_array != 0, max_label=1)[0]
    if bounds is None:
        raise ValueError("A máscara não pode estar vazia.")

    # Valida o espaçamento físico positivo e a margem em milímetros.
    spacing_array = np.asarray(tuple(spacing), dtype=float)
    if spacing_array.shape != (3,):
        raise ValueError("O espaçamento deve conter exatamente três valores.")
    if not np.all(np.isfinite(spacing_array)) or np.any(spacing_array <= 0):
        raise ValueError("O espaçamento deve conter valores finitos e positivos.")

    resolved_margin = float(margin_mm)
    if not np.isfinite(resolved_margin) or resolved_margin < 0:
        raise ValueError("A margem deve ser finita e não negativa.")

    # Converte a margem física em voxels por eixo (arredondamento para cima).
    margin_voxels = np.ceil(resolved_margin / spacing_array).astype(int)

    def padded(axis: int) -> slice:
        """Recorta o slice do eixo respeitando os limites do volume."""
        margin = int(margin_voxels[axis])
        return slice(
            max(0, bounds[axis].start - margin),
            min(mask_array.shape[axis], bounds[axis].stop + margin),
        )

    return padded(0), padded(1), padded(2)


def mask_voxel_bbox_slices(
    mask: NDArray[Any],
    pad_voxels: int = 0,
) -> tuple[slice, slice, slice] | None:
    """Calcula a bounding box 3D da máscara com margem fixa em voxels.

    Args:
        mask (NDArray[Any]): Máscara binária 3D onde a região de interesse é avaliada.
        pad_voxels (int): Margem inteira não negativa (em voxels) adicionada a cada
            eixo.

    Returns:
        tuple[slice, slice, slice] | None: Tupla ``(s0, s1, s2)`` limitada ao shape
            de ``mask``, ou ``None`` se vazia.
    """
    # Garante volume 3D e margem em voxels inteira não negativa (rejeitando bool).
    mask_array = np.asarray(mask)
    if mask_array.ndim != 3:
        raise ValueError("A máscara deve ser tridimensional.")
    if isinstance(pad_voxels, bool) or int(pad_voxels) != pad_voxels or pad_voxels < 0:
        raise ValueError("pad_voxels deve ser um inteiro não negativo.")

    # Extrai a bounding box inicial e interrompe com None caso a máscara esteja vazia.
    pad = int(pad_voxels)
    bounds = find_objects(mask_array != 0, max_label=1)
    if not bounds or bounds[0] is None:
        return None
    bbox = bounds[0]

    # Expande a caixa pela margem solicitada sem ultrapassar as bordas do volume.
    return (
        slice(
            max(0, bbox[0].start - pad),
            min(mask_array.shape[0], bbox[0].stop + pad),
        ),
        slice(
            max(0, bbox[1].start - pad),
            min(mask_array.shape[1], bbox[1].stop + pad),
        ),
        slice(
            max(0, bbox[2].start - pad),
            min(mask_array.shape[2], bbox[2].stop + pad),
        ),
    )


__all__ = [
    "mask_bounding_box_slices",
    "mask_voxel_bbox_slices",
]
