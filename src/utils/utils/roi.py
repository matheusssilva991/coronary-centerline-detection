"""Funções auxiliares para extração de regiões de interesse em volumes 3D."""

from collections.abc import Sequence
from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def mask_bounding_box_slices(
    mask: NDArray[Any],
    spacing: Sequence[float],
    margin_mm: float,
) -> tuple[slice, slice, slice]:
    """Calcula o recorte 3D da máscara com uma margem física por eixo."""
    mask_array = np.asarray(mask)
    if mask_array.ndim != 3:
        raise ValueError("A máscara deve ser tridimensional.")
    if not np.any(mask_array):
        raise ValueError("A máscara não pode estar vazia.")

    spacing_array = np.asarray(tuple(spacing), dtype=float)
    if spacing_array.shape != (3,):
        raise ValueError("O espaçamento deve conter exatamente três valores.")
    if not np.all(np.isfinite(spacing_array)) or np.any(spacing_array <= 0):
        raise ValueError("O espaçamento deve conter valores finitos e positivos.")

    resolved_margin = float(margin_mm)
    if not np.isfinite(resolved_margin) or resolved_margin < 0:
        raise ValueError("A margem deve ser finita e não negativa.")

    coordinates = np.argwhere(mask_array != 0)
    margin_voxels = np.ceil(resolved_margin / spacing_array).astype(int)
    lower = np.maximum(coordinates.min(axis=0) - margin_voxels, 0)
    upper = np.minimum(
        coordinates.max(axis=0) + margin_voxels + 1,
        np.asarray(mask_array.shape),
    )
    return (
        slice(int(lower[0]), int(upper[0])),
        slice(int(lower[1]), int(upper[1])),
        slice(int(lower[2]), int(upper[2])),
    )


def extract_square_region(
    image: NDArray, x_min: int, x_max: int, y_min: int, y_max: int
) -> NDArray:
    """Extrai uma ROI retangular de um volume 3D.

    Argumentos:        image: Volume 3D com shape (H, W, D) ou similar.
        x_min/x_max/y_min/y_max: Coordenadas inteiras da ROI.

    Retorna:        Sub-volume recortado como NDArray.
    """
    h, w, _ = image.shape

    x_min = max(0, x_min)
    x_max = min(h, x_max)
    y_min = max(0, y_min)
    y_max = min(w, y_max)

    if x_min >= x_max or y_min >= y_max:
        raise ValueError(
            "Coordenadas inválidas: x_min deve ser menor que x_max e y_min deve ser menor que y_max"
        )

    return image[x_min:x_max, y_min:y_max, :]


def extract_circular_region(
    image: NDArray,
    center: Optional[Tuple[int, int]] = None,
    radius: Optional[int] = None,
    mask_background: bool = True,
) -> NDArray:
    """Extrai uma ROI circular de um volume 3D mascarando cada fatia 2D.

    Argumentos:        image: Volume 3D (H, W, D).
        center: Tupla (y, x) do centro; se None usa centro da imagem.
        radius: Raio em pixels; se None usa min(H,W)//4.
        mask_background: Se True, aplica máscara circular nas fatias.

    Retorna:        Sub-volume (com máscara aplicada se solicitado).
    """
    h, w, _ = image.shape

    resolved_center = center if center is not None else (h // 2, w // 2)
    resolved_radius = radius if radius is not None else min(h, w) // 4

    x_min = max(0, resolved_center[0] - resolved_radius)
    x_max = min(h, resolved_center[0] + resolved_radius)
    y_min = max(0, resolved_center[1] - resolved_radius)
    y_max = min(w, resolved_center[1] + resolved_radius)
    sub_volume = image[x_min:x_max, y_min:y_max, :]

    if mask_background:
        sub_h, sub_w = sub_volume.shape[0], sub_volume.shape[1]
        sub_center = (sub_h // 2, sub_w // 2)

        y, x = np.ogrid[:sub_h, :sub_w]
        dist_from_center = (x - sub_center[1]) ** 2 + (y - sub_center[0]) ** 2
        mask = dist_from_center <= resolved_radius**2

        masked_volume = np.zeros_like(sub_volume)
        for z in range(sub_volume.shape[2]):
            masked_volume[:, :, z] = sub_volume[:, :, z] * mask

        return masked_volume

    return sub_volume


__all__ = [
    "extract_circular_region",
    "extract_square_region",
    "mask_bounding_box_slices",
]
