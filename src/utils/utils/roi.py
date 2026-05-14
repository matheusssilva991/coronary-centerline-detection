"""Funções auxiliares para extração de regiões de interesse em volumes 3D."""

import numpy as np
from typing import Optional, Sequence, Tuple
from numpy.typing import NDArray


def extract_square_region(
    image: NDArray, x_min: int, x_max: int, y_min: int, y_max: int
) -> NDArray:
    """Extrai uma ROI retangular de um volume 3D.

    Args:
        image: Volume 3D com shape (H, W, D) ou similar.
        x_min/x_max/y_min/y_max: Coordenadas inteiras da ROI.

    Returns:
        Sub-volume recortado como NDArray.
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

    Args:
        image: Volume 3D (H, W, D).
        center: Tupla (y, x) do centro; se None usa centro da imagem.
        radius: Raio em pixels; se None usa min(H,W)//4.
        mask_background: Se True, aplica máscara circular nas fatias.

    Returns:
        Sub-volume (com máscara aplicada se solicitado).
    """
    h, w, _ = image.shape

    if center is None:
        center = (h // 2, w // 2)

    if radius is None:
        radius = min(h, w) // 4

    x_min, x_max = max(0, center[0] - radius), min(h, center[0] + radius)
    y_min, y_max = max(0, center[1] - radius), min(w, center[1] + radius)
    sub_volume = image[x_min:x_max, y_min:y_max, :]

    if mask_background:
        sub_h, sub_w = sub_volume.shape[0], sub_volume.shape[1]
        sub_center = (sub_h // 2, sub_w // 2)

        y, x = np.ogrid[:sub_h, :sub_w]
        dist_from_center = (x - sub_center[1]) ** 2 + (y - sub_center[0]) ** 2
        mask = dist_from_center <= radius**2

        masked_volume = np.zeros_like(sub_volume)
        for z in range(sub_volume.shape[2]):
            masked_volume[:, :, z] = sub_volume[:, :, z] * mask

        return masked_volume

    return sub_volume


__all__ = [
    "extract_circular_region",
    "extract_square_region",
]
