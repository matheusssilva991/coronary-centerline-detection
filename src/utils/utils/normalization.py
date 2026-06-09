"""Funções auxiliares de normalização de imagem para pré-processamento de volumes CT."""

from typing import Any
from numpy.typing import NDArray
from ..processing.gpu_utils import get_array_module


def normalize_image(img: NDArray[Any]) -> NDArray[Any]:
    """Normaliza a imagem para [0, 1] com escalonamento min-max."""
    xp = get_array_module(img)
    min_val, max_val = xp.min(img), xp.max(img)
    if float(max_val - min_val) == 0:
        return img
    return (img - min_val) / (max_val - min_val)


def robust_normalize(
    img: NDArray[Any], p_min: float = 0, p_max: float = 99.8
) -> NDArray[Any]:
    """Normaliza de forma robusta usando percentis para reduzir influência de outliers."""
    xp = get_array_module(img)

    if img.size == 0:
        return img

    val_min = xp.percentile(img, p_min)
    val_max = xp.percentile(img, p_max)
    img_clipped = xp.clip(img, val_min, val_max)

    if float(val_max - val_min) == 0:
        return xp.zeros_like(img, dtype=float)

    return (img_clipped - val_min) / (val_max - val_min)


__all__ = [
    "normalize_image",
    "robust_normalize",
]
