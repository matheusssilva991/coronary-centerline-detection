"""Funções auxiliares de normalização de imagem para pré-processamento de volumes CT."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..processing.gpu_utils import get_array_module


def normalize_image(img: NDArray[Any]) -> NDArray[Any]:
    """Normaliza a imagem para [0, 1] com escalonamento min-max.

    Args:
        img (NDArray[Any]): Array NumPy ou CuPy de intensidades a normalizar.

    Returns:
        NDArray[Any]: Array normalizado em ``[0, 1]`` (ou inalterado se o intervalo
            for nulo).
    """
    xp = get_array_module(img)
    min_val, max_val = xp.min(img), xp.max(img)
    if float(max_val - min_val) == 0:
        return img
    return (img - min_val) / (max_val - min_val)


def robust_normalize(
    img: NDArray[Any], p_min: float = 0, p_max: float = 99.8
) -> NDArray[Any]:
    """Normaliza de forma robusta usando percentis para reduzir influência de outliers.

    Args:
        img (NDArray[Any]): Array NumPy ou CuPy de intensidades a normalizar.
        p_min (float): Percentil inferior em ``[0, 100]`` usado como piso de recorte.
        p_max (float): Percentil superior em ``[0, 100]`` usado como teto de recorte.

    Returns:
        NDArray[Any]: Array recortado e escalado para ``[0, 1]`` (ou zerado se o
            intervalo for nulo).
    """
    xp = get_array_module(img)

    if img.size == 0:
        return img

    # Calcula os limites por percentil para conter outliers antes da escala linear.
    val_min = xp.percentile(img, p_min)
    val_max = xp.percentile(img, p_max)
    img_clipped = xp.clip(img, val_min, val_max)

    if float(val_max - val_min) == 0:
        return xp.zeros_like(img, dtype=float)

    return (img_clipped - val_min) / (val_max - val_min)


def normalize_vesselness(vesselness: NDArray[Any]) -> NDArray[np.float32]:
    """Normaliza um mapa de vesselness para [0, 1] em ``float32`` na CPU.

    Diferente de ``normalize_image``, trata mapas sem valores finitos ou sem
    resposta positiva como zero antes de regiões/fuzzy connectedness.

    Args:
        vesselness (NDArray[Any]): Mapa de resposta tubular retornado pelo filtro de
            Frangi.

    Returns:
        NDArray[np.float32]: Array ``float32`` no intervalo ``[0, 1]`` (ou zeros
            quando inválido/não positivo).
    """
    values = np.asarray(vesselness, dtype=np.float32)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return np.zeros_like(values, dtype=np.float32)

    max_value = float(np.max(values[finite_mask]))
    if max_value <= 0:
        return np.zeros_like(values, dtype=np.float32)

    return np.clip(values / max_value, 0.0, 1.0).astype(np.float32)


__all__ = [
    "normalize_image",
    "normalize_vesselness",
    "robust_normalize",
]
