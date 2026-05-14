"""Utilitários de métricas de avaliação para saídas de segmentação."""

import numpy as np
from typing import Any
from numpy.typing import NDArray


def dice_score(pred: NDArray[Any], target: NDArray[Any]) -> float:
    """Calcula o coeficiente de Dice para máscaras binárias de segmentação.

    Args:
        pred: Array de predição (qualquer dtype numérico)
        target: Array de ground-truth (qualquer dtype numérico)

    Returns:
        Dice score no intervalo [0.0, 1.0].
    """
    pred_binary = (pred > 0).astype(bool)
    target_binary = (target > 0).astype(bool)

    intersection = np.sum(pred_binary & target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return 2.0 * float(intersection) / float(union)


__all__ = ["dice_score"]
