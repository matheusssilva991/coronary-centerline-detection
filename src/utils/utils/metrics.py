"""Utilitários de métricas de avaliação para saídas de segmentação."""

import numpy as np


def dice_score(pred, target):
    """Calcula o coeficiente de Dice para máscaras binárias de segmentação."""
    pred_binary = (pred > 0).astype(bool)
    target_binary = (target > 0).astype(bool)

    intersection = np.sum(pred_binary & target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return 2.0 * intersection / union


__all__ = ["dice_score"]
