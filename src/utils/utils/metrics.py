"""Utilitários de métricas de avaliação para saídas de segmentação."""

import numpy as np
from typing import Any, Dict
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


def binary_segmentation_metrics(
    prediction: NDArray[Any],
    ground_truth: NDArray[Any],
) -> Dict[str, float | int]:
    """Calcula Dice, contagens, sensibilidade e precisão de máscaras binárias."""
    prediction_binary = np.asarray(prediction) > 0
    ground_truth_binary = np.asarray(ground_truth) > 0
    true_positives = int(np.sum(prediction_binary & ground_truth_binary))
    predicted_voxels = int(prediction_binary.sum())
    ground_truth_voxels = int(ground_truth_binary.sum())

    return {
        "dice": float(dice_score(prediction_binary, ground_truth_binary)),
        "predicted_voxels": predicted_voxels,
        "ground_truth_voxels": ground_truth_voxels,
        "true_positives": true_positives,
        "sensitivity": (
            true_positives / ground_truth_voxels if ground_truth_voxels else np.nan
        ),
        "precision": (
            true_positives / predicted_voxels if predicted_voxels else np.nan
        ),
    }


def print_segmentation_metrics(
    title: str,
    metrics: Dict[str, float | int],
) -> None:
    """Imprime o resumo compacto usado nas análises interativas."""
    print(title)
    print(f"  Dice: {metrics['dice']:.4f}")
    print(f"  Voxels preditos: {metrics['predicted_voxels']:,}")
    print(f"  Voxels ground truth: {metrics['ground_truth_voxels']:,}")
    print(f"  Interseção: {metrics['true_positives']:,}")
    print(f"  Sensibilidade: {metrics['sensitivity']:.4f}")
    print(f"  Valor preditivo positivo: {metrics['precision']:.4f}")
    print()


__all__ = [
    "binary_segmentation_metrics",
    "dice_score",
    "print_segmentation_metrics",
]
