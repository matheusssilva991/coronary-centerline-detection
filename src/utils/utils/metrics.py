"""Utilitários de métricas de avaliação para saídas de segmentação."""

import numpy as np
from typing import TYPE_CHECKING, Any, Dict, Sequence
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion, generate_binary_structure

# Os stubs locais tipam KDTree; a consulta tem a mesma interface de cKDTree.
if TYPE_CHECKING:
    from scipy.spatial import KDTree as cKDTree
else:
    from scipy.spatial import cKDTree


def surface_distance_metrics(
    prediction: NDArray[Any],
    reference: NDArray[Any],
    spacing: Sequence[float],
) -> dict[str, float | str | None]:
    """Calcula HD e AHD entre superfícies de máscaras 3D, em milímetros.

    A superfície contém os voxels removidos por uma erosão de seis vizinhos,
    com exterior igual a fundo. AHD = (média A->B + média B->A) / 2; cada
    sentido recebe o mesmo peso, independentemente da quantidade de pontos.
    Máscaras vazias não possuem distância definida e retornam valores nulos.
    """
    pred = np.asarray(prediction) > 0
    target = np.asarray(reference) > 0
    voxel_spacing = np.asarray(spacing, dtype=float)
    if (
        pred.ndim != 3
        or pred.shape != target.shape
        or any(size == 0 for size in pred.shape)
    ):
        raise ValueError(
            "As máscaras devem ser volumes 3D com o mesmo shape não vazio."
        )
    if (
        voxel_spacing.shape != (3,)
        or not np.isfinite(voxel_spacing).all()
        or np.any(voxel_spacing <= 0)
    ):
        raise ValueError(
            "O espaçamento deve conter três valores positivos e finitos em mm."
        )

    pred_exists, target_exists = bool(pred.any()), bool(target.any())
    if not pred_exists or not target_exists:
        status = (
            "both_empty"
            if not pred_exists and not target_exists
            else "prediction_empty"
            if not pred_exists
            else "reference_empty"
        )
        return {"hd_mm": None, "ahd_mm": None, "status": status}

    # Preserva ilhas e vazamentos; não suaviza nem subamostra as superfícies.
    structure = generate_binary_structure(3, 1)
    pred_surface = pred & ~np.asarray(
        binary_erosion(pred, structure=structure, border_value=0), dtype=bool
    )
    target_surface = target & ~np.asarray(
        binary_erosion(target, structure=structure, border_value=0), dtype=bool
    )
    pred_points = np.argwhere(pred_surface) * voxel_spacing
    target_points = np.argwhere(target_surface) * voxel_spacing
    forward = cKDTree(target_points).query(pred_points, k=1, eps=0, workers=1)[0]
    backward = cKDTree(pred_points).query(target_points, k=1, eps=0, workers=1)[0]
    if not np.isfinite(forward).all() or not np.isfinite(backward).all():
        raise ValueError("As distâncias físicas calculadas não são finitas.")
    return {
        "hd_mm": float(max(forward.max(), backward.max())),
        "ahd_mm": float((forward.mean() + backward.mean()) / 2.0),
        "status": "ok",
    }


def dice_score(pred: NDArray[Any], target: NDArray[Any]) -> float:
    """Calcula o coeficiente de Dice para máscaras binárias de segmentação.

    Argumentos:        pred: Array de predição (qualquer dtype numérico)
        target: Array de ground-truth (qualquer dtype numérico)

    Retorna:        Dice score no intervalo [0.0, 1.0].
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
    "surface_distance_metrics",
    "binary_segmentation_metrics",
    "dice_score",
    "print_segmentation_metrics",
]
