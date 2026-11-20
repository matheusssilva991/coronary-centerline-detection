"""Utilitários de métricas de avaliação para saídas de segmentação."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion, generate_binary_structure

from utils.utils.roi import mask_voxel_bbox_slices

# Os stubs locais tipam KDTree; a consulta tem a mesma interface de cKDTree.
if TYPE_CHECKING:
    from scipy.spatial import KDTree as cKDTree
else:
    from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SurfaceDistanceSamples:
    """Coordenadas físicas das duas superfícies e distâncias alinhadas aos pontos de origem."""

    prediction_points_mm: NDArray[np.float64]
    reference_points_mm: NDArray[np.float64]
    prediction_to_reference_mm: NDArray[np.float64] | None
    reference_to_prediction_mm: NDArray[np.float64] | None

    def hausdorff_distances(self) -> tuple[float, float] | None:
        """Calcula ``(HD, AHD)`` em ``mm``, com o mesmo peso para cada sentido da AHD.

        Returns:
            tuple[float, float] | None: Tupla ``(hd_mm, ahd_mm)``, ou ``None`` se
                uma das superfícies estiver vazia.
        """
        forward = self.prediction_to_reference_mm
        backward = self.reference_to_prediction_mm
        if forward is None or backward is None:
            return None
        return (
            float(max(forward.max(), backward.max())),
            float((forward.mean() + backward.mean()) / 2),
        )


def _surface_points(
    mask: NDArray[np.bool_], spacing: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Extrai centros dos voxels de superfície de seis vizinhos.

    Args:
        mask (NDArray[np.bool_]): Máscara booleana 3D em ordem de eixos ``(y, x, z)``.
        spacing (NDArray[np.float64]): Vetor de espaçamento físico ``(dy, dx, dz)``
            em ``mm``.

    Returns:
        NDArray[np.float64]: Matriz ``(N, 3)`` com as coordenadas físicas dos voxels
            de superfície em ``mm``.
    """
    # O recorte com margem mantém a erosão equivalente à do volume completo.
    bbox = mask_voxel_bbox_slices(mask, pad_voxels=1)
    if bbox is None:
        return np.empty((0, 3), dtype=float)
    sub_mask = mask[bbox]
    eroded = np.asarray(
        binary_erosion(
            sub_mask, structure=generate_binary_structure(3, 1), border_value=0
        ),
        dtype=bool,
    )
    # Converte os índices locais da subcaixa para coordenadas físicas globais.
    offset = np.array([bbox[0].start, bbox[1].start, bbox[2].start], dtype=int)
    return (np.argwhere(sub_mask & ~eroded) + offset) * spacing


def sample_surface_distances(
    prediction: NDArray[np.bool_],
    reference: NDArray[np.bool_],
    spacing: Sequence[float],
) -> SurfaceDistanceSamples:
    """Amostra distâncias exatas ao vizinho mais próximo entre as duas superfícies de seis vizinhos.

    Args:
        prediction (NDArray[np.bool_]): Máscara binária 3D predita em ordem de eixos
            ``(y, x, z)``.
        reference (NDArray[np.bool_]): Máscara binária 3D de referência em ordem de
            eixos ``(y, x, z)``.
        spacing (Sequence[float]): Espaçamento físico em ``mm`` na ordem
            ``(dy, dx, dz)``.

    Returns:
        SurfaceDistanceSamples: Pontos físicos e distâncias nos dois sentidos
            (``None`` se alguma máscara estiver vazia).
    """
    # Valida dimensionalidade, compatibilidade de shapes e espaçamento físico em mm.
    pred = np.asarray(prediction, dtype=bool)
    target = np.asarray(reference, dtype=bool)
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

    # Extrai pontos de superfície e retorna cedo quando uma das máscaras está vazia.
    pred_points = _surface_points(pred, voxel_spacing)
    target_points = _surface_points(target, voxel_spacing)
    if not pred_points.size or not target_points.size:
        return SurfaceDistanceSamples(pred_points, target_points, None, None)

    # Consulta distâncias euclidianas exatas ao vizinho mais próximo via KD-tree.
    forward, _ = cKDTree(target_points).query(pred_points, k=1, eps=0, workers=-1)
    backward, _ = cKDTree(pred_points).query(target_points, k=1, eps=0, workers=-1)
    if not np.isfinite(forward).all() or not np.isfinite(backward).all():
        raise ValueError("As distâncias físicas calculadas não são finitas.")
    return SurfaceDistanceSamples(pred_points, target_points, forward, backward)


def surface_hd95(
    prediction: NDArray[np.bool_],
    reference: NDArray[np.bool_],
    spacing: Sequence[float],
) -> float:
    """Calcula o HD95 simétrico (``mm``) entre superfícies binárias de seis vizinhos não vazias.

    Args:
        prediction (NDArray[np.bool_]): Máscara binária 3D predita em ordem de eixos
            ``(y, x, z)``.
        reference (NDArray[np.bool_]): Máscara binária 3D de referência em ordem de
            eixos ``(y, x, z)``.
        spacing (Sequence[float]): Espaçamento físico em ``mm`` na ordem
            ``(dy, dx, dz)``.

    Returns:
        float: Percentil 95 simétrico das distâncias direcionais de superfície,
            em ``mm``.
    """
    samples = sample_surface_distances(prediction, reference, spacing)
    forward = samples.prediction_to_reference_mm
    backward = samples.reference_to_prediction_mm
    if forward is None or backward is None:
        raise ValueError("O HD95 requer duas máscaras não vazias.")
    return float(
        max(
            np.percentile(forward, 95, method="linear"),
            np.percentile(backward, 95, method="linear"),
        )
    )


def foreground_hausdorff_distances(
    prediction: NDArray[Any], reference: NDArray[Any], spacing: Sequence[float]
) -> tuple[float, float]:
    """Calcula HD e AvgHD nas máscaras inteiras não vazias, em milímetros.

    Args:
        prediction (NDArray[Any]): Máscara binária 3D predita em ordem de eixos
            ``(y, x, z)``.
        reference (NDArray[Any]): Máscara binária 3D de referência em ordem de eixos
            ``(y, x, z)``.
        spacing (Sequence[float]): Espaçamento físico por eixo em ``mm`` na ordem
            ``(dy, dx, dz)``.

    Returns:
        tuple[float, float]: Tupla ``(hd, avg_hd)`` em milímetros calculada via
            SimpleITK.
    """
    import SimpleITK as sitk

    def to_image(mask: NDArray[Any]) -> Any:
        """Converte máscara NumPy ``(y, x, z)`` para ``SimpleITK.Image`` preservando spacing."""
        image = sitk.GetImageFromArray((np.asarray(mask) > 0).astype(np.uint8))
        # GetImageFromArray inverte os eixos do array na imagem ITK.
        image.SetSpacing(tuple(float(step) for step in reversed(spacing)))
        return image

    distance = sitk.HausdorffDistanceImageFilter()
    distance.Execute(to_image(reference), to_image(prediction))
    return distance.GetHausdorffDistance(), distance.GetAverageHausdorffDistance()


def _dice_from_counts(intersection: int, total: int) -> float:
    """Calcula o coeficiente de Dice a partir da interseção e soma de voxels.

    Args:
        intersection (int): Quantidade de voxels positivos em comum entre as
            máscaras.
        total (int): Soma total de voxels positivos nas duas máscaras.

    Returns:
        float: Coeficiente de Dice em ``[0.0, 1.0]`` (retorna ``1.0`` se ambas
            estiverem vazias).
    """
    return 2.0 * intersection / total if total else 1.0


def dice_score(pred: NDArray[Any], target: NDArray[Any]) -> float:
    """Calcula o coeficiente de Dice no intervalo ``[0.0, 1.0]`` para duas máscaras binárias.

    Args:
        pred (NDArray[Any]): Máscara binária ou booleana predita (voxels ``> 0``
            contam como positivos).
        target (NDArray[Any]): Máscara binária ou booleana de referência.

    Returns:
        float: Coeficiente de Dice em ``[0.0, 1.0]`` (retorna ``1.0`` se ambas
            estiverem vazias).
    """
    pred_binary = pred > 0
    target_binary = target > 0
    return _dice_from_counts(
        int(np.count_nonzero(pred_binary & target_binary)),
        int(np.count_nonzero(pred_binary) + np.count_nonzero(target_binary)),
    )


def binary_segmentation_metrics(
    prediction: NDArray[Any],
    ground_truth: NDArray[Any],
) -> Dict[str, float | int]:
    """Calcula Dice, contagens, sensibilidade e precisão de máscaras binárias.

    Args:
        prediction (NDArray[Any]): Máscara binária ou numérica predita (``> 0`` é
            positivo).
        ground_truth (NDArray[Any]): Máscara binária ou numérica de referência.

    Returns:
        Dict[str, float | int]: Dicionário com ``dice``, ``predicted_voxels``,
            ``ground_truth_voxels``, ``true_positives``, ``sensitivity`` e
            ``precision``.
    """
    # Binariza as entradas e contabiliza positivos preditos, referência e interseção.
    prediction_binary = np.asarray(prediction) > 0
    ground_truth_binary = np.asarray(ground_truth) > 0
    true_positives = int(np.sum(prediction_binary & ground_truth_binary))
    predicted_voxels = int(prediction_binary.sum())
    ground_truth_voxels = int(ground_truth_binary.sum())

    # Calcula Dice, sensibilidade e precisão protegendo denominadores nulos com NaN.
    return {
        "dice": _dice_from_counts(
            true_positives, predicted_voxels + ground_truth_voxels
        ),
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
    """Imprime o resumo compacto de métricas usado nas análises interativas.

    Args:
        title (str): Cabeçalho textual exibido antes das linhas de métrica.
        metrics (Dict[str, float | int]): Dicionário produzido por
            ``binary_segmentation_metrics``.
    """
    print(title)
    print(f"  Dice: {metrics['dice']:.4f}")
    print(f"  Voxels preditos: {metrics['predicted_voxels']:,}")
    print(f"  Voxels ground truth: {metrics['ground_truth_voxels']:,}")
    print(f"  Interseção: {metrics['true_positives']:,}")
    print(f"  Sensibilidade: {metrics['sensitivity']:.4f}")
    print(f"  Valor preditivo positivo: {metrics['precision']:.4f}")
    print()


__all__ = [
    "SurfaceDistanceSamples",
    "foreground_hausdorff_distances",
    "sample_surface_distances",
    "surface_hd95",
    "binary_segmentation_metrics",
    "dice_score",
    "print_segmentation_metrics",
]
