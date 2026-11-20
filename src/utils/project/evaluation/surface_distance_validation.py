"""Cálculo independente para conferir as distâncias de superfície arteriais."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve, distance_transform_edt

from utils.utils.metrics import SurfaceDistanceSamples


@dataclass(frozen=True)
class SurfaceDistanceCheck:
    """Valores de referência e maior erro entre distâncias de pontos correspondentes."""

    reference_hd_mm: float
    reference_ahd_mm: float
    max_directional_error_mm: float


def reference_surface_distances(
    prediction: NDArray[np.bool_],
    reference: NDArray[np.bool_],
    spacing: Sequence[float],
) -> SurfaceDistanceSamples | None:
    """Calcula distâncias de referência por EDT em máscaras 3D alinhadas.

    A superfície é extraída pela contagem de vizinhos de face. O espaçamento
    segue a ordem dos eixos do volume, em mm.

    Args:
        prediction (NDArray[np.bool_]): Máscara booleana 3D predita.
        reference (NDArray[np.bool_]): Máscara booleana 3D de referência.
        spacing (Sequence[float]): Espaçamento físico em ``mm`` alinhado aos
            eixos das máscaras.

    Returns:
        SurfaceDistanceSamples | None: Amostras físicas de superfície e distâncias
        direcionais em ``mm``, ou ``None`` quando alguma máscara está vazia.
    """
    if not prediction.any() or not reference.any():
        return None

    # Recorta a bounding box da união das máscaras para acelerar convolução e EDT.
    union = prediction | reference
    occupied = [
        np.flatnonzero(union.any(axis=axes)) for axes in ((1, 2), (0, 2), (0, 1))
    ]
    start = np.array([axis[0] for axis in occupied])
    stop = [axis[-1] + 1 for axis in occupied]
    region = tuple(slice(lo, hi) for lo, hi in zip(start, stop, strict=True))

    # Monta núcleo 6-conectado para identificar voxels de borda sem face interna.
    kernel = np.zeros((3, 3, 3), dtype=np.uint8)
    kernel[:, 1, 1] = kernel[1, :, 1] = kernel[1, 1, :] = 1
    kernel[1, 1, 1] = 0

    pred_crop = prediction[region]
    ref_crop = reference[region]
    pred_surface = pred_crop & (
        convolve(pred_crop.astype(np.uint8), kernel, mode="constant", cval=0) < 6
    )
    ref_surface = ref_crop & (
        convolve(ref_crop.astype(np.uint8), kernel, mode="constant", cval=0) < 6
    )

    # A EDT mede até o zero mais próximo, que representa a superfície de destino.
    pred_to_ref = distance_transform_edt(~ref_surface, sampling=spacing)[pred_surface]
    ref_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)[ref_surface]
    return SurfaceDistanceSamples(
        prediction_points_mm=(np.argwhere(pred_surface) + start) * spacing,
        reference_points_mm=(np.argwhere(ref_surface) + start) * spacing,
        prediction_to_reference_mm=pred_to_ref,
        reference_to_prediction_mm=ref_to_pred,
    )


def compare_surface_distance_samples(
    calculated: SurfaceDistanceSamples, reference: SurfaceDistanceSamples
) -> SurfaceDistanceCheck:
    """Confere distâncias por ponto e resume a referência sem o agregador avaliado.

    A ausência de correspondência entre os pontos é sinalizada por erro infinito.

    Args:
        calculated (SurfaceDistanceSamples): Amostras produzidas pelo avaliador
            sob validação.
        reference (SurfaceDistanceSamples): Amostras de referência calculadas de
            forma independente.

    Returns:
        SurfaceDistanceCheck: Métricas de referência (HD e AHD em ``mm``) e
        maior erro absoluto direcional observado entre os pontos.
    """
    # Valida se a referência possui distâncias calculadas nos dois sentidos.
    expected_forward = reference.prediction_to_reference_mm
    expected_backward = reference.reference_to_prediction_mm
    if expected_forward is None or expected_backward is None:
        raise ValueError(
            "A conferência exige duas superfícies de referência não vazias."
        )

    # Recalcula HD e AHD simétricas diretamente sobre os arrays de referência.
    reference_hd = float(np.concatenate((expected_forward, expected_backward)).max())
    reference_ahd = float(
        expected_forward.sum() / (2 * len(expected_forward))
        + expected_backward.sum() / (2 * len(expected_backward))
    )

    # Verifica alinhamento exato das coordenadas e dos vetores antes de comparar.
    forward = calculated.prediction_to_reference_mm
    backward = calculated.reference_to_prediction_mm
    if (
        forward is None
        or backward is None
        or forward.shape != expected_forward.shape
        or backward.shape != expected_backward.shape
        or not np.array_equal(
            calculated.prediction_points_mm, reference.prediction_points_mm
        )
        or not np.array_equal(
            calculated.reference_points_mm, reference.reference_points_mm
        )
    ):
        return SurfaceDistanceCheck(reference_hd, reference_ahd, np.inf)

    # Calcula o maior desvio pontual absoluto em mm nos dois sentidos.
    max_error = np.maximum(
        np.abs(forward - expected_forward).max(),
        np.abs(backward - expected_backward).max(),
    )
    return SurfaceDistanceCheck(reference_hd, reference_ahd, float(max_error))
