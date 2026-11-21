"""Distâncias direcionais para visualizar as máscaras arteriais inteiras."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

from utils.utils.roi import mask_bounding_box_slices


def sample_foreground_distances(
    prediction: NDArray[np.bool_],
    reference: NDArray[np.bool_],
    spacing_mm: Sequence[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Amostra a distância ao voxel positivo mais próximo nos dois sentidos.

    As máscaras são booleanas, 3D e alinhadas. O espaçamento em mm segue a ordem
    dos eixos. Cada vetor corresponde à indexação booleana da máscara de origem.
    Retorna `None` quando pelo menos uma das máscaras está vazia.
    """
    if not prediction.any() or not reference.any():
        return None

    # O recorte mantém todos os voxels positivos nas duas máscaras.
    region = mask_bounding_box_slices(prediction | reference, spacing_mm, margin_mm=0)
    pred = prediction[region]
    ref = reference[region]
    return (
        distance_transform_edt(~ref, sampling=spacing_mm)[pred],
        distance_transform_edt(~pred, sampling=spacing_mm)[ref],
    )
