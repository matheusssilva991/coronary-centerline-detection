"""Alinha máscaras arteriais NIfTI e mede a sobreposição na grade do pipeline."""

from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine
from nibabel.processing import resample_from_to
from nibabel.spatialimages import SpatialImage
from numpy.typing import NDArray

from utils.utils.metrics import dice_score


def load_artery_prediction(path: Path) -> SpatialImage:
    """Carrega a geometria 3D de uma previsão NIfTI com affine espacial válido."""
    image = nib.load(path)
    if not isinstance(image, SpatialImage) or len(image.shape) != 3:
        raise ValueError(f"Previsão NIfTI 3D esperada: {path.name}")
    affine = image.affine
    if (
        affine is None
        or not np.isfinite(affine).all()
        or np.linalg.matrix_rank(affine[:3, :3]) != 3
    ):
        raise ValueError(f"Affine espacial inválido: {path.name}")
    return image


def binary_artery_mask(image: SpatialImage) -> NDArray[np.bool_]:
    """Lê a máscara preservando os eixos NIfTI, com valores 0 para fundo e 1 para artéria."""
    data = np.asanyarray(image.dataobj)
    mask = data == 1
    if np.count_nonzero(data) != np.count_nonzero(mask):
        raise ValueError("A máscara contém valores diferentes de 0 e 1.")
    return mask


def align_artery_prediction(
    prediction: SpatialImage, target: SpatialImage
) -> tuple[NDArray[np.bool_], bool, int]:
    """Alinha a máscara de IA à grade alvo por vizinho mais próximo.

    A transformação usa os affines NIfTI da mesma aquisição. Os centros da
    grade alvo devem estar no campo de visão da IA. Retorna a máscara alinhada,
    o indicador de reamostragem e a contagem de voxels positivos antes do ajuste.
    """
    if prediction.affine is None or target.affine is None:
        raise ValueError("As máscaras devem possuir affine.")
    transform = np.linalg.solve(prediction.affine, target.affine)
    same_grid = prediction.shape == target.shape and np.allclose(
        transform, np.eye(4), atol=1e-5, rtol=0
    )
    corners = list(product(*((0, size - 1) for size in target.shape)))
    mapped = apply_affine(transform, corners)
    if np.any(mapped < -0.5001) or np.any(
        mapped > np.asarray(prediction.shape) - 0.4999
    ):
        raise ValueError("O campo de visão da IA não cobre a grade do pipeline.")
    mask = binary_artery_mask(prediction)
    source_voxels = int(np.count_nonzero(mask))
    if same_grid:
        return mask, False, source_voxels
    # O modo nearest também representa a meia célula de suporte na borda da imagem.
    binary_image = nib.Nifti1Image(mask.astype(np.uint8), prediction.affine)
    aligned = resample_from_to(
        binary_image, (target.shape, target.affine), order=0, mode="nearest"
    )
    return np.asarray(aligned.dataobj, dtype=bool), True, source_voxels


def artery_pair_metrics(
    pipeline: NDArray[np.bool_],
    aligned_ai: NDArray[np.bool_],
    ai_source_voxels: int,
    voxel_volume_mm3: float,
) -> dict[str, float | int | str | None]:
    """Mede Dice e volumes na grade comum, excluindo pares em que ambas são vazias."""
    pipeline_voxels = int(np.count_nonzero(pipeline))
    ai_voxels = int(np.count_nonzero(aligned_ai))
    if pipeline_voxels == ai_voxels == 0:
        status = "both_empty"
        dice = None
    elif pipeline_voxels == 0:
        status, dice = "pipeline_empty", 0.0
    elif ai_voxels == 0:
        status = "ai_lost_in_resampling" if ai_source_voxels else "ai_empty"
        dice = 0.0
    else:
        status, dice = "ok", dice_score(pipeline, aligned_ai)
    return {
        "status": status,
        "dice": dice,
        "pipeline_voxels": pipeline_voxels,
        "ai_aligned_voxels": ai_voxels,
        "pipeline_volume_ml": pipeline_voxels * voxel_volume_mm3 / 1000,
        "ai_volume_ml": ai_voxels * voxel_volume_mm3 / 1000,
        "ai_source_voxels": ai_source_voxels,
    }
