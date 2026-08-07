"""Funções auxiliares de I/O NIfTI e NumPy para volumes de imagem médica."""

import logging
from typing import Any, Optional, Tuple, cast

import numpy as np
from nibabel.loadsave import load as load_nifti
from nibabel.loadsave import save as save_nifti
from nibabel.nifti1 import Nifti1Image
from nibabel.spatialimages import SpatialImage
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def load_img_and_label(
    img_path: str, label_path: Optional[str] = None
) -> Tuple[NDArray[Any], Optional[NDArray[Any]]]:
    """Carrega imagem NIfTI e rótulo opcional como arrays NumPy.

    Args:
        img_path: Caminho para o arquivo de imagem NIfTI.
        label_path: Caminho opcional para arquivo de rótulo NIfTI.

    Returns:
        Tupla (img, label) com arrays NumPy ou None quando não informado.
    """
    img_object = cast(SpatialImage, load_nifti(img_path))
    img = np.asarray(img_object.get_fdata())
    label: Optional[NDArray[Any]] = None

    if label_path:
        label_object = cast(SpatialImage, load_nifti(label_path))
        label = np.asarray(label_object.get_fdata())

    return img, label


def load_raw_img_and_label(
    img_path: str, label_path: Optional[str] = None
) -> Tuple[SpatialImage, Optional[SpatialImage]]:
    """Carrega imagem NIfTI e rótulo opcional como objetos nibabel.

    Retorna objetos nibabel (ex.: Nifti1Image) ou None quando não informado.
    """
    img = cast(SpatialImage, load_nifti(img_path))
    label: Optional[SpatialImage] = None

    if label_path:
        label = cast(SpatialImage, load_nifti(label_path))

    return img, label


def save_nii_image(
    image: NDArray[Any], affine: NDArray[Any], path_to_save: str = "."
) -> None:
    """Salva um volume NumPy como imagem NIfTI.

    Args:
        image: Array NumPy do volume.
        affine: Matriz afin para o NIfTI.
        path_to_save: Caminho de saída.
    """
    nifti_img = Nifti1Image(image, affine)

    try:
        save_nifti(nifti_img, path_to_save)
        logger.info("Imagem salva em: %s", path_to_save)
    except Exception:
        logger.exception("Erro ao salvar a imagem em: %s", path_to_save)


def save_npy_array(array: NDArray[Any], path: str) -> None:
    """Salva um array em arquivo .npy."""
    try:
        np.save(path, array)
        logger.info("Array salvo em: %s", path)
    except Exception:
        logger.exception("Erro ao salvar o array em: %s", path)


__all__ = [
    "load_img_and_label",
    "load_raw_img_and_label",
    "save_nii_image",
    "save_npy_array",
]
