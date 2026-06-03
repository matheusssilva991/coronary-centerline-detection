"""Funções auxiliares de I/O NIfTI e NumPy para volumes de imagem médica."""

import logging

import nibabel as nib
import numpy as np
from typing import Any, Optional, Tuple
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def load_img_and_label(
    img_path: str, label_path: Optional[str] = None
) -> Tuple[Optional[NDArray[Any]], Optional[NDArray[Any]]]:
    """Carrega imagem NIfTI e rótulo opcional como arrays NumPy.

    Args:
        img_path: Caminho para o arquivo de imagem NIfTI.
        label_path: Caminho opcional para arquivo de rótulo NIfTI.

    Returns:
        Tupla (img, label) com arrays NumPy ou None quando não informado.
    """
    img: Optional[NDArray[Any]] = None
    label: Optional[NDArray[Any]] = None

    if img_path:
        img = nib.load(img_path).get_fdata()
    if label_path:
        label = nib.load(label_path).get_fdata()

    return img, label


def load_raw_img_and_label(
    img_path: str, label_path: Optional[str] = None
) -> Tuple[Optional[Any], Optional[Any]]:
    """Carrega imagem NIfTI e rótulo opcional como objetos nibabel.

    Retorna objetos nibabel (ex.: Nifti1Image) ou None quando não informado.
    """
    img: Optional[Any] = None
    label: Optional[Any] = None

    if img_path:
        img = nib.load(img_path)
    if label_path:
        label = nib.load(label_path)

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
    nifti_img = nib.Nifti1Image(image, affine)

    try:
        nib.save(nifti_img, path_to_save)
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
