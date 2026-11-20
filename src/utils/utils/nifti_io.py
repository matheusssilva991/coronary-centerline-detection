"""Funções auxiliares de I/O NIfTI e NumPy para volumes de imagem médica."""

import logging
from typing import Any, Optional, Tuple, cast

import numpy as np
from nibabel.loadsave import load as load_nifti
from nibabel.spatialimages import SpatialImage
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def load_img_and_label(
    img_path: str, label_path: Optional[str] = None
) -> Tuple[NDArray[Any], Optional[NDArray[Any]]]:
    """Carrega imagem NIfTI e rótulo opcional em arrays NumPy ``(img, label)``.

    Args:
        img_path (str): Caminho do arquivo NIfTI da imagem.
        label_path (Optional[str]): Caminho opcional do arquivo NIfTI do rótulo;
            ``None`` omite o rótulo.

    Returns:
        Tuple[NDArray[Any], Optional[NDArray[Any]]]: Tupla ``(img, label)`` com os
            dados em arrays NumPy (ou ``None`` no rótulo).
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

    Args:
        img_path (str): Caminho do arquivo NIfTI da imagem.
        label_path (Optional[str]): Caminho opcional do arquivo NIfTI do rótulo;
            ``None`` omite o rótulo.

    Returns:
        Tuple[SpatialImage, Optional[SpatialImage]]: Tupla ``(img, label)`` com
            objetos nibabel (``SpatialImage``) ou ``None`` no rótulo.
    """
    img = cast(SpatialImage, load_nifti(img_path))
    label: Optional[SpatialImage] = None

    if label_path:
        label = cast(SpatialImage, load_nifti(label_path))

    return img, label


__all__ = [
    "load_img_and_label",
    "load_raw_img_and_label",
]
