"""Funções auxiliares de I/O NIfTI e NumPy para volumes de imagem médica."""

import nibabel as nib
import numpy as np
from numpy.typing import NDArray


def load_img_and_label(
    img_path: str, label_path: str = None
) -> tuple[NDArray, NDArray]:
    """Carrega imagem NIfTI e rótulo opcional como arrays NumPy."""
    img, label = None, None

    if img_path:
        img = nib.load(img_path).get_fdata()
    if label_path:
        label = nib.load(label_path).get_fdata()

    return img, label


def load_raw_img_and_label(img_path: str, label_path: str = None) -> tuple:
    """Carrega imagem NIfTI e rótulo opcional como objetos nibabel."""
    img, label = None, None

    if img_path:
        img = nib.load(img_path)
    if label_path:
        label = nib.load(label_path)

    return img, label


def save_nii_image(image, affine, path_to_save="."):
    """Salva um volume NumPy como imagem NIfTI."""
    nifti_img = nib.Nifti1Image(image, affine)

    try:
        nib.save(nifti_img, path_to_save)
        print("Imagem salva em:", path_to_save, ".")
    except Exception as e:
        print("Erro ao salvar a imagem:", e)


def save_npy_array(array: NDArray, path: str):
    """Salva um array em arquivo .npy."""
    try:
        np.save(path, array)
        print(f"Array salvo em: {path}")
    except Exception as e:
        print(f"Erro ao salvar o array: {e}")


__all__ = [
    "load_img_and_label",
    "load_raw_img_and_label",
    "save_nii_image",
    "save_npy_array",
]
