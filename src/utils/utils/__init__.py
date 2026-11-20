"""Reúne submódulos utilitários gerais.

A API pública permanece disponível por ``utils.utils``.
"""

from .json_io import load_json_file, make_json_safe, save_json_atomic, save_json_file
from .metrics import dice_score
from .nifti_io import (
    load_img_and_label,
    load_raw_img_and_label,
)
from .normalization import normalize_image, normalize_vesselness, robust_normalize
from .roi import (
    mask_bounding_box_slices,
    mask_voxel_bbox_slices,
)

__all__ = [
    "normalize_image",
    "normalize_vesselness",
    "robust_normalize",
    "load_json_file",
    "save_json_file",
    "make_json_safe",
    "save_json_atomic",
    "load_img_and_label",
    "load_raw_img_and_label",
    "mask_bounding_box_slices",
    "mask_voxel_bbox_slices",
    "dice_score",
]
