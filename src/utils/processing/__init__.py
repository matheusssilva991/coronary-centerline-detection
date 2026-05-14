"""Processing domain subpackage.

Contains morphology, vesselness, preprocessing and acceleration helpers.
"""

from .binary_operations import *
from .frangi import *
from .gpu_utils import *
from .preprocessing import *

__all__ = [
    # Binary operations
    "binary_closing",
    "binary_dilation",
    "binary_erosion",
    "binary_opening",
    "label",
    "keep_largest_component",
    # Frangi vesselness
    "get_gf",
    "get_gd",
    "get_vesselness",
    "get_vesselness_optimized",
    "save_vesselness_cache",
    "load_vesselness_cache",
    # GPU utilities
    "use_gpu",
    "to_gpu",
    "to_cpu",
    "get_array_module",
    "ensure_cpu",
    "ensure_gpu",
    # Preprocessing
    "downscale_image_ndi",
    "downscale_image_opencv",
    "downscale_image",
    "threshold_image",
    "threshold_image_with_offset",
    "largest_connected_component",
    "run_core_preprocessing_pipeline",
]
