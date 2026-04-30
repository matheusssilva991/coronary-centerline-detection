"""Grouped exports for image processing, morphology and acceleration utilities - lazy loaded."""

from importlib import import_module

_SYMBOL_TO_MODULE = {
    # binary_operations
    "binary_closing": "binary_operations",
    "binary_dilation": "binary_operations",
    "binary_erosion": "binary_operations",
    "binary_opening": "binary_operations",
    "keep_largest_component": "binary_operations",
    "label": "binary_operations",
    # frangi
    "get_vesselness": "frangi",
    "get_vesselness_optimized": "frangi",
    "load_vesselness_cache": "frangi",
    "save_vesselness_cache": "frangi",
    # gpu_utils
    "get_array_module": "gpu_utils",
    "to_cpu": "gpu_utils",
    "to_gpu": "gpu_utils",
    "use_gpu": "gpu_utils",
    # preprocessing
    "downscale_image": "preprocessing",
    "downscale_image_ndi": "preprocessing",
    "downscale_image_opencv": "preprocessing",
    "largest_connected_component": "preprocessing",
    "run_core_preprocessing_pipeline": "preprocessing",
    "threshold_image": "preprocessing",
    "threshold_image_with_offset": "preprocessing",
}


def __getattr__(name):
    if name in _SYMBOL_TO_MODULE:
        submodule = _SYMBOL_TO_MODULE[name]
        module = import_module(f"utils.processing.{submodule}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
