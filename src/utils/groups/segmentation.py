"""Grouped exports for segmentation-related utilities - lazy loaded."""

from importlib import import_module

_SYMBOL_TO_MODULE = {
    # ostia_detection
    "find_aorta_surface": "ostia_detection",
    "calculate_robust_diameter": "ostia_detection",
    "check_ostium_intersection": "ostia_detection",
    "find_ostia": "ostia_detection",
    # artery_segmentation
    "region_growing_article": "artery_segmentation",
    "region_growing_segmentation": "artery_segmentation",
    # aorta_segmentation
    "level_set_segmentation": "aorta_segmentation",
    "remove_leaks_morphology": "aorta_segmentation",
    # aorta_localization
    "detect_aorta_circles": "aorta_localization",
    "detect_initial_circle": "aorta_localization",
    "refine_circle_with_neighbors": "aorta_localization",
    "get_initial_circle_diagnostics": "aorta_localization",
    # pipeline_steps
    "detect_and_evaluate_ostia": "pipeline_steps",
    "get_or_compute_vesselness": "pipeline_steps",
    "get_or_detect_aorta_circles": "pipeline_steps",
    "get_or_segment_aorta": "pipeline_steps",
    "load_and_preprocess_image": "pipeline_steps",
    "segment_arteries_from_ostia": "pipeline_steps",
}


def __getattr__(name):
    if name in _SYMBOL_TO_MODULE:
        submodule = _SYMBOL_TO_MODULE[name]
        module = import_module(f"utils.segmentation.{submodule}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
