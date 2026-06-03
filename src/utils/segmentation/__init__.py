"""Segmentation domain subpackage - public symbols loaded on demand."""

from importlib import import_module

_SYMBOL_TO_MODULE = {
    # ostia_detection
    "calculate_robust_diameter": "ostia_detection",
    "check_ostium_intersection": "ostia_detection",
    "find_aorta_surface": "ostia_detection",
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
    "get_initial_circle_diagnostics": "aorta_localization",
    "refine_circle_with_neighbors": "aorta_localization",
    # pipeline modules
    "build_lcc_image_from_mask": "fuzzy_threshold",
    "detect_and_evaluate_ostia": "pipeline_detection",
    "fuzzy_trapezoid_threshold": "fuzzy_threshold",
    "get_or_compute_vesselness": "pipeline_preprocessing",
    "get_or_detect_aorta_circles": "pipeline_detection",
    "get_or_segment_aorta": "pipeline_detection",
    "load_and_preprocess_image": "pipeline_preprocessing",
    "segment_arteries_from_ostia": "pipeline_arteries",
}

__all__ = list(_SYMBOL_TO_MODULE)


def __getattr__(name):
    if name in _SYMBOL_TO_MODULE:
        module = import_module(f".{_SYMBOL_TO_MODULE[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
