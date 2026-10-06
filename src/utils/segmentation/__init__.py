"""Subpacote de segmentação com exports públicos carregados sob demanda.

Os módulos de segmentação são relativamente pesados porque podem importar
dependências de imagem, GPU e notebooks. Este arquivo mantém uma API curta em
``utils.segmentation`` e só importa cada módulo quando o símbolo é usado.
"""

from importlib import import_module

_SYMBOL_TO_MODULE = {
    # ostia_detection
    "calculate_robust_diameter": "ostia_detection",
    "check_ostium_intersection": "ostia_detection",
    "find_aorta_surface": "ostia_detection",
    "find_ostia": "ostia_detection",
    # fuzzy_connectedness
    "collect_local_object_seeds": "fuzzy.connectedness",
    "edge_affinity": "fuzzy.connectedness",
    "fuzzy_connectedness_map": "fuzzy.connectedness",
    "fuzzy_connectedness_segmentation": "fuzzy.connectedness",
    "limit_candidate_mask_by_vesselness": "fuzzy.connectedness",
    "neighbor_offsets_3d": "fuzzy.connectedness",
    "segment_artery_fuzzy_connectedness": "fuzzy.connectedness",
    "valid_seed": "fuzzy.connectedness",
    "vesselness_affinity": "fuzzy.connectedness",
    # aorta_segmentation
    "build_circle_trajectory_envelope": "aorta.segmentation",
    "classify_aorta_segmentation_feedback": "aorta.segmentation",
    "level_set_segmentation": "aorta.segmentation",
    "remove_leaks_morphology": "aorta.segmentation",
    "restrict_mask_to_circle_trajectory": "aorta.segmentation",
    # aorta_localization
    "detect_aorta_circles": "aorta.localization",
    "detect_initial_circle": "aorta.localization",
    "get_initial_circle_diagnostics": "aorta.localization",
    "refine_circle_with_neighbors": "aorta.localization",
    # pipeline modules
    "AortaCircleTrackingResult": "pipeline.detection",
    "ArterySegmentationResult": "pipeline.arteries",
    "detect_and_evaluate_ostia": "pipeline.detection",
    "detect_ostia": "pipeline.detection",
    "estimate_fuzzy_centers": "fuzzy.threshold",
    "fuzzy_threshold_from_config": "fuzzy.threshold",
    "fuzzy_threshold_outputs": "fuzzy.threshold",
    "get_thresholding_config": "fuzzy.threshold",
    "get_lower_threshold_config": "lower_threshold",
    "compute_vesselness": "pipeline.preprocessing",
    "locate_and_filter_aorta_circles": "pipeline.detection",
    "locate_aorta_circles": "pipeline.detection",
    "segment_aorta": "pipeline.detection",
    "load_and_preprocess_image": "pipeline.preprocessing",
    "preprocess_ccta_volume": "pipeline.preprocessing",
    "normalize_lower_threshold_method": "lower_threshold",
    "normal_region_growing_from_ostia": "artery_segmentation",
    "get_artery_postprocessing_stages": "pipeline.arteries",
    "postprocess_artery_mask": "pipeline.arteries",
    "resolve_lower_threshold": "lower_threshold",
    "segment_artery_masks_from_vesselness": "pipeline.arteries",
    "segment_arteries_from_ostia": "pipeline.arteries",
    "segment_arteries_from_vesselness": "pipeline.arteries",
    "summarize_aorta_circles": "aorta.diagnostics",
}

__all__ = list(_SYMBOL_TO_MODULE)


def __getattr__(name):
    if name in _SYMBOL_TO_MODULE:
        module = import_module(f".{_SYMBOL_TO_MODULE[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
