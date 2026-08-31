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
    "collect_local_object_seeds": "fuzzy_connectedness",
    "edge_affinity": "fuzzy_connectedness",
    "fuzzy_connectedness_map": "fuzzy_connectedness",
    "fuzzy_connectedness_segmentation": "fuzzy_connectedness",
    "limit_candidate_mask_by_vesselness": "fuzzy_connectedness",
    "neighbor_offsets_3d": "fuzzy_connectedness",
    "segment_artery_fuzzy_connectedness": "fuzzy_connectedness",
    "valid_seed": "fuzzy_connectedness",
    "vesselness_affinity": "fuzzy_connectedness",
    # aorta_segmentation
    "build_circle_trajectory_envelope": "aorta_segmentation",
    "level_set_segmentation": "aorta_segmentation",
    "remove_leaks_morphology": "aorta_segmentation",
    "restrict_mask_to_circle_trajectory": "aorta_segmentation",
    # aorta_localization
    "detect_aorta_circles": "aorta_localization",
    "detect_initial_circle": "aorta_localization",
    "get_initial_circle_diagnostics": "aorta_localization",
    "refine_circle_with_neighbors": "aorta_localization",
    # pipeline modules
    "detect_and_evaluate_ostia": "pipeline_detection",
    "estimate_fuzzy_centers": "fuzzy_threshold",
    "fuzzy_threshold_from_config": "fuzzy_threshold",
    "fuzzy_threshold_outputs": "fuzzy_threshold",
    "get_thresholding_config": "fuzzy_threshold",
    "get_lower_threshold_config": "lower_threshold",
    "compute_vesselness": "pipeline_preprocessing",
    "locate_aorta_circles": "pipeline_detection",
    "segment_aorta": "pipeline_detection",
    "load_and_preprocess_image": "pipeline_preprocessing",
    "normalize_lower_threshold_method": "lower_threshold",
    "normal_region_growing_from_ostia": "artery_segmentation",
    "get_artery_postprocessing_stages": "pipeline_arteries",
    "postprocess_artery_mask": "pipeline_arteries",
    "resolve_lower_threshold": "lower_threshold",
    "segment_arteries_from_ostia": "pipeline_arteries",
    "segment_arteries_from_vesselness": "pipeline_arteries",
    "summarize_aorta_circles": "pipeline_orchestration",
}

__all__ = list(_SYMBOL_TO_MODULE)


def __getattr__(name):
    if name in _SYMBOL_TO_MODULE:
        module = import_module(f".{_SYMBOL_TO_MODULE[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
