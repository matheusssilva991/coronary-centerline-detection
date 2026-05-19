"""Pacote utils para segmentação de artérias coronárias.

Os símbolos são carregados de forma preguiçosa para evitar imports caros ou
cíclicos durante a inicialização do pacote.
"""

from importlib import import_module

_LAZY_EXPORTS = {
    # Normalization
    "normalize_image": ".utils.normalization",
    "robust_normalize": ".utils.normalization",
    # IO / config / results / metrics / nifti / geometry
    "create_timestamped_output_dir": ".groups.io",
    "deep_update_dict": ".groups.io",
    "dice_score": ".groups.io",
    "extract_circular_region": ".groups.io",
    "extract_square_region": ".groups.io",
    "get_data_splits": ".groups.io",
    "load_config_json": ".groups.io",
    "load_img_and_label": ".groups.io",
    "load_json_file": ".groups.io",
    "load_raw_img_and_label": ".groups.io",
    "make_result_dataframe": ".groups.io",
    "merge_batch_results": ".groups.io",
    "normalize_runtime_config": ".groups.io",
    "save_config_json": ".groups.io",
    "save_json_file": ".groups.io",
    "save_metadata": ".groups.io",
    "save_nii_image": ".groups.io",
    "save_npy_array": ".groups.io",
    "save_results": ".groups.io",
    "scale_config_to_resolution": ".groups.io",
    "segment_by_hu": ".groups.io",
    "serialize_config_for_json": ".groups.io",
    # Processing / GPU / preprocessing
    "binary_closing": ".groups.processing",
    "binary_dilation": ".groups.processing",
    "binary_erosion": ".groups.processing",
    "binary_opening": ".groups.processing",
    "downscale_image": ".groups.processing",
    "downscale_image_ndi": ".groups.processing",
    "downscale_image_opencv": ".groups.processing",
    "get_array_module": ".groups.processing",
    "get_vesselness": ".groups.processing",
    "get_vesselness_optimized": ".groups.processing",
    "keep_largest_component": ".groups.processing",
    "label": ".groups.processing",
    "largest_connected_component": ".groups.processing",
    "load_vesselness_cache": ".groups.processing",
    "run_core_preprocessing_pipeline": ".groups.processing",
    "save_vesselness_cache": ".groups.processing",
    "threshold_image": ".groups.processing",
    "threshold_image_with_offset": ".groups.processing",
    "to_cpu": ".groups.processing",
    "to_gpu": ".groups.processing",
    "use_gpu": ".groups.processing",
    # Segmentation pipeline
    "calculate_robust_diameter": ".groups.segmentation",
    "check_ostium_intersection": ".groups.segmentation",
    "detect_and_evaluate_ostia": ".groups.segmentation",
    "detect_aorta_circles": ".groups.segmentation",
    "detect_initial_circle": ".groups.segmentation",
    "find_aorta_surface": ".groups.segmentation",
    "find_ostia": ".groups.segmentation",
    "get_or_compute_vesselness": ".groups.segmentation",
    "get_or_detect_aorta_circles": ".groups.segmentation",
    "get_or_segment_aorta": ".groups.segmentation",
    "get_initial_circle_diagnostics": ".groups.segmentation",
    "level_set_segmentation": ".groups.segmentation",
    "load_and_preprocess_image": ".groups.segmentation",
    "refine_circle_with_neighbors": ".groups.segmentation",
    "region_growing_article": ".groups.segmentation",
    "region_growing_segmentation": ".groups.segmentation",
    "remove_leaks_morphology": ".groups.segmentation",
    "segment_arteries_from_ostia": ".groups.segmentation",
}

_VISUALIZATION_EXPORTS = {
    "build_comparison_agg_df",
    "compare_shared_bad_cases",
    "get_bad_cases",
    "get_execution_time_seconds",
    "get_num_images",
    "get_total_success_percent",
    "load_ia_results_for_comparison",
    "load_math_results_for_comparison",
    "load_split_metadata",
    "load_split_summary",
    "map_ia_resolution_to_target",
    "plot_bad_cases_by_subset",
    "plot_comparison_bar_by_resolution",
    "plot_dice_distribution_by_subset",
    "plot_downscale_dice",
    "plot_downscale_execution_time",
    "plot_downscale_ostia_success",
    "plot_mip_projection",
    "plot_preprocessing_grid",
    "plot_stage",
    "compute_vesselness_maps",
    "plot_vesselness_mip_grid",
    "plot_vesselness_mip",
    "plot_hough_initial_diagnostics",
    "plot_hough_initial_circle",
    "plot_hough_refinement_candidates",
    "plot_hough_refined_circle",
    "plot_spaced_detected_circles",
    "plot_slices",
    "plot_subset_execution_time_by_resolution",
    "plot_subset_metric_by_resolution",
    "plot_subset_ostia_success_by_resolution",
    "plot_validation_dice",
    "plot_validation_execution_time",
    "plot_validation_ostia_success",
    "prettify_method_label",
    "visualize_arteries_comparison",
    "visualize_3d_k3d",
    "visualize_aorta_with_ostia",
    "visualize_circles_on_slices",
}


def __getattr__(name):
    if name in _VISUALIZATION_EXPORTS:
        module = import_module(".groups.visualization", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _LAZY_EXPORTS:
        module = import_module(_LAZY_EXPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "groups":
        module = import_module(".groups", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Ostia detection
    "find_aorta_surface",
    "calculate_robust_diameter",
    "check_ostium_intersection",
    "find_ostia",
    # Artery segmentation
    "region_growing_segmentation",
    "region_growing_article",
    # Aorta segmentation
    "level_set_segmentation",
    "remove_leaks_morphology",
    # Aorta localization
    "detect_initial_circle",
    "refine_circle_with_neighbors",
    "detect_aorta_circles",
    "get_initial_circle_diagnostics",
    # Frangi filter
    "get_vesselness",
    "get_vesselness_optimized",
    "save_vesselness_cache",
    "load_vesselness_cache",
    # GPU Utils
    "use_gpu",
    "to_gpu",
    "to_cpu",
    "get_array_module",
    # Config
    "deep_update_dict",
    "normalize_runtime_config",
    "serialize_config_for_json",
    "load_config_json",
    "save_config_json",
    "scale_config_to_resolution",
    # Dataset
    "get_data_splits",
    # Results
    "create_timestamped_output_dir",
    "make_result_dataframe",
    "merge_batch_results",
    "save_results",
    "save_metadata",
    "load_split_metadata",
    "load_split_summary",
    "get_bad_cases",
    "map_ia_resolution_to_target",
    "prettify_method_label",
    "load_ia_results_for_comparison",
    "load_math_results_for_comparison",
    "build_comparison_agg_df",
    "compare_shared_bad_cases",
    "get_execution_time_seconds",
    "get_num_images",
    "get_total_success_percent",
    "plot_bad_cases_by_subset",
    "plot_comparison_bar_by_resolution",
    "plot_dice_distribution_by_subset",
    "plot_downscale_dice",
    "plot_downscale_execution_time",
    "plot_downscale_ostia_success",
    "plot_subset_execution_time_by_resolution",
    "plot_subset_metric_by_resolution",
    "plot_subset_ostia_success_by_resolution",
    "plot_validation_dice",
    "plot_validation_execution_time",
    "plot_validation_ostia_success",
    # Pipeline steps
    "load_and_preprocess_image",
    "get_or_compute_vesselness",
    "get_or_detect_aorta_circles",
    "get_or_segment_aorta",
    "detect_and_evaluate_ostia",
    "segment_arteries_from_ostia",
    # Morphology
    "binary_closing",
    "binary_dilation",
    "binary_erosion",
    "binary_opening",
    "label",
    "keep_largest_component",
    # Preprocessing
    "downscale_image_ndi",
    "downscale_image_opencv",
    "downscale_image",
    "threshold_image",
    "threshold_image_with_offset",
    "largest_connected_component",
    "run_core_preprocessing_pipeline",
    "plot_preprocessing_grid",
    "plot_stage",
    "compute_vesselness_maps",
    "plot_vesselness_mip_grid",
    "plot_vesselness_mip",
    "plot_hough_initial_diagnostics",
    "plot_hough_initial_circle",
    "plot_hough_refinement_candidates",
    "plot_hough_refined_circle",
    "plot_spaced_detected_circles",
    # Plots
    "plot_mip_projection",
    "plot_slices",
    "visualize_circles_on_slices",
    "visualize_arteries_comparison",
    "visualize_3d_k3d",
    "visualize_aorta_with_ostia",
    # Utils
    "normalize_image",
    "robust_normalize",
    "load_json_file",
    "save_json_file",
    "load_img_and_label",
    "load_raw_img_and_label",
    "save_nii_image",
    "extract_square_region",
    "extract_circular_region",
    "segment_by_hu",
    "dice_score",
    "save_npy_array",
    # Grouped namespaces
    "groups",
]
