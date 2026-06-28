"""Pacote utilitário com exports preguiçosos e compatíveis.

Prefira imports diretos dos submódulos em código novo, por exemplo:

```
from utils.project.config import load_config_json
from utils.processing.preprocessing import downscale_image_ndi
```

Os exports abaixo continuam disponíveis para notebooks e scripts antigos.
"""

from importlib import import_module

_LAZY_EXPORTS = {
    # Config / dataset / results
    "deep_update_dict": ".project.config",
    "load_config_json": ".project.config",
    "normalize_runtime_config": ".project.config",
    "save_config_json": ".project.config",
    "scale_config_to_resolution": ".project.config",
    "serialize_config_for_json": ".project.config",
    "get_data_splits": ".project.dataset",
    "create_timestamped_output_dir": ".project.results",
    "get_batch_result_file": ".project.results",
    "list_batch_result_files": ".project.results",
    "make_result_dataframe": ".project.results",
    "merge_batch_results": ".project.results",
    "save_metadata": ".project.results",
    "save_results": ".project.results",
    # General utils
    "dice_score": ".utils.metrics",
    "extract_circular_region": ".utils.roi",
    "extract_square_region": ".utils.roi",
    "load_img_and_label": ".utils.nifti_io",
    "load_json_file": ".utils.json_io",
    "load_raw_img_and_label": ".utils.nifti_io",
    "normalize_image": ".utils.normalization",
    "normalize_vesselness": ".utils.normalization",
    "robust_normalize": ".utils.normalization",
    "save_json_file": ".utils.json_io",
    "save_nii_image": ".utils.nifti_io",
    "save_npy_array": ".utils.nifti_io",
    "segment_by_hu": ".utils.segmentation",
    # Processing
    "binary_closing": ".processing.binary_operations",
    "binary_dilation": ".processing.binary_operations",
    "binary_erosion": ".processing.binary_operations",
    "binary_opening": ".processing.binary_operations",
    "keep_largest_component": ".processing.binary_operations",
    "label": ".processing.binary_operations",
    "downscale_image": ".processing.preprocessing",
    "downscale_image_ndi": ".processing.preprocessing",
    "downscale_image_opencv": ".processing.preprocessing",
    "build_lcc_image_from_mask": ".processing.preprocessing",
    "largest_connected_component": ".processing.preprocessing",
    "run_core_preprocessing_pipeline": ".processing.preprocessing",
    "threshold_image": ".processing.preprocessing",
    "threshold_image_with_offset": ".processing.preprocessing",
    "get_array_module": ".processing.gpu_utils",
    "to_cpu": ".processing.gpu_utils",
    "to_gpu": ".processing.gpu_utils",
    "use_gpu": ".processing.gpu_utils",
    "get_modified_vesselness": ".processing.frangi",
    "get_vesselness": ".processing.frangi",
    "load_vesselness_cache": ".processing.frangi",
    "save_vesselness_cache": ".processing.frangi",
    # Segmentation
    "calculate_robust_diameter": ".segmentation.ostia_detection",
    "check_ostium_intersection": ".segmentation.ostia_detection",
    "find_aorta_surface": ".segmentation.ostia_detection",
    "find_ostia": ".segmentation.ostia_detection",
    "region_growing_article": ".segmentation.artery_segmentation",
    "level_set_segmentation": ".segmentation.aorta_segmentation",
    "remove_leaks_morphology": ".segmentation.aorta_segmentation",
    "detect_aorta_circles": ".segmentation.aorta_localization",
    "detect_initial_circle": ".segmentation.aorta_localization",
    "get_initial_circle_diagnostics": ".segmentation.aorta_localization",
    "refine_circle_with_neighbors": ".segmentation.aorta_localization",
    "detect_and_evaluate_ostia": ".segmentation.pipeline_detection",
    "get_or_detect_aorta_circles": ".segmentation.pipeline_detection",
    "get_or_segment_aorta": ".segmentation.pipeline_detection",
    "get_or_compute_vesselness": ".segmentation.pipeline_preprocessing",
    "load_and_preprocess_image": ".segmentation.pipeline_preprocessing",
    "segment_arteries_from_ostia": ".segmentation.pipeline_arteries",
    # Comparison helpers
    "build_comparison_agg_df": ".comparison_utils.ia_math",
    "filter_to_common_ia_math_ids": ".comparison_utils.ia_math",
    "get_common_ia_math_keys": ".comparison_utils.ia_math",
    "load_ia_results_for_comparison": ".comparison_utils.ia_math",
    "load_math_results_for_comparison": ".comparison_utils.ia_math",
    "map_ia_resolution_to_target": ".comparison_utils.ia_math",
    "prettify_method_label": ".comparison_utils.ia_math",
    "get_bad_cases": ".comparison_utils.bad_cases",
    "get_execution_time_seconds": ".comparison_utils.metadata",
    "get_num_images": ".comparison_utils.metadata",
    "get_total_success_percent": ".comparison_utils.metadata",
    "load_split_metadata": ".comparison_utils.io",
    "load_split_summary": ".comparison_utils.io",
    # Visualization
    "compare_shared_bad_cases": ".visualization.bad_cases",
    "plot_bad_cases_by_subset": ".visualization.bad_cases",
    "plot_comparison_bar_by_resolution": ".visualization.comparison",
    "plot_dice_distribution_by_subset": ".visualization.comparison",
    "plot_downscale_dice": ".visualization.category",
    "plot_downscale_execution_time": ".visualization.category",
    "plot_downscale_ostia_success": ".visualization.category",
    "plot_hough_initial_circle": ".visualization.hough",
    "plot_hough_initial_diagnostics": ".visualization.hough",
    "plot_hough_refined_circle": ".visualization.hough",
    "plot_hough_refinement_candidates": ".visualization.hough",
    "plot_mip_projection": ".visualization.image_slices",
    "plot_preprocessing_grid": ".visualization.preprocessing_views",
    "plot_spaced_detected_circles": ".visualization.hough",
    "plot_stage": ".visualization.preprocessing_views",
    "plot_subset_execution_time_by_resolution": ".visualization.subset",
    "plot_subset_metric_by_resolution": ".visualization.subset",
    "plot_subset_ostia_success_by_resolution": ".visualization.subset",
    "plot_validation_dice": ".visualization.category",
    "plot_validation_execution_time": ".visualization.category",
    "plot_validation_ostia_success": ".visualization.category",
    "compute_vesselness_maps": ".visualization.vesselness",
    "plot_vesselness_mip": ".visualization.vesselness",
    "plot_vesselness_mip_grid": ".visualization.vesselness",
    "plot_slices": ".visualization.image_slices",
    "add_pair_ostia_status_groups": ".visualization.variant_comparison",
    "normalize_ostia_status_group": ".visualization.variant_comparison",
    "select_qualitative_pair_cases": ".visualization.variant_comparison",
    "visualize_3d_k3d": ".visualization.volume",
    "visualize_aorta_ostia_artery": ".visualization.volume",
    "visualize_aorta_with_ostia": ".visualization.volume",
    "visualize_arteries_comparison": ".visualization.volume",
    "visualize_circles_on_slices": ".visualization.image_slices",
}


def __getattr__(name):
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


__all__ = sorted(_LAZY_EXPORTS) + ["groups"]
