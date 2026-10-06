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
    "load_notebook_pipeline_config": ".project.runtime.notebook_env",
    "serialize_config_for_json": ".project.config",
    "get_data_splits": ".project.datasets.imagecas",
    "load_ccta_aorta_ground_truth": ".project.datasets.ccta",
    # General utils
    "dice_score": ".utils.metrics",
    "binary_segmentation_metrics": ".utils.metrics",
    "print_segmentation_metrics": ".utils.metrics",
    "extract_circular_region": ".utils.roi",
    "extract_square_region": ".utils.roi",
    "mask_bounding_box_slices": ".utils.roi",
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
    "get_vesselness": ".processing.frangi",
    # Segmentation
    "calculate_robust_diameter": ".segmentation.ostia_detection",
    "check_ostium_intersection": ".segmentation.ostia_detection",
    "find_aorta_surface": ".segmentation.ostia_detection",
    "find_ostia": ".segmentation.ostia_detection",
    "level_set_segmentation": ".segmentation.aorta.segmentation",
    "remove_leaks_morphology": ".segmentation.aorta.segmentation",
    "detect_aorta_circles": ".segmentation.aorta.localization",
    "detect_initial_circle": ".segmentation.aorta.localization",
    "get_initial_circle_diagnostics": ".segmentation.aorta.localization",
    "refine_circle_with_neighbors": ".segmentation.aorta.localization",
    "detect_and_evaluate_ostia": ".segmentation.pipeline.detection",
    "detect_ostia": ".segmentation.pipeline.detection",
    "locate_aorta_circles": ".segmentation.pipeline.detection",
    "segment_aorta": ".segmentation.pipeline.detection",
    "compute_vesselness": ".segmentation.pipeline.preprocessing",
    "load_and_preprocess_image": ".segmentation.pipeline.preprocessing",
    "preprocess_ccta_volume": ".segmentation.pipeline.preprocessing",
    "segment_arteries_from_ostia": ".segmentation.pipeline.arteries",
    "segment_arteries_from_vesselness": ".segmentation.pipeline.arteries",
    "get_artery_postprocessing_stages": ".segmentation.pipeline.arteries",
    "summarize_aorta_circles": ".segmentation.aorta.diagnostics",
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
    "summarize_split_results": ".comparison_utils.metadata",
    "load_split_batch_timings": ".comparison_utils.io",
    "load_split_metadata": ".comparison_utils.io",
    "load_split_results": ".comparison_utils.io",
    "load_split_summary": ".comparison_utils.io",
    # Visualization
    "plot_hough_initial_circle": ".visualization.pipeline.hough",
    "plot_hough_initial_diagnostics": ".visualization.pipeline.hough",
    "plot_hough_refined_circle": ".visualization.pipeline.hough",
    "plot_hough_refinement_candidates": ".visualization.pipeline.hough",
    "plot_mip_projection": ".visualization.images.image_slices",
    "save_volume_slice_figure": ".visualization.images.image_slices",
    "plot_preprocessing_grid": ".visualization.pipeline.preprocessing_views",
    "plot_pipeline_preprocessing_stages": ".visualization.pipeline.preprocessing_views",
    "plot_spaced_detected_circles": ".visualization.pipeline.hough",
    "plot_stage": ".visualization.pipeline.preprocessing_views",
    "save_detected_circles_figure": ".visualization.pipeline.pipeline_artifacts",
    "save_stage_views": ".visualization.pipeline.pipeline_artifacts",
    "compute_vesselness_maps": ".visualization.pipeline.vesselness",
    "display_vesselness_summary": ".visualization.pipeline.vesselness",
    "plot_vesselness_mip": ".visualization.pipeline.vesselness",
    "plot_vesselness_mip_grid": ".visualization.pipeline.vesselness",
    "plot_slices": ".visualization.images.image_slices",
    "add_pair_ostia_status_groups": ".comparison_utils.variant_comparison",
    "normalize_ostia_status_group": ".comparison_utils.variant_comparison",
    "select_qualitative_pair_cases": ".comparison_utils.variant_comparison",
    "visualize_3d_k3d": ".visualization.images.volume",
    "visualize_label_map_3d": ".visualization.images.volume",
    "visualize_aorta_ostia_artery": ".visualization.images.volume",
    "visualize_aorta_with_ostia": ".visualization.images.volume",
    "visualize_arteries_comparison": ".visualization.images.volume",
    "visualize_binary_masks_comparison": ".visualization.images.volume",
    "visualize_circles_on_slices": ".visualization.images.image_slices",
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
