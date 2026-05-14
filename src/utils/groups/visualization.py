"""Grouped exports for plotting and EDA comparison visualization helpers."""

from importlib import import_module

_COMPARISON_EXPORTS = {
    "build_comparison_agg_df",
    "get_bad_cases",
    "get_execution_time_seconds",
    "get_num_images",
    "get_total_success_percent",
    "load_ia_results_for_comparison",
    "load_math_results_for_comparison",
    "load_split_metadata",
    "load_split_summary",
    "map_ia_resolution_to_target",
    "prettify_method_label",
}

_VISUALIZATION_EXPORTS = {
    "compare_shared_bad_cases",
    "plot_bad_cases_by_subset",
    "plot_category_metric_bar",
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
    "save_k3d_plot_html",
    "visualize_arteries_comparison",
    "visualize_3d_k3d",
    "visualize_aorta_with_ostia",
    "visualize_circles_on_slices",
}


def __getattr__(name):
    if name in _COMPARISON_EXPORTS:
        module = import_module("utils.comparison_utils")
    elif name in _VISUALIZATION_EXPORTS:
        module = import_module("utils.visualization")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(module, name)
    globals()[name] = value
    return value
