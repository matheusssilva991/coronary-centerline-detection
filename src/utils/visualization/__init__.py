"""Visualization domain subpackage - all exports loaded on-demand."""

from importlib import import_module

_SYMBOL_TO_MODULE = {
    # bad_cases
    "compare_shared_bad_cases": "bad_cases",
    "plot_bad_dice_indicator": "bad_cases",
    "change_status_label_for_plot": "bad_cases",
    "plot_bad_cases_by_subset": "bad_cases",
    # comparison
    "plot_comparison_bar_by_resolution": "comparison",
    "plot_image_dice_scatter_by_resolution": "comparison",
    "plot_ia_vs_math_scatter_by_resolution": "comparison",
    "plot_image_dice_scatter_interactive": "comparison",
    "plot_ia_vs_math_scatter_interactive": "comparison",
    "plot_dice_distribution_by_subset": "comparison",
    # segmentation_eda
    "build_dice_summary_by_subset": "segmentation_eda",
    "plot_distance_distribution_by_subset": "segmentation_eda",
    "plot_status_distribution_by_subset": "segmentation_eda",
    "plot_success_error_by_subset": "segmentation_eda",
    # images
    "plot_preprocessing_grid": "images",
    "plot_stage": "images",
    "plot_mip_projection": "images",
    "plot_slices": "images",
    "visualize_circles_on_slices": "images",
    # subset
    "prepare_subset_plot_df": "subset",
    "plot_subset_metric_by_resolution": "subset",
    "plot_subset_execution_time_by_resolution": "subset",
    "plot_subset_ostia_success_by_resolution": "subset",
    # volume
    "visualize_3d_k3d": "volume",
    "visualize_aorta_with_ostia": "volume",
    "visualize_arteries_comparison": "volume",
    "save_k3d_plot_html": "volume",
    # category
    "plot_category_metric_bar": "category",
    "plot_downscale_execution_time": "category",
    "plot_downscale_dice": "category",
    "plot_downscale_ostia_success": "category",
    "plot_validation_dice": "category",
    "plot_validation_execution_time": "category",
    "plot_validation_ostia_success": "category",
}


def __getattr__(name):
    if name in _SYMBOL_TO_MODULE:
        submodule = _SYMBOL_TO_MODULE[name]
        module = import_module(f".{submodule}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
