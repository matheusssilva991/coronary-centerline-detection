"""Reúne visualizações carregadas sob demanda."""

from importlib import import_module

_SYMBOL_TO_MODULE = {
    # bad_cases
    # comparison
    "plot_grouped_metric_panels": "results.comparison",
    "plot_image_dice_scatter_by_resolution": "results.comparison",
    "plot_ia_vs_math_scatter_by_resolution": "results.comparison",
    # segmentation_eda
    "build_dice_summary_by_subset": "utils.comparison_utils.segmentation_eda",
    # image_slices
    "plot_mip_projection": "images.image_slices",
    "plot_slices": "images.image_slices",
    "plot_volume_slice": "images.image_slices",
    "save_volume_slice_figure": "images.image_slices",
    "visualize_circles_on_slices": "images.image_slices",
    # intensity
    "calculate_binned_intensity_mean_median": "images.intensity",
    "plot_binned_intensity_histogram": "images.intensity",
    # preprocessing_views
    "plot_preprocessing_grid": "pipeline.preprocessing_views",
    "plot_pipeline_preprocessing_stages": "pipeline.preprocessing_views",
    "plot_stage": "pipeline.preprocessing_views",
    # pipeline_artifacts
    "save_detected_circles_figure": "pipeline.pipeline_artifacts",
    "save_stage_views": "pipeline.pipeline_artifacts",
    # vesselness
    "compute_vesselness_maps": "pipeline.vesselness",
    "display_vesselness_summary": "pipeline.vesselness",
    "plot_vesselness_mip_grid": "pipeline.vesselness",
    "plot_vesselness_mip": "pipeline.vesselness",
    # hough
    "plot_hough_initial_diagnostics": "pipeline.hough",
    "plot_hough_initial_circle": "pipeline.hough",
    "plot_hough_refinement_candidates": "pipeline.hough",
    "plot_hough_refined_circle": "pipeline.hough",
    "plot_spaced_detected_circles": "pipeline.hough",
    # subset
    # volume
    "visualize_3d_k3d": "images.volume",
    "visualize_label_map_3d": "images.volume",
    "visualize_aorta_ostia_artery": "images.volume",
    "visualize_aorta_with_ostia": "images.volume",
    "visualize_arteries_comparison": "images.volume",
    "visualize_binary_masks_comparison": "images.volume",
    "save_k3d_plot_html": "images.volume",
    # variant_comparison
    "best_variant_by_suffix": "utils.comparison_utils.variant_comparison",
    "add_pair_ostia_status_groups": "utils.comparison_utils.variant_comparison",
    "build_dice_stats_by_variant": "utils.comparison_utils.variant_comparison",
    "build_delta_summary_vs_reference": "utils.comparison_utils.variant_comparison",
    "build_pair_outcome_counts": "utils.comparison_utils.variant_comparison",
    "build_pair_curve_auc": "utils.comparison_utils.variant_comparison",
    "build_ranking_table": "utils.comparison_utils.variant_comparison",
    "largest_pair_changes": "utils.comparison_utils.variant_comparison",
    "load_variant_results": "utils.comparison_utils.variant_comparison",
    "make_pair_delta": "utils.comparison_utils.variant_comparison",
    "normalize_ostia_status_group": "utils.comparison_utils.variant_comparison",
    "pair_summary": "utils.comparison_utils.variant_comparison",
    "plot_largest_pair_changes": "results.variant_comparison",
    "select_qualitative_pair_cases": "utils.comparison_utils.variant_comparison",
}


def __getattr__(name):
    if name in _SYMBOL_TO_MODULE:
        submodule = _SYMBOL_TO_MODULE[name]
        module = import_module(
            submodule if submodule.startswith("utils.") else f".{submodule}",
            __name__,
        )
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
