"""Reúne utilitários de comparação entre IA e métodos matemáticos nas EDAs."""

from importlib import import_module

_SUBMODULES = {
    "bad_cases",
    "failure_analysis",
    "ia_math",
    "io",
    "metadata",
    "ostia_scenarios",
    "run_comparison",
    "segmentation_eda",
    "variant_comparison",
}

_SYMBOL_TO_MODULE = {
    # Análises por conjunto e variante.
    "build_success_status_summary_by_subset": "segmentation_eda",
    "build_dice_summary_by_subset": "segmentation_eda",
    "CORRECT_LABELS": "variant_comparison",
    "OSTIA_STATUS_COLUMNS": "variant_comparison",
    "OSTIA_STATUS_GROUP_LABELS": "variant_comparison",
    "SUCCESS_LABELS": "variant_comparison",
    "TOLERABLE_LABELS": "variant_comparison",
    "WRONG_LABELS": "variant_comparison",
    "add_pair_ostia_status_groups": "variant_comparison",
    "best_variant_by_suffix": "variant_comparison",
    "build_dice_stats_by_variant": "variant_comparison",
    "build_delta_summary_vs_reference": "variant_comparison",
    "build_pair_outcome_counts": "variant_comparison",
    "build_pair_curve_auc": "variant_comparison",
    "build_ranking_table": "variant_comparison",
    "first_existing_value": "variant_comparison",
    "largest_pair_changes": "variant_comparison",
    "load_variant_results": "variant_comparison",
    "load_variant_run": "variant_comparison",
    "make_pair_delta": "variant_comparison",
    "normalize_ostia_status_group": "variant_comparison",
    "order_variants": "variant_comparison",
    "pair_summary": "variant_comparison",
    "select_qualitative_pair_cases": "variant_comparison",
    "yes_no_to_bool": "variant_comparison",
    # io
    "load_split_metadata": "io",
    "load_split_batch_timings": "io",
    "load_split_results": "io",
    "load_split_summary": "io",
    # metadata
    "get_execution_time_seconds": "metadata",
    "get_num_images": "metadata",
    "get_total_success_percent": "metadata",
    "build_split_resolution_summary": "metadata",
    "summarize_split_results": "metadata",
    # bad_cases
    "build_bad_cases_export_df": "bad_cases",
    "filter_correct_ostia_cases": "bad_cases",
    "get_bad_cases": "bad_cases",
    "prepare_bad_cases_for_subset": "bad_cases",
    "save_bad_cases_artifacts": "bad_cases",
    "summarize_bad_dice_with_threshold": "bad_cases",
    # failure_analysis
    "build_failure_case_catalog": "failure_analysis",
    "compact_focused_failure_cohort": "failure_analysis",
    "select_focused_failure_cohort": "failure_analysis",
    "summarize_failure_categories": "failure_analysis",
    # ia_math
    "build_comparison_agg_df": "ia_math",
    "filter_to_common_ia_math_ids": "ia_math",
    "get_common_ia_math_keys": "ia_math",
    "load_ia_results_for_comparison": "ia_math",
    "load_math_results_for_comparison": "ia_math",
    "map_ia_resolution_to_target": "ia_math",
    "prettify_method_label": "ia_math",
    # ostia_scenarios
    "build_ostia_image_comparison_df": "ostia_scenarios",
    "load_math_results_for_ostia_scenario": "ostia_scenarios",
    "load_ostia_comparison_scenario": "ostia_scenarios",
    # run_comparison
    "OSTIA_SUCCESS_STATUSES": "run_comparison",
    "build_dice_ostia_overview": "run_comparison",
    "compare_paired_run_matrix": "run_comparison",
    "load_validated_comparison_runs": "run_comparison",
    "ostia_success_mask": "run_comparison",
}

__all__ = sorted(_SUBMODULES) + list(_SYMBOL_TO_MODULE)


def __getattr__(name):
    if name in _SUBMODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _SYMBOL_TO_MODULE:
        module = import_module(f".{_SYMBOL_TO_MODULE[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
