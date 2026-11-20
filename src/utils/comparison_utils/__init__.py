"""Reúne utilitários de comparação entre IA e métodos matemáticos nas EDAs."""

from importlib import import_module

_SUBMODULES = {
    "bad_cases",
    "ia_math",
    "io",
    "ostia_scenarios",
    "run_comparison",
    "segmentation_eda",
    "variant_comparison",
}

_SYMBOL_TO_MODULE = {
    # Análises por conjunto e variante.
    "build_success_status_summary_by_subset": "segmentation_eda",
    "CORRECT_LABELS": "variant_comparison",
    "OSTIA_STATUS_COLUMNS": "variant_comparison",
    "OSTIA_STATUS_GROUP_LABELS": "variant_comparison",
    "SUCCESS_LABELS": "variant_comparison",
    "TOLERABLE_LABELS": "variant_comparison",
    "WRONG_LABELS": "variant_comparison",
    "build_dice_stats_by_variant": "variant_comparison",
    "build_delta_summary_vs_reference": "variant_comparison",
    "largest_pair_changes": "variant_comparison",
    "make_pair_delta": "variant_comparison",
    "order_variants": "variant_comparison",
    "yes_no_to_bool": "variant_comparison",
    # io
    "build_split_resolution_summary": "io",
    "load_experiment_variant_results": "io",
    "load_split_batch_timings": "io",
    "load_split_results": "io",
    # bad_cases
    "filter_correct_ostia_cases": "bad_cases",
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
