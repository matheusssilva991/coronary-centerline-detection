"""Utilities package for IA vs mathematical comparison in EDA notebooks."""

from importlib import import_module

_SUBMODULES = {
    "bad_cases",
    "failure_analysis",
    "ia_math",
    "io",
    "metadata",
    "ostia_scenarios",
}

_SYMBOL_TO_MODULE = {
    # io
    "load_split_metadata": "io",
    "load_split_summary": "io",
    # metadata
    "get_execution_time_seconds": "metadata",
    "get_num_images": "metadata",
    "get_total_success_percent": "metadata",
    # bad_cases
    "build_bad_cases_export_df": "bad_cases",
    "filter_correct_ostia_cases": "bad_cases",
    "get_bad_cases": "bad_cases",
    "prepare_bad_cases_for_subset": "bad_cases",
    "save_bad_cases_artifacts": "bad_cases",
    "summarize_bad_dice_with_threshold": "bad_cases",
    # failure_analysis
    "build_failure_case_catalog": "failure_analysis",
    "select_focused_failure_cohort": "failure_analysis",
    "select_representative_failure_cases": "failure_analysis",
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
    "filter_ia_results_for_math_ids": "ostia_scenarios",
    "load_math_results_for_ostia_scenario": "ostia_scenarios",
    "load_ostia_comparison_scenario": "ostia_scenarios",
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
