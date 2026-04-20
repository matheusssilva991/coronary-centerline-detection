"""Utilities package for IA vs mathematical comparison in EDA notebooks."""

from . import bad_cases, ia_math, io, metadata
from .io import load_split_metadata, load_split_summary
from .metadata import (
    get_execution_time_seconds,
    get_num_images,
    get_total_success_percent,
)
from .bad_cases import (
    build_bad_cases_export_df,
    filter_correct_ostia_cases,
    get_bad_cases,
    prepare_bad_cases_for_subset,
    save_bad_cases_artifacts,
    summarize_bad_dice_with_threshold,
)
from .ia_math import (
    build_comparison_agg_df,
    load_ia_results_for_comparison,
    load_math_results_for_comparison,
    map_ia_resolution_to_target,
    prettify_method_label,
)

__all__ = [
    "io",
    "metadata",
    "bad_cases",
    "ia_math",
    "load_split_metadata",
    "load_split_summary",
    "get_execution_time_seconds",
    "get_num_images",
    "get_total_success_percent",
    "get_bad_cases",
    "filter_correct_ostia_cases",
    "build_bad_cases_export_df",
    "save_bad_cases_artifacts",
    "prepare_bad_cases_for_subset",
    "summarize_bad_dice_with_threshold",
    "build_comparison_agg_df",
    "load_ia_results_for_comparison",
    "load_math_results_for_comparison",
    "map_ia_resolution_to_target",
    "prettify_method_label",
]
