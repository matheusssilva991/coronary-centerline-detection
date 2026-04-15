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
    get_bad_cases,
    save_bad_cases_artifacts,
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
    "build_bad_cases_export_df",
    "save_bad_cases_artifacts",
    "build_comparison_agg_df",
    "load_ia_results_for_comparison",
    "load_math_results_for_comparison",
    "map_ia_resolution_to_target",
    "prettify_method_label",
]
