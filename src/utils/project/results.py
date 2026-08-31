"""Fachada compatível para relatórios e persistência de resultados.

As implementações ficam separadas por responsabilidade. Este módulo preserva
os imports históricos ``utils.project.results`` usados pelo pipeline e pelas
análises.
"""

from .results_io import (
    batch_result_number,
    create_timestamped_output_dir,
    get_batch_result_file,
    list_batch_result_files,
    merge_batch_results,
    save_results,
)
from .results_metadata import build_metadata, make_json_safe, save_metadata
from .results_columns import (
    CANONICAL_COLUMN_NAMES,
    OSTIA_STATUS_INTERNAL_LABELS,
    OSTIA_STATUS_READABLE_LABELS,
    READABLE_BOOL_COLUMNS,
    READABLE_COLUMN_NAMES,
    RESULT_COLUMNS,
    STATUS_LABELS,
)
from .results_schema import (
    add_config_columns,
    add_internal_result_aliases,
    build_result_row,
    classify_result_status,
    make_readable_results_dataframe,
    make_result_dataframe,
    summarize_results_df,
)
from .results_timing import (
    BATCH_TIMING_COLUMNS,
    batch_timing_manifest_path,
    duration_breakdown,
    load_batch_timing_records,
    save_batch_timing_record,
    summarize_batch_timing_records,
)


__all__ = [
    "BATCH_TIMING_COLUMNS",
    "CANONICAL_COLUMN_NAMES",
    "OSTIA_STATUS_INTERNAL_LABELS",
    "OSTIA_STATUS_READABLE_LABELS",
    "READABLE_BOOL_COLUMNS",
    "READABLE_COLUMN_NAMES",
    "RESULT_COLUMNS",
    "STATUS_LABELS",
    "add_config_columns",
    "add_internal_result_aliases",
    "batch_result_number",
    "batch_timing_manifest_path",
    "build_metadata",
    "build_result_row",
    "classify_result_status",
    "create_timestamped_output_dir",
    "duration_breakdown",
    "get_batch_result_file",
    "list_batch_result_files",
    "load_batch_timing_records",
    "make_json_safe",
    "make_readable_results_dataframe",
    "make_result_dataframe",
    "merge_batch_results",
    "save_batch_timing_record",
    "save_metadata",
    "save_results",
    "summarize_batch_timing_records",
    "summarize_results_df",
]
