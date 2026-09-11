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
    EDA_REQUIRED_RESULT_COLUMNS,
    EDA_REQUIRED_RESULT_COLUMN_UNION,
    RESULT_COLUMNS,
    RESULT_CONFIGURATION_COLUMNS,
    STATUS_LABELS,
)
from .results_schema import (
    add_config_columns,
    add_internal_result_aliases,
    build_result_row,
    classify_result_status,
    make_readable_results_dataframe,
    make_result_dataframe,
    select_per_image_result_columns,
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
from .run_summary import (
    ResultIntegrityError,
    build_run_summary_row,
    effective_config_sha256,
    infer_run_identity,
    validate_result_integrity,
)


__all__ = [
    "BATCH_TIMING_COLUMNS",
    "CANONICAL_COLUMN_NAMES",
    "EDA_REQUIRED_RESULT_COLUMNS",
    "EDA_REQUIRED_RESULT_COLUMN_UNION",
    "OSTIA_STATUS_INTERNAL_LABELS",
    "OSTIA_STATUS_READABLE_LABELS",
    "READABLE_BOOL_COLUMNS",
    "READABLE_COLUMN_NAMES",
    "RESULT_COLUMNS",
    "RESULT_CONFIGURATION_COLUMNS",
    "ResultIntegrityError",
    "STATUS_LABELS",
    "add_config_columns",
    "add_internal_result_aliases",
    "batch_result_number",
    "batch_timing_manifest_path",
    "build_metadata",
    "build_result_row",
    "build_run_summary_row",
    "classify_result_status",
    "create_timestamped_output_dir",
    "duration_breakdown",
    "effective_config_sha256",
    "get_batch_result_file",
    "infer_run_identity",
    "list_batch_result_files",
    "load_batch_timing_records",
    "make_json_safe",
    "make_readable_results_dataframe",
    "make_result_dataframe",
    "merge_batch_results",
    "save_batch_timing_record",
    "save_metadata",
    "save_results",
    "select_per_image_result_columns",
    "summarize_batch_timing_records",
    "summarize_results_df",
    "validate_result_integrity",
]
