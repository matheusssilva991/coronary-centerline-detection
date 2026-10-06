"""Fachada compatível para relatórios e persistência de resultados.

As implementações ficam separadas por responsabilidade. Este módulo preserva
os imports históricos ``utils.project.results`` usados pelo pipeline e pelas
análises.
"""

from utils.project.results.io import (
    batch_result_number,
    create_timestamped_output_dir,
    get_batch_result_file,
    list_batch_result_files,
    merge_batch_results,
    save_results,
    save_dataframe_atomic,
)
from utils.project.results.metadata import (
    build_metadata,
    build_metadata_results,
    make_json_safe,
    save_metadata,
)
from utils.project.results.columns import (
    CANONICAL_COLUMN_NAMES,
    OSTIA_STATUS_INTERNAL_LABELS,
    OSTIA_STATUS_PORTUGUESE_LABELS,
    OSTIA_STATUS_READABLE_LABELS,
    READABLE_BOOL_COLUMNS,
    READABLE_COLUMN_NAMES,
    EDA_REQUIRED_RESULT_COLUMNS,
    EDA_REQUIRED_RESULT_COLUMN_UNION,
    RESULT_COLUMNS,
    RESULT_CONFIGURATION_COLUMNS,
    STATUS_LABELS,
    STATUS_PORTUGUESE_LABELS,
)
from utils.project.results.schema import (
    add_config_columns,
    add_internal_result_aliases,
    build_result_row,
    classify_result_status,
    make_readable_results_dataframe,
    make_result_dataframe,
    normalize_ostia_status,
    normalize_result_status,
    ostia_status_label_pt,
    result_status_label_pt,
    select_per_image_result_columns,
    summarize_results_df,
)
from utils.project.results.timing import (
    BATCH_TIMING_COLUMNS,
    batch_timing_manifest_path,
    duration_breakdown,
    load_batch_timing_records,
    save_batch_timing_record,
    summarize_batch_timing_records,
)
from utils.project.results.summary import (
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
    "OSTIA_STATUS_PORTUGUESE_LABELS",
    "OSTIA_STATUS_READABLE_LABELS",
    "READABLE_BOOL_COLUMNS",
    "READABLE_COLUMN_NAMES",
    "RESULT_COLUMNS",
    "RESULT_CONFIGURATION_COLUMNS",
    "ResultIntegrityError",
    "STATUS_LABELS",
    "STATUS_PORTUGUESE_LABELS",
    "add_config_columns",
    "add_internal_result_aliases",
    "batch_result_number",
    "batch_timing_manifest_path",
    "build_metadata",
    "build_metadata_results",
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
    "normalize_ostia_status",
    "normalize_result_status",
    "ostia_status_label_pt",
    "result_status_label_pt",
    "merge_batch_results",
    "save_batch_timing_record",
    "save_metadata",
    "save_results",
    "save_dataframe_atomic",
    "select_per_image_result_columns",
    "summarize_batch_timing_records",
    "summarize_results_df",
    "validate_result_integrity",
]
