"""Utilitários de infraestrutura do projeto.

Este subpacote agrupa funções que não pertencem diretamente a processamento,
segmentação ou visualização: cache em disco, configuração, splits do dataset,
ambiente de notebooks e persistência de resultados.
"""

from .cache import load_json_cache, load_npy_cache, save_json_cache, save_npy_cache
from .config import (
    deep_update_dict,
    load_config_json,
    normalize_runtime_config,
    save_config_json,
    scale_config_to_resolution,
    serialize_config_for_json,
)
from .dataset import get_data_splits
from .notebook_env import (
    configure_notebook_environment,
    get_bad_cases_export_dir,
    get_cases_analysis_cache_dir,
    get_cases_analysis_output_dir,
    get_default_split_paths,
    resolve_existing_path,
    resolve_imagecas_base_path,
    resolve_processed_imagecas_path,
)
from .results import (
    add_internal_result_aliases,
    create_timestamped_output_dir,
    duration_breakdown,
    get_batch_result_file,
    list_batch_result_files,
    load_batch_timing_records,
    make_json_safe,
    make_readable_results_dataframe,
    make_result_dataframe,
    merge_batch_results,
    save_batch_timing_record,
    save_metadata,
    save_results,
    summarize_batch_timing_records,
)

__all__ = [
    "add_internal_result_aliases",
    "configure_notebook_environment",
    "create_timestamped_output_dir",
    "deep_update_dict",
    "duration_breakdown",
    "get_bad_cases_export_dir",
    "get_batch_result_file",
    "get_cases_analysis_cache_dir",
    "get_cases_analysis_output_dir",
    "get_data_splits",
    "get_default_split_paths",
    "list_batch_result_files",
    "load_batch_timing_records",
    "load_config_json",
    "load_json_cache",
    "load_npy_cache",
    "make_json_safe",
    "make_readable_results_dataframe",
    "make_result_dataframe",
    "merge_batch_results",
    "normalize_runtime_config",
    "resolve_existing_path",
    "resolve_imagecas_base_path",
    "resolve_processed_imagecas_path",
    "save_batch_timing_record",
    "save_config_json",
    "save_json_cache",
    "save_metadata",
    "save_npy_cache",
    "save_results",
    "scale_config_to_resolution",
    "serialize_config_for_json",
    "summarize_batch_timing_records",
]
