"""Utilitários de infraestrutura do projeto.

Este subpacote agrupa funções que não pertencem diretamente a processamento,
segmentação ou visualização: configuração, splits do dataset,
ambiente de notebooks e persistência de resultados.
"""

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
    get_cases_analysis_output_dir,
    get_default_split_paths,
    resolve_existing_path,
    resolve_imagecas_base_path,
)

__all__ = [
    "configure_notebook_environment",
    "deep_update_dict",
    "get_bad_cases_export_dir",
    "get_cases_analysis_output_dir",
    "get_data_splits",
    "get_default_split_paths",
    "load_config_json",
    "normalize_runtime_config",
    "resolve_existing_path",
    "resolve_imagecas_base_path",
    "save_config_json",
    "scale_config_to_resolution",
    "serialize_config_for_json",
]
