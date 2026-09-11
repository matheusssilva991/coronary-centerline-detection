"""Construção e persistência dos metadados compactos de uma execução."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .result_paths import metadata_filename


METADATA_SCHEMA_VERSION = 2
EFFECTIVE_CONFIG_RELATIVE_PATH = "../config/effective_pipeline_config.json"


def make_json_safe(value: Any) -> Any:
    """Converte valores comuns de pandas/numpy/pathlib para JSON nativo."""
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "as_posix"):
        return value.as_posix()
    if hasattr(value, "tolist"):
        return make_json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def effective_config_sha256(config: dict[str, Any]) -> str:
    """Calcula o hash determinístico do snapshot efetivo da configuração."""
    payload = json.dumps(
        make_json_safe(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _compact_configuration(
    config: dict[str, Any],
    legacy_result_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Mantém somente rótulos principais; parâmetros completos ficam no snapshot."""
    legacy = legacy_result_config or {}
    thresholding = config.get("THRESHOLDING") or {}
    artery = config.get("ARTERY_SEGMENTATION") or {}
    compact = {
        "use_gpu": _first_present(config.get("USE_GPU"), legacy.get("use_gpu")),
        "downscale_method": _first_present(
            config.get("DOWNSCALE_METHOD"), legacy.get("downscale_method")
        ),
        "downscale_factors": _first_present(
            config.get("DOWNSCALE_FACTORS"), legacy.get("downscale_factors")
        ),
        "min_threshold_hu": _first_present(
            config.get("MIN_THRESHOLD"), legacy.get("min_threshold_hu")
        ),
        "max_threshold_percentile": _first_present(
            config.get("MAX_THRESHOLD_PERCENTILE"),
            legacy.get("max_threshold_percentile"),
        ),
        "threshold_method": _first_present(
            thresholding.get("method"),
            legacy.get("threshold_method"),
            legacy.get("threshold_mode"),
        ),
        "artery_segmentation_method": _first_present(
            artery.get("method"),
            legacy.get("artery_segmentation_method"),
            legacy.get("configured_artery_segmentation_method"),
        ),
    }
    return make_json_safe(compact)


def build_metadata(
    split_name: str,
    config: dict[str, Any],
    *,
    resolution: str | None = None,
    config_source: str = "effective_pipeline_config",
    config_sha256: str | None = None,
    legacy_result_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Monta metadata portátil, sem IDs, tempos ou estatísticas deriváveis."""
    configuration = {
        "source": config_source,
        "sha256": (
            effective_config_sha256(config)
            if config_sha256 is None and config_source == "effective_pipeline_config"
            else (config_sha256 or "")
        ),
        "snapshot_file": (
            EFFECTIVE_CONFIG_RELATIVE_PATH
            if config_source == "effective_pipeline_config"
            else None
        ),
        **_compact_configuration(config, legacy_result_config),
    }
    return {
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "run": {
            "split": split_name,
            "resolution": resolution,
        },
        "configuration": configuration,
    }


def save_metadata(
    split_name: str,
    output_dir: str | Path,
    config: dict[str, Any],
    *,
    resolution: str | None = None,
    config_source: str = "effective_pipeline_config",
    config_sha256: str | None = None,
    legacy_result_config: dict[str, Any] | None = None,
) -> str:
    """Salva metadata compacto de forma atômica."""
    metadata = build_metadata(
        split_name,
        config,
        resolution=resolution,
        config_source=config_source,
        config_sha256=config_sha256,
        legacy_result_config=legacy_result_config,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / metadata_filename(split_name)
    temporary_path = metadata_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2, ensure_ascii=False)
    temporary_path.replace(metadata_path)
    return str(metadata_path)


__all__ = [
    "EFFECTIVE_CONFIG_RELATIVE_PATH",
    "METADATA_SCHEMA_VERSION",
    "build_metadata",
    "effective_config_sha256",
    "make_json_safe",
    "save_metadata",
]
