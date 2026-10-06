"""Construção e persistência dos metadados compactos de uma execução."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from utils.utils.json_io import make_json_safe, save_json_atomic

from utils.project.dataframe import require_series_column, to_numeric_series
from utils.project.results.paths import metadata_filename
from utils.project.results.schema import add_internal_result_aliases


METADATA_SCHEMA_VERSION = 3
EFFECTIVE_CONFIG_RELATIVE_PATH = "../config/effective_pipeline_config.json"


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
    """Retorna o primeiro valor disponível na ordem informada."""
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


def _truthy_series(dataframe: pd.DataFrame, column: str) -> pd.Series:
    """Normaliza flags legadas ou legíveis sem transformar ausentes em sucesso."""
    if column not in dataframe.columns:
        return pd.Series(False, index=dataframe.index, dtype=bool)
    values = dataframe[column]
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "sim", "s", "yes", "y"})


def _is_missing_scalar(value: Any) -> bool:
    """Reconhece valores escalares ausentes sem avaliar arrays como booleanos."""
    if value is None:
        return True
    missing = pd.isna(value)
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _has_text(value: Any) -> bool:
    """Indica se um valor escalar contém texto útil."""
    return not _is_missing_scalar(value) and bool(str(value).strip())


def _metric_summary(values: pd.Series) -> dict[str, Any]:
    """Resume a média e a quantidade válida de uma métrica."""
    numeric = to_numeric_series(values).dropna()
    return {
        "mean": float(np.asarray(numeric.mean()).item()) if not numeric.empty else None,
        "valid_exam_count": int(len(numeric)),
    }


def _percent_entry(count: int, total: int) -> dict[str, Any]:
    """Monta a contagem e o percentual usando o total processado."""
    return {
        "count": int(count),
        "percent": (float(count / total * 100) if total else None),
    }


def _execution_time(
    records: Sequence[dict[str, Any]],
    expected_batches: Sequence[int] | None,
) -> dict[str, float | None]:
    """Soma somente manifests completos, sem aceitar totais parciais."""
    invalid = False
    expected: set[int] | None = None
    if expected_batches is not None:
        expected = {int(number) for number in expected_batches}

    declared_totals: set[int] = set()
    for record in records:
        value = record.get("total_batches")
        if value is None or pd.isna(value):
            continue
        try:
            declared_total = int(value)
        except (TypeError, ValueError, OverflowError):
            invalid = True
            continue
        if declared_total <= 0:
            invalid = True
            continue
        declared_totals.add(declared_total)
    if len(declared_totals) > 1:
        invalid = True
    elif len(declared_totals) == 1:
        declared_expected = set(range(1, next(iter(declared_totals)) + 1))
        if expected is not None and expected != declared_expected:
            invalid = True
        expected = declared_expected

    durations: dict[int, float] = {}
    invalid = invalid or expected is None or not expected
    for record in records:
        try:
            batch_number = int(record["batch_number"])
            duration = float(record["duration_seconds"])
        except (KeyError, TypeError, ValueError, OverflowError):
            invalid = True
            continue
        if batch_number in durations or not math.isfinite(duration) or duration < 0:
            invalid = True
            continue
        durations[batch_number] = duration

    if invalid or set(durations) != expected:
        return {"seconds": None, "minutes": None, "hours": None}

    seconds = float(sum(durations.values()))
    return {
        "seconds": seconds,
        "minutes": seconds / 60,
        "hours": seconds / 3600,
    }


def build_metadata_results(
    results: pd.DataFrame,
    batch_timings: Sequence[dict[str, Any]],
    *,
    expected_batches: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Calcula os poucos agregados finais a partir dos artefatos persistidos."""
    dataframe = add_internal_result_aliases(results)
    total = len(dataframe)

    empty_numeric = pd.Series(index=dataframe.index, dtype=float)
    final_dice = (
        require_series_column(dataframe, "dice_artery")
        if "dice_artery" in dataframe.columns
        else empty_numeric
    )
    before_dice = (
        require_series_column(dataframe, "dice_artery_before_morphology")
        if "dice_artery_before_morphology" in dataframe.columns
        else empty_numeric
    )
    after_values = (
        require_series_column(dataframe, "dice_artery_after_morphology")
        if "dice_artery_after_morphology" in dataframe.columns
        else empty_numeric
    )
    explicit_after = to_numeric_series(after_values)
    # O Dice final histórico representa a saída após morfologia. O fallback é
    # aplicado por exame para também tolerar arquivos mistos.
    after_dice = explicit_after.fillna(to_numeric_series(final_dice))

    both_correct = _truthy_series(dataframe, "both_correct")
    both_tolerable = _truthy_series(dataframe, "both_tolerable") & ~both_correct
    ostia_found = _truthy_series(dataframe, "ostia_found")
    status_values = (
        require_series_column(dataframe, "ostia_status")
        if "ostia_status" in dataframe.columns
        else pd.Series("", index=dataframe.index, dtype=str)
    )
    statuses = (
        status_values.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    category_counts = {
        "both_correct": 0,
        "both_tolerable": 0,
        "found_but_incorrect": 0,
        "not_found": 0,
        "not_evaluated_or_error": 0,
    }
    for position, (_, row) in enumerate(dataframe.iterrows()):
        status = statuses.iloc[position]
        if status in {"both_correct", "both_ostia_correct"}:
            category = "both_correct"
        elif status in {"both_tolerable", "both_ostia_tolerable"}:
            category = "both_tolerable"
        elif status in {
            "found_but_wrong",
            "found_but_incorrect",
            "one_ostium_correct",
            "no_ostium_correct",
        }:
            category = "found_but_incorrect"
        elif status in {"not_found", "ostia_not_found"}:
            category = "not_found"
        elif status in {"not_evaluated", "pipeline_error", "error"}:
            category = "not_evaluated_or_error"
        elif both_correct.iloc[position]:
            category = "both_correct"
        elif both_tolerable.iloc[position]:
            category = "both_tolerable"
        elif _has_text(row.get("error")) or _has_text(row.get("ostia_error")):
            category = "not_evaluated_or_error"
        elif ostia_found.iloc[position]:
            category = "found_but_incorrect"
        elif "ostia_found" in dataframe.columns and not _is_missing_scalar(
            row.get("ostia_found")
        ):
            category = "not_found"
        else:
            category = "not_evaluated_or_error"
        category_counts[category] += 1

    success_count = category_counts["both_correct"] + category_counts["both_tolerable"]
    return make_json_safe(
        {
            "execution_time": _execution_time(batch_timings, expected_batches),
            "dice": {
                "before_morphology": _metric_summary(before_dice),
                "after_morphology": _metric_summary(after_dice),
            },
            "ostia": {
                "processed_exam_count": total,
                "success": _percent_entry(success_count, total),
                "types": {
                    name: _percent_entry(count, total)
                    for name, count in category_counts.items()
                },
                "sides": {
                    "left_correct": _percent_entry(
                        int(_truthy_series(dataframe, "left_intersects").sum()),
                        total,
                    ),
                    "right_correct": _percent_entry(
                        int(_truthy_series(dataframe, "right_intersects").sum()),
                        total,
                    ),
                },
            },
        }
    )


def build_metadata(
    split_name: str,
    config: dict[str, Any],
    *,
    resolution: str | None = None,
    config_source: str = "effective_pipeline_config",
    config_sha256: str | None = None,
    legacy_result_config: dict[str, Any] | None = None,
    results: pd.DataFrame | None = None,
    batch_timings: Sequence[dict[str, Any]] = (),
    expected_batches: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Monta metadata portátil com configuração e resultados essenciais."""
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
        "results": build_metadata_results(
            results if results is not None else pd.DataFrame(),
            batch_timings,
            expected_batches=expected_batches,
        ),
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
    results: pd.DataFrame | None = None,
    batch_timings: Sequence[dict[str, Any]] = (),
    expected_batches: Sequence[int] | None = None,
) -> str:
    """Salva metadata compacto de forma atômica."""
    metadata = build_metadata(
        split_name,
        config,
        resolution=resolution,
        config_source=config_source,
        config_sha256=config_sha256,
        legacy_result_config=legacy_result_config,
        results=results,
        batch_timings=batch_timings,
        expected_batches=expected_batches,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / metadata_filename(split_name)
    save_json_atomic(metadata, metadata_path)
    return str(metadata_path)


__all__ = [
    "EFFECTIVE_CONFIG_RELATIVE_PATH",
    "METADATA_SCHEMA_VERSION",
    "build_metadata",
    "build_metadata_results",
    "effective_config_sha256",
    "make_json_safe",
    "save_metadata",
]
