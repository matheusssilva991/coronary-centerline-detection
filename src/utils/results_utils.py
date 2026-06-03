"""Utilities para criação de relatórios e persistência de resultados."""

from __future__ import annotations

import json
import platform
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


RESULT_COLUMNS = [
    "IMG_ID",
    "dice_artery",
    "artery_voxels",
    "ostia_found",
    "ostia_status",
    "segmentation_attempted",
    "proceeded_with_bad_ostia",
    "skip_reason",
    "ostia_error",
    "both_correct",
    "both_tolerable",
    "left_intersects",
    "right_intersects",
    "left_dist_mm",
    "right_dist_mm",
    "ostia_left",
    "ostia_right",
    "error",
    "status",
]

READABLE_COLUMN_NAMES = {
    "dice_artery": "artery_dice",
    "artery_voxels": "artery_voxel_count",
    "ostia_found": "ostia_detected",
    "ostia_status": "ostia_detection_status",
    "segmentation_attempted": "artery_segmentation_run",
    "proceeded_with_bad_ostia": "segmented_with_incorrect_ostia",
    "skip_reason": "segmentation_skip_reason",
    "ostia_error": "ostia_detection_error",
    "both_correct": "both_ostia_correct",
    "both_tolerable": "both_ostia_tolerable",
    "left_intersects": "left_ostium_correct",
    "right_intersects": "right_ostium_correct",
    "left_dist_mm": "left_ostium_distance_mm",
    "right_dist_mm": "right_ostium_distance_mm",
    "ostia_left": "left_ostium",
    "ostia_right": "right_ostium",
    "error": "pipeline_error",
    "downscale_method": "downscale_method",
    "opencv_interpolation": "opencv_interpolation",
    "downscale_factors": "downscale_factors",
    "max_threshold_percentile": "max_threshold_percentile",
}

CANONICAL_COLUMN_NAMES = {value: key for key, value in READABLE_COLUMN_NAMES.items()}

READABLE_BOOL_COLUMNS = {
    "ostia_detected",
    "artery_segmentation_run",
    "segmented_with_incorrect_ostia",
    "both_ostia_correct",
    "both_ostia_tolerable",
    "left_ostium_correct",
    "right_ostium_correct",
}

OSTIA_STATUS_READABLE_LABELS = {
    "not_evaluated": "not evaluated",
    "not_found": "not found",
    "both_correct": "both correct",
    "both_tolerable": "both tolerable",
    "found_but_wrong": "found but incorrect",
}

STATUS_LABELS = {
    "not_found": "ostia not found",
    "both_correct": "both ostia correct",
    "both_tolerable": "both ostia tolerable",
    "one_correct": "one ostium correct",
    "error": "pipeline error",
    "none_correct": "no ostium correct",
}


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


def _readable_column_name(column: str) -> str:
    return READABLE_COLUMN_NAMES.get(column, column)


def _get_result_value(result: dict[str, Any], column: str, default: Any = None) -> Any:
    if column in result:
        return result.get(column)

    readable_column = _readable_column_name(column)
    if readable_column in result:
        return result.get(readable_column)

    return default


def _as_bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)

    normalized = str(value).strip().lower()
    return normalized in {"true", "1", "sim", "s", "yes", "y"}


def _format_bool_readable(value: Any) -> str:
    return "yes" if _as_bool_value(value) else "no"


def make_readable_results_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas técnicas do resultado para nomes/valores mais legíveis."""
    readable_df = df.rename(columns=READABLE_COLUMN_NAMES).copy()

    for column in READABLE_BOOL_COLUMNS.intersection(readable_df.columns):
        readable_df[column] = readable_df[column].map(_format_bool_readable)

    for column in ("ostia_detection_status", "status_ostios"):
        if column in readable_df.columns:
            readable_df[column] = readable_df[column].map(
                lambda value: OSTIA_STATUS_READABLE_LABELS.get(value, value)
            )

    return readable_df


def _series_from_aliases(df: pd.DataFrame, column: str, dtype=None) -> pd.Series:
    column_candidates = (column, _readable_column_name(column))
    for column_candidate in column_candidates:
        if column_candidate in df.columns:
            return df[column_candidate]
    return pd.Series(index=df.index, dtype=dtype)


def create_timestamped_output_dir(base_output_dir, experiment_name="segmentation"):
    """Cria diretório de saída com timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(base_output_dir) / experiment_name / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def classify_result_status(result: dict[str, Any]) -> str:
    """Classifica uma linha de resultado no rótulo textual usado nos CSVs."""
    if result.get("ostia_status") == "not_found":
        return STATUS_LABELS["not_found"]
    if result.get("both_correct", False):
        return STATUS_LABELS["both_correct"]
    if result.get("both_tolerable", False):
        return STATUS_LABELS["both_tolerable"]
    if result.get("left_intersects", False) or result.get("right_intersects", False):
        return STATUS_LABELS["one_correct"]
    if result.get("error"):
        return STATUS_LABELS["error"]
    return STATUS_LABELS["none_correct"]


def build_result_row(result: dict[str, Any]) -> dict[str, Any]:
    """Converte um resultado bruto do pipeline em uma linha CSV padronizada."""
    row = {
        "IMG_ID": result.get("IMG_ID"),
        "dice_artery": _get_result_value(result, "dice_artery"),
        "artery_voxels": _get_result_value(result, "artery_voxels"),
        "ostia_found": _as_bool_value(_get_result_value(result, "ostia_found", False)),
        "ostia_status": _get_result_value(result, "ostia_status"),
        "segmentation_attempted": _as_bool_value(
            _get_result_value(result, "segmentation_attempted", False)
        ),
        "proceeded_with_bad_ostia": _as_bool_value(
            _get_result_value(result, "proceeded_with_bad_ostia", False)
        ),
        "skip_reason": _get_result_value(result, "skip_reason"),
        "ostia_error": _get_result_value(result, "ostia_error"),
        "both_correct": _as_bool_value(
            _get_result_value(result, "both_correct", False)
        ),
        "both_tolerable": _as_bool_value(
            _get_result_value(result, "both_tolerable", False)
        ),
        "left_intersects": _as_bool_value(
            _get_result_value(result, "left_intersects", False)
        ),
        "right_intersects": _as_bool_value(
            _get_result_value(result, "right_intersects", False)
        ),
        "left_dist_mm": _get_result_value(result, "left_dist_mm"),
        "right_dist_mm": _get_result_value(result, "right_dist_mm"),
        "ostia_left": _get_result_value(result, "ostia_left"),
        "ostia_right": _get_result_value(result, "ostia_right"),
        "error": _get_result_value(result, "error", None),
    }
    row["status"] = result.get("status") or classify_result_status(row)
    return row


def make_result_dataframe(results):
    """Converte lista de resultados em DataFrame formatado."""
    rows = [build_result_row(result) for result in results]
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def add_config_columns(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Adiciona ao DataFrame as colunas de configuração salvas no CSV."""
    df = df.copy()
    df["downscale_method"] = config.get("DOWNSCALE_METHOD", "N/A")
    df["opencv_interpolation"] = (
        config.get("OPENCV_INTERPOLATION", "N/A")
        if config.get("DOWNSCALE_METHOD") == "opencv"
        else "N/A"
    )
    df["downscale_factors"] = str(config.get("DOWNSCALE_FACTORS", "N/A"))
    df["max_threshold_percentile"] = config.get("MAX_THRESHOLD_PERCENTILE", "N/A")
    return df


def save_results(results, split_name, output_dir, config=None):
    """Salva resultados em CSV."""
    df = make_result_dataframe(results)
    if config is not None:
        df = add_config_columns(df, config)
    df = make_readable_results_dataframe(df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ostios_{split_name}_summary.csv"
    df.to_csv(output_path, index=False)
    return str(output_path)


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    series = _series_from_aliases(df, column, dtype=bool)
    return series.map(_as_bool_value)


def summarize_results_df(df: pd.DataFrame) -> dict[str, Any]:
    """Calcula contagens e métricas agregadas de um DataFrame de resultados."""
    both_correct_series = _bool_series(df, "both_correct")
    both_tolerable_series = _bool_series(df, "both_tolerable")
    ostia_found_series = _bool_series(df, "ostia_found")
    segmentation_attempted_series = _bool_series(df, "segmentation_attempted")
    proceeded_with_bad_ostia_series = _bool_series(df, "proceeded_with_bad_ostia")
    ostia_status_series = _series_from_aliases(df, "ostia_status")
    ostia_status_normalized = ostia_status_series.fillna("").astype(str).str.lower()
    ostia_not_found_series = ostia_status_normalized.isin(
        {"not_found", "not found", "não encontrados", "óstios não encontrados"}
    )
    error_series = _series_from_aliases(df, "error")
    dice_series = pd.to_numeric(
        _series_from_aliases(df, "dice_artery", dtype=float), errors="coerce"
    )

    total_success_series = both_correct_series | both_tolerable_series
    summary = {
        "total_processed": len(df),
        "ostia_found": int(ostia_found_series.sum()),
        "ostia_found_percent": float(ostia_found_series.mean() * 100),
        "ostia_status_not_found": int(ostia_not_found_series.sum()),
        "ostia_status_not_found_percent": float(ostia_not_found_series.mean() * 100),
        "both_correct": int(both_correct_series.sum()),
        "both_correct_percent": float(both_correct_series.mean() * 100),
        "both_tolerable": int(both_tolerable_series.sum()),
        "both_tolerable_percent": float(both_tolerable_series.mean() * 100),
        "segmentation_attempted": int(segmentation_attempted_series.sum()),
        "segmentation_attempted_percent": float(
            segmentation_attempted_series.mean() * 100
        ),
        "proceeded_with_bad_ostia": int(proceeded_with_bad_ostia_series.sum()),
        "proceeded_with_bad_ostia_percent": float(
            proceeded_with_bad_ostia_series.mean() * 100
        ),
        "total_success": int(total_success_series.sum()),
        "total_success_percent": float(total_success_series.mean() * 100),
        "left_correct": int(_bool_series(df, "left_intersects").sum()),
        "right_correct": int(_bool_series(df, "right_intersects").sum()),
        "error_not_null": int(error_series.notna().sum()),
    }

    if dice_series.notna().any():
        summary.update(
            {
                "dice_artery_mean": float(dice_series.mean()),
                "dice_artery_std": float(dice_series.std()),
                "dice_artery_median": float(dice_series.median()),
            }
        )
    else:
        summary.update(
            {
                "dice_artery_mean": None,
                "dice_artery_std": None,
                "dice_artery_median": None,
            }
        )
    return summary


def _vesselness_metadata(config: dict[str, Any], key: str) -> dict[str, Any]:
    vesselness_config = config[key]
    sigmas = vesselness_config["sigmas"]
    return {
        "sigmas": sigmas.tolist() if hasattr(sigmas, "tolist") else list(sigmas),
        "alpha": vesselness_config["alpha"],
        "beta": vesselness_config["beta"],
        "gamma": vesselness_config["gamma"],
    }


def build_metadata(
    split_name,
    config,
    ids,
    results,
    execution_time=None,
    base_path=None,
    base_save_path=None,
    root_output_dir=None,
):
    """Monta a estrutura JSON de metadados sem gravar arquivo."""
    df = make_result_dataframe(results)
    results_summary = summarize_results_df(df)

    metadata = {
        "execution_info": {
            "timestamp": datetime.now().isoformat(),
            "split_name": split_name,
            "num_images": len(ids),
            "image_ids": ids,
            "execution_time_seconds": execution_time,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "state_counters": {
                "ostia_found": results_summary["ostia_found"],
                "ostia_status_not_found": results_summary["ostia_status_not_found"],
                "segmentation_attempted": results_summary["segmentation_attempted"],
                "proceeded_with_bad_ostia": results_summary[
                    "proceeded_with_bad_ostia"
                ],
                "error_not_null": results_summary["error_not_null"],
            },
        },
        "preprocessing_config": {
            "downscale_method": config.get("DOWNSCALE_METHOD"),
            "opencv_interpolation": config.get("OPENCV_INTERPOLATION")
            if config.get("DOWNSCALE_METHOD") == "opencv"
            else None,
            "downscale_factors": config.get("DOWNSCALE_FACTORS"),
            "max_threshold_percentile": config.get("MAX_THRESHOLD_PERCENTILE"),
        },
        "vesselness_config": {
            "ostios": _vesselness_metadata(config, "VESSELNESS_AORTA"),
            "artery": _vesselness_metadata(config, "VESSELNESS_ARTERY"),
        },
        "circle_detection_config": config.get("CIRCLE_DETECTION"),
        "level_set_config": config.get("LEVEL_SET"),
        "ostia_detection_config": config.get("OSTIA_DETECTION"),
        "region_growing_config": config.get("REGION_GROWING"),
        "postprocessing_config": config.get("POSTPROCESSING"),
        "cache_config": {
            "load_cache": config.get("LOAD_CACHE"),
            "save_cache": config.get("SAVE_CACHE"),
        },
        "evaluation_config": {
            "tolerable_distance_mm": config["OSTIA_VALIDATION"][
                "distance_threshold_mm"
            ],
        },
        "results_summary": results_summary,
    }

    if (
        base_path is not None
        or base_save_path is not None
        or root_output_dir is not None
    ):
        metadata["paths"] = {
            "base_path": base_path,
            "base_save_path": base_save_path,
            "output_dir": root_output_dir,
        }
    return metadata


def save_metadata(
    split_name,
    output_dir,
    config,
    ids,
    results,
    execution_time=None,
    base_path=None,
    base_save_path=None,
    root_output_dir=None,
):
    """Salva metadados da execução em arquivo JSON."""
    metadata = build_metadata(
        split_name,
        config,
        ids,
        results,
        execution_time=execution_time,
        base_path=base_path,
        base_save_path=base_save_path,
        root_output_dir=root_output_dir,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / f"ostios_{split_name}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file_handle:
        json.dump(make_json_safe(metadata), file_handle, indent=2, ensure_ascii=False)

    return str(metadata_path)


def batch_result_number(path, split_name):
    """Extrai o número do lote de um arquivo de resultado."""
    filename = Path(path).name
    match = re.match(
        rf"^ostios_{re.escape(split_name)}_lote_(\d+)(?:_summary)?\.csv$",
        filename,
    )
    return int(match.group(1)) if match else None


def list_batch_result_files(split_name, output_dir):
    """Lista CSVs de lote em ordem numérica, preferindo o formato atual."""
    output_dir = Path(output_dir)
    batch_files_by_number = {}

    def current_format_first(path):
        return 0 if path.name.endswith("_summary.csv") else 1

    candidates = [
        path
        for path in output_dir.glob(f"ostios_{split_name}_lote_*.csv")
        if batch_result_number(path, split_name) is not None
    ]
    for batch_file in sorted(
        candidates,
        key=lambda path: (
            batch_result_number(path, split_name),
            current_format_first(path),
        ),
    ):
        batch_files_by_number.setdefault(
            batch_result_number(batch_file, split_name), batch_file
        )

    return list(batch_files_by_number.values())


def get_batch_result_file(output_dir, split_name, batch_number):
    """Retorna o CSV exato de um lote, preferindo o formato atual *_summary.csv."""
    output_dir = Path(output_dir)
    candidates = (
        output_dir / f"ostios_{split_name}_lote_{batch_number}_summary.csv",
        output_dir / f"ostios_{split_name}_lote_{batch_number}.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for batch_file in list_batch_result_files(split_name, output_dir):
        if batch_result_number(batch_file, split_name) == batch_number:
            return batch_file
    return None


def merge_batch_results(split_name, output_dir):
    """Mescla todos os CSVs de lotes em um único arquivo final."""
    output_dir = Path(output_dir)
    batch_files = list_batch_result_files(split_name, output_dir)

    if not batch_files:
        print(f"⚠️  Nenhum arquivo de lote encontrado em {output_dir}")
        return None

    print(f"\n🔄 Mesclando {len(batch_files)} arquivo(s) de lote...")
    dfs = []
    for batch_file in batch_files:
        df = pd.read_csv(batch_file)
        df = make_readable_results_dataframe(df)
        dfs.append(df)
        print(f"   ✓ {batch_file.name} ({len(df)} registros)")

    merged_df = pd.concat(dfs, ignore_index=True)
    final_path = output_dir / f"ostios_{split_name}_summary.csv"
    merged_df.to_csv(final_path, index=False)

    print(
        f"✅ Arquivo final mesclado: {final_path} ({len(merged_df)} registros totais)\n"
    )
    return str(final_path)
