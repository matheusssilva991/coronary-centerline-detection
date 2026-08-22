"""Schema, aliases e métricas agregadas dos resultados do pipeline."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd


RESULT_COLUMNS: list[str] = [
    "IMG_ID",
    "dice_artery",
    "dice_artery_before_morphology",
    "dice_artery_after_morphology",
    "dice_artery_morphology_delta",
    "artery_voxels",
    "artery_voxels_before_morphology",
    "artery_voxels_after_morphology",
    "artery_segmentation_method",
    "fc_processed_voxels",
    "fc_effective_alpha",
    "fc_object_seed_count",
    "fc_candidate_voxels_final",
    "threshold_mode",
    "fuzzy_mask_strategy",
    "min_threshold",
    "max_threshold",
    "lower_threshold_method",
    "lower_threshold_percentile",
    "threshold_voxels",
    "lcc_voxels",
    "image_slice_count",
    "image_voxels",
    "aorta_circle_count",
    "aorta_detected_circle_count",
    "aorta_interpolated_circle_count",
    "aorta_circle_first_slice",
    "aorta_circle_last_slice",
    "aorta_circle_coverage",
    "aorta_recovered_initialization",
    "aorta_mask_voxels",
    "aorta_segmented_slice_count",
    "aorta_voxels_per_segmented_slice",
    "aorta_volume_fraction",
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

READABLE_COLUMN_NAMES: dict[str, str] = {
    "dice_artery": "artery_dice",
    "dice_artery_before_morphology": "artery_dice_before_morphology",
    "dice_artery_after_morphology": "artery_dice_after_morphology",
    "dice_artery_morphology_delta": "artery_dice_morphology_delta",
    "artery_voxels": "artery_voxel_count",
    "artery_voxels_before_morphology": "artery_voxel_count_before_morphology",
    "artery_voxels_after_morphology": "artery_voxel_count_after_morphology",
    "artery_segmentation_method": "artery_segmentation_method",
    "fc_processed_voxels": "fc_processed_voxels",
    "fc_effective_alpha": "fc_effective_alpha",
    "fc_object_seed_count": "fc_object_seed_count",
    "fc_candidate_voxels_final": "fc_candidate_voxels_final",
    "threshold_mode": "threshold_mode",
    "fuzzy_mask_strategy": "fuzzy_mask_strategy",
    "min_threshold": "min_threshold_hu",
    "max_threshold": "max_threshold_hu",
    "lower_threshold_method": "lower_threshold_method",
    "lower_threshold_percentile": "lower_threshold_percentile",
    "threshold_voxels": "threshold_voxel_count",
    "lcc_voxels": "lcc_voxel_count",
    "image_slice_count": "image_slice_count",
    "image_voxels": "image_voxel_count",
    "aorta_circle_count": "aorta_circle_count",
    "aorta_detected_circle_count": "aorta_detected_circle_count",
    "aorta_interpolated_circle_count": "aorta_interpolated_circle_count",
    "aorta_circle_first_slice": "aorta_circle_first_slice",
    "aorta_circle_last_slice": "aorta_circle_last_slice",
    "aorta_circle_coverage": "aorta_circle_coverage",
    "aorta_recovered_initialization": "aorta_recovered_initialization",
    "aorta_mask_voxels": "aorta_mask_voxel_count",
    "aorta_segmented_slice_count": "aorta_segmented_slice_count",
    "aorta_voxels_per_segmented_slice": "aorta_voxels_per_segmented_slice",
    "aorta_volume_fraction": "aorta_volume_fraction",
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
    "lcc_per_slice": "lcc_per_slice",
    "lcc_mode": "lcc_mode",
    "configured_artery_segmentation_method": "configured_artery_segmentation_method",
    "aorta_ostia_method": "aorta_ostia_method",
    "aorta_miss_count": "aorta_miss_count",
    "aorta_interpolate_missed_circles": "aorta_interpolate_missed_circles",
}

CANONICAL_COLUMN_NAMES: dict[str, str] = {
    value: key for key, value in READABLE_COLUMN_NAMES.items()
}

READABLE_BOOL_COLUMNS: set[str] = {
    "ostia_detected",
    "artery_segmentation_run",
    "segmented_with_incorrect_ostia",
    "both_ostia_correct",
    "both_ostia_tolerable",
    "left_ostium_correct",
    "right_ostium_correct",
    "lcc_per_slice",
    "aorta_interpolate_missed_circles",
    "aorta_recovered_initialization",
}

OSTIA_STATUS_READABLE_LABELS: dict[str, str] = {
    "not_evaluated": "not evaluated",
    "not_found": "not found",
    "both_correct": "both correct",
    "both_tolerable": "both tolerable",
    "found_but_wrong": "found but incorrect",
}
OSTIA_STATUS_INTERNAL_LABELS: dict[str, str] = {
    readable: internal for internal, readable in OSTIA_STATUS_READABLE_LABELS.items()
}

STATUS_LABELS: dict[str, str] = {
    "not_found": "ostia not found",
    "both_correct": "both ostia correct",
    "both_tolerable": "both ostia tolerable",
    "one_correct": "one ostium correct",
    "error": "pipeline error",
    "none_correct": "no ostium correct",
}


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
    if value is None:
        return False
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return False
        return bool(value)

    normalized = str(value).strip().lower()
    return normalized in {"true", "1", "sim", "s", "yes", "y"}


def _as_optional_float(value: Any) -> float | None:
    """Converte um valor escalar para float, preservando ausentes como None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def _format_bool_readable(value: Any) -> str:
    return "yes" if _as_bool_value(value) else "no"


def _configured_artery_segmentation_method(config: dict[str, Any]) -> str:
    """Retorna o método arterial selecionado na configuração efetiva."""
    return str(config.get("ARTERY_SEGMENTATION", {}).get("method", "region_growing"))


def make_readable_results_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas técnicas do resultado para nomes/valores mais legíveis."""
    # Renomeia em uma cópia para não alterar o DataFrame usado pelo pipeline.
    readable_df = df.rename(columns=READABLE_COLUMN_NAMES).copy()

    # Converte apenas colunas booleanas conhecidas para rótulos de apresentação.
    dataframe_columns = {str(column) for column in readable_df.columns}
    for column in READABLE_BOOL_COLUMNS.intersection(dataframe_columns):
        readable_df[column] = readable_df[column].map(_format_bool_readable)

    if "ostia_detection_status" in readable_df.columns:
        readable_df["ostia_detection_status"] = readable_df[
            "ostia_detection_status"
        ].map(lambda value: OSTIA_STATUS_READABLE_LABELS.get(value, value))

    return readable_df


def add_internal_result_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona aliases internos sem remover as colunas legíveis persistidas."""
    normalized_df = df.copy()
    # Resultados antigos e novos podem usar lados diferentes do mapa de aliases.
    for readable_column, internal_column in CANONICAL_COLUMN_NAMES.items():
        if (
            internal_column not in normalized_df.columns
            and readable_column in normalized_df.columns
        ):
            alias = normalized_df[readable_column]
            if readable_column in READABLE_BOOL_COLUMNS:
                alias = alias.map(_as_bool_value)
            elif internal_column == "ostia_status":
                alias = alias.map(
                    lambda value: OSTIA_STATUS_INTERNAL_LABELS.get(value, value)
                )
            normalized_df[internal_column] = alias
    return normalized_df


def _series_from_aliases(
    df: pd.DataFrame,
    column: str,
    dtype: Any = None,
) -> pd.Series:
    column_candidates = (column, _readable_column_name(column))
    for column_candidate in column_candidates:
        if column_candidate in df.columns:
            return cast(pd.Series, df[column_candidate])
    return pd.Series(index=df.index, dtype=dtype)


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
        # Métricas e diagnósticos da segmentação arterial.
        "IMG_ID": result.get("IMG_ID"),
        "dice_artery": _get_result_value(result, "dice_artery"),
        "dice_artery_before_morphology": _get_result_value(
            result, "dice_artery_before_morphology"
        ),
        "dice_artery_after_morphology": _get_result_value(
            result, "dice_artery_after_morphology"
        ),
        "dice_artery_morphology_delta": _get_result_value(
            result, "dice_artery_morphology_delta"
        ),
        "artery_voxels": _get_result_value(result, "artery_voxels"),
        "artery_voxels_before_morphology": _get_result_value(
            result, "artery_voxels_before_morphology"
        ),
        "artery_voxels_after_morphology": _get_result_value(
            result, "artery_voxels_after_morphology"
        ),
        "artery_segmentation_method": _get_result_value(
            result, "artery_segmentation_method", "region_growing"
        ),
        "fc_processed_voxels": _get_result_value(result, "fc_processed_voxels"),
        "fc_effective_alpha": _get_result_value(result, "fc_effective_alpha"),
        "fc_object_seed_count": _get_result_value(result, "fc_object_seed_count"),
        "fc_candidate_voxels_final": _get_result_value(
            result, "fc_candidate_voxels_final"
        ),
        # Parâmetros efetivos e volumes intermediários do pré-processamento.
        "threshold_mode": _get_result_value(result, "threshold_mode"),
        "fuzzy_mask_strategy": _get_result_value(result, "fuzzy_mask_strategy"),
        "min_threshold": _get_result_value(result, "min_threshold"),
        "max_threshold": _get_result_value(result, "max_threshold"),
        "lower_threshold_method": _get_result_value(result, "lower_threshold_method"),
        "lower_threshold_percentile": _get_result_value(
            result, "lower_threshold_percentile"
        ),
        "threshold_voxels": _get_result_value(result, "threshold_voxels"),
        "lcc_voxels": _get_result_value(result, "lcc_voxels"),
        "image_slice_count": _get_result_value(result, "image_slice_count"),
        "image_voxels": _get_result_value(result, "image_voxels"),
        "aorta_circle_count": _get_result_value(result, "aorta_circle_count"),
        "aorta_detected_circle_count": _get_result_value(
            result, "aorta_detected_circle_count"
        ),
        "aorta_interpolated_circle_count": _get_result_value(
            result, "aorta_interpolated_circle_count"
        ),
        "aorta_circle_first_slice": _get_result_value(
            result, "aorta_circle_first_slice"
        ),
        "aorta_circle_last_slice": _get_result_value(result, "aorta_circle_last_slice"),
        "aorta_circle_coverage": _get_result_value(result, "aorta_circle_coverage"),
        "aorta_recovered_initialization": _as_bool_value(
            _get_result_value(result, "aorta_recovered_initialization", False)
        ),
        "aorta_mask_voxels": _get_result_value(result, "aorta_mask_voxels"),
        "aorta_segmented_slice_count": _get_result_value(
            result, "aorta_segmented_slice_count"
        ),
        "aorta_voxels_per_segmented_slice": _get_result_value(
            result, "aorta_voxels_per_segmented_slice"
        ),
        "aorta_volume_fraction": _get_result_value(
            result, "aorta_volume_fraction"
        ),
        # Resultado da localização e validação dos óstios.
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


def make_result_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Converte lista de resultados em DataFrame formatado."""
    rows = [build_result_row(result) for result in results]
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def add_config_columns(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Adiciona ao DataFrame as colunas de configuração salvas no CSV."""
    df = df.copy()
    # Registra opções que ajudam a reproduzir e comparar execuções futuras.
    circle_config = config.get("CIRCLE_DETECTION", {})
    df["downscale_method"] = config.get("DOWNSCALE_METHOD", "N/A")
    df["opencv_interpolation"] = (
        config.get("OPENCV_INTERPOLATION", "N/A")
        if config.get("DOWNSCALE_METHOD") == "opencv"
        else "N/A"
    )
    df["downscale_factors"] = str(config.get("DOWNSCALE_FACTORS", "N/A"))
    df["max_threshold_percentile"] = config.get("MAX_THRESHOLD_PERCENTILE", "N/A")
    thresholding_config = config.get("THRESHOLDING", {})
    lower_threshold_config = config.get("LOWER_THRESHOLD", {})
    df["threshold_mode"] = thresholding_config.get("method", "normal")
    df["configured_lower_threshold_method"] = lower_threshold_config.get(
        "method", "fixed"
    )
    df["lcc_per_slice"] = True
    df["lcc_mode"] = "per_slice"
    df["configured_artery_segmentation_method"] = (
        _configured_artery_segmentation_method(config)
    )
    df["aorta_ostia_method"] = config.get("AORTA_OSTIA_METHOD", {}).get(
        "method", "standard"
    )
    df["aorta_miss_count"] = circle_config.get("max_slice_miss_threshold", "N/A")
    df["aorta_interpolate_missed_circles"] = circle_config.get(
        "interpolate_missed_circles", "N/A"
    )
    return df


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    series = _series_from_aliases(df, column, dtype=bool)
    return series.map(_as_bool_value)


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Converte uma coluna e seus aliases em uma Series numérica anulável."""
    series = _series_from_aliases(df, column, dtype=float)
    return series.map(_as_optional_float)


def summarize_results_df(df: pd.DataFrame) -> dict[str, Any]:
    """Calcula contagens e métricas agregadas de um DataFrame de resultados."""
    # Resolve aliases primeiro para aceitar tanto CSVs legíveis quanto internos.
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
    dice_series = _numeric_series(df, "dice_artery")
    dice_before_series = _numeric_series(df, "dice_artery_before_morphology")
    dice_delta_series = _numeric_series(df, "dice_artery_morphology_delta")

    # Correto e tolerável são considerados sucesso na avaliação dos óstios.
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

    # Métricas de Dice permanecem nulas quando nenhuma artéria foi segmentada.
    if dice_series.notna().any():
        summary.update(
            {
                "dice_artery_mean": float(dice_series.mean()),
                "dice_artery_std": float(cast(float, dice_series.std())),
                "dice_artery_median": float(dice_series.median()),
                "dice_artery_before_morphology_mean": (
                    float(dice_before_series.mean())
                    if dice_before_series.notna().any()
                    else None
                ),
                "dice_artery_after_morphology_mean": float(dice_series.mean()),
                "dice_artery_morphology_delta_mean": (
                    float(dice_delta_series.mean())
                    if dice_delta_series.notna().any()
                    else None
                ),
            }
        )
    else:
        summary.update(
            {
                "dice_artery_mean": None,
                "dice_artery_std": None,
                "dice_artery_median": None,
                "dice_artery_before_morphology_mean": None,
                "dice_artery_after_morphology_mean": None,
                "dice_artery_morphology_delta_mean": None,
            }
        )
    return summary
