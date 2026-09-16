"""Schema, aliases e métricas agregadas dos resultados do pipeline."""

from __future__ import annotations

import unicodedata
from typing import Any, cast

import pandas as pd

from .results_columns import (
    ARTERY_BRANCH_COLUMNS,
    CANONICAL_COLUMN_NAMES,
    OSTIA_STATUS_PORTUGUESE_LABELS,
    READABLE_BOOL_COLUMNS,
    READABLE_COLUMN_NAMES,
    RESULT_COLUMNS,
    STATUS_LABELS,
    STATUS_PORTUGUESE_LABELS,
)


OSTIA_STATUS_ALIASES: dict[str, str] = {
    "not_evaluated": "not_evaluated",
    "nao_avaliado": "not_evaluated",
    "not_found": "not_found",
    "ostia_not_found": "not_found",
    "nao_encontrados": "not_found",
    "ostios_nao_encontrados": "not_found",
    "both_correct": "both_correct",
    "both_ostia_correct": "both_correct",
    "ambos_corretos": "both_correct",
    "both_tolerable": "both_tolerable",
    "both_ostia_tolerable": "both_tolerable",
    "ambos_toleraveis": "both_tolerable",
    "found_but_wrong": "found_but_wrong",
    "found_but_incorrect": "found_but_wrong",
    "encontrados_mas_incorretos": "found_but_wrong",
    "one_correct": "found_but_wrong",
    "one_ostium_correct": "found_but_wrong",
    "um_correto": "found_but_wrong",
    "none_correct": "found_but_wrong",
    "no_ostium_correct": "found_but_wrong",
    "nenhum_correto": "found_but_wrong",
}

RESULT_STATUS_ALIASES: dict[str, str] = {
    "not_found": "not_found",
    "ostia_not_found": "not_found",
    "nao_encontrados": "not_found",
    "ostios_nao_encontrados": "not_found",
    "both_correct": "both_correct",
    "both_ostia_correct": "both_correct",
    "ambos_corretos": "both_correct",
    "both_tolerable": "both_tolerable",
    "both_ostia_tolerable": "both_tolerable",
    "ambos_toleraveis": "both_tolerable",
    "one_correct": "one_correct",
    "one_ostium_correct": "one_correct",
    "um_correto": "one_correct",
    "none_correct": "none_correct",
    "no_ostium_correct": "none_correct",
    "nenhum_correto": "none_correct",
    "error": "error",
    "pipeline_error": "error",
    "erro": "error",
    "erro_no_pipeline": "error",
    "found_but_wrong": "found_but_wrong",
    "found_but_incorrect": "found_but_wrong",
    "not_evaluated": "not_evaluated",
}


def _status_key(value: Any) -> str | None:
    """Convert a status scalar to an accent-free snake-case lookup key."""
    if value is None or pd.isna(value):
        return None
    normalized = unicodedata.normalize("NFKD", str(value).strip().casefold())
    ascii_value = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    key = ascii_value.replace("-", "_").replace(" ", "_")
    while "__" in key:
        key = key.replace("__", "_")
    return key.strip("_")


def normalize_ostia_status(value: Any) -> str | None:
    """Normalize persisted and legacy ostia statuses to English status codes."""
    key = _status_key(value)
    return None if key is None else OSTIA_STATUS_ALIASES.get(key, key)


def normalize_result_status(value: Any) -> str | None:
    """Normalize persisted and legacy result statuses to English status codes."""
    key = _status_key(value)
    return None if key is None else RESULT_STATUS_ALIASES.get(key, key)


def ostia_status_label_pt(value: Any) -> str:
    """Return the Portuguese presentation label for an ostia status code."""
    normalized = normalize_ostia_status(value)
    if normalized is None:
        return "sem status"
    return OSTIA_STATUS_PORTUGUESE_LABELS.get(normalized, normalized)


def result_status_label_pt(value: Any) -> str:
    """Return the Portuguese presentation label for a result status code."""
    normalized = normalize_result_status(value)
    if normalized is None:
        return "sem status"
    return STATUS_PORTUGUESE_LABELS.get(normalized, normalized)


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
        ].map(normalize_ostia_status)
    if "status" in readable_df.columns:
        readable_df["status"] = readable_df["status"].map(normalize_result_status)

    return readable_df


def select_per_image_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém somente resultados e diagnósticos que pertencem a cada exame."""
    readable_df = make_readable_results_dataframe(df)
    if "effective_upper_threshold_hu" not in readable_df.columns:
        # Runs antigos registravam o threshold efetivo como ``max_threshold_hu``.
        if "max_threshold_hu" in readable_df.columns:
            readable_df["effective_upper_threshold_hu"] = readable_df[
                "max_threshold_hu"
            ]
    elif "max_threshold_hu" in readable_df.columns:
        readable_df["effective_upper_threshold_hu"] = readable_df[
            "effective_upper_threshold_hu"
        ].fillna(readable_df["max_threshold_hu"])
    result_columns = [_readable_column_name(column) for column in RESULT_COLUMNS]
    for column in result_columns:
        if column not in readable_df.columns:
            readable_df[column] = None
    return readable_df.loc[:, result_columns]


def add_internal_result_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona aliases internos sem remover as colunas legíveis persistidas."""
    normalized_df = df.copy()
    if "ostia_detection_status" in normalized_df.columns:
        normalized_df["ostia_detection_status"] = normalized_df[
            "ostia_detection_status"
        ].map(normalize_ostia_status)
    if "ostia_status" in normalized_df.columns:
        normalized_df["ostia_status"] = normalized_df["ostia_status"].map(
            normalize_ostia_status
        )
    if "status" in normalized_df.columns:
        normalized_df["status"] = normalized_df["status"].map(normalize_result_status)
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
                alias = alias.map(normalize_ostia_status)
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
    effective_upper_threshold_hu = _get_result_value(
        result, "effective_upper_threshold_hu"
    )
    if effective_upper_threshold_hu is None:
        # Permite retomar runs normais antigos que salvavam apenas max_threshold_hu.
        effective_upper_threshold_hu = _get_result_value(result, "max_threshold")

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
        **{
            column: _get_result_value(result, column)
            for column in ARTERY_BRANCH_COLUMNS
        },
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
        "effective_upper_threshold_hu": effective_upper_threshold_hu,
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
        "aorta_circle_radius_min_px": _get_result_value(
            result, "aorta_circle_radius_min_px"
        ),
        "aorta_circle_radius_max_px": _get_result_value(
            result, "aorta_circle_radius_max_px"
        ),
        "aorta_circle_radius_mean_px": _get_result_value(
            result, "aorta_circle_radius_mean_px"
        ),
        "aorta_circle_radius_median_px": _get_result_value(
            result, "aorta_circle_radius_median_px"
        ),
        "aorta_circle_radius_std_px": _get_result_value(
            result, "aorta_circle_radius_std_px"
        ),
        "aorta_circle_radius_p10_px": _get_result_value(
            result, "aorta_circle_radius_p10_px"
        ),
        "aorta_circle_radius_p90_px": _get_result_value(
            result, "aorta_circle_radius_p90_px"
        ),
        "aorta_circle_radius_min_mm": _get_result_value(
            result, "aorta_circle_radius_min_mm"
        ),
        "aorta_circle_radius_max_mm": _get_result_value(
            result, "aorta_circle_radius_max_mm"
        ),
        "aorta_circle_radius_mean_mm": _get_result_value(
            result, "aorta_circle_radius_mean_mm"
        ),
        "aorta_circle_radius_median_mm": _get_result_value(
            result, "aorta_circle_radius_median_mm"
        ),
        "aorta_circle_radius_std_mm": _get_result_value(
            result, "aorta_circle_radius_std_mm"
        ),
        "aorta_circle_radius_p10_mm": _get_result_value(
            result, "aorta_circle_radius_p10_mm"
        ),
        "aorta_circle_radius_p90_mm": _get_result_value(
            result, "aorta_circle_radius_p90_mm"
        ),
        "aorta_detected_circle_radius_median_mm": _get_result_value(
            result, "aorta_detected_circle_radius_median_mm"
        ),
        "aorta_interpolated_circle_radius_median_mm": _get_result_value(
            result, "aorta_interpolated_circle_radius_median_mm"
        ),
        "aorta_circle_radius_first_mm": _get_result_value(
            result, "aorta_circle_radius_first_mm"
        ),
        "aorta_circle_radius_last_mm": _get_result_value(
            result, "aorta_circle_radius_last_mm"
        ),
        "aorta_circle_radius_max_step_change_mm": _get_result_value(
            result, "aorta_circle_radius_max_step_change_mm"
        ),
        "aorta_circle_radius_p90_step_change_mm": _get_result_value(
            result, "aorta_circle_radius_p90_step_change_mm"
        ),
        "aorta_circle_mean_hough_accumulator": _get_result_value(
            result, "aorta_circle_mean_hough_accumulator"
        ),
        "aorta_circle_lower_radius_bound_fraction": _get_result_value(
            result, "aorta_circle_lower_radius_bound_fraction"
        ),
        "aorta_circle_upper_radius_bound_fraction": _get_result_value(
            result, "aorta_circle_upper_radius_bound_fraction"
        ),
        "aorta_circle_filter_method": _get_result_value(
            result, "aorta_circle_filter_method", "none"
        ),
        "aorta_circle_filter_applied": _as_bool_value(
            _get_result_value(result, "aorta_circle_filter_applied", False)
        ),
        "aorta_circle_original_count": _get_result_value(
            result, "aorta_circle_original_count"
        ),
        "aorta_circle_used_count": _get_result_value(result, "aorta_circle_used_count"),
        "aorta_circle_filter_synthetic_tail_count": _get_result_value(
            result, "aorta_circle_filter_synthetic_tail_count", 0
        ),
        "aorta_circle_filter_trimmed_tail_count": _get_result_value(
            result, "aorta_circle_filter_trimmed_tail_count", 0
        ),
        "aorta_circle_filter_trim_start_slice": _get_result_value(
            result, "aorta_circle_filter_trim_start_slice"
        ),
        "aorta_circle_filter_detected_tail_start_slice": _get_result_value(
            result, "aorta_circle_filter_detected_tail_start_slice"
        ),
        "aorta_circle_filter_original_coverage": _get_result_value(
            result, "aorta_circle_filter_original_coverage"
        ),
        "aorta_circle_filter_used_coverage": _get_result_value(
            result, "aorta_circle_filter_used_coverage"
        ),
        "aorta_circle_filter_reason": _get_result_value(
            result, "aorta_circle_filter_reason"
        ),
        "aorta_mask_voxels": _get_result_value(result, "aorta_mask_voxels"),
        "aorta_segmented_slice_count": _get_result_value(
            result, "aorta_segmented_slice_count"
        ),
        "aorta_voxels_per_segmented_slice": _get_result_value(
            result, "aorta_voxels_per_segmented_slice"
        ),
        "aorta_volume_fraction": _get_result_value(result, "aorta_volume_fraction"),
        "aorta_level_set_initial_voxel_count": _get_result_value(
            result, "aorta_level_set_initial_voxel_count"
        ),
        "aorta_level_set_raw_voxel_count": _get_result_value(
            result, "aorta_level_set_raw_voxel_count"
        ),
        "aorta_level_set_initial_volume_fraction": _get_result_value(
            result, "aorta_level_set_initial_volume_fraction"
        ),
        "aorta_level_set_raw_volume_fraction": _get_result_value(
            result, "aorta_level_set_raw_volume_fraction"
        ),
        "aorta_level_set_iterations_used": _get_result_value(
            result, "aorta_level_set_iterations_used"
        ),
        "aorta_level_set_circle_fill_q25": _get_result_value(
            result, "aorta_level_set_circle_fill_q25"
        ),
        "aorta_level_set_circle_area_ratio_p90": _get_result_value(
            result, "aorta_level_set_circle_area_ratio_p90"
        ),
        "aorta_slice_area_jump_p95": _get_result_value(
            result, "aorta_slice_area_jump_p95"
        ),
        "aorta_segmentation_feedback": _get_result_value(
            result,
            "aorta_segmentation_feedback",
            "insufficient_data",
        ),
        # Resultado da localização e validação dos óstios.
        "ostia_found": _as_bool_value(_get_result_value(result, "ostia_found", False)),
        "ostia_status": normalize_ostia_status(
            _get_result_value(result, "ostia_status")
        ),
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
    row["status"] = normalize_result_status(
        result.get("status") or classify_result_status(row)
    )
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
    level_set_config = config.get("LEVEL_SET", {})
    df["downscale_method"] = config.get("DOWNSCALE_METHOD", "N/A")
    df["opencv_interpolation"] = (
        config.get("OPENCV_INTERPOLATION", "N/A")
        if config.get("DOWNSCALE_METHOD") == "opencv"
        else "N/A"
    )
    df["downscale_factors"] = str(config.get("DOWNSCALE_FACTORS", "N/A"))
    df["max_threshold_percentile"] = config.get("MAX_THRESHOLD_PERCENTILE", "N/A")
    thresholding_config = config.get("THRESHOLDING", {})
    df["threshold_mode"] = thresholding_config.get("method", "normal")
    df["lcc_per_slice"] = True
    df["lcc_mode"] = "per_slice"
    df["aorta_miss_count"] = circle_config.get("max_slice_miss_threshold", "N/A")
    df["configured_aorta_hough_radii_start_px"] = circle_config.get(
        "radii_start_px",
        "N/A",
    )
    df["configured_aorta_hough_radii_end_px"] = circle_config.get(
        "radii_end_px",
        "N/A",
    )
    df["aorta_interpolate_missed_circles"] = circle_config.get(
        "interpolate_missed_circles", "N/A"
    )
    df["aorta_trajectory_radius_factor"] = level_set_config.get(
        "trajectory_radius_factor"
    )
    df["aorta_trajectory_axial_margin_slices"] = level_set_config.get(
        "trajectory_axial_margin_slices", 0
    )
    df["aorta_opening_radius"] = level_set_config.get("leak_removal_radius", 0)
    return df


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    series = _series_from_aliases(df, column, dtype=bool)
    return series.map(_as_bool_value)


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Converte uma coluna e seus aliases em uma Series numérica anulável."""
    series = _series_from_aliases(df, column, dtype=float)
    return series.map(_as_optional_float)


def _numeric_stats(
    df: pd.DataFrame,
    column: str,
    *,
    include_sum: bool = False,
) -> dict[str, Any]:
    """Calcula estatísticas estáveis para uma medição opcional por exame."""
    values = _numeric_series(df, column).dropna()
    prefix = _readable_column_name(column)
    stats: dict[str, Any] = {f"{prefix}_count": int(len(values))}
    if values.empty:
        for suffix in ("mean", "std", "median", "q1", "q3", "min", "max"):
            stats[f"{prefix}_{suffix}"] = None
        if include_sum:
            stats[f"{prefix}_sum"] = None
        return stats

    stats.update(
        {
            f"{prefix}_mean": float(values.mean()),
            f"{prefix}_std": (float(values.std()) if len(values) > 1 else None),
            f"{prefix}_median": float(values.median()),
            f"{prefix}_q1": float(values.quantile(0.25)),
            f"{prefix}_q3": float(values.quantile(0.75)),
            f"{prefix}_min": float(values.min()),
            f"{prefix}_max": float(values.max()),
        }
    )
    if include_sum:
        stats[f"{prefix}_sum"] = float(values.sum())
    return stats


def summarize_results_df(df: pd.DataFrame) -> dict[str, Any]:
    """Calcula contagens e métricas agregadas de um DataFrame de resultados."""
    # Resolve aliases primeiro para aceitar tanto CSVs legíveis quanto internos.
    both_correct_series = _bool_series(df, "both_correct")
    both_tolerable_series = _bool_series(df, "both_tolerable")
    ostia_found_series = _bool_series(df, "ostia_found")
    segmentation_attempted_series = _bool_series(df, "segmentation_attempted")
    proceeded_with_bad_ostia_series = _bool_series(df, "proceeded_with_bad_ostia")
    ostia_status_series = _series_from_aliases(df, "ostia_status")
    ostia_status_normalized = ostia_status_series.map(normalize_ostia_status)
    ostia_not_found_series = ostia_status_normalized.eq("not_found")
    error_series = _series_from_aliases(df, "error")
    ostia_error_series = _series_from_aliases(df, "ostia_error")
    dice_series = _numeric_series(df, "dice_artery")
    dice_before_series = _numeric_series(df, "dice_artery_before_morphology")
    dice_after_series = _numeric_series(df, "dice_artery_after_morphology")
    if not dice_after_series.notna().any():
        dice_after_series = dice_series
    dice_delta_series = _numeric_series(df, "dice_artery_morphology_delta")
    effective_upper_threshold_series = _numeric_series(
        df, "effective_upper_threshold_hu"
    )
    if not effective_upper_threshold_series.notna().any():
        effective_upper_threshold_series = _numeric_series(df, "max_threshold")

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
        "ostia_error_not_null": int(ostia_error_series.notna().sum()),
    }

    feedback_series = _series_from_aliases(df, "aorta_segmentation_feedback")
    feedback_counts = (
        feedback_series.fillna("insufficient_data").astype(str).value_counts()
    )
    summary["aorta_segmentation_feedback_counts"] = {
        label: int(count) for label, count in feedback_counts.items()
    }

    # Medições científicas compactas usadas para caracterizar a coorte e a
    # qualidade da segmentação, sem repetir parâmetros de configuração.
    for column, include_sum in (
        ("image_slice_count", True),
        ("artery_voxels", False),
        ("artery_voxels_before_morphology", False),
        ("artery_voxels_after_morphology", False),
        ("threshold_voxels", False),
        ("lcc_voxels", False),
        ("aorta_circle_count", False),
        ("aorta_detected_circle_count", False),
        ("aorta_interpolated_circle_count", False),
        ("aorta_segmented_slice_count", False),
        ("aorta_circle_coverage", False),
        ("aorta_mask_voxels", False),
        ("aorta_voxels_per_segmented_slice", False),
        ("aorta_volume_fraction", False),
    ):
        summary.update(_numeric_stats(df, column, include_sum=include_sum))

    # Resume os thresholds efetivos sem inventar um valor escalar para o fuzzy.
    valid_upper_thresholds = effective_upper_threshold_series.dropna()
    summary["effective_upper_threshold_hu_count"] = int(len(valid_upper_thresholds))
    summary["effective_upper_threshold_hu_mean"] = (
        float(valid_upper_thresholds.mean())
        if not valid_upper_thresholds.empty
        else None
    )
    summary["effective_upper_threshold_hu_min"] = (
        float(valid_upper_thresholds.min())
        if not valid_upper_thresholds.empty
        else None
    )
    summary["effective_upper_threshold_hu_max"] = (
        float(valid_upper_thresholds.max())
        if not valid_upper_thresholds.empty
        else None
    )

    # Métricas de Dice permanecem nulas quando nenhuma artéria foi segmentada.
    if dice_series.notna().any():
        valid_ostia_dice = dice_series[total_success_series & dice_series.notna()]
        invalid_ostia_dice = dice_series[(~total_success_series) & dice_series.notna()]
        summary.update(
            {
                "dice_artery_mean": float(dice_series.mean()),
                "dice_artery_std": float(cast(float, dice_series.std())),
                "dice_artery_median": float(dice_series.median()),
                "dice_artery_q1": float(dice_series.quantile(0.25)),
                "dice_artery_q3": float(dice_series.quantile(0.75)),
                "dice_artery_valid_ostia_mean": (
                    float(valid_ostia_dice.mean())
                    if not valid_ostia_dice.empty
                    else None
                ),
                "dice_artery_invalid_ostia_mean": (
                    float(invalid_ostia_dice.mean())
                    if not invalid_ostia_dice.empty
                    else None
                ),
                "dice_artery_before_morphology_mean": (
                    float(dice_before_series.mean())
                    if dice_before_series.notna().any()
                    else None
                ),
                "dice_artery_after_morphology_mean": float(dice_after_series.mean()),
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
                "dice_artery_q1": None,
                "dice_artery_q3": None,
                "dice_artery_valid_ostia_mean": None,
                "dice_artery_invalid_ostia_mean": None,
                "dice_artery_before_morphology_mean": None,
                "dice_artery_after_morphology_mean": None,
                "dice_artery_morphology_delta_mean": None,
            }
        )
    return summary
