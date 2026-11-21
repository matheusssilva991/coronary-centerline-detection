"""Schema, aliases e métricas agregadas dos resultados do pipeline."""

from __future__ import annotations

import unicodedata
from typing import Any, cast

import numpy as np
import pandas as pd

from utils.project.evaluation.artery_distances import (
    DISTANCE_METRIC_COLUMNS,
    LEGACY_DISTANCE_COLUMNS,
    summarize_artery_distances,
)
from utils.project.results.columns import (
    ARTERY_MASK_EXPORT_COLUMNS,
    CANONICAL_COLUMN_NAMES,
    OSTIA_STATUS_PORTUGUESE_LABELS,
    READABLE_BOOL_COLUMNS,
    READABLE_COLUMN_NAMES,
    RESULT_COLUMNS,
    STATUS_LABELS,
    STATUS_PORTUGUESE_LABELS,
)

OSTIA_SUCCESS_STATUSES = frozenset({"both_correct", "both_tolerable"})

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
    """Converte um status escalar em chave snake case sem acentos.

    Args:
        value (Any): Valor de status bruto a normalizar.

    Returns:
        str | None: Chave em minúsculas com underscores, ou ``None`` se ausente.
    """
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
    """Normaliza status atuais e legados dos óstios para códigos em inglês.

    Args:
        value (Any): Valor textual ou legado representando o status dos óstios.

    Returns:
        str | None: Código canônico em inglês dos óstios, ou ``None`` se ausente.
    """
    key = _status_key(value)
    return None if key is None else OSTIA_STATUS_ALIASES.get(key, key)


def normalize_result_status(value: Any) -> str | None:
    """Normaliza status atuais e legados dos resultados para códigos em inglês.

    Args:
        value (Any): Valor textual ou legado representando o status geral do exame.

    Returns:
        str | None: Código canônico em inglês do resultado, ou ``None`` se ausente.
    """
    key = _status_key(value)
    return None if key is None else RESULT_STATUS_ALIASES.get(key, key)


def ostia_status_label_pt(value: Any) -> str:
    """Retorna o rótulo em português de um status dos óstios.

    Args:
        value (Any): Código atual ou legado do status dos óstios.

    Returns:
        str: Rótulo legível em português correspondente ao status (ou ``"sem status"``).
    """
    normalized = normalize_ostia_status(value)
    if normalized is None:
        return "sem status"
    return OSTIA_STATUS_PORTUGUESE_LABELS.get(normalized, normalized)


def result_status_label_pt(value: Any) -> str:
    """Retorna o rótulo em português de um status de resultado.

    Args:
        value (Any): Código atual ou legado do status do resultado.

    Returns:
        str: Rótulo legível em português correspondente ao status (ou ``"sem status"``).
    """
    normalized = normalize_result_status(value)
    if normalized is None:
        return "sem status"
    return STATUS_PORTUGUESE_LABELS.get(normalized, normalized)


def _readable_column_name(column: str) -> str:
    """Retorna o nome legível associado a uma coluna interna do schema.

    Args:
        column (str): Nome interno da coluna de resultado.

    Returns:
        str: Nome legível para exportação ou o próprio nome original.
    """
    return READABLE_COLUMN_NAMES.get(column, column)


def _get_result_value(result: dict[str, Any], column: str, default: Any = None) -> Any:
    """Busca o valor de uma coluna aceitando chave interna ou nome legível.

    Args:
        result (dict[str, Any]): Dicionário com os campos de resultado de um exame.
        column (str): Nome interno da coluna procurada.
        default (Any): Valor padrão retornado quando nenhuma das chaves existe.

    Returns:
        Any: Valor associado à coluna encontrada ou ``default``.
    """
    if column in result:
        return result.get(column)

    readable_column = _readable_column_name(column)
    if readable_column in result:
        return result.get(readable_column)

    return default


def _as_bool_value(value: Any) -> bool:
    """Converte flags escalares atuais e legadas em booleanos.

    Args:
        value (Any): Valor booleano, numérico ou textual a converter.

    Returns:
        bool: Booleano equivalente para marcadores positivos em PT e EN.
    """
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
    """Converte um valor escalar para float, preservando ausentes como None.

    Args:
        value (Any): Valor escalar bruto a converter para ponto flutuante.

    Returns:
        float | None: Número convertido ou ``None`` quando nulo, NaN ou inválido.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def _format_bool_readable(value: Any) -> str:
    """Formata uma flag booleana para exibição padronizada em CSV.

    Args:
        value (Any): Valor booleano ou legado a formatar.

    Returns:
        str: Texto ``"yes"`` quando verdadeiro ou ``"no"`` caso contrário.
    """
    return "yes" if _as_bool_value(value) else "no"


def make_readable_results_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas técnicas do resultado para nomes/valores mais legíveis.

    Args:
        df (pd.DataFrame): DataFrame com colunas internas de resultado do pipeline.

    Returns:
        pd.DataFrame: Cópia com colunas renomeadas, flags formatadas e status
            normalizados.
    """
    # Renomeia em uma cópia para não alterar o DataFrame usado pelo pipeline.
    readable_df = df.rename(columns=READABLE_COLUMN_NAMES).copy()

    # Converte apenas colunas booleanas conhecidas para rótulos de apresentação.
    dataframe_columns = {str(column) for column in readable_df.columns}
    for column in READABLE_BOOL_COLUMNS.intersection(dataframe_columns):
        readable_df[column] = readable_df[column].map(_format_bool_readable)

    # Garante códigos de status uniformizados após a renomeação das colunas.
    if "ostia_detection_status" in readable_df.columns:
        readable_df["ostia_detection_status"] = readable_df[
            "ostia_detection_status"
        ].map(normalize_ostia_status)
    if "status" in readable_df.columns:
        readable_df["status"] = readable_df["status"].map(normalize_result_status)

    return readable_df


def select_per_image_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém somente resultados e diagnósticos que pertencem a cada exame.

    Args:
        df (pd.DataFrame): DataFrame bruto ou legado com resultados por exame.

    Returns:
        pd.DataFrame: DataFrame projetado com apenas as colunas esperadas por imagem.
    """
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
    # Projeta apenas as colunas do contrato por imagem, preenchendo ausentes com None.
    result_columns = [_readable_column_name(column) for column in RESULT_COLUMNS]
    result_columns.extend(
        c
        for c in (*LEGACY_DISTANCE_COLUMNS, *ARTERY_MASK_EXPORT_COLUMNS)
        if c in readable_df
    )
    for column in result_columns:
        if column not in readable_df.columns:
            readable_df[column] = None
    return readable_df.loc[:, result_columns]


def add_internal_result_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona aliases internos sem remover as colunas legíveis persistidas.

    Args:
        df (pd.DataFrame): DataFrame carregado de CSV com colunas legíveis ou
            internas.

    Returns:
        pd.DataFrame: Cópia com colunas internas populadas e status normalizados.
    """
    # Normaliza status existentes antes de mapear colunas para aliases internos.
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
    """Obtém uma coluna por nome interno ou alias legível, ou retorna série vazia.

    Args:
        df (pd.DataFrame): DataFrame de resultados consultado.
        column (str): Nome interno da coluna desejada.
        dtype (Any): Tipo padrão utilizado caso nenhuma candidata exista no DataFrame.

    Returns:
        pd.Series: Série encontrada no DataFrame ou série vazia indexada com ``dtype``.
    """
    column_candidates = (column, _readable_column_name(column))
    for column_candidate in column_candidates:
        if column_candidate in df.columns:
            return cast(pd.Series, df[column_candidate])
    return pd.Series(index=df.index, dtype=dtype)


def classify_result_status(result: dict[str, Any]) -> str:
    """Classifica uma linha de resultado no rótulo textual usado nos CSVs.

    Args:
        result (dict[str, Any]): Dicionário com desfecho dos óstios e erros.

    Returns:
        str: Rótulo de status padronizado definido em ``STATUS_LABELS``.
    """
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


_EXTRA_RESULT_ROW_COLUMNS: tuple[str, ...] = (
    "artery_segmentation_method",
    "threshold_mode",
    "fuzzy_mask_strategy",
    "min_threshold",
    "max_threshold",
    "lower_threshold_method",
    "lower_threshold_percentile",
    "aorta_circle_filter_method",
)
_ROW_BOOL_COLUMNS: frozenset[str] = frozenset(
    {
        "aorta_circle_filter_applied",
        "ostia_found",
        "segmentation_attempted",
        "proceeded_with_bad_ostia",
        "both_correct",
        "both_tolerable",
        "left_intersects",
        "right_intersects",
    }
)
_ROW_DEFAULTS: dict[str, Any] = {
    "artery_segmentation_method": "region_growing",
    "aorta_circle_filter_method": "none",
    "aorta_circle_filter_synthetic_tail_count": 0,
    "aorta_circle_filter_trimmed_tail_count": 0,
    "aorta_segmentation_feedback": "insufficient_data",
}


def build_result_row(result: dict[str, Any]) -> dict[str, Any]:
    """Converte um resultado bruto do pipeline em uma linha CSV padronizada.

    Args:
        result (dict[str, Any]): Dicionário bruto produzido pelo pipeline para um exame.

    Returns:
        dict[str, Any]: Linha normalizada contendo todas as colunas previstas no schema.
    """
    effective_upper_threshold_hu = _get_result_value(
        result, "effective_upper_threshold_hu"
    )
    if effective_upper_threshold_hu is None:
        # Permite retomar runs normais antigos que salvavam apenas max_threshold_hu.
        effective_upper_threshold_hu = _get_result_value(result, "max_threshold")

    # Preenche cada coluna do schema normalizando flags booleanas, status e defaults.
    row: dict[str, Any] = {"IMG_ID": result.get("IMG_ID")}
    for column in (*RESULT_COLUMNS, *_EXTRA_RESULT_ROW_COLUMNS):
        if column in {"IMG_ID", "status"}:
            continue
        if column == "effective_upper_threshold_hu":
            row[column] = effective_upper_threshold_hu
        elif column in _ROW_BOOL_COLUMNS:
            row[column] = _as_bool_value(_get_result_value(result, column, False))
        elif column == "ostia_status":
            row[column] = normalize_ostia_status(_get_result_value(result, column))
        else:
            row[column] = _get_result_value(result, column, _ROW_DEFAULTS.get(column))
    row["status"] = normalize_result_status(
        result.get("status") or classify_result_status(row)
    )
    return row


def make_result_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Converte lista de resultados em DataFrame formatado.

    Args:
        results (list[dict[str, Any]]): Lista de resultados individuais brutos.

    Returns:
        pd.DataFrame: DataFrame estruturado com colunas canônicas e históricas.
    """
    rows = [build_result_row(result) for result in results]
    optional = [
        c
        for c in (*LEGACY_DISTANCE_COLUMNS, *ARTERY_MASK_EXPORT_COLUMNS)
        if any(c in row for row in results)
    ]
    for row, original in zip(rows, results):
        row.update({c: original.get(c) for c in optional})
    return pd.DataFrame(rows, columns=[*RESULT_COLUMNS, *optional])


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Obtém uma coluna booleana por alias e converte seus valores.

    Args:
        df (pd.DataFrame): DataFrame com resultados por exame.
        column (str): Nome interno da coluna booleana buscada.

    Returns:
        pd.Series: Série com valores booleanos convertidos.
    """
    series = _series_from_aliases(df, column, dtype=bool)
    return series.map(_as_bool_value)


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Converte uma coluna e seus aliases em uma Series numérica anulável.

    Args:
        df (pd.DataFrame): DataFrame com resultados por exame.
        column (str): Nome interno da coluna numérica buscada.

    Returns:
        pd.Series: Série numérica com ausentes ou inválidos convertidos em ``NaN``.
    """
    series = _series_from_aliases(df, column, dtype=float)
    return series.map(_as_optional_float)


def _scalar_float(value: Any) -> float:
    """Converte uma redução numérica do pandas em float escalar.

    Args:
        value (Any): Resultado escalar de agregação numérica (pandas ou NumPy).

    Returns:
        float: Valor convertido para ``float`` nativo.
    """
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError("Era esperado um valor numérico escalar.")
    return float(array.item())


def _numeric_stats(
    df: pd.DataFrame,
    column: str,
    *,
    include_sum: bool = False,
) -> dict[str, Any]:
    """Calcula estatísticas estáveis para uma medição opcional por exame.

    Args:
        df (pd.DataFrame): DataFrame com resultados por exame.
        column (str): Nome interno da coluna numérica a sumarizar.
        include_sum (bool): Se ``True``, inclui o somatório total (``<prefix>_sum``).

    Returns:
        dict[str, Any]: Dicionário prefixado com contagem, média, desvio, mediana,
            quartis e extremos.
    """
    # Filtra observações válidas e devolve estatísticas nulas se não houver medições.
    values = _numeric_series(df, column).dropna()
    prefix = _readable_column_name(column)
    stats: dict[str, Any] = {f"{prefix}_count": int(len(values))}
    if values.empty:
        for suffix in ("mean", "std", "median", "q1", "q3", "min", "max"):
            stats[f"{prefix}_{suffix}"] = None
        if include_sum:
            stats[f"{prefix}_sum"] = None
        return stats

    # Calcula tendência central, dispersão, quartis e soma opcional para a coorte.
    stats.update(
        {
            f"{prefix}_mean": _scalar_float(values.mean()),
            f"{prefix}_std": (_scalar_float(values.std()) if len(values) > 1 else None),
            f"{prefix}_median": _scalar_float(values.median()),
            f"{prefix}_q1": _scalar_float(values.quantile(0.25)),
            f"{prefix}_q3": _scalar_float(values.quantile(0.75)),
            f"{prefix}_min": _scalar_float(values.min()),
            f"{prefix}_max": _scalar_float(values.max()),
        }
    )
    if include_sum:
        stats[f"{prefix}_sum"] = _scalar_float(values.sum())
    return stats


def summarize_results_df(df: pd.DataFrame) -> dict[str, Any]:
    """Calcula contagens e métricas agregadas de um DataFrame de resultados.

    Args:
        df (pd.DataFrame): DataFrame consolidado com resultados dos exames.

    Returns:
        dict[str, Any]: Dicionário com contagens, percentuais, distâncias e Dice.
    """
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
        ("artery_voxels_after_closing", False),
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

    # Medidas intermediárias não usam o Dice final como fallback em runs antigos.
    for column in (
        "dice_artery_after_closing",
        "dice_artery_closing_delta",
        "dice_artery_dilation_delta",
    ):
        values = _numeric_series(df, column).dropna()
        summary[f"{column}_count"] = int(len(values))
        summary[f"{column}_mean"] = float(values.mean()) if len(values) else None

    distances = summarize_artery_distances(df)
    for column in (
        *DISTANCE_METRIC_COLUMNS,
        *(c for c in LEGACY_DISTANCE_COLUMNS if c in df),
    ):
        summary.update(
            {f"{column}_{key}": value for key, value in distances[column].items()}
        )
    summary["artery_distance_metrics_status_counts"] = distances["status_counts"]
    summary["artery_distance_metrics_missing_status_count"] = distances[
        "missing_status_count"
    ]

    # Métricas de Dice permanecem nulas quando nenhuma artéria foi segmentada.
    if dice_series.notna().any():
        valid_ostia_dice = dice_series.where(total_success_series).dropna()
        invalid_ostia_dice = dice_series.where(~total_success_series).dropna()
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
