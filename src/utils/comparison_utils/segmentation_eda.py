"""Calcula resumos arteriais e dos óstios por conjunto e resolução."""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Sequence

from utils.project.dataframe import numeric_series
from utils.project.results.schema import (
    normalize_ostia_status,
    ostia_status_label_pt,
)

ARTERY_SURFACE_METRICS = {"artery_hd_mm": "HD", "artery_ahd_mm": "AHD"}
ARTERY_SURFACE_STATUS = "artery_distance_metrics_status"


def has_artery_surface_distances(frame: pd.DataFrame | None) -> bool:
    """Identifica resultados com as colunas de distância e status."""
    return frame is not None and {
        ARTERY_SURFACE_STATUS,
        *ARTERY_SURFACE_METRICS,
    }.issubset(frame.columns)


def artery_surface_distance_cases(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Retorna exames com HD e AHD válidos, indexados pelo ID do exame."""
    if frame is None or not has_artery_surface_distances(frame):
        return pd.DataFrame(
            columns=[*ARTERY_SURFACE_METRICS, "dice_artery"],
            index=pd.Index([], name="IMG_ID"),
        )
    if frame["IMG_ID"].isna().any() or frame["IMG_ID"].duplicated().any():
        raise ValueError("A comparação de HD/AHD exige IDs presentes e únicos.")
    evaluated = frame.loc[frame[ARTERY_SURFACE_STATUS].eq("ok")].set_index("IMG_ID")
    cases = pd.DataFrame(
        {column: numeric_series(evaluated, column) for column in ARTERY_SURFACE_METRICS}
    )
    if "dice_artery" in evaluated:
        dice = numeric_series(evaluated, "dice_artery")
        cases["dice_artery"] = dice.where(dice.between(0, 1))
    else:
        cases["dice_artery"] = np.nan
    hd = cases["artery_hd_mm"]
    ahd = cases["artery_ahd_mm"]
    return cases.loc[np.isfinite(hd) & np.isfinite(ahd) & ahd.ge(0) & hd.ge(ahd)]


def summarize_artery_surface_distance_cases(
    source: pd.DataFrame | None, cases: pd.DataFrame
) -> dict[str, Any]:
    """Resume a cobertura e as distâncias usando somente os exames válidos."""
    row: dict[str, Any] = {
        "Exames no CSV": len(source) if source is not None else None,
        "Exames válidos": len(cases),
        "Cobertura": (
            "Disponível"
            if not cases.empty
            else "Sem medidas válidas"
            if has_artery_surface_distances(source)
            else "Indisponível"
        ),
    }
    for column, label in ARTERY_SURFACE_METRICS.items():
        row[f"{label} média (mm)"] = cases[column].mean()
        row[f"{label} mediana (mm)"] = cases[column].median()
    row["Dice médio"] = cases["dice_artery"].mean()
    return row


def pair_artery_surface_distance_cases(
    mid: pd.DataFrame, high: pd.DataFrame
) -> pd.DataFrame:
    """Pareia exames válidos nas duas resoluções pelo ID, em um único split."""
    columns = list(ARTERY_SURFACE_METRICS)
    return mid[columns].join(
        high[columns], how="inner", lsuffix="_mid", rsuffix="_high", validate="1:1"
    )


def _normalize_success_status(value: Any) -> str | None:
    """Normaliza rótulos legados e legíveis para status internos."""
    return normalize_ostia_status(value)


def _success_status_series(
    df_split: pd.DataFrame,
    status_column: str,
) -> pd.Series:
    """Retorna status normalizados dos óstios priorizando o alias canônico."""
    source_column = "ostia_status" if "ostia_status" in df_split else status_column
    return df_split[source_column].map(_normalize_success_status)


def _has_success_status_column(df_split: pd.DataFrame, status_column: str) -> bool:
    """Verifica se existe um status legível ou canônico dos óstios."""
    return "ostia_status" in df_split or status_column in df_split


def summarize_surface_distances_with_successful_ostia(
    source: pd.DataFrame | None, cases: pd.DataFrame
) -> dict[str, Any]:
    """Resume HD e AHD dos exames com ambos os óstios corretos ou toleráveis."""
    valid = cases.iloc[:0]
    success_count: int | None = None
    if source is not None:
        status_column = (
            "ostia_detection_status" if "ostia_detection_status" in source else "status"
        )
        if _has_success_status_column(source, status_column):
            successful = _success_status_series(source, status_column).isin(
                ["both_correct", "both_tolerable"]
            )
            success_count = int(successful.sum())
            valid = cases.loc[cases.index.isin(source.loc[successful, "IMG_ID"])]
    summary = {
        "Ambos corretos ou toleráveis": success_count,
        "Exames com HD e AHD válidos": len(valid),
    }
    for column, label in ARTERY_SURFACE_METRICS.items():
        summary[f"{label} média (mm)"] = valid[column].mean()
        summary[f"{label} mediana (mm)"] = valid[column].median()
    return summary


def _normalized_success_statuses(success_status: Sequence[str]) -> list[str]:
    """Normaliza rótulos solicitados preservando a ordem de exibição."""
    return [
        normalized
        for status in success_status
        if (normalized := _normalize_success_status(status)) is not None
    ]


def _get_split_df(
    data_by_resolution: Optional[Dict[str, Any]], resolution: str, split_name: str
) -> Optional[pd.DataFrame]:
    if data_by_resolution is None:
        return None
    resolution_data = data_by_resolution.get(resolution, {})
    return resolution_data.get(split_name)


def build_success_status_summary_by_subset(
    data_by_resolution: Optional[Dict[str, Any]],
    split_name: str,
    success_status: Sequence[str],
    status_column: str = "status",
) -> pd.DataFrame:
    """Resume os acertos por status usando o total do split como denominador."""
    rows = []

    for resolution in ["high", "mid"]:
        df_split = _get_split_df(data_by_resolution, resolution, split_name)
        if (
            df_split is None
            or df_split.empty
            or not _has_success_status_column(df_split, status_column)
        ):
            rows.append(
                {
                    "split": split_name,
                    "resolution": resolution,
                    "status": "sem dados",
                    "quantidade": 0,
                    "percentual_do_total": 0.0,
                    "total_acertos": 0,
                    "total_imagens": 0,
                }
            )
            continue

        total_images = len(df_split)
        normalized_status = _success_status_series(df_split, status_column)
        normalized_success = _normalized_success_statuses(success_status)
        status_counts = normalized_status[
            normalized_status.isin(normalized_success)
        ].value_counts()
        total_success = int(status_counts.sum())

        for status, normalized_status_name in zip(
            success_status, normalized_success, strict=True
        ):
            count = int(status_counts.get(normalized_status_name, 0))
            percentage = 100 * count / total_images if total_images > 0 else 0.0
            rows.append(
                {
                    "split": split_name,
                    "resolution": resolution,
                    "status": ostia_status_label_pt(normalized_status_name),
                    "quantidade": count,
                    "percentual_do_total": round(percentage, 2),
                    "total_acertos": total_success,
                    "total_imagens": total_images,
                }
            )

        success_percentage = (
            100 * total_success / total_images if total_images > 0 else 0.0
        )
        rows.append(
            {
                "split": split_name,
                "resolution": resolution,
                "status": "total acertos",
                "quantidade": total_success,
                "percentual_do_total": round(success_percentage, 2),
                "total_acertos": total_success,
                "total_imagens": total_images,
            }
        )

    return pd.DataFrame(rows)


def build_dice_summary_by_subset(
    data_by_resolution: Optional[Dict[str, Any]],
    split_name: str,
    dice_column: str = "dice_artery",
) -> pd.DataFrame:
    """Retorna um resumo de Dice por resolução para um subset."""
    rows = []

    for resolution in ["high", "mid"]:
        df_split = _get_split_df(data_by_resolution, resolution, split_name)
        if df_split is None or df_split.empty or dice_column not in df_split.columns:
            dice_values = pd.Series(dtype=float)
        else:
            dice_values = pd.to_numeric(df_split[dice_column], errors="coerce").dropna()

        rows.append(
            {
                "resolution": resolution.upper(),
                "split": str(split_name).upper(),
                "count": int(dice_values.shape[0]),
                "mean": float(dice_values.mean()) if not dice_values.empty else pd.NA,
                "median": float(dice_values.median())
                if not dice_values.empty
                else pd.NA,
                "std": float(dice_values.std()) if not dice_values.empty else pd.NA,
                "min": float(dice_values.min()) if not dice_values.empty else pd.NA,
                "max": float(dice_values.max()) if not dice_values.empty else pd.NA,
            }
        )

    return pd.DataFrame(rows)
