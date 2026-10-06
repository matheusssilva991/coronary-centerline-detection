"""Calcula resumos de Dice e acertos dos óstios por conjunto e resolução."""

import pandas as pd
from typing import Any, Dict, Optional, Sequence

from utils.project.results.schema import (
    normalize_ostia_status,
    ostia_status_label_pt,
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
