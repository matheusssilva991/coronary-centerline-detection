"""Utilitários tipados para acessar dados tabulares do projeto."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Literal

import pandas as pd


def require_series_column(frame: pd.DataFrame, column: Hashable) -> pd.Series:
    """Retorna uma coluna única e rejeita nomes duplicados no DataFrame."""
    values = frame[column]
    if isinstance(values, pd.DataFrame):
        raise ValueError(f"A coluna {column!r} está duplicada.")
    return values


def numeric_series(frame: pd.DataFrame, column: Hashable) -> pd.Series:
    """Converte uma coluna única para valores numéricos anuláveis."""
    return to_numeric_series(require_series_column(frame, column))


def to_numeric_series(
    values: pd.Series,
    *,
    errors: Literal["raise", "coerce"] = "coerce",
) -> pd.Series:
    """Converte uma Series e preserva explicitamente seu tipo tabular."""
    converted = pd.to_numeric(values, errors=errors)
    if not isinstance(converted, pd.Series):
        raise TypeError("A conversão numérica não retornou uma Series.")
    return converted


__all__ = ["numeric_series", "require_series_column", "to_numeric_series"]
