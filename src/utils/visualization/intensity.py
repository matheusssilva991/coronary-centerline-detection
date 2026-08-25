"""Helpers de visualização para histogramas de intensidade em HU."""

from typing import Any

import pandas as pd


def calculate_binned_intensity_mean_median(
    histogram: pd.DataFrame,
    *,
    value_column: str = "count",
) -> tuple[float, float]:
    """Estima média e mediana usando os centros e pesos dos bins."""
    required_columns = {"bin_center_hu", value_column}
    missing = required_columns.difference(histogram.columns)
    if missing:
        raise ValueError(f"Colunas de histograma ausentes: {sorted(missing)}")

    centers = pd.to_numeric(histogram["bin_center_hu"], errors="coerce")
    weights = pd.to_numeric(histogram[value_column], errors="coerce").fillna(0)
    valid = centers.notna() & weights.gt(0)
    centers = centers.loc[valid]
    weights = weights.loc[valid]
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("O histograma não possui pesos positivos.")

    mean_hu = float((centers * weights).sum() / total)
    median_index = weights.cumsum().ge(total / 2).idxmax()
    return mean_hu, float(centers.loc[median_index])


def plot_binned_intensity_histogram(
    ax: Any,
    histogram: pd.DataFrame,
    *,
    value_column: str = "count",
    mean_hu: float,
    median_hu: float,
    distribution_label: str = "Distribuição HU",
    color: str = "#31688e",
    linewidth: float = 1.3,
    log_scale: bool = False,
) -> None:
    """Plota um histograma discretizado e marca sua média e mediana em HU."""
    required_columns = {"bin_center_hu", value_column}
    missing = required_columns.difference(histogram.columns)
    if missing:
        raise ValueError(f"Colunas de histograma ausentes: {sorted(missing)}")

    ax.step(
        histogram["bin_center_hu"],
        histogram[value_column],
        where="mid",
        color=color,
        linewidth=linewidth,
        label=distribution_label,
    )
    ax.axvline(
        mean_hu,
        color="#2a9d8f",
        linestyle=":",
        linewidth=1.4,
        label=f"Média: {mean_hu:.1f} HU",
    )
    ax.axvline(
        median_hu,
        color="#555555",
        linestyle="--",
        linewidth=1.3,
        label=f"Mediana: {median_hu:.1f} HU",
    )
    if log_scale:
        ax.set_yscale("log")
