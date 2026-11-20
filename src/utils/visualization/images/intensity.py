"""Reúne visualizações de histogramas de intensidade em HU."""

from typing import Any

import pandas as pd

from utils.project.dataframe import numeric_series, require_series_column


def calculate_binned_intensity_mean_median(
    histogram: pd.DataFrame,
    *,
    value_column: str = "count",
) -> tuple[float, float]:
    """Estima média e mediana de intensidade usando centros e pesos dos bins.

    Args:
        histogram (pd.DataFrame): DataFrame com a coluna ``bin_center_hu`` e a
            coluna de pesos indicada por ``value_column``.
        value_column (str): Nome da coluna usada como peso de cada bin (por
            exemplo, contagem de voxels ou fração de volume).

    Returns:
        tuple[float, float]: Par ``(média_hu, mediana_hu)`` estimado a partir dos
        bins válidos em unidades Hounsfield (HU).
    """
    required_columns = {"bin_center_hu", value_column}
    missing = required_columns.difference(histogram.columns)
    if missing:
        raise ValueError(f"Colunas de histograma ausentes: {sorted(missing)}")

    # Converte centros e pesos para série numérica e filtra bins inválidos ou vazios.
    centers = numeric_series(histogram, "bin_center_hu")
    weights = numeric_series(histogram, value_column).fillna(0)
    valid = centers.notna() & weights.gt(0)
    centers = centers.loc[valid]
    weights = weights.loc[valid]
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("O histograma não possui pesos positivos.")

    # Calcula a média ponderada e localiza o bin cuja soma acumulada atinge 50%.
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
    """Plota um histograma discretizado e marca sua média e mediana em HU.

    Args:
        ax (Any): Eixo do Matplotlib onde o histograma será desenhado.
        histogram (pd.DataFrame): DataFrame contendo ``bin_center_hu`` e
            ``value_column``.
        value_column (str): Nome da coluna de frequências ou densidades do
            histograma.
        mean_hu (float): Valor médio da distribuição em HU exibido como linha
            vertical pontilhada.
        median_hu (float): Valor mediano da distribuição em HU exibido como
            linha vertical tracejada.
        distribution_label (str): Rótulo da curva na legenda do gráfico.
        color (str): Cor da linha principal do histograma em formato aceito pelo
            Matplotlib.
        linewidth (float): Espessura do traço da curva em degraus.
        log_scale (bool): Se ``True``, configura o eixo Y em escala logarítmica.
    """
    # Garante a presença das colunas obrigatórias do histograma.
    required_columns = {"bin_center_hu", value_column}
    missing = required_columns.difference(histogram.columns)
    if missing:
        raise ValueError(f"Colunas de histograma ausentes: {sorted(missing)}")

    # Traça a curva em degraus centrada nos bins do histograma.
    ax.step(
        require_series_column(histogram, "bin_center_hu"),
        require_series_column(histogram, value_column),
        where="mid",
        color=color,
        linewidth=linewidth,
        label=distribution_label,
    )
    # Destaca média e mediana em HU e aplica escala logarítmica se solicitada.
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
