"""Gráficos comparativos de métricas por subconjunto (train/val/test) e resolução."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def prepare_subset_plot_df(subset_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Filtra subconjuntos disponíveis e traduz rótulos para o eixo X.

    Args:
        subset_summary_df (pd.DataFrame): Resumo por split e resolução com coluna ``disponivel``.

    Returns:
        pd.DataFrame: Cópia filtrada contendo a coluna traduzida ``conjunto``.
    """
    # Filtra apenas linhas marcadas como disponíveis.
    plot_subset_df = subset_summary_df[subset_summary_df["disponivel"]].copy()
    # Traduz nome do subset para rótulo do eixo X.
    plot_subset_df["conjunto"] = plot_subset_df["subset"].map(
        {"train": "Treino", "val": "Validacao", "test": "Teste"}
    )
    return plot_subset_df


def plot_subset_metric_by_resolution(
    subset_summary_df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    palette: Any,
    ylim: tuple[float, float] | None = None,
    hline_y: float | None = None,
    hline_kwargs: dict[str, Any] | None = None,
    bar_label_fmt: str = "%.3f",
) -> Any:
    """Plota barras agrupadas de uma métrica por subconjunto e resolução.

    Args:
        subset_summary_df (pd.DataFrame): Resumo por subconjunto e resolução.
        metric_col (str): Nome da coluna numérica a plotar no eixo Y.
        title (str): Título do gráfico.
        ylabel (str): Rótulo do eixo Y.
        palette (Any): Paleta de cores repassada ao ``seaborn.barplot``.
        ylim (tuple[float, float] | None): Limites inferior e superior opcionais do eixo Y.
        hline_y (float | None): Posição vertical opcional para linha horizontal de referência.
        hline_kwargs (dict[str, Any] | None): Estilo adicional da linha de referência.
        bar_label_fmt (str): Formato numérico aplicado aos rótulos de barra.

    Returns:
        Any: Eixo Matplotlib (``Axes``) contendo o gráfico renderizado.
    """
    # Gera base de plot com subsets válidos.
    plot_subset_df = prepare_subset_plot_df(subset_summary_df)

    # Constrói gráfico de barras agrupadas por subset e resolução.
    _, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax = sns.barplot(
        data=plot_subset_df,
        x="conjunto",
        y=metric_col,
        hue="resolucao",
        palette=palette,
    )

    if ylim is not None:
        ax.set_ylim(*ylim)

    if hline_y is not None:
        line_kwargs: dict[str, Any] = {
            "color": "black",
            "linewidth": 1,
            "linestyle": "--",
        }
        if hline_kwargs:
            line_kwargs.update(hline_kwargs)
        ax.axhline(hline_y, **line_kwargs)

    # Configura títulos, grade, legenda e rótulos numéricos nas barras.
    ax.set_title(title, pad=16)
    ax.set_xlabel("Conjunto")
    ax.set_ylabel(ylabel, labelpad=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Resolucao")

    for container in ax.containers:
        ax.bar_label(container, fmt=bar_label_fmt, padding=3)

    plt.show()
    return ax


def plot_subset_execution_time_by_resolution(
    subset_summary_df: pd.DataFrame,
    palette: Sequence[str] | None = None,
    ymax_factor: float = 1.12,
) -> Any:
    """Plota o tempo total de execução (em minutos) por subconjunto e resolução.

    Args:
        subset_summary_df (pd.DataFrame): Resumo com a coluna ``tempo_execucao_min``.
        palette (Sequence[str] | None): Paleta opcional de cores por resolução.
        ymax_factor (float): Fator multiplicativo aplicado ao maior tempo para folga no eixo Y.

    Returns:
        Any: Eixo Matplotlib retornado por ``plot_subset_metric_by_resolution``.
    """
    if palette is None:
        palette = ["#F58518", "#E45756"]

    # Calcula teto do eixo Y com margem sobre o tempo máximo disponível.
    plot_subset_df = prepare_subset_plot_df(subset_summary_df)
    max_time = plot_subset_df["tempo_execucao_min"].max()
    ylim = (0.0, float(max_time) * ymax_factor) if pd.notna(max_time) else (0.0, 1.0)

    return plot_subset_metric_by_resolution(
        subset_summary_df=subset_summary_df,
        metric_col="tempo_execucao_min",
        title="Tempo de execução por conjunto e resolução",
        ylabel="Tempo de execução (min)",
        palette=palette,
        ylim=ylim,
        bar_label_fmt="%.1f",
    )


def plot_subset_ostia_success_by_resolution(
    subset_summary_df: pd.DataFrame,
    palette: Sequence[str] | None = None,
) -> Any:
    """Plota o percentual de sucesso na detecção de óstios por subconjunto e resolução.

    Args:
        subset_summary_df (pd.DataFrame): Resumo com a coluna ``sucesso_total_percent``.
        palette (Sequence[str] | None): Paleta opcional de cores por resolução.

    Returns:
        Any: Eixo Matplotlib retornado por ``plot_subset_metric_by_resolution``.
    """
    if palette is None:
        palette = ["#54A24B", "#2E8B57"]

    return plot_subset_metric_by_resolution(
        subset_summary_df=subset_summary_df,
        metric_col="sucesso_total_percent",
        title="Percentual de óstios detectados com sucesso por conjunto e resolução",
        ylabel="Óstios detectados com sucesso (%)",
        palette=palette,
        ylim=(0, 100),
        bar_label_fmt="%.1f%%",
    )
