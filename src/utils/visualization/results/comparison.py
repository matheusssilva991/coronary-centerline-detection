"""Gráficos em painéis agrupados para comparação de métricas entre métodos e resoluções."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    go = None
    px = None


def plot_grouped_metric_panels(
    summary: pd.DataFrame,
    *,
    panel_col: str,
    group_col: str,
    x_col: str,
    metric_col: str,
    panel_order: list[str],
    group_order: list[str],
    x_order: list[str],
    colors: dict[str, str],
    ylabel: str,
    title: str,
    ylim: tuple[float, float],
    value_format: str = "%.3f",
) -> tuple[Any, Any]:
    """Plota uma métrica agrupada em barras divididas por painéis de comparação.

    Args:
        summary (pd.DataFrame): Tabela agregada contendo as colunas de painel,
            grupo, categoria no eixo X e métrica.
        panel_col (str): Nome da coluna usada para separar os painéis
            horizontais.
        group_col (str): Nome da coluna que define as séries agrupadas por cor.
        x_col (str): Nome da coluna com as categorias exibidas no eixo X.
        metric_col (str): Nome da coluna numérica plotada na altura das barras.
        panel_order (list[str]): Ordem explícita dos valores de ``panel_col``.
        group_order (list[str]): Ordem explícita das séries de ``group_col``.
        x_order (list[str]): Ordem explícita das categorias de ``x_col``.
        colors (dict[str, str]): Mapeamento de cada nome de grupo para sua cor.
        ylabel (str): Rótulo exibido no eixo Y dos painéis.
        title (str): Título principal exibido acima de todos os painéis.
        ylim (tuple[float, float]): Limites ``(mínimo, máximo)`` do eixo Y.
        value_format (str): Formato estilo printf aplicado aos rótulos acima
            das barras.

    Returns:
        tuple[Any, Any]: Par ``(fig, axes)`` do Matplotlib com os painéis
        renderizados.
    """
    # Inicializa a grade de painéis e calcula largura e deslocamentos das barras.
    fig, axes = plt.subplots(
        1,
        len(panel_order),
        figsize=(7 * len(panel_order), 5),
        sharey=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    width = 0.8 / len(group_order)
    x_positions = np.arange(len(x_order))
    offsets = (np.arange(len(group_order)) - (len(group_order) - 1) / 2) * width

    # Filtra os dados de cada painel e plota as séries agrupadas em barras.
    for axis, panel in zip(axes_array, panel_order, strict=True):
        panel_data = summary.loc[summary[panel_col].eq(panel)]
        for offset, group in zip(offsets, group_order, strict=True):
            values = (
                panel_data.loc[panel_data[group_col].eq(group)]
                .set_index(x_col)
                .reindex(x_order)[metric_col]
            )
            if values.isna().any():
                raise ValueError(
                    f"Dados incompletos para painel={panel!r}, grupo={group!r}"
                )
            bars = axis.bar(
                x_positions + offset,
                values,
                width,
                label=group,
                color=colors[group],
            )
            axis.bar_label(bars, fmt=value_format, padding=3)
        # Configura rótulos, limites do eixo Y e legenda do painel atual.
        axis.set_title(panel)
        axis.set_xticks(x_positions, x_order)
        axis.set_ylim(*ylim)
        axis.set_ylabel(ylabel)
        axis.legend(loc="lower right")
    fig.suptitle(title, fontsize=14)
    return fig, axes
