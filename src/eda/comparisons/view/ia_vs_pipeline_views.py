"""Auxiliares de visualização para comparação entre modelos de IA e pipeline matemático."""

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.comparison_utils.ia_math import (
    build_comparison_agg_df,
    prettify_method_label,
)

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    go = None
    px = None

__all__ = [
    "display_comparison_scenario",
    "plot_comparison_bar_by_resolution",
    "plot_ia_vs_math_scatter_interactive",
    "plot_image_dice_scatter_interactive",
]


def plot_comparison_bar_by_resolution(
    agg: pd.DataFrame, resolution: str, comparison_title: Optional[str] = None
) -> None:
    """Plota gráfico de barras comparando o Dice médio por método em uma resolução.

    Args:
        agg (pd.DataFrame): DataFrame agregado com médias e desvios-padrão de Dice.
        resolution (str): Resolução alvo a ser filtrada (`mid_res` ou `high_res`).
        comparison_title (Optional[str]): Título base opcional para o gráfico.
    """
    # Exibe aviso visual caso o DataFrame agregado esteja vazio ou sem coluna de resolução.
    if agg.empty or "target_resolution" not in agg.columns:
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, f"Sem dados para {resolution}", ha="center", va="center")
        plt.axis("off")
        plt.show()
        return

    # Filtra somente a resolução solicitada e verifica disponibilidade de linhas.
    subset = agg[agg["target_resolution"] == resolution].copy()
    if subset.empty:
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, f"Sem dados para {resolution}", ha="center", va="center")
        plt.axis("off")
        plt.show()
        return

    # Formata rótulos de origem e ordena os métodos pela média de Dice decrescente.
    subset["origin_label"] = subset["source"].map({"ia": "IA", "math": "Matematico"})
    subset["method_label"] = subset["method"].apply(prettify_method_label)
    method_order = subset.sort_values("mean_dice", ascending=False)[
        "method_label"
    ].tolist()
    subset["method_label"] = pd.Categorical(
        subset["method_label"], categories=method_order, ordered=True
    )
    subset = subset.sort_values("method_label")

    # Plota as barras separando métodos de IA e pipeline matemático por cor.
    plt.figure(figsize=(12, 5))
    ax = sns.barplot(
        data=subset,
        x="method_label",
        y="mean_dice",
        hue="origin_label",
        palette={"IA": "#4C78A8", "Matematico": "#F58518"},
    )

    # Sobrepõe o desvio-padrão como barra de erro vertical em cada método.
    for idx, row in subset.reset_index(drop=True).iterrows():
        std_val = row["std_dice"]
        if pd.notna(std_val):
            ax.errorbar(
                x=idx,
                y=row["mean_dice"],
                yerr=std_val,
                fmt="none",
                ecolor="black",
                elinewidth=1.2,
                capsize=3,
            )

    # Ajusta títulos, rótulos numéricos das barras e formatação dos eixos.
    if comparison_title:
        ax.set_title(f"{comparison_title} - {resolution}")
    else:
        ax.set_title(f"Comparacao de Dice por metodo - {resolution}")
    ax.set_xlabel("Metodo")
    ax.set_ylabel("Dice medio")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=35)
    for tick_label in ax.get_xticklabels():
        tick_label.set_ha("right")

    for container in ax.containers:
        if hasattr(container, "patches") and container.patches:
            ax.bar_label(container, fmt="%.3f", padding=3)

    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Origem")
    plt.tight_layout()
    plt.show()


def plot_image_dice_scatter_interactive(
    comparison_df: pd.DataFrame, resolution: str, comparison_title: Optional[str] = None
) -> None:
    """Plota dispersão interativa de Dice por exame para métodos de IA e pipeline matemático.

    Args:
        comparison_df (pd.DataFrame): Dados pareados por imagem entre IA e método matemático.
        resolution (str): Resolução alvo a ser filtrada (`mid_res` ou `high_res`).
        comparison_title (Optional[str]): Título base opcional para o gráfico interativo.
    """
    if px is None or go is None:
        raise ImportError(
            "Plotly is required for interactive plots. Please install plotly."
        )

    # Filtra a resolução desejada e trata o caso sem dados disponíveis.
    subset = comparison_df[comparison_df["target_resolution"] == resolution].copy()
    if subset.empty:
        fig = go.Figure()
        title = comparison_title or "Comparacao de Dice por imagem"
        fig.add_annotation(
            text=f"Sem dados para {title} - {resolution}", showarrow=False
        )
        fig.show()
        return

    subset = subset.sort_values(["ia_method", "img_id"]).reset_index(drop=True)

    # Adiciona uma série de marcadores para cada modelo de IA avaliado.
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for method_index, (method, method_df) in enumerate(subset.groupby("ia_method")):
        method_label = prettify_method_label(str(method))
        fig.add_trace(
            go.Scatter(
                x=method_df["img_id"].astype(str),
                y=method_df["ia_dice"],
                mode="markers",
                marker=dict(color=colors[method_index % len(colors)], size=7),
                name=method_label,
                customdata=method_df[["img_id", "ia_method"]].values,
                hovertemplate=(
                    "img_id: %{customdata[0]}<br>Dice: %{y:.3f}"
                    "<br>Metodo: %{customdata[1]}<extra></extra>"
                ),
            )
        )

    # Adiciona a série única do pipeline matemático sem duplicar exames por método de IA.
    math_df = subset.drop_duplicates(["target_resolution", "img_id"])
    fig.add_trace(
        go.Scatter(
            x=math_df["img_id"].astype(str),
            y=math_df["math_dice"],
            mode="markers",
            marker=dict(color="#222222", size=8, symbol="x"),
            name="Pipeline matemático",
            customdata=math_df[["img_id", "math_method"]].values,
            hovertemplate="img_id: %{customdata[0]}<br>Dice: %{y:.3f}<br>Metodo: %{customdata[1]}<extra></extra>",
        )
    )

    # Configura layout, limites verticais e inclinação dos rótulos de exames.
    title = comparison_title or "Comparacao Dice por imagem"
    fig.update_layout(
        title=f"{title} - {resolution}",
        xaxis_title="Imagem",
        yaxis_title="Dice",
        yaxis=dict(range=[0, 1.05]),
        template="simple_white",
        legend_title_text="Método",
        margin=dict(l=40, r=20, t=60, b=120),
    )
    fig.update_xaxes(tickangle=45)
    fig.show()


def plot_ia_vs_math_scatter_interactive(
    comparison_df: pd.DataFrame, resolution: str, comparison_title: Optional[str] = None
) -> None:
    """Plota dispersão interativa entre o Dice da IA e o Dice do pipeline matemático.

    Args:
        comparison_df (pd.DataFrame): Dados pareados por imagem entre IA e método matemático.
        resolution (str): Resolução alvo a ser filtrada (`mid_res` ou `high_res`).
        comparison_title (Optional[str]): Título base opcional para o gráfico interativo.
    """
    if px is None or go is None:
        raise ImportError(
            "Plotly is required for interactive plots. Please install plotly."
        )

    # Filtra a resolução selecionada e exibe anotação caso não existam pares.
    subset = comparison_df[comparison_df["target_resolution"] == resolution].copy()
    if subset.empty:
        fig = go.Figure()
        title = comparison_title or "Comparacao IA vs Matematico"
        fig.add_annotation(
            text=f"Sem dados para {title} - {resolution}", showarrow=False
        )
        fig.show()
        return

    # Constrói o gráfico de dispersão colorido por método de IA com linha de identidade.
    subset["ia_method_label"] = subset["ia_method"].apply(prettify_method_label)
    fig = px.scatter(
        subset,
        x="ia_dice",
        y="math_dice",
        hover_data=["img_id", "ia_method", "math_method"],
        color="ia_method_label",
        height=600,
        width=760,
    )
    fig.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray")
    )
    title = comparison_title or "Comparacao IA vs Matematico"
    fig.update_layout(
        title=f"{title} - {resolution}",
        legend_title_text="Método de IA",
    )
    fig.update_xaxes(range=[0, 1.05], title_text="Dice IA")
    fig.update_yaxes(range=[0, 1.05], title_text="Dice Matematico")
    fig.show()


def display_comparison_scenario(
    scenario_label: str,
    scenario_math_df: pd.DataFrame,
    scenario_ia_df: pd.DataFrame,
    resolution: str,
) -> None:
    """Agrega resultados de um cenário de óstios e exibe o gráfico de barras por resolução.

    Args:
        scenario_label (str): Nome descritivo do cenário avaliado (ex.: Completo).
        scenario_math_df (pd.DataFrame): Resultados filtrados do pipeline matemático.
        scenario_ia_df (pd.DataFrame): Resultados filtrados dos métodos de IA.
        resolution (str): Resolução alvo para exibição (`mid_res` ou `high_res`).
    """
    # Reúne IA e pipeline matemático antes de agregar as métricas do cenário.
    scenario_raw_df = pd.concat([scenario_ia_df, scenario_math_df], ignore_index=True)
    scenario_agg_df = build_comparison_agg_df(scenario_raw_df)

    # Renderiza o gráfico de barras para a resolução solicitada.
    plot_comparison_bar_by_resolution(
        scenario_agg_df,
        resolution,
        comparison_title=scenario_label,
    )
