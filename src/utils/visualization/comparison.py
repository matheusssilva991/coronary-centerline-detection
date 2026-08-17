from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
except Exception:
    go = None
    px = None

from ..comparison_utils.ia_math import prettify_method_label


def plot_dice_distribution_for_publication(
    dice_values: pd.Series,
    *,
    output_path: str | Path | None = None,
    dpi: int = 300,
    color: str = "skyblue",
    bins: int = 20,
) -> tuple[Any, dict[str, float | int]]:
    """Plota Dice, média e faixa de um desvio padrão para publicação."""
    values = pd.to_numeric(dice_values, errors="coerce").dropna()
    if values.empty:
        raise ValueError("dice_values não contém valores numéricos válidos.")

    mean = float(values.mean())
    std = float(values.std())
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=dpi)
    sns.histplot(
        values,
        bins=bins,
        kde=True,
        color=color,
        edgecolor="black",
        ax=ax,
    )
    ax.axvline(mean, color="red", linewidth=2, label=f"Média = {mean:.3f}")
    ax.axvline(
        mean - std,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"-1σ = {mean - std:.3f}",
    )
    ax.axvline(
        mean + std,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"+1σ = {mean + std:.3f}",
    )
    ax.axvspan(mean - std, mean + std, color="orange", alpha=0.15)
    ax.set_xlabel("Dice Score", fontsize=16)
    ax.set_ylabel("Frequência", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=12)
    fig.tight_layout()

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=dpi, bbox_inches="tight")

    return fig, {"count": int(values.size), "mean": mean, "std": std}


def plot_comparison_bar_by_resolution(
    agg: pd.DataFrame, resolution: str, comparison_title: Optional[str] = None
) -> None:
    """Plota comparação de Dice por método para uma resolução alvo."""
    # Filtra somente a resolução solicitada.
    subset = agg[agg["target_resolution"] == resolution].copy()
    if subset.empty:
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, f"Sem dados para {resolution}", ha="center", va="center")
        plt.axis("off")
        plt.show()
        return

    subset["origin_label"] = subset["source"].map({"ia": "IA", "math": "Matematico"})
    subset["method_label"] = subset["method"].apply(prettify_method_label)
    # Ordena métodos por média de Dice.
    method_order = subset.sort_values("mean_dice", ascending=False)[
        "method_label"
    ].tolist()
    subset["method_label"] = pd.Categorical(
        subset["method_label"], categories=method_order, ordered=True
    )
    subset = subset.sort_values("method_label")

    # Plota barras separando IA e Matemático por cor.
    plt.figure(figsize=(12, 5))
    ax = sns.barplot(
        data=subset,
        x="method_label",
        y="mean_dice",
        hue="origin_label",
        palette={"IA": "#4C78A8", "Matematico": "#F58518"},
    )

    for idx, row in subset.reset_index(drop=True).iterrows():
        # Usa desvio padrão como barra de erro.
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


def plot_image_dice_scatter_by_resolution(
    comparison_df: pd.DataFrame, resolution: str, comparison_title: Optional[str] = None
) -> None:
    """Plota Dice por imagem para IA e método matemático no mesmo eixo X."""
    subset = comparison_df[comparison_df["target_resolution"] == resolution].copy()
    if subset.empty:
        plt.figure(figsize=(10, 5))
        title = comparison_title or "Comparacao de Dice por imagem"
        plt.text(
            0.5, 0.5, f"Sem dados para {title} - {resolution}", ha="center", va="center"
        )
        plt.axis("off")
        plt.show()
        return

    subset = subset.sort_values("img_id").reset_index(drop=True)
    x_positions = np.arange(len(subset))
    tick_step = max(1, len(subset) // 12)

    plt.figure(figsize=(14, 5))
    ax = plt.gca()
    ax.scatter(
        x_positions - 0.12,
        subset["ia_dice"],
        color="#D62728",
        alpha=0.75,
        s=18,
        label="IA",
    )
    ax.scatter(
        x_positions + 0.12,
        subset["math_dice"],
        color="#1F77B4",
        alpha=0.75,
        s=18,
        label="Matematico",
    )

    ax.set_xticks(x_positions[::tick_step])
    ax.set_xticklabels(
        subset["img_id"].astype(str).iloc[::tick_step], rotation=45, ha="right"
    )
    ax.set_xlim(-1, len(subset))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Imagem")
    ax.set_ylabel("Dice")
    if comparison_title:
        ax.set_title(f"{comparison_title} - {resolution}")
    else:
        ax.set_title(f"Comparacao Dice por imagem - {resolution}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Origem")
    plt.tight_layout()
    plt.show()


def plot_ia_vs_math_scatter_by_resolution(
    comparison_df: pd.DataFrame, resolution: str, comparison_title: Optional[str] = None
) -> None:
    """Plota Dice da IA versus Dice do método matemático por imagem."""
    subset = comparison_df[comparison_df["target_resolution"] == resolution].copy()
    if subset.empty:
        plt.figure(figsize=(6, 6))
        title = comparison_title or "Comparacao IA vs Matematico"
        plt.text(
            0.5, 0.5, f"Sem dados para {title} - {resolution}", ha="center", va="center"
        )
        plt.axis("off")
        plt.show()
        return

    plt.figure(figsize=(6.5, 6.5))
    ax = plt.gca()
    ax.scatter(
        subset["ia_dice"],
        subset["math_dice"],
        color="#5B8FF9",
        alpha=0.75,
        s=22,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, alpha=0.8)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Dice IA")
    ax.set_ylabel("Dice Matematico")
    if comparison_title:
        ax.set_title(f"{comparison_title} - {resolution}")
    else:
        ax.set_title(f"Comparacao IA vs Matematico - {resolution}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_image_dice_scatter_interactive(
    comparison_df: pd.DataFrame, resolution: str, comparison_title: Optional[str] = None
) -> None:
    """Interactive per-image scatter (IA and Math) using Plotly.

    Hover shows `img_id`, dice and method.
    """
    if px is None or go is None:
        raise ImportError(
            "Plotly is required for interactive plots. Please install plotly."
        )

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

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for method_index, (method, method_df) in enumerate(subset.groupby("ia_method")):
        method_label = prettify_method_label(method)
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

    # O Dice matemático é igual nas linhas repetidas por método de IA.
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
    """Interactive IA vs Math scatter using Plotly.

    Hover shows `img_id` plus methods and scores.
    """
    if px is None:
        raise ImportError(
            "Plotly is required for interactive plots. Please install plotly."
        )

    subset = comparison_df[comparison_df["target_resolution"] == resolution].copy()
    if subset.empty:
        fig = go.Figure()
        title = comparison_title or "Comparacao IA vs Matematico"
        fig.add_annotation(
            text=f"Sem dados para {title} - {resolution}", showarrow=False
        )
        fig.show()
        return

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


def plot_dice_distribution_by_subset(
    df_mid: Optional[pd.DataFrame], df_high: Optional[pd.DataFrame], subset_label: str
) -> None:
    """Plota distribuição dos Dice scores para mid e high por subconjunto."""
    # Mostra histogramas de Dice para Mid e High no mesmo painel.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if df_mid is not None and not df_mid.empty:
        # Converte e remove valores inválidos de Dice.
        dice_scores_mid = pd.to_numeric(
            df_mid["dice_artery"],
            errors="coerce",
        ).dropna()
        mu_mid = dice_scores_mid.mean()
        sigma_mid = dice_scores_mid.std()
        # KDE mostra a forma da distribuição.
        sns.histplot(
            dice_scores_mid,
            bins=20,
            kde=True,
            color="skyblue",
            edgecolor="black",
            ax=axes[0],
        )
        axes[0].axvline(
            mu_mid,
            color="red",
            linestyle="-",
            linewidth=2,
            label=f"Media = {mu_mid:.3f}",
        )
        axes[0].axvline(
            mu_mid - sigma_mid,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"-1sigma = {mu_mid - sigma_mid:.3f}",
        )
        axes[0].axvline(
            mu_mid + sigma_mid,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"+1sigma = {mu_mid + sigma_mid:.3f}",
        )
        axes[0].axvspan(
            mu_mid - sigma_mid, mu_mid + sigma_mid, color="orange", alpha=0.15
        )
        # Marca média e faixa de 1 sigma.
        axes[0].set_title(f"Distribuicao dos Dice Scores - Mid Res ({subset_label})")
        axes[0].set_xlabel("Dice Score")
        axes[0].set_ylabel("Frequencia")
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].legend()
    else:
        axes[0].text(
            0.5,
            0.5,
            f"Sem dados de {subset_label.lower()} para Mid Res",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        axes[0].set_title(f"Distribuicao dos Dice Scores - Mid Res ({subset_label})")

    if df_high is not None and not df_high.empty:
        # Repete o mesmo fluxo para High.
        dice_scores_high = pd.to_numeric(
            df_high["dice_artery"], errors="coerce"
        ).dropna()
        mu_high = dice_scores_high.mean()
        sigma_high = dice_scores_high.std()
        sns.histplot(
            dice_scores_high,
            bins=20,
            kde=True,
            color="lightgreen",
            edgecolor="black",
            ax=axes[1],
        )
        axes[1].axvline(
            mu_high,
            color="red",
            linestyle="-",
            linewidth=2,
            label=f"Media = {mu_high:.3f}",
        )
        axes[1].axvline(
            mu_high - sigma_high,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"-1sigma = {mu_high - sigma_high:.3f}",
        )
        axes[1].axvline(
            mu_high + sigma_high,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"+1sigma = {mu_high + sigma_high:.3f}",
        )
        axes[1].axvspan(
            mu_high - sigma_high, mu_high + sigma_high, color="orange", alpha=0.15
        )
        axes[1].set_title(f"Distribuicao dos Dice Scores - High Res ({subset_label})")
        axes[1].set_xlabel("Dice Score")
        axes[1].set_ylabel("Frequencia")
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].legend()
    else:
        axes[1].text(
            0.5,
            0.5,
            f"Sem dados de {subset_label.lower()} para High Res\n(Em preparacao)",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            fontsize=11,
            color="gray",
        )
        axes[1].set_title(f"Distribuicao dos Dice Scores - High Res ({subset_label})")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

    plt.tight_layout()
    plt.show()
