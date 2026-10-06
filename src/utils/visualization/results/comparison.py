from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
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
    """Plota uma métrica agrupada em um painel por dimensão de comparação."""
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
        axis.set_title(panel)
        axis.set_xticks(x_positions, x_order)
        axis.set_ylim(*ylim)
        axis.set_ylabel(ylabel)
        axis.legend(loc="lower right")
    fig.suptitle(title, fontsize=14)
    return fig, axes


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
