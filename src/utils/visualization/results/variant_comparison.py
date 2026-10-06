"""Organiza rótulos e gráficos de comparação entre variantes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from utils.comparison_utils.variant_comparison import (
    largest_pair_changes,
    order_variants,
)

OSTIA_STATUS_LABELS = [
    "Ambos corretos",
    "Ambos toleráveis",
    "Encontrados, mas incorretos",
    "Não encontrados/erro",
]


OSTIA_STATUS_COLORS = ["#2ca02c", "#8fd175", "#ff9f1a", "#d62728"]


def add_variant_labels(
    df: pd.DataFrame,
    pretty_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Adiciona/atualiza a coluna legível ``variant_label``."""
    out = df.copy()
    names = pretty_names or {}
    out["variant_label"] = (
        out["folder_variant"].map(names).fillna(out["folder_variant"])
    )
    return out


def prepare_variant_for_plot(
    df: pd.DataFrame,
    preferred_order: Sequence[str] | None = None,
    pretty_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Prepara labels categóricos e ordenação para gráficos por variante."""
    out = add_variant_labels(df, pretty_names)
    out = order_variants(out, preferred_order)
    if preferred_order:
        labels = [
            pretty_names.get(name, name) if pretty_names else name
            for name in preferred_order
        ]
        out["variant_label"] = pd.Categorical(
            out["variant_label"],
            categories=labels,
            ordered=True,
        )
        out = out.sort_values("variant_label")
    return out.reset_index(drop=True)


def plot_largest_pair_changes(
    results_df: pd.DataFrame,
    reference_variant: str,
    comparison_variant: str,
    *,
    title: str,
    top_n: int = 15,
    save_path: Path | None = None,
    ax: Any | None = None,
) -> Any:
    """Plota as maiores variações de Dice entre duas variantes."""
    plot_df = largest_pair_changes(
        results_df, reference_variant, comparison_variant, top_n=top_n
    ).sort_values("dice_delta")
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in plot_df["dice_delta"]]
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.barh(plot_df["IMG_ID"].astype(str), plot_df["dice_delta"], color=colors)
    for bar, value in zip(bars, plot_df["dice_delta"]):
        ha = "left" if value >= 0 else "right"
        offset = 0.003 if value >= 0 else -0.003
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f}",
            va="center",
            ha=ha,
            fontsize=9,
        )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Delta Dice: comparação - referência", fontsize=12)
    ax.set_ylabel("IMG_ID", fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")
    return ax


__all__ = [
    "OSTIA_STATUS_COLORS",
    "OSTIA_STATUS_LABELS",
    "add_variant_labels",
    "plot_largest_pair_changes",
    "prepare_variant_for_plot",
]
