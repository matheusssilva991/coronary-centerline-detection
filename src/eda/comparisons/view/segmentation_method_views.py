"""Auxiliares de visualização para comparação de métodos de segmentação arterial."""

from pathlib import Path
from typing import Any, Sequence

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.comparison_utils.variant_comparison import (
    OSTIA_STATUS_COLUMNS,
    make_pair_delta,
)
from utils.visualization.results.variant_comparison import (
    OSTIA_STATUS_COLORS,
    OSTIA_STATUS_LABELS,
    prepare_variant_for_plot,
)

__all__ = [
    "OSTIA_STATUS_COLORS",
    "OSTIA_STATUS_COLUMNS",
    "OSTIA_STATUS_LABELS",
    "_set_image_ticks",
    "plot_ostia_status_by_variant",
    "plot_pair_delta_by_image",
    "plot_pair_dice_by_image",
]


def plot_ostia_status_by_variant(
    summary_df: pd.DataFrame,
    *,
    preferred_order: Sequence[str] | None = None,
    pretty_names: dict[str, str] | None = None,
    ax: Any | None = None,
    figsize: tuple[float, float] = (12, 5),
    save_path: Path | None = None,
) -> Any:
    """Plota a distribuição de status dos óstios por variante em barras empilhadas.

    Args:
        summary_df (pd.DataFrame): Resumo agregado por variante com contagens de status.
        preferred_order (Sequence[str] | None): Ordem preferencial das variantes no eixo X.
        pretty_names (dict[str, str] | None): Mapeamento opcional de nomes amigáveis.
        ax (Any | None): Eixo Matplotlib existente para desenho opcional.
        figsize (tuple[float, float]): Dimensões da figura caso um novo eixo seja criado.
        save_path (Path | None): Caminho opcional para exportar a figura gerada.

    Returns:
        Any: Eixo Matplotlib contendo o gráfico de barras empilhadas.
    """
    # Ordena e rotula as variantes antes de construir as barras empilhadas.
    plot_df = prepare_variant_for_plot(summary_df, preferred_order, pretty_names)
    label_outline = [path_effects.withStroke(linewidth=3.2, foreground="white")]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    x_labels = plot_df["variant_label"].astype(str)
    bottom = np.zeros(len(plot_df))
    # Empilha cada categoria de status sobre o total acumulado das anteriores.
    for column, label, color in zip(
        OSTIA_STATUS_COLUMNS,
        OSTIA_STATUS_LABELS,
        OSTIA_STATUS_COLORS,
    ):
        values = plot_df[column].to_numpy()
        bars = ax.bar(x_labels, values, bottom=bottom, label=label, color=color)
        for bar, value, base in zip(bars, values, bottom):
            if value <= 0:
                continue
            text = ax.text(
                bar.get_x() + bar.get_width() / 2,
                base + value / 2,
                f"{int(value)}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="black",
            )
            text.set_path_effects(label_outline)
        bottom += values

    # Anota o total de exames no topo de cada barra empilhada.
    max_total = float(max(bottom)) if len(bottom) else 1.0
    for x_pos, total in zip(range(len(x_labels)), bottom):
        ax.text(
            x_pos,
            total + max_total * 0.045,
            f"{int(total)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.18",
                "fc": "white",
                "ec": "0.75",
                "alpha": 0.9,
            },
        )

    # Configura limites, legenda e exportação opcional da figura.
    ax.set_ylabel("Número de imagens", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylim(0, max_total * 1.24)
    ax.tick_params(axis="x", rotation=35, labelsize=10)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
        framealpha=0.98,
        facecolor="white",
        edgecolor="0.85",
    )
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")
    return ax


def _set_image_ticks(ax: Any, image_ids: Sequence[int], max_ticks: int = 20) -> None:
    """Distribui rótulos de IDs de exames no eixo X sem sobreposição excessiva.

    Args:
        ax (Any): Eixo Matplotlib cujo eixo X será configurado.
        image_ids (Sequence[int]): Sequência ordenada de identificadores de exames.
        max_ticks (int): Quantidade máxima de marcações exibidas no eixo X.
    """
    if not image_ids:
        return
    # Seleciona posições igualmente espaçadas ao longo da lista de exames.
    count = len(image_ids)
    tick_count = min(max_ticks, count)
    positions = np.unique(np.linspace(0, count - 1, tick_count, dtype=int))
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [str(image_ids[position]) for position in positions],
        rotation=60,
        ha="right",
    )


def plot_pair_dice_by_image(
    results_df: pd.DataFrame,
    reference_variant: str,
    comparison_variant: str,
    *,
    pretty_names: dict[str, str] | None = None,
    axes: Sequence[Any] | None = None,
    figsize: tuple[float, float] = (15, 8),
    save_path: Path | None = None,
) -> np.ndarray:
    """Plota as séries de Dice por exame de duas variantes em painéis verticais.

    Args:
        results_df (pd.DataFrame): Resultados por exame contendo as variantes comparadas.
        reference_variant (str): Nome da variante de referência (baseline).
        comparison_variant (str): Nome da variante candidata comparada.
        pretty_names (dict[str, str] | None): Mapeamento opcional de títulos amigáveis.
        axes (Sequence[Any] | None): Par de eixos Matplotlib pré-existentes.
        figsize (tuple[float, float]): Tamanho da figura caso novos eixos sejam criados.
        save_path (Path | None): Caminho opcional para salvar a figura.

    Returns:
        np.ndarray: Array com os dois eixos Matplotlib utilizados na plotagem.
    """
    # Pareia os resultados por exame e remove entradas sem Dice válido nas duas variantes.
    pair_df = make_pair_delta(
        results_df,
        reference_variant,
        comparison_variant,
    ).sort_values("IMG_ID")
    pair_df = pair_df.dropna(subset=["reference_dice", "comparison_dice"])
    names = pretty_names or {}
    x = np.arange(len(pair_df))

    # Inicializa ou valida o par de eixos verticais compartilhados.
    if axes is None:
        _, axes_array = plt.subplots(
            2,
            1,
            figsize=figsize,
            sharex=True,
            sharey=True,
            constrained_layout=True,
            gridspec_kw={"hspace": 0.06},
        )
    else:
        axes_array = np.asarray(axes, dtype=object)
        if axes_array.size != 2:
            raise ValueError("axes deve conter exatamente dois eixos.")
        axes_array = axes_array.reshape(2)

    comparison_ax, reference_ax = axes_array
    panels = [
        (
            comparison_ax,
            names.get(comparison_variant, comparison_variant),
            pair_df["comparison_dice"],
            "#f58518",
        ),
        (
            reference_ax,
            names.get(reference_variant, reference_variant),
            pair_df["reference_dice"],
            "#4c78a8",
        ),
    ]
    # Desenha a curva de Dice por exame em cada painel.
    for panel_ax, title, values, color in panels:
        panel_ax.plot(
            x,
            values,
            color=color,
            linewidth=0.85,
            alpha=0.9,
        )
        panel_ax.set_ylim(0, 1)
        panel_ax.set_ylabel("Dice", fontsize=22, labelpad=10)
        panel_ax.set_title(title, fontsize=18)
        panel_ax.tick_params(axis="y", labelsize=16)
        panel_ax.grid(axis="y", alpha=0.25)

    # Ajusta os rótulos de exames no eixo inferior e exporta se solicitado.
    comparison_ax.tick_params(axis="x", labelbottom=False)
    reference_ax.set_xlabel(
        "Exames",
        fontsize=22,
        labelpad=14,
    )
    _set_image_ticks(reference_ax, pair_df["IMG_ID"].astype(int).tolist())
    reference_ax.tick_params(axis="x", labelsize=16)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        axes_array[0].figure.savefig(save_path, dpi=300, bbox_inches="tight")
    return axes_array


def plot_pair_delta_by_image(
    results_df: pd.DataFrame,
    reference_variant: str,
    comparison_variant: str,
    *,
    pretty_names: dict[str, str] | None = None,
    ax: Any | None = None,
    figsize: tuple[float, float] = (15, 5.5),
    save_path: Path | None = None,
) -> Any:
    """Plota a diferença de Dice (comparação menos referência) por exame.

    Args:
        results_df (pd.DataFrame): Resultados por exame contendo as variantes comparadas.
        reference_variant (str): Nome da variante de referência (baseline).
        comparison_variant (str): Nome da variante candidata comparada.
        pretty_names (dict[str, str] | None): Mapeamento opcional de nomes amigáveis.
        ax (Any | None): Eixo Matplotlib existente para desenho opcional.
        figsize (tuple[float, float]): Dimensões da figura caso um novo eixo seja criado.
        save_path (Path | None): Caminho opcional para salvar a figura gerada.

    Returns:
        Any: Eixo Matplotlib contendo a curva de delta de Dice por exame.
    """
    _ = pretty_names
    # Pareia os resultados ordenados por IMG_ID e extrai o vetor de diferenças.
    pair_df = make_pair_delta(
        results_df,
        reference_variant,
        comparison_variant,
    ).sort_values("IMG_ID")
    pair_df = pair_df.dropna(subset=["reference_dice", "comparison_dice"])
    x = np.arange(len(pair_df))
    delta = pair_df["dice_delta"].to_numpy(dtype=float)

    # Desenha a linha de diferença com preenchimento verde para ganhos e vermelho para perdas.
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.plot(x, delta, color="#444444", linewidth=1.1)
    ax.fill_between(x, 0, delta, where=delta >= 0, color="#2ca02c", alpha=0.55)
    ax.fill_between(x, 0, delta, where=delta < 0, color="#d62728", alpha=0.55)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel(
        "Delta Dice",
        fontsize=22,
        labelpad=10,
    )
    ax.set_xlabel(
        "Exames",
        fontsize=22,
        labelpad=14,
    )
    ax.set_ylim(-1, 1)
    _set_image_ticks(ax, pair_df["IMG_ID"].astype(int).tolist())
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")
    return ax
