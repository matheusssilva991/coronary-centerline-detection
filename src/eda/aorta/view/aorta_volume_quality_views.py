"""Gráficos da análise exploratória de volume e qualidade visual da aorta."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def plot_ostia_relationship(cohort_df: pd.DataFrame, cohort_name: str) -> None:
    """Relaciona ocupação percentual da aorta e maior erro físico entre os dois óstios.

    Args:
        cohort_df (pd.DataFrame): Coorte com ocupação da aorta, erro máximo e status dos óstios.
        cohort_name (str): Nome da coorte exibido no título.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = cohort_df["ostia_success"].map({True: "#2a9d8f", False: "#d1495b"})

    # Plota dispersão colorida pelo sucesso dos óstios e rotula exames com falha.
    ax.scatter(
        cohort_df["aorta_volume_percentage"],
        cohort_df["max_ostium_distance_mm"],
        c=colors,
        s=70,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.7,
    )
    for row in cohort_df.loc[~cohort_df["ostia_success"]].itertuples():
        ax.annotate(
            str(row.IMG_ID),
            (float(row.aorta_volume_percentage), float(row.max_ostium_distance_mm)),
            xytext=(5, 5),
            textcoords="offset points",
        )

    # Destaca a linha de tolerância física usada na validação dos óstios.
    ax.axhline(
        7.0,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label="tolerância de 7 mm",
    )
    ax.set_xlabel("Volume da aorta em relação à imagem (%)")
    ax.set_ylabel("Maior distância dos óstios (mm)")
    ax.set_title(f"Ocupação da aorta e localização dos óstios — {cohort_name}")
    ax.grid(alpha=0.2)
    ax.legend()
    plt.show()


def plot_visual_quality(cohort_df: pd.DataFrame, cohort_name: str) -> None:
    """Compara ocupação no volume e área por fatia entre aortas boas e ruins.

    Args:
        cohort_df (pd.DataFrame): Coorte contendo ``visual_aorta_quality`` e métricas de volume.
        cohort_name (str): Nome da coorte exibido nos títulos dos subpainéis.
    """
    good = cohort_df.loc[cohort_df["visual_aorta_quality"].eq("boa")]
    bad = cohort_df.loc[cohort_df["visual_aorta_quality"].eq("ruim")]

    # Desenha boxplots lado a lado para ocupação global e largura média por fatia.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].boxplot(
        [good["aorta_volume_percentage"], bad["aorta_volume_percentage"]],
        tick_labels=["Boa", "Ruim"],
        patch_artist=True,
        boxprops={"facecolor": "#9fd3c7"},
    )
    axes[0].set_ylabel("Volume da aorta (%)")
    axes[0].set_title(f"Ocupação no volume — {cohort_name}")

    axes[1].boxplot(
        [
            good["aorta_voxels_per_segmented_slice"],
            bad["aorta_voxels_per_segmented_slice"],
        ],
        tick_labels=["Boa", "Ruim"],
        patch_artist=True,
        boxprops={"facecolor": "#f4a261"},
    )
    axes[1].set_ylabel("Voxels por fatia segmentada")
    axes[1].set_title(f"Largura média da máscara — {cohort_name}")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_ordered_volume(cohort_df: pd.DataFrame, cohort_name: str) -> None:
    """Exibe barras ordenadas de ocupação da aorta com qualidade visual e status dos óstios.

    Args:
        cohort_df (pd.DataFrame): Coorte com ``IMG_ID``, ocupação, qualidade visual e óstios.
        cohort_name (str): Nome da coorte exibido no título do gráfico.
    """
    ordered = cohort_df.sort_values("aorta_volume_percentage").reset_index(drop=True)
    bar_colors = ordered["visual_aorta_quality"].map(
        {"boa": "#2a9d8f", "ruim": "#d1495b"}
    )
    x_positions = np.arange(len(ordered))
    volume_values = ordered["aorta_volume_percentage"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(
        x_positions,
        volume_values,
        color=bar_colors,
        edgecolor="#333333",
        linewidth=0.45,
    )

    # Aplica hachura nas barras correspondentes a aortas classificadas como ruins.
    for bar, is_bad in zip(
        bars, ordered["visual_aorta_quality"].eq("ruim"), strict=True
    ):
        if is_bad:
            bar.set_hatch("///")

    # Posiciona marcadores acima das barras para indicar sucesso ou falha dos óstios.
    marker_offset = max(float(np.nanmax(volume_values)) * 0.025, 0.03)
    marker_y = volume_values + marker_offset
    ostia_success = ordered["ostia_success"].to_numpy(dtype=bool)
    ax.scatter(
        x_positions[ostia_success],
        marker_y[ostia_success],
        marker="o",
        s=34,
        color="#1b4332",
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax.scatter(
        x_positions[~ostia_success],
        marker_y[~ostia_success],
        marker="X",
        s=52,
        color="#111111",
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
    )

    # Configura eixo X ordenado por IMG_ID, legenda combinada e limites verticais.
    ax.set_xticks(x_positions, ordered["IMG_ID"].astype(str))
    ax.set_xlabel("IMG_ID ordenado pela porcentagem da aorta")
    ax.set_ylabel("Volume da aorta (%)")
    ax.set_title(f"Ocupação da máscara e classificação visual — {cohort_name}")
    ax.tick_params(axis="x", rotation=70)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(
        handles=[
            Patch(facecolor="#2a9d8f", edgecolor="#333333", label="Aorta boa"),
            Patch(
                facecolor="#d1495b",
                edgecolor="#333333",
                hatch="///",
                label="Aorta ruim",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#1b4332",
                markeredgecolor="white",
                markersize=7,
                label="Óstios: sucesso",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="none",
                markerfacecolor="#111111",
                markeredgecolor="white",
                markersize=8,
                label="Óstios: falha",
            ),
        ],
        loc="upper left",
        ncols=2,
        frameon=True,
    )
    ax.set_ylim(top=float(np.nanmax(marker_y)) + marker_offset * 3.0)
    fig.tight_layout()
    plt.show()
