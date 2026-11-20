"""Gráficos de distribuição de status, distâncias de óstios, Dice e superfície arterial."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.comparison_utils.segmentation_eda import (
    ARTERY_SURFACE_METRICS,
    _get_split_df,
    _has_success_status_column,
    _normalized_success_statuses,
    _success_status_series,
)
from utils.project.results.schema import normalize_result_status, result_status_label_pt


def _get_split_title(split_name: str) -> str:
    """Converte o identificador do split para título em maiúsculas.

    Args:
        split_name (str): Nome do subconjunto (``train``, ``val`` ou ``test``).

    Returns:
        str: Rótulo do subconjunto em letras maiúsculas.
    """
    return str(split_name).upper()


def _get_status_color(split_name: str) -> str:
    """Seleciona a cor padrão das barras de status para o subconjunto informado.

    Args:
        split_name (str): Nome do subconjunto (``train``, ``val`` ou ``test``).

    Returns:
        str: Nome da cor Matplotlib associada ao split.
    """
    return {
        "train": "skyblue",
        "val": "lightgreen",
        "test": "salmon",
    }.get(split_name, "steelblue")


def plot_status_distribution_by_subset(
    data_by_resolution: dict[str, Any] | None,
    split_name: str,
    status_column: str = "status",
) -> None:
    """Plota a distribuição de status dos óstios para High e Mid em um subconjunto.

    Args:
        data_by_resolution (dict[str, Any] | None): Dicionário aninhado por resolução e split.
        split_name (str): Subconjunto avaliado (``train``, ``val`` ou ``test``).
        status_column (str): Coluna de status a contabilizar.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    title_suffix = _get_split_title(split_name)
    color = _get_status_color(split_name)

    # Renderiza painel para cada resolução normalizando códigos de status para português.
    for idx, resolution in enumerate(["high", "mid"]):
        df_split = _get_split_df(data_by_resolution, resolution, split_name)
        ax = axes[idx]

        if df_split is None or df_split.empty or status_column not in df_split.columns:
            ax.text(
                0.5,
                0.5,
                f"Sem dados para {resolution.upper()}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(
                f"{resolution.upper()} Resolution - Distribuição dos Status ({title_suffix})"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        status_counts = (
            df_split[status_column]
            .map(normalize_result_status)
            .map(result_status_label_pt)
            .value_counts()
        )
        status_counts.plot(kind="bar", color=color, edgecolor="black", ax=ax)

        # Ajusta o limite superior do eixo Y para acomodar os rótulos de contagem.
        max_count = (
            int(max(status_counts.values)) if len(status_counts.values) > 0 else 0
        )
        ax.set_ylim(0, max_count * 1.18 if max_count > 0 else 1)
        ax.set_title(
            f"{resolution.upper()} Resolution - Distribuição dos Status ({title_suffix})",
            pad=12,
        )
        ax.set_xlabel("Status")
        ax.set_ylabel("Número de Imagens")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        for i, value in enumerate(status_counts.values):
            text_offset = max_count * 0.03 if max_count > 0 else 0.1
            ax.text(
                i,
                float(value) + text_offset,
                str(value),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.show()


def plot_success_error_by_subset(
    data_by_resolution: dict[str, Any] | None,
    split_name: str,
    success_status: Sequence[str],
    status_column: str = "status",
    success_color: str = "#4CAF50",
    error_color: str = "#F44336",
) -> None:
    """Plota contagem e percentual de acertos vs. erros para High e Mid em um subconjunto.

    Args:
        data_by_resolution (dict[str, Any] | None): Dicionário por resolução e split.
        split_name (str): Subconjunto avaliado (``train``, ``val`` ou ``test``).
        success_status (Sequence[str]): Status considerados sucesso na detecção dos óstios.
        status_column (str): Nome preferencial da coluna de status.
        success_color (str): Cor das barras de acerto.
        error_color (str): Cor das barras de erro.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    title_suffix = _get_split_title(split_name)

    # Calcula acertos e erros em cada resolução usando a mesma regra normalizada do resumo.
    for idx, resolution in enumerate(["high", "mid"]):
        df_split = _get_split_df(data_by_resolution, resolution, split_name)
        ax = axes[idx]

        if (
            df_split is None
            or df_split.empty
            or not _has_success_status_column(df_split, status_column)
        ):
            ax.text(
                0.5,
                0.5,
                f"Sem dados para {resolution.upper()}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(
                f"{resolution.upper()} Resolution - Acertos vs Erros ({title_suffix})"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        total = len(df_split)
        normalized_status = _success_status_series(df_split, status_column)
        normalized_success = _normalized_success_statuses(success_status)
        successful_count = int(normalized_status.isin(normalized_success).sum())
        error_count = total - successful_count

        success_pct = 100 * successful_count / total if total > 0 else 0.0
        error_pct = 100 * error_count / total if total > 0 else 0.0

        bars = ax.bar(
            ["Acertos", "Erros"],
            [successful_count, error_count],
            color=[success_color, error_color],
            alpha=0.8,
            edgecolor="black",
        )

        # Ajusta limites e anota contagem absoluta e percentual acima de cada barra.
        ax.set_ylabel("Número de Imagens")
        max_count = max(successful_count, error_count)
        ax.set_ylim(0, max_count * 1.18 if max_count > 0 else 1)
        ax.set_title(
            f"{resolution.upper()} Resolution - Acertos vs Erros ({title_suffix})",
            pad=12,
        )
        ax.grid(axis="y", alpha=0.3)

        for i, (bar, value) in enumerate(zip(bars, [successful_count, error_count])):
            percentage = [success_pct, error_pct][i]
            height = bar.get_height()
            text_offset = max_count * 0.03 if max_count > 0 else 0.1
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + text_offset,
                f"{int(value)}\n({percentage:.1f}%)",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.show()


def plot_distance_distribution_by_subset(
    data_by_resolution: dict[str, Any] | None,
    split_name: str,
    left_column: str = "left_dist_mm",
    right_column: str = "right_dist_mm",
    tolerance_mm: float = 7.0,
    bins: int | None = None,
    high_color: str = "#FF6B6B",
    mid_color: str = "#4ECDC4",
) -> None:
    """Plota histogramas de distâncias dos óstios esquerdo/direito para High e Mid.

    Args:
        data_by_resolution (dict[str, Any] | None): Dados carregados por resolução e split.
        split_name (str): Subconjunto avaliado (``train``, ``val`` ou ``test``).
        left_column (str): Coluna com a distância física do óstio esquerdo em ``mm``.
        right_column (str): Coluna com a distância física do óstio direito em ``mm``.
        tolerance_mm (float): Limiar de tolerância em ``mm`` destacado nos histogramas.
        bins (int | None): Número de bins do histograma; ``None`` usa padrão por split.
        high_color (str): Cor dos histogramas de High Resolution.
        mid_color (str): Cor dos histogramas de Mid Resolution.
    """
    if bins is None:
        bins = {"train": 25, "val": 15, "test": 25}.get(split_name, 25)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title_suffix = _get_split_title(split_name)

    # Constrói os histogramas de distância esquerda/direita para cada resolução.
    for idx, resolution in enumerate(["high", "mid"]):
        df_split = _get_split_df(data_by_resolution, resolution, split_name)
        ax_left = axes[0, idx]
        ax_right = axes[1, idx]

        if df_split is None or df_split.empty:
            for ax in [ax_left, ax_right]:
                ax.text(
                    0.5,
                    0.5,
                    f"Sem dados para {resolution.upper()}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
            ax_left.set_title(
                f"{resolution.upper()} - Left Distance Distribution ({title_suffix})"
            )
            ax_right.set_title(
                f"{resolution.upper()} - Right Distance Distribution ({title_suffix})"
            )
            continue

        left_dist = (
            pd.to_numeric(df_split[left_column], errors="coerce").dropna()
            if left_column in df_split.columns
            else pd.Series(dtype=float)
        )
        right_dist = (
            pd.to_numeric(df_split[right_column], errors="coerce").dropna()
            if right_column in df_split.columns
            else pd.Series(dtype=float)
        )

        # Remove valores infinitos que impedem o cálculo dos intervalos do histograma.
        if not left_dist.empty:
            left_dist = left_dist[np.isfinite(left_dist)]
        if not right_dist.empty:
            right_dist = right_dist[np.isfinite(right_dist)]
        color = high_color if resolution == "high" else mid_color

        # Imprime resumo de contagens abaixo e acima do limiar de tolerância.
        left_below = int((left_dist <= tolerance_mm).sum())
        left_above = int((left_dist > tolerance_mm).sum())
        right_below = int((right_dist <= tolerance_mm).sum())
        right_above = int((right_dist > tolerance_mm).sum())

        print(f"\n{resolution.upper()} - {title_suffix}")
        print(
            f"  Left Distance: {left_below} abaixo de {tolerance_mm}mm, {left_above} acima"
        )
        print(
            f"  Right Distance: {right_below} abaixo de {tolerance_mm}mm, {right_above} acima"
        )

        ax_left.hist(left_dist, bins=bins, alpha=0.8, color=color, edgecolor="black")
        ax_left.axvline(
            tolerance_mm,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Tolerância ({tolerance_mm:g}mm)",
        )
        ax_left.set_xlabel("Left Distance (mm)")
        ax_left.set_ylabel("Frequência")
        ax_left.set_title(
            f"{resolution.upper()} - Left Distance Distribution ({title_suffix})"
        )
        ax_left.legend()
        ax_left.grid(axis="y", alpha=0.3)

        ax_right.hist(right_dist, bins=bins, alpha=0.8, color=color, edgecolor="black")
        ax_right.axvline(
            tolerance_mm,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Tolerância ({tolerance_mm:g}mm)",
        )
        ax_right.set_xlabel("Right Distance (mm)")
        ax_right.set_ylabel("Frequência")
        ax_right.set_title(
            f"{resolution.upper()} - Right Distance Distribution ({title_suffix})"
        )
        ax_right.legend()
        ax_right.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_dice_distribution_by_subset(
    df_mid: pd.DataFrame | None,
    df_high: pd.DataFrame | None,
    subset_label: str,
) -> None:
    """Plota histogramas com KDE, média e faixa de 1 desvio-padrão do Dice para Mid e High.

    Args:
        df_mid (pd.DataFrame | None): Resultados de Mid Resolution para o subconjunto.
        df_high (pd.DataFrame | None): Resultados de High Resolution para o subconjunto.
        subset_label (str): Rótulo descritivo do subconjunto exibido nos títulos.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Renderiza distribuição de Dice de Mid Resolution com média e intervalo de ±1 sigma.
    if df_mid is not None and not df_mid.empty:
        dice_scores_mid = pd.to_numeric(
            df_mid["dice_artery"],
            errors="coerce",
        ).dropna()
        mu_mid = dice_scores_mid.mean()
        sigma_mid = dice_scores_mid.std()
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

    # Renderiza distribuição de Dice de High Resolution no painel direito.
    if df_high is not None and not df_high.empty:
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


def plot_paired_artery_surface_distances(
    paired: pd.DataFrame,
    mid_source: str,
    high_source: str,
) -> None:
    """Plota comparação pareada de HD e AHD de superfície entre Mid e High por exame.

    Args:
        paired (pd.DataFrame): DataFrame pareado por exame com métricas ``*_mid`` e ``*_high``.
        mid_source (str): Nome da fonte de dados usada para Mid Resolution.
        high_source (str): Nome da fonte de dados usada para High Resolution.
    """
    if paired.empty:
        print(
            "Comparação pareada indisponível. Nenhum exame tem HD e AHD válidos nas duas resoluções."
        )
        return

    # Desenha gráficos de dispersão com linha de identidade para HD e AHD.
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.1), layout="constrained")
    fig.set_layout_engine("constrained", h_pad=0.13, w_pad=0.10)
    for ax, (column, label) in zip(axes, ARTERY_SURFACE_METRICS.items(), strict=True):
        mid = paired[f"{column}_mid"]
        high = paired[f"{column}_high"]
        limit = max(1.0, float(max(mid.max(), high.max())) * 1.06)
        ax.plot([0, limit], [0, limit], "--", color="#65717d", label="Mid = High")
        ax.scatter(mid, high, s=18, alpha=0.42, color="#16697a", edgecolors="none")
        ax.set(
            xlim=(0, limit),
            ylim=(0, limit),
            xlabel=f"Mid ({mid_source}) · {label} (mm)",
            ylabel=f"High ({high_source}) · {label} (mm)",
            title=f"{label} · médias mid {mid.mean():.2f} | high {high.mean():.2f} mm",
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left", frameon=False)
    fig.suptitle(
        f"Distâncias arteriais no teste ImageCAS · {len(paired)} exames pareados"
    )
    plt.show()
