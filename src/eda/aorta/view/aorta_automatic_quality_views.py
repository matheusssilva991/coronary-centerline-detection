"""Visualizações e tabelas para avaliação automática da qualidade da aorta."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.patches import Patch

from utils.project.analysis.aorta_automatic_quality import (
    IMAGECAS_AORTA_GROUPS,
    OSTIA_OUTCOMES,
)
from utils.project.analysis.external_visual_assessment import AORTA_FEEDBACK_ORDER

DATASETS = ["ImageCAS", "MM-WHS", "OrCaScore"]
SUBSETS = {"train": "Treino", "val": "Validação", "test": "Teste"}
FEEDBACK_COLORS = ["#22A65A", "#E6AE00", "#DA4A49", "#9CA3AF", "#64748B"]
SHORT_LABELS = [
    "Adequada",
    "Suspeita sub",
    "Suspeita sobre",
    "Insuficientes",
    "Sem feedback",
]
OSTIA_COLORS = ["#22A65A", "#79C9B1", "#DA4A49", "#E6AE00", "#9CA3AF"]


def cohort_rows(
    frame: pd.DataFrame,
    dataset: str,
    resolution: str,
    subset: str,
) -> pd.DataFrame:
    """Filtra linhas pertencentes a uma combinação de banco, resolução e subconjunto.

    Args:
        frame (pd.DataFrame): Tabela consolidada de registros ou cobertura.
        dataset (str): Nome do banco avaliado.
        resolution (str): Resolução alvo (``mid`` ou ``high``).
        subset (str): Subconjunto avaliado (``train``, ``val`` ou ``test``).

    Returns:
        pd.DataFrame: Subconjunto filtrado de ``frame``.
    """
    return frame.loc[
        frame["dataset"].eq(dataset)
        & frame["resolution"].eq(resolution)
        & frame["subset"].eq(subset)
    ]


def show_source_coverage(selected_coverage: pd.DataFrame) -> None:
    """Exibe matriz visual de disponibilidade das fontes por banco, resolução e split.

    Args:
        selected_coverage (pd.DataFrame): Tabela de cobertura filtrada para a
            referência selecionada.
    """
    statuses = [
        "Disponível",
        "Diagnóstico não persistido",
        "Arquivo ausente",
        "Não se aplica",
        "Coorte vazia",
    ]
    colors = ["#D8F0E0", "#FFF0BD", "#F9D3D3", "#E5E7EB", "#DBEAFE"]
    fig, ax = plt.subplots(figsize=(15, 3.5), layout="constrained")

    # Preenche cada célula da grade com cor de status e contagem observada/esperada.
    for y, dataset in enumerate(DATASETS):
        for x, (resolution, subset) in enumerate(
            (resolution, subset) for resolution in ("mid", "high") for subset in SUBSETS
        ):
            row = cohort_rows(selected_coverage, dataset, resolution, subset)
            status = row["status"].iloc[0] if not row.empty else "Arquivo ausente"
            color = colors[statuses.index(status)]
            ax.add_patch(
                plt.Rectangle(
                    (x - 0.48, y - 0.45),
                    0.96,
                    0.9,
                    facecolor=color,
                    edgecolor="white",
                )
            )
            label = status
            if not row.empty and pd.notna(row["exam_count"].iloc[0]):
                count = int(row["exam_count"].iloc[0])
                expected = row["expected_count"].iloc[0]
                label += f"\nN={count}" + (
                    f"/{int(expected)}" if pd.notna(expected) else ""
                )
            ax.text(
                x,
                y,
                label.replace(
                    "Diagnóstico não persistido", "Diagnóstico\nnão persistido"
                ),
                ha="center",
                va="center",
                fontsize=9,
            )

    # Configura rótulos dos eixos e título da matriz de cobertura.
    ax.set(
        xticks=range(6),
        xticklabels=[
            f"{r.upper()} / {name}"
            for r in ("mid", "high")
            for name in SUBSETS.values()
        ],
        yticks=range(3),
        yticklabels=DATASETS,
        xlim=(-0.5, 5.5),
        ylim=(2.5, -0.5),
        title="Cobertura das fontes atuais por dataset e subconjunto",
    )
    ax.grid(False)
    plt.show()


def show_dataset_comparison(
    selected_summary: pd.DataFrame,
    selected_coverage: pd.DataFrame,
    title: str,
    resolutions: Sequence[str] = ("mid", "high"),
) -> None:
    """Exibe barras horizontais empilhadas das classes de feedback da aorta por banco.

    Args:
        selected_summary (pd.DataFrame): Resumo percentual por categoria de feedback.
        selected_coverage (pd.DataFrame): Cobertura por banco, resolução e subconjunto.
        title (str): Título geral da figura.
        resolutions (Sequence[str]): Resoluções exibidas nas linhas da grade.
    """
    fig, axes = plt.subplots(
        len(resolutions),
        3,
        figsize=(18, 4 * len(resolutions) + 1),
        squeeze=False,
        layout="constrained",
    )

    # Desenha barras empilhadas para cada combinação de resolução e subconjunto.
    for row_index, resolution in enumerate(resolutions):
        for column_index, (subset, subset_label) in enumerate(SUBSETS.items()):
            ax = axes[row_index, column_index]
            labels = []
            for y, dataset in enumerate([*DATASETS, "Geral"]):
                group = cohort_rows(selected_summary, dataset, resolution, subset)
                source = cohort_rows(selected_coverage, dataset, resolution, subset)
                count = source["exam_count"].iloc[0] if not source.empty else pd.NA
                label = dataset
                if (
                    dataset == "Geral"
                    and not source.empty
                    and source["status"].iloc[0] == "Parcial"
                ):
                    label = "Geral parcial"
                labels.append(label + (f" (N={int(count)})" if pd.notna(count) else ""))
                if group.empty:
                    status = (
                        source["status"].iloc[0]
                        if not source.empty
                        else "Arquivo ausente"
                    )
                    ax.text(
                        50,
                        y,
                        status,
                        ha="center",
                        va="center",
                        color="#666666",
                        fontsize=9,
                    )
                    continue
                values = group.set_index("status")["percent"].reindex(
                    AORTA_FEEDBACK_ORDER, fill_value=0
                )
                left = 0.0
                for percent, color in zip(values, FEEDBACK_COLORS, strict=True):
                    ax.barh(
                        y,
                        percent,
                        left=left,
                        color=color,
                        height=0.55,
                        edgecolor="#333333" if dataset == "Geral" else "none",
                        linewidth=0.8,
                    )
                    if percent >= 8:
                        ax.text(
                            left + percent / 2,
                            y,
                            f"{percent:.1f}%".replace(".", ","),
                            ha="center",
                            va="center",
                            fontsize=9,
                        )
                    left += float(percent)
            ax.set(
                yticks=range(4),
                yticklabels=labels,
                ylim=(3.65, -0.65),
                xlim=(0, 100),
                xlabel="Exames presentes no CSV (%)",
                title=f"{subset_label} — {resolution.upper()}",
            )
            ax.grid(axis="y", visible=False)

    # Adiciona título global e legenda compartilhada das categorias de feedback.
    fig.suptitle(title, fontsize=15)
    fig.legend(
        handles=[
            Patch(facecolor=color, label=status)
            for status, color in zip(AORTA_FEEDBACK_ORDER, FEEDBACK_COLORS, strict=True)
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    plt.show()


def show_paired_transitions(
    paired_records: pd.DataFrame,
    paired_coverage: pd.DataFrame,
) -> None:
    """Plota matrizes de transição de classificação da aorta entre MID e HIGH.

    Args:
        paired_records (pd.DataFrame): Registros pareados por exame entre MID e HIGH.
        paired_coverage (pd.DataFrame): Cobertura e IDs exclusivos por resolução.
    """
    # Filtra apenas exames que possuem diagnóstico disponível nas duas resoluções.
    available = (
        paired_records.loc[
            paired_records["diagnostic_mid_available"].eq(True)
            & paired_records["diagnostic_high_available"].eq(True)
        ]
        if not paired_records.empty
        else paired_records
    )
    if available.empty:
        fig, ax = plt.subplots(figsize=(9, 2.5), layout="constrained")
        ax.text(
            0.5,
            0.5,
            "Sem diagnósticos mid/high pareados nas fontes disponíveis",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        plt.show()
        return

    # Organiza os subpainéis por banco e subconjunto com dados pareados.
    groups = list(available.groupby(["dataset", "subset"], sort=False))
    columns = min(3, len(groups))
    rows = (len(groups) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(6 * columns, 5 * rows),
        squeeze=False,
        layout="constrained",
    )
    image: Any = None
    for ax, ((dataset, subset), group) in zip(
        axes.ravel()[: len(groups)], groups, strict=True
    ):
        counts = pd.crosstab(group["feedback_mid"], group["feedback_high"]).reindex(
            index=AORTA_FEEDBACK_ORDER, columns=AORTA_FEEDBACK_ORDER, fill_value=0
        )
        percentages = counts / len(group) * 100
        image = ax.imshow(percentages, vmin=0, vmax=100, cmap="Blues")
        for y in range(len(counts)):
            for x in range(len(counts.columns)):
                value = float(percentages.iloc[y, x])
                ax.text(
                    x,
                    y,
                    f"{value:.1f}%" if value else "—",
                    ha="center",
                    va="center",
                    color="white" if value > 50 else "#333333",
                    fontsize=8,
                )
        info = paired_coverage.loc[
            paired_coverage["dataset"].eq(dataset)
            & paired_coverage["subset"].eq(subset)
        ].iloc[0]
        ax.set(
            xticks=range(5),
            xticklabels=SHORT_LABELS,
            yticks=range(5),
            yticklabels=SHORT_LABELS,
            xlabel="Classificação HIGH",
            ylabel="Classificação MID",
            title=(
                f"{dataset} / {SUBSETS[str(subset)]}\n"
                f"N={len(group)} pares; exclusivos mid/high: "
                f"{len(info['mid_only_ids'])}/{len(info['high_only_ids'])}"
            ),
        )
        ax.tick_params(axis="x", labelrotation=65)
        ax.grid(False)

    # Oculta eixos sobressalentes e adiciona barra de cores percentual.
    for ax in axes.ravel()[len(groups) :]:
        ax.set_axis_off()
    if image is not None:
        fig.colorbar(
            image,
            ax=axes.ravel()[: len(groups)].tolist(),
            label="Exames pareados (%)",
            shrink=0.7,
        )
    plt.show()


def show_absolute_relative_table(
    selected_summary: pd.DataFrame,
    selected_coverage: pd.DataFrame,
) -> None:
    """Exibe tabela consolidada com contagens absolutas e percentuais de feedback.

    Args:
        selected_summary (pd.DataFrame): Resumo por classe de feedback da aorta.
        selected_coverage (pd.DataFrame): Metadados de cobertura das coortes.
    """
    rows = []
    # Monta linhas combinando cobertura e distribuição por categoria de feedback.
    for resolution in ("mid", "high"):
        for subset, subset_label in SUBSETS.items():
            for dataset in [*DATASETS, "Geral"]:
                group = cohort_rows(selected_summary, dataset, resolution, subset)
                source = cohort_rows(
                    selected_coverage, dataset, resolution, subset
                ).iloc[0]
                label = (
                    "Geral parcial"
                    if dataset == "Geral" and source["status"] == "Parcial"
                    else dataset
                )
                status = source["status"]
                if dataset == "Geral" and source.get("included_datasets", ""):
                    status += "; inclui: " + source["included_datasets"]
                row: dict[str, Any] = {
                    "Resolução": resolution.upper(),
                    "Subconjunto": subset_label,
                    "Banco": label,
                    "N": source["exam_count"],
                    "N esperado": source["expected_count"],
                    "Cobertura": status,
                }
                for category in AORTA_FEEDBACK_ORDER:
                    value = group.loc[group["status"].eq(category)]
                    row[category] = (
                        f"{int(value['count'].iloc[0])} ({value['percent'].iloc[0]:.2f}%)".replace(
                            ".", ","
                        )
                        if not value.empty
                        else "—"
                    )
                rows.append(row)

    # Formata contagens inteiras e exibe a tabela completa sem truncar linhas.
    table = pd.DataFrame(rows).set_index(["Resolução", "Subconjunto", "Banco"])
    for column in ("N", "N esperado"):
        table[column] = table[column].map(
            lambda value: str(int(value)) if pd.notna(value) else "—"
        )
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        display(table)


def count_percent(count: Any, percent: Any) -> str:
    """Formata par ``contagem (percentual%)`` com vírgula decimal ou traço quando ausente.

    Args:
        count (Any): Contagem absoluta de exames.
        percent (Any): Valor percentual correspondente.

    Returns:
        str: Texto formatado para exibição tabular.
    """
    return (
        f"{int(count)} ({percent:.1f}%)".replace(".", ",") if pd.notna(percent) else "—"
    )


def show_imagecas_ostia_comparison(ostia_summary: pd.DataFrame) -> None:
    """Plota e tabula o desfecho dos óstios por grupo de qualidade da aorta no ImageCAS.

    Args:
        ostia_summary (pd.DataFrame): Resumo de desfechos dos óstios por grupo da aorta.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), layout="constrained")

    # Plota barras horizontais empilhadas de desfechos dos óstios por resolução e split.
    for i, resolution in enumerate(("mid", "high")):
        for j, (subset, label) in enumerate(SUBSETS.items()):
            ax = axes[i, j]
            cohort = cohort_rows(ostia_summary, "ImageCAS", resolution, subset)
            labels = []
            for y, category in enumerate(IMAGECAS_AORTA_GROUPS):
                group = cohort.loc[cohort["aorta_group"].eq(category)]
                total = int(group["total"].iloc[0]) if not group.empty else 0
                labels.append(
                    f"{category} (N={total})" if not group.empty else category
                )
                if not total:
                    ax.text(
                        50,
                        y,
                        "Sem casos" if not cohort.empty else "Fonte ausente",
                        ha="center",
                        va="center",
                        color="#666666",
                    )
                    continue
                row = group.iloc[0]
                left = 0.0
                for status, color in zip(OSTIA_OUTCOMES, OSTIA_COLORS, strict=True):
                    percent = float(row[f"{status}_percent"])
                    ax.barh(y, percent, left=left, height=0.55, color=color)
                    if percent >= 9:
                        ax.text(
                            left + percent / 2,
                            y,
                            f"{percent:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=9,
                        )
                    left += percent
            ax.set(
                yticks=range(3),
                yticklabels=labels,
                ylim=(2.6, -0.6),
                xlim=(0, 100),
                xlabel="Exames do grupo (%)",
                title=f"{label} — {resolution.upper()}",
            )
            ax.grid(axis="y", visible=False)

    fig.suptitle(
        "ImageCAS: qualidade automática da aorta × resultado dos óstios", fontsize=15
    )
    fig.legend(
        handles=[
            Patch(facecolor=color, label=label)
            for label, color in zip(OSTIA_OUTCOMES.values(), OSTIA_COLORS, strict=True)
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    plt.show()

    # Constrói e exibe tabela detalhada de contagens e percentuais por grupo.
    rows = []
    for _, row in ostia_summary.iterrows():
        item: dict[str, Any] = {
            "Resolução": str(row["resolution"]).upper(),
            "Subconjunto": SUBSETS[str(row["subset"])],
            "Aorta": row["aorta_group"],
            "N": int(row["total"]),
        }
        for status, label in OSTIA_OUTCOMES.items():
            item[label] = count_percent(row[status], row[f"{status}_percent"])
        item["Total aceito"] = count_percent(row["accepted"], row["accepted_percent"])
        rows.append(item)
    if rows:
        table = pd.DataFrame(rows).set_index(["Resolução", "Subconjunto", "Aorta"])
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            display(table)


def show_mmwhs_dice_comparison(
    cases: pd.DataFrame,
    dice_summary: pd.DataFrame,
    dice_coverage: pd.DataFrame,
    source_coverage: pd.DataFrame,
    threshold: float,
) -> None:
    """Exibe boxplots e tabela relacionando feedback automático e Dice da aorta no MM-WHS.

    Args:
        cases (pd.DataFrame): Casos individuais com elegibilidade e Dice da aorta.
        dice_summary (pd.DataFrame): Resumo estatístico do Dice por classe de feedback.
        dice_coverage (pd.DataFrame): Cobertura e protocolo da avaliação de Dice.
        source_coverage (pd.DataFrame): Cobertura das fontes primárias do MM-WHS.
        threshold (float): Limiar exploratório de Dice destacado nos gráficos e na tabela.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")

    # Plota distribuição de Dice por categoria de diagnóstico em cada resolução e split.
    for i, resolution in enumerate(("mid", "high")):
        for j, subset in enumerate(("train", "test")):
            ax = axes[i, j]
            cohort = cohort_rows(cases, "MM-WHS", resolution, subset)
            info = cohort_rows(dice_coverage, "MM-WHS", resolution, subset)
            source = cohort_rows(source_coverage, "MM-WHS", resolution, subset)
            total = len(cohort)
            valid = cohort.loc[cohort["dice_eligible"]]
            ax.set(
                title=(
                    f"MM-WHS / {SUBSETS[subset]} — {resolution.upper()}\n"
                    f"Dice incluído: {len(valid)}/{total}"
                    if not cohort.empty
                    else "Fonte ausente"
                ),
                ylim=(-0.03, 1.03),
                ylabel="Dice da aorta",
            )
            if valid.empty:
                message = (
                    info["status"].iloc[0]
                    if not info.empty
                    else source["status"].iloc[0]
                    if not source.empty
                    else "Fonte ausente"
                )
                ax.text(
                    0.5,
                    0.5,
                    str(message).replace(": ", ":\n"),
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    wrap=True,
                )
                ax.set_xticks([])
                continue
            categories = [
                category
                for category in [*AORTA_FEEDBACK_ORDER, "Diagnóstico não persistido"]
                if cohort["feedback"].eq(category).any()
            ]
            labels = []
            for x, category in enumerate(categories):
                group = valid.loc[valid["feedback"].eq(category)].sort_values(
                    "image_id"
                )
                values = group["aorta_dice"].to_numpy(dtype=float)
                labels.append(
                    f"{category.replace('Suspeita de ', '')}\nN={len(values)}"
                )
                if not len(values):
                    continue
                ax.boxplot(
                    [values],
                    positions=[x],
                    widths=0.42,
                    showfliers=False,
                    patch_artist=True,
                    boxprops={"facecolor": "#DBEAFE", "edgecolor": "#4C78A8"},
                    medianprops={"color": "#172554"},
                )
                # Aplica deslocamento horizontal determinístico para separar pontos individuais.
                offsets = (
                    np.linspace(-0.15, 0.15, len(values))
                    if len(values) > 1
                    else np.array([0.0])
                )
                ax.scatter(
                    x + np.asarray(offsets),
                    values,
                    color="#4C78A8",
                    s=25,
                    alpha=0.8,
                    zorder=3,
                )
            ax.axhline(
                threshold,
                color="#DA4A49",
                linestyle="--",
                label=f"Corte exploratório: {threshold:.2f}",
            )
            ax.set(
                xticks=range(len(categories)),
                xticklabels=labels,
                xlim=(-0.6, len(categories) - 0.4),
            )
            ax.tick_params(axis="x", labelsize=9)
            ax.legend(loc="lower right", fontsize=9, frameon=False)
            protocol = info["protocol"].iloc[0]
            ax.set_xlabel(protocol)

    fig.suptitle("MM-WHS: diagnóstico automático × Dice da aorta", fontsize=15)
    plt.show()

    # Consolida tabela com medianas e contagens abaixo/acima do corte exploratório.
    rows = []
    for resolution in ("mid", "high"):
        for subset in ("train", "test"):
            info = cohort_rows(dice_coverage, "MM-WHS", resolution, subset)
            source = cohort_rows(source_coverage, "MM-WHS", resolution, subset)
            summary = cohort_rows(dice_summary, "MM-WHS", resolution, subset)
            status = (
                info["status"].iloc[0]
                if not info.empty
                else source["status"].iloc[0]
                if not source.empty
                else "Fonte ausente"
            )
            protocol = info["protocol"].iloc[0] if not info.empty else "—"
            if summary.empty:
                rows.append(
                    {
                        "Resolução": resolution.upper(),
                        "Subconjunto": SUBSETS[subset],
                        "Diagnóstico": "—",
                        "Cobertura": status,
                        "Protocolo": protocol,
                        "N": pd.NA,
                        "N avaliado": pd.NA,
                        "Excluídos": pd.NA,
                        "Mediana Dice": "—",
                        f"Dice < {threshold:.2f}": "—",
                        f"Dice ≥ {threshold:.2f}": "—",
                    }
                )
            for _, row in summary.iterrows():
                rows.append(
                    {
                        "Resolução": resolution.upper(),
                        "Subconjunto": SUBSETS[subset],
                        "Diagnóstico": row["feedback"],
                        "Cobertura": status,
                        "Protocolo": protocol,
                        "N": int(row["exam_count"]),
                        "N avaliado": int(row["evaluated_count"]),
                        "Excluídos": int(row["excluded_count"]),
                        "Mediana Dice": (
                            f"{row['median_dice']:.3f}".replace(".", ",")
                            if pd.notna(row["median_dice"])
                            else "—"
                        ),
                        f"Dice < {threshold:.2f}": count_percent(
                            row["below_threshold"], row["below_percent"]
                        ),
                        f"Dice ≥ {threshold:.2f}": count_percent(
                            row["above_threshold"], row["above_percent"]
                        ),
                    }
                )
    table = pd.DataFrame(rows).set_index(["Resolução", "Subconjunto", "Diagnóstico"])
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        display(table)
