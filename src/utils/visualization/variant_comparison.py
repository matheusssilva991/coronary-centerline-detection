"""Análise e visualização de variantes do pipeline de segmentação.

As funções deste módulo são genéricas para qualquer comparação organizada como:

```
<result_root>/<variant>/<timestamp>/numeric/ostios_<split>_summary.csv
```

Elas foram extraídas do notebook de comparação fuzzy para reutilizar o resumo
de Dice, status dos óstios e deltas entre variantes em outros experimentos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUCCESS_LABELS = {
    "both correct",
    "both tolerable",
    "both ostia correct",
    "both ostia tolerable",
}
CORRECT_LABELS = {"both correct", "both ostia correct"}
TOLERABLE_LABELS = {"both tolerable", "both ostia tolerable"}
WRONG_LABELS = {"found but incorrect", "found_but_wrong"}
OSTIA_STATUS_GROUP_LABELS = {
    "success": "success",
    "wrong": "wrong",
    "not_found_or_error": "not_found_or_error",
}

OSTIA_STATUS_COLUMNS = [
    "both_correct_n",
    "both_tolerable_n",
    "found_wrong_n",
    "not_found_or_error_n",
]
OSTIA_STATUS_LABELS = [
    "Both correct",
    "Both tolerable",
    "Found but wrong",
    "Not found/error",
]
OSTIA_STATUS_COLORS = ["#2ca02c", "#8fd175", "#ff9f1a", "#d62728"]


def yes_no_to_bool(series: pd.Series) -> pd.Series:
    """Converte valores textuais comuns para booleanos."""
    return series.astype(str).str.lower().isin(["yes", "true", "1", "sim"])


def first_existing_value(
    df: pd.DataFrame,
    columns: Sequence[str],
    default: Any = "",
) -> Any:
    """Retorna o primeiro valor válido dentre uma lista de colunas possíveis."""
    for column in columns:
        if column in df.columns and not df[column].empty:
            value = df[column].iloc[0]
            if pd.notna(value):
                return value
    return default


def normalize_ostia_status_group(status: object) -> str:
    """Agrupa status de óstios para comparações qualitativas.

    ``both correct`` e ``both tolerable`` são tratados como o mesmo grupo de
    sucesso, porque ambos indicam óstios aceitáveis para a análise.
    """
    status_text = str(status).strip().lower()
    if status_text in SUCCESS_LABELS:
        return "success"
    if status_text in WRONG_LABELS:
        return "wrong"
    return "not_found_or_error"


def order_variants(
    df: pd.DataFrame,
    preferred_order: Sequence[str] | None = None,
    variant_column: str = "folder_variant",
) -> pd.DataFrame:
    """Ordena um DataFrame usando uma ordem preferida de variantes."""
    ordered = df.copy()
    if not preferred_order:
        return ordered.sort_values(variant_column).reset_index(drop=True)

    order_map = {name: idx for idx, name in enumerate(preferred_order)}
    ordered["_variant_order"] = ordered[variant_column].map(order_map)
    ordered["_variant_order"] = ordered["_variant_order"].fillna(len(order_map))
    ordered = ordered.sort_values(["_variant_order", variant_column])
    return ordered.drop(columns="_variant_order").reset_index(drop=True)


def add_variant_labels(
    df: pd.DataFrame,
    pretty_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Adiciona/atualiza a coluna legível ``variant_label``."""
    out = df.copy()
    names = pretty_names or {}
    out["variant_label"] = out["folder_variant"].map(names).fillna(out["folder_variant"])
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
        labels = [pretty_names.get(name, name) if pretty_names else name for name in preferred_order]
        out["variant_label"] = pd.Categorical(
            out["variant_label"],
            categories=labels,
            ordered=True,
        )
        out = out.sort_values("variant_label")
    return out.reset_index(drop=True)


def _safe_relative_path(path: Path, root: Path | None) -> str:
    """Retorna caminho relativo quando possível."""
    if root is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def load_variant_run(
    summary_path: Path,
    *,
    result_root: Path | None = None,
    repo_root: Path | None = None,
    pretty_names: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Carrega um CSV consolidado de uma variante e gera seu resumo."""
    summary_path = Path(summary_path)
    variant = summary_path.parents[2].name
    run_dir = summary_path.parents[1]
    names = pretty_names or {}

    df = pd.read_csv(summary_path)
    df["folder_variant"] = variant
    df["variant_label"] = names.get(variant, variant)
    df["run_timestamp"] = run_dir.name
    df["run_dir"] = _safe_relative_path(run_dir, repo_root)

    if "artery_dice" not in df.columns and "dice_artery" in df.columns:
        df["artery_dice"] = df["dice_artery"]
    df["artery_dice"] = pd.to_numeric(df.get("artery_dice"), errors="coerce")

    if "ostia_detected" in df.columns:
        df["ostia_detected_bool"] = yes_no_to_bool(df["ostia_detected"])
    else:
        df["ostia_detected_bool"] = False

    status = df.get("ostia_detection_status", pd.Series(index=df.index, dtype=str))
    status = status.astype(str)
    df["ostia_success"] = status.isin(SUCCESS_LABELS)
    df["both_correct"] = status.isin(CORRECT_LABELS)
    df["both_tolerable"] = status.isin(TOLERABLE_LABELS)
    df["found_wrong"] = status.isin(WRONG_LABELS)

    run_dir_value = _safe_relative_path(run_dir, repo_root)
    summary = {
        "folder_variant": variant,
        "variant_label": names.get(variant, variant),
        "run_timestamp": run_dir.name,
        "run_dir": run_dir_value,
        "n_images": len(df),
        "threshold_mode": first_existing_value(df, ["threshold_mode"], "normal"),
        "artery_method": first_existing_value(
            df,
            ["configured_artery_segmentation_method", "artery_segmentation_method"],
            "",
        ),
        "ostia_detected_rate": df["ostia_detected_bool"].mean(),
        "ostia_success_rate": df["ostia_success"].mean(),
        "both_correct_n": int(df["both_correct"].sum()),
        "both_tolerable_n": int(df["both_tolerable"].sum()),
        "found_wrong_n": int(df["found_wrong"].sum()),
        "not_found_or_error_n": int((~(df["ostia_success"] | df["found_wrong"])).sum()),
        "mean_dice": df["artery_dice"].mean(),
        "median_dice": df["artery_dice"].median(),
        "std_dice": df["artery_dice"].std(),
        "mean_dice_success_ostia": df.loc[df["ostia_success"], "artery_dice"].mean(),
    }
    if result_root is not None:
        summary["result_root"] = str(result_root)
    return df, summary


def load_variant_results(
    result_root: Path,
    *,
    split: str = "test",
    preferred_order: Sequence[str] | None = None,
    pretty_names: dict[str, str] | None = None,
    repo_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega todas as variantes encontradas em ``result_root`` para um split."""
    result_root = Path(result_root)
    all_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    pattern = f"*/*/numeric/ostios_{split}_summary.csv"

    allowed = set(preferred_order or [])
    for summary_path in sorted(result_root.glob(pattern)):
        variant = summary_path.parents[2].name
        if allowed and variant not in allowed:
            continue
        df_variant, summary = load_variant_run(
            summary_path,
            result_root=result_root,
            repo_root=repo_root,
            pretty_names=pretty_names,
        )
        all_frames.append(df_variant)
        summary_rows.append(summary)

    if not all_frames:
        raise FileNotFoundError(
            f"Nenhum ostios_{split}_summary.csv encontrado em {result_root}"
        )

    results_df = pd.concat(all_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df = order_variants(summary_df, preferred_order)
    return results_df, summary_df


def build_ranking_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Ordena variantes por sucesso dos óstios e Dice."""
    ranking = summary_df.sort_values(
        ["ostia_success_rate", "mean_dice", "mean_dice_success_ostia"],
        ascending=False,
    ).copy()
    columns = [
        "variant_label",
        "threshold_mode",
        "artery_method",
        "n_images",
        "ostia_detected_rate",
        "ostia_success_rate",
        "both_correct_n",
        "both_tolerable_n",
        "found_wrong_n",
        "not_found_or_error_n",
        "mean_dice",
        "median_dice",
        "mean_dice_success_ostia",
    ]
    return ranking[[column for column in columns if column in ranking.columns]].copy()


def build_dice_stats_by_variant(
    results_df: pd.DataFrame,
    preferred_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Calcula média, máximo, mínimo, desvio padrão e mediana do Dice."""
    stats = (
        results_df.groupby(["folder_variant", "variant_label"], as_index=False)["artery_dice"]
        .agg(
            mean_dice="mean",
            max_dice="max",
            min_dice="min",
            std_dice="std",
            median_dice="median",
        )
    )
    return order_variants(stats, preferred_order)


def make_pair_delta(
    results_df: pd.DataFrame,
    reference_variant: str,
    comparison_variant: str,
) -> pd.DataFrame:
    """Compara Dice por imagem entre duas variantes."""
    df = results_df.copy()
    if "artery_voxel_count" not in df.columns:
        df["artery_voxel_count"] = df.get("artery_voxels", np.nan)
    if "ostia_detection_status" not in df.columns:
        df["ostia_detection_status"] = ""
    for column in ("left_ostium", "right_ostium"):
        if column not in df.columns:
            df[column] = ""

    base_columns = [
        "IMG_ID",
        "artery_dice",
        "ostia_detection_status",
        "artery_voxel_count",
        "left_ostium",
        "right_ostium",
    ]
    reference = df.loc[
        df["folder_variant"] == reference_variant,
        base_columns,
    ].rename(
        columns={
            "artery_dice": "reference_dice",
            "ostia_detection_status": "reference_ostia_status",
            "artery_voxel_count": "reference_artery_voxels",
            "left_ostium": "reference_left_ostium",
            "right_ostium": "reference_right_ostium",
        }
    )
    comparison = df.loc[
        df["folder_variant"] == comparison_variant,
        base_columns,
    ].rename(
        columns={
            "artery_dice": "comparison_dice",
            "ostia_detection_status": "comparison_ostia_status",
            "artery_voxel_count": "comparison_artery_voxels",
            "left_ostium": "comparison_left_ostium",
            "right_ostium": "comparison_right_ostium",
        }
    )
    pair_df = reference.merge(comparison, on="IMG_ID", how="inner")
    pair_df["dice_delta"] = pair_df["comparison_dice"] - pair_df["reference_dice"]
    pair_df["abs_delta"] = pair_df["dice_delta"].abs()
    pair_df["same_ostia_points"] = (
        pair_df["reference_left_ostium"].astype(str)
        == pair_df["comparison_left_ostium"].astype(str)
    ) & (
        pair_df["reference_right_ostium"].astype(str)
        == pair_df["comparison_right_ostium"].astype(str)
    )
    pair_df["reference_variant"] = reference_variant
    pair_df["comparison_variant"] = comparison_variant
    return pair_df


def add_pair_ostia_status_groups(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona grupos de status dos óstios a uma comparação par-a-par."""
    out = pair_df.copy()
    out["reference_status_group"] = out["reference_ostia_status"].map(
        normalize_ostia_status_group
    )
    out["comparison_status_group"] = out["comparison_ostia_status"].map(
        normalize_ostia_status_group
    )
    out["same_ostia_status_group"] = (
        out["reference_status_group"] == out["comparison_status_group"]
    )
    return out


def pair_summary(
    results_df: pd.DataFrame,
    reference_variant: str,
    comparison_variant: str,
    pretty_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Resume o delta de Dice entre duas variantes."""
    pair_df = make_pair_delta(results_df, reference_variant, comparison_variant)
    names = pretty_names or {}
    min_delta_label = "0_02"
    return pd.DataFrame(
        [
            {
                "reference": names.get(reference_variant, reference_variant),
                "comparison": names.get(comparison_variant, comparison_variant),
                "n_images": len(pair_df),
                "mean_delta": pair_df["dice_delta"].mean(),
                "median_delta": pair_df["dice_delta"].median(),
                f"comparison_better_by_{min_delta_label}_n": int(
                    (pair_df["dice_delta"] >= 0.02).sum()
                ),
                f"reference_better_by_{min_delta_label}_n": int(
                    (pair_df["dice_delta"] <= -0.02).sum()
                ),
                "max_gain": pair_df["dice_delta"].max(),
                "max_loss": pair_df["dice_delta"].min(),
            }
        ]
    )


def build_pair_outcome_counts(
    results_df: pd.DataFrame,
    pair_comparisons: Iterable[Sequence[str]],
    *,
    pretty_names: dict[str, str] | None = None,
    min_delta: float = 0.02,
) -> pd.DataFrame:
    """Conta quantos exames melhoraram, pioraram ou ficaram estáveis por par.

    ``dice_delta`` é calculado como ``comparison - reference``. Portanto,
    valores positivos favorecem a variante de comparação e valores negativos
    favorecem a variante de referência.
    """
    names = pretty_names or {}
    rows: list[dict[str, Any]] = []
    min_delta_label = str(min_delta).replace(".", "_")

    for pair in pair_comparisons:
        reference_variant, comparison_variant = pair[:2]
        comparison_name = pair[2] if len(pair) > 2 else None
        pair_df = make_pair_delta(results_df, reference_variant, comparison_variant)
        valid_delta = pair_df["dice_delta"].dropna()
        rows.append(
            {
                "comparison_name": comparison_name
                or f"{names.get(comparison_variant, comparison_variant)} vs {names.get(reference_variant, reference_variant)}",
                "reference_variant": reference_variant,
                "comparison_variant": comparison_variant,
                "reference_label": names.get(reference_variant, reference_variant),
                "comparison_label": names.get(comparison_variant, comparison_variant),
                "n_images": int(valid_delta.shape[0]),
                "comparison_better_n": int((valid_delta > 0).sum()),
                "reference_better_n": int((valid_delta < 0).sum()),
                "same_dice_n": int((valid_delta == 0).sum()),
                f"comparison_better_by_{min_delta_label}_n": int(
                    (valid_delta >= min_delta).sum()
                ),
                f"reference_better_by_{min_delta_label}_n": int(
                    (valid_delta <= -min_delta).sum()
                ),
                "mean_delta": valid_delta.mean(),
                "median_delta": valid_delta.median(),
                "max_gain": valid_delta.max(),
                "max_loss": valid_delta.min(),
            }
        )

    return pd.DataFrame(rows)


def select_qualitative_pair_cases(
    results_df: pd.DataFrame,
    reference_variant: str,
    comparison_variant: str,
    *,
    min_dice: float = 0.02,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Seleciona casos qualitativos para comparar duas variantes.

    Retorna casos com status de óstios equivalente, status de óstios diferente
    e Dice próximo das médias de cada variante.
    """
    pair_df = add_pair_ostia_status_groups(
        make_pair_delta(results_df, reference_variant, comparison_variant)
    )
    valid_mask = (
        (pair_df["reference_dice"].fillna(0) > min_dice)
        & (pair_df["comparison_dice"].fillna(0) > min_dice)
    )

    reference_mean_dice = results_df.loc[
        results_df["folder_variant"] == reference_variant, "artery_dice"
    ].mean()
    comparison_mean_dice = results_df.loc[
        results_df["folder_variant"] == comparison_variant, "artery_dice"
    ].mean()

    same_candidates = (
        pair_df.loc[valid_mask & pair_df["same_ostia_status_group"]]
        .sort_values("abs_delta", ascending=False)
    )
    different_candidates = (
        pair_df.loc[valid_mask & ~pair_df["same_ostia_status_group"]]
        .sort_values("abs_delta", ascending=False)
    )
    near_mean_candidates = (
        pair_df.loc[valid_mask]
        .assign(
            distance_to_variant_means=lambda df: (
                (df["reference_dice"] - reference_mean_dice).abs()
                + (df["comparison_dice"] - comparison_mean_dice).abs()
            )
        )
        .sort_values("distance_to_variant_means")
    )

    def pick_case(
        case_key: str,
        label: str,
        candidates: pd.DataFrame,
    ) -> dict[str, Any]:
        if candidates.empty:
            candidates = near_mean_candidates
        if candidates.empty:
            raise ValueError("Não há casos válidos para a seleção qualitativa.")
        row = candidates.iloc[0].to_dict()
        row["case_key"] = case_key
        row["case_label"] = label
        row["reference_mean_dice"] = reference_mean_dice
        row["comparison_mean_dice"] = comparison_mean_dice
        row["reference_distance_to_mean"] = abs(
            row["reference_dice"] - reference_mean_dice
        )
        row["comparison_distance_to_mean"] = abs(
            row["comparison_dice"] - comparison_mean_dice
        )
        return row

    cases = {
        "same_ostia": pick_case(
            "same_ostia",
            "Status de óstios iguais",
            same_candidates,
        ),
        "different_ostia": pick_case(
            "different_ostia",
            "Status de óstios diferentes",
            different_candidates,
        ),
        "near_mean": pick_case(
            "near_mean",
            "Dice próximo das médias das variantes",
            near_mean_candidates,
        ),
    }
    return cases, pd.DataFrame(cases.values())


def largest_pair_changes(
    results_df: pd.DataFrame,
    reference_variant: str,
    comparison_variant: str,
    top_n: int = 15,
) -> pd.DataFrame:
    """Retorna imagens com maior variação absoluta de Dice entre duas variantes."""
    pair_df = make_pair_delta(results_df, reference_variant, comparison_variant)
    columns = [
        "IMG_ID",
        "reference_dice",
        "comparison_dice",
        "dice_delta",
        "reference_ostia_status",
        "comparison_ostia_status",
        "same_ostia_points",
        "reference_left_ostium",
        "comparison_left_ostium",
        "reference_right_ostium",
        "comparison_right_ostium",
        "reference_artery_voxels",
        "comparison_artery_voxels",
    ]
    return pair_df.sort_values("abs_delta", ascending=False).head(top_n)[columns]


def build_delta_summary_vs_reference(
    results_df: pd.DataFrame,
    reference_variant: str,
    variants: Iterable[str] | None = None,
    pretty_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Resume o ganho/perda de Dice de várias variantes contra uma referência."""
    names = pretty_names or {}
    available = list(dict.fromkeys(results_df["folder_variant"].astype(str)))
    variants_to_compare = list(variants) if variants is not None else available
    rows = []

    for variant in variants_to_compare:
        if variant == reference_variant or variant not in available:
            continue
        pair_df = make_pair_delta(results_df, reference_variant, variant)
        rows.append(
            {
                "folder_variant": variant,
                "variant_label": names.get(variant, variant),
                "mean_delta": pair_df["dice_delta"].mean(),
                "median_delta": pair_df["dice_delta"].median(),
                "variant_better_by_0_02_n": int(
                    (pair_df["dice_delta"] >= 0.02).sum()
                ),
                "reference_better_by_0_02_n": int(
                    (pair_df["dice_delta"] <= -0.02).sum()
                ),
                "max_gain": pair_df["dice_delta"].max(),
                "max_loss": pair_df["dice_delta"].min(),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("mean_delta", ascending=False)


def best_variant_by_suffix(
    ranking_df: pd.DataFrame,
    suffix: str,
    variant_column: str = "folder_variant",
) -> str:
    """Seleciona a melhor variante do ranking por sufixo de nome."""
    candidates = ranking_df[ranking_df[variant_column].astype(str).str.endswith(suffix)]
    if candidates.empty:
        raise ValueError(f"Nenhuma variante encontrada com sufixo {suffix!r}.")
    return str(candidates.iloc[0][variant_column])


def plot_ostia_status_by_variant(
    summary_df: pd.DataFrame,
    *,
    preferred_order: Sequence[str] | None = None,
    pretty_names: dict[str, str] | None = None,
    ax: Any | None = None,
    figsize: tuple[float, float] = (12, 5),
    save_path: Path | None = None,
) -> Any:
    """Plota status dos óstios por variante em barras empilhadas."""
    plot_df = prepare_variant_for_plot(summary_df, preferred_order, pretty_names)
    label_outline = [path_effects.withStroke(linewidth=3.2, foreground="white")]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    x_labels = plot_df["variant_label"].astype(str)
    bottom = np.zeros(len(plot_df))
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
    pair_df = make_pair_delta(results_df, reference_variant, comparison_variant)
    plot_df = (
        pair_df.sort_values("abs_delta", ascending=False)
        .head(top_n)
        .sort_values("dice_delta")
    )
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
    "CORRECT_LABELS",
    "OSTIA_STATUS_COLORS",
    "OSTIA_STATUS_COLUMNS",
    "OSTIA_STATUS_GROUP_LABELS",
    "OSTIA_STATUS_LABELS",
    "SUCCESS_LABELS",
    "TOLERABLE_LABELS",
    "WRONG_LABELS",
    "add_pair_ostia_status_groups",
    "add_variant_labels",
    "best_variant_by_suffix",
    "build_dice_stats_by_variant",
    "build_delta_summary_vs_reference",
    "build_pair_outcome_counts",
    "build_ranking_table",
    "first_existing_value",
    "largest_pair_changes",
    "load_variant_results",
    "load_variant_run",
    "make_pair_delta",
    "normalize_ostia_status_group",
    "order_variants",
    "pair_summary",
    "plot_largest_pair_changes",
    "plot_ostia_status_by_variant",
    "prepare_variant_for_plot",
    "select_qualitative_pair_cases",
    "yes_no_to_bool",
]
