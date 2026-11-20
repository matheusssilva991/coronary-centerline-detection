"""Carrega, pareia e resume variantes sem dependências de apresentação."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

SUCCESS_LABELS = {
    "both_correct",
    "both_tolerable",
}

CORRECT_LABELS = {"both_correct"}

TOLERABLE_LABELS = {"both_tolerable"}

WRONG_LABELS = {"found_but_wrong"}

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


def yes_no_to_bool(series: pd.Series) -> pd.Series:
    """Converte valores textuais comuns para booleanos.

    Args:
        series (pd.Series): Série contendo valores booleanos ou textuais de
            sim/não.

    Returns:
        pd.Series: Série booleana resultante da normalização textual.
    """
    return series.astype(str).str.lower().isin(["yes", "true", "1", "sim"])


def order_variants(
    df: pd.DataFrame,
    preferred_order: Sequence[str] | None = None,
    variant_column: str = "folder_variant",
) -> pd.DataFrame:
    """Ordena um DataFrame usando uma ordem preferida de variantes.

    Args:
        df (pd.DataFrame): Tabela contendo a coluna identificadora das variantes.
        preferred_order (Sequence[str] | None): Sequência explícita de nomes de
            variantes; quando ``None``, ordena alfabeticamente.
        variant_column (str): Nome da coluna usada na ordenação.

    Returns:
        pd.DataFrame: Cópia ordenada das linhas com índice reiniciado.
    """
    ordered = df.copy()
    if not preferred_order:
        return ordered.sort_values(variant_column).reset_index(drop=True)

    order_map = {name: idx for idx, name in enumerate(preferred_order)}
    ordered["_variant_order"] = ordered[variant_column].map(order_map)
    ordered["_variant_order"] = ordered["_variant_order"].fillna(len(order_map))
    ordered = ordered.sort_values(["_variant_order", variant_column])
    return ordered.drop(columns="_variant_order").reset_index(drop=True)


def build_dice_stats_by_variant(
    results_df: pd.DataFrame,
    preferred_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Calcula média, máximo, mínimo, desvio padrão e mediana do Dice.

    Args:
        results_df (pd.DataFrame): Resultados por exame com ``folder_variant``,
            ``variant_label`` e ``artery_dice``.
        preferred_order (Sequence[str] | None): Ordem preferida opcional para
            as variantes na saída.

    Returns:
        pd.DataFrame: Resumo agregado do Dice arterial por variante.
    """
    stats = results_df.groupby(["folder_variant", "variant_label"], as_index=False)[
        "artery_dice"
    ].agg(
        mean_dice="mean",
        max_dice="max",
        min_dice="min",
        std_dice="std",
        median_dice="median",
    )
    return order_variants(stats, preferred_order)


def make_pair_delta(
    results_df: pd.DataFrame,
    reference_variant: str,
    comparison_variant: str,
) -> pd.DataFrame:
    """Compara Dice por imagem entre duas variantes.

    Args:
        results_df (pd.DataFrame): Tabela consolidada de resultados de múltiplas
            variantes por exame.
        reference_variant (str): Nome da variante de referência.
        comparison_variant (str): Nome da variante avaliada contra a referência.

    Returns:
        pd.DataFrame: Tabela pareada por ``IMG_ID`` contendo deltas de Dice,
        indicadores booleanos dos óstios e metadados das variantes.
    """
    # Garante presença de colunas opcionais de óstios e contagem de voxels.
    df = results_df.copy()
    if "artery_voxel_count" not in df.columns:
        df["artery_voxel_count"] = df.get("artery_voxels", np.nan)
    if "ostia_detection_status" not in df.columns:
        df["ostia_detection_status"] = ""
    for column in ("left_ostium", "right_ostium"):
        if column not in df.columns:
            df[column] = ""
    for column in (
        "both_ostia_correct",
        "both_ostia_tolerable",
        "left_ostium_correct",
        "right_ostium_correct",
    ):
        if column not in df.columns:
            df[column] = False

    base_columns = [
        "IMG_ID",
        "artery_dice",
        "ostia_detection_status",
        "artery_voxel_count",
        "left_ostium",
        "right_ostium",
        "both_ostia_correct",
        "both_ostia_tolerable",
        "left_ostium_correct",
        "right_ostium_correct",
    ]
    # Renomeia os lados antes do merge para preservar ambas as medições.
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
            "both_ostia_correct": "reference_both_ostia_correct",
            "both_ostia_tolerable": "reference_both_ostia_tolerable",
            "left_ostium_correct": "reference_left_ostium_correct",
            "right_ostium_correct": "reference_right_ostium_correct",
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
            "both_ostia_correct": "comparison_both_ostia_correct",
            "both_ostia_tolerable": "comparison_both_ostia_tolerable",
            "left_ostium_correct": "comparison_left_ostium_correct",
            "right_ostium_correct": "comparison_right_ostium_correct",
        }
    )
    # Pareia variantes por IMG_ID e deriva deltas de Dice e flags dos óstios.
    pair_df = reference.merge(comparison, on="IMG_ID", how="inner")
    pair_df["dice_delta"] = pair_df["comparison_dice"] - pair_df["reference_dice"]
    pair_df["abs_delta"] = pair_df["dice_delta"].abs()
    for prefix in ("reference", "comparison"):
        for column in (
            "both_ostia_correct",
            "both_ostia_tolerable",
            "left_ostium_correct",
            "right_ostium_correct",
        ):
            full_column = f"{prefix}_{column}"
            pair_df[f"{full_column}_bool"] = yes_no_to_bool(pair_df[full_column])

    pair_df["reference_one_ostium_correct"] = (
        pair_df["reference_left_ostium_correct_bool"]
        ^ pair_df["reference_right_ostium_correct_bool"]
    )
    pair_df["comparison_ostia_success"] = (
        pair_df["comparison_both_ostia_correct_bool"]
        | pair_df["comparison_both_ostia_tolerable_bool"]
    )
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


def largest_pair_changes(
    results_df: pd.DataFrame,
    reference_variant: str,
    comparison_variant: str,
    top_n: int = 15,
) -> pd.DataFrame:
    """Retorna imagens com maior variação absoluta de Dice entre duas variantes.

    Args:
        results_df (pd.DataFrame): Tabela consolidada de resultados por variante.
        reference_variant (str): Nome da variante de referência.
        comparison_variant (str): Nome da variante comparada.
        top_n (int): Quantidade máxima de casos retornados (padrão ``15``).

    Returns:
        pd.DataFrame: Subconjunto ordenado decrescentemente por ``abs_delta``
        contendo métricas de Dice, óstios e contagem de voxels do par.
    """
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
    """Resume o ganho/perda de Dice de várias variantes contra uma referência.

    Args:
        results_df (pd.DataFrame): Tabela consolidada de resultados por variante.
        reference_variant (str): Nome da variante usada como baseline.
        variants (Iterable[str] | None): Coleção opcional de variantes a avaliar;
            quando ``None``, compara todas as variantes disponíveis.
        pretty_names (dict[str, str] | None): Mapa opcional de nomes técnicos
            para rótulos legíveis de exibição.

    Returns:
        pd.DataFrame: Resumo ordenado por ``mean_delta`` decrescente com média,
        mediana, extremos e contagens de ganho/perda relevantes (>= 0,02).
    """
    # Resolve o conjunto de variantes disponíveis a comparar contra a referência.
    names = pretty_names or {}
    available = list(dict.fromkeys(results_df["folder_variant"].astype(str)))
    variants_to_compare = list(variants) if variants is not None else available
    rows = []

    # Calcula pareamento e estatísticas de variação de Dice para cada variante.
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
                "variant_better_by_0_02_n": int((pair_df["dice_delta"] >= 0.02).sum()),
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


__all__ = [
    "CORRECT_LABELS",
    "OSTIA_STATUS_COLUMNS",
    "OSTIA_STATUS_GROUP_LABELS",
    "SUCCESS_LABELS",
    "TOLERABLE_LABELS",
    "WRONG_LABELS",
    "build_delta_summary_vs_reference",
    "build_dice_stats_by_variant",
    "largest_pair_changes",
    "make_pair_delta",
    "order_variants",
    "yes_no_to_bool",
]
