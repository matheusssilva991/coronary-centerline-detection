"""Organiza rótulos e gráficos de comparação entre variantes."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from utils.comparison_utils.variant_comparison import order_variants

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
    """Adiciona ou atualiza a coluna legível ``variant_label`` no DataFrame.

    Args:
        df (pd.DataFrame): DataFrame contendo a coluna ``folder_variant``.
        pretty_names (dict[str, str] | None): Mapeamento opcional de nomes de
            variante para rótulos legíveis; ``None`` mantém ``folder_variant``.

    Returns:
        pd.DataFrame: Cópia de ``df`` com a coluna ``variant_label`` preenchida.
    """
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
    """Prepara rótulos categóricos e ordenação de variantes para visualização.

    Args:
        df (pd.DataFrame): DataFrame de resultados contendo ``folder_variant``.
        preferred_order (Sequence[str] | None): Ordem desejada das variantes
            originais; ``None`` ordena alfabeticamente.
        pretty_names (dict[str, str] | None): Mapeamento opcional de variantes
            para rótulos legíveis; ``None`` preserva os nomes originais.

    Returns:
        pd.DataFrame: DataFrame copiado, ordenado e com ``variant_label``
        pronto para plotagem.
    """
    out = add_variant_labels(df, pretty_names)
    out = order_variants(out, preferred_order)
    # Converte variant_label em categórica ordenada para preservar a ordem nos gráficos.
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


__all__ = [
    "OSTIA_STATUS_COLORS",
    "OSTIA_STATUS_LABELS",
    "add_variant_labels",
    "prepare_variant_for_plot",
]
