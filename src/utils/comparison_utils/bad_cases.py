"""Filtragem de exames com detecção bem-sucedida dos óstios coronarianos."""

import pandas as pd

from utils.project.results.schema import (
    OSTIA_SUCCESS_STATUSES,
    normalize_ostia_status,
    normalize_result_status,
)


def _compute_success_mask(df, success_status=None):
    """Calcula a máscara de sucesso aceitando diferentes schemas.

    Args:
        df (pd.DataFrame): Tabela de resultados por exame em qualquer schema suportado.
        success_status (Iterable[str] | None): Lista opcional de rótulos aceitos como
            sucesso no schema legado; usa ``OSTIA_SUCCESS_STATUSES`` se ``None``.

    Returns:
        pd.Series: Série booleana indicando exames com óstios bem-sucedidos.
    """
    if "ostia_status" in df.columns:
        # Schema intermediário: status consolidado por linha.
        return (
            df["ostia_status"].map(normalize_ostia_status).isin(OSTIA_SUCCESS_STATUSES)
        )

    if {"both_correct", "both_tolerable"}.issubset(df.columns):
        # Schema novo: sucesso = correto OU tolerável.
        truthy = {"true", "1", "sim", "s", "yes", "y"}
        both_correct = df["both_correct"].map(
            lambda value: (
                value if isinstance(value, bool) else str(value).lower() in truthy
            )
        )
        both_tolerable = df["both_tolerable"].map(
            lambda value: (
                value if isinstance(value, bool) else str(value).lower() in truthy
            )
        )
        return both_correct | both_tolerable

    # Schema antigo: converte coluna textual para padrão interno.
    status_series = df.get("status", pd.Series(index=df.index, dtype="object"))
    status_english = status_series.map(normalize_result_status)
    if success_status is None:
        success_status = OSTIA_SUCCESS_STATUSES
    success_status_english = {
        normalize_result_status(status) for status in success_status
    }
    return status_english.isin(success_status_english)


def filter_correct_ostia_cases(df, success_status=None):
    """Retorna apenas casos com detecção dos óstios bem-sucedida.

    Args:
        df (pd.DataFrame | None): Tabela de resultados por exame a ser filtrada.
        success_status (Iterable[str] | None): Lista opcional de status considerados
            sucesso no schema legado; usa ``OSTIA_SUCCESS_STATUSES`` se ``None``.

    Returns:
        pd.DataFrame: Cópia filtrada contendo somente exames com óstios aceitos.
    """
    # Mantém somente linhas com sucesso de óstio.
    if df is None or df.empty:
        # Retorno vazio com mesmo schema.
        return pd.DataFrame(columns=df.columns if df is not None else None)

    # Reaproveita a mesma regra de sucesso deste módulo.
    success_mask = _compute_success_mask(df, success_status)
    return df.loc[success_mask].copy()


__all__ = ["filter_correct_ostia_cases"]
