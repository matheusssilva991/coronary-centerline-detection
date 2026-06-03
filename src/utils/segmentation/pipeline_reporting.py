"""Relatórios de console para o pipeline de segmentação."""

from __future__ import annotations

import pandas as pd

from ..results_utils import _as_bool_value, _readable_column_name


def _series_from_aliases(df, column, dtype=None):
    for column_candidate in (column, _readable_column_name(column)):
        if column_candidate in df.columns:
            return df[column_candidate]
    return pd.Series(index=df.index, dtype=dtype)


def _bool_series(df, column):
    return _series_from_aliases(df, column, dtype=bool).map(_as_bool_value)


def print_statistics(train_ids, val_ids, test_ids, all_ids):
    """Imprime estatísticas dos conjuntos de dados."""
    print("\n" + "=" * 50)
    print("ESTATÍSTICAS DOS CONJUNTOS")
    print("=" * 50)
    print(
        f"Treino:    {len(train_ids):3d} imagens ({len(train_ids) / len(all_ids) * 100:5.1f}%)"
    )
    print(
        f"Validação: {len(val_ids):3d} imagens ({len(val_ids) / len(all_ids) * 100:5.1f}%)"
    )
    print(
        f"Teste:     {len(test_ids):3d} imagens ({len(test_ids) / len(all_ids) * 100:5.1f}%)"
    )
    print(f"Total:     {len(all_ids):3d} imagens")
    print("=" * 50 + "\n")


def _format_duration(seconds):
    if seconds is None or pd.isna(seconds):
        return "N/A"
    return f"{seconds:.1f}s ({seconds / 60:.2f}min, {seconds / 3600:.3f}h)"


def print_split_summary(
    df,
    split_name,
    config,
    execution_time=None,
    timing_summary=None,
    current_run_execution_time=None,
):
    """Imprime estatísticas consolidadas de um split."""
    if df.empty:
        return

    both_correct_series = _bool_series(df, "both_correct")
    both_tolerable_series = _bool_series(df, "both_tolerable")
    ostia_found_series = _bool_series(df, "ostia_found")
    segmentation_attempted_series = _bool_series(df, "segmentation_attempted")
    proceeded_with_bad_ostia_series = _bool_series(df, "proceeded_with_bad_ostia")
    ostia_status_series = _series_from_aliases(df, "ostia_status")
    ostia_not_found_series = ostia_status_series.fillna("").astype(str).str.lower().isin(
        {"not_found", "not found", "não encontrados", "óstios não encontrados"}
    )
    dice_series = pd.to_numeric(
        _series_from_aliases(df, "dice_artery", dtype=float), errors="coerce"
    )
    tolerance_mm = config["OSTIA_VALIDATION"]["distance_threshold_mm"]

    print(f"\n📊 Estatísticas do conjunto {split_name}:")
    print(
        f"   - Óstios encontrados:         {ostia_found_series.sum():3d} ({ostia_found_series.mean() * 100:5.1f}%)"
    )
    print(
        f"   - Óstios não encontrados:     {ostia_not_found_series.sum():3d} ({ostia_not_found_series.mean() * 100:5.1f}%)"
    )
    print(
        f"   - Ambos corretos (estrito): {both_correct_series.sum():3d} ({both_correct_series.mean() * 100:5.1f}%)"
    )
    print(
        f"   - Tolerável apenas:         {both_tolerable_series.sum():3d} ({both_tolerable_series.mean() * 100:5.1f}%)"
    )
    print(
        f"   - Segmentação tentada:      {segmentation_attempted_series.sum():3d} ({segmentation_attempted_series.mean() * 100:5.1f}%)"
    )
    print(
        f"   - Prosseguiu com óstio ruim:{proceeded_with_bad_ostia_series.sum():3d} ({proceeded_with_bad_ostia_series.mean() * 100:5.1f}%)"
    )
    print(
        f"   - Total sucesso (<= {tolerance_mm}mm): {(both_correct_series | both_tolerable_series).sum():3d} ({(both_correct_series | both_tolerable_series).mean() * 100:5.1f}%)"
    )
    if dice_series.notna().any():
        print(f"   - Dice médio:       {dice_series.mean():.4f}")
    if execution_time:
        print(f"   - Tempo total conhecido: {_format_duration(execution_time)}")
    if current_run_execution_time and current_run_execution_time != execution_time:
        print(
            f"   - Tempo desta execução: {_format_duration(current_run_execution_time)}"
        )
    if timing_summary:
        missing_batches = timing_summary.get("missing_timing_batches") or []
        if missing_batches:
            missing_text = ", ".join(str(batch) for batch in missing_batches)
            print(f"   - Lotes sem tempo salvo: {missing_text}")
