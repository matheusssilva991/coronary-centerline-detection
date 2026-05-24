"""Relatórios de console para o pipeline de segmentação."""

from __future__ import annotations


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


def print_split_summary(df, split_name, config, execution_time=None):
    """Imprime estatísticas consolidadas de um split."""
    if df.empty:
        return

    both_correct_series = df["both_correct"].fillna(False)
    both_tolerable_series = df["both_tolerable"].fillna(False)
    ostia_found_series = df["ostia_found"].fillna(False)
    ostia_not_found_series = df["ostia_status"].eq("not_found")
    segmentation_attempted_series = df["segmentation_attempted"].fillna(False)
    proceeded_with_bad_ostia_series = df["proceeded_with_bad_ostia"].fillna(False)
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
    if "dice_artery" in df.columns and df["dice_artery"].notna().any():
        print(f"   - Dice médio:       {df['dice_artery'].mean():.4f}")
    if execution_time:
        print(f"   - Tempo de execução: {execution_time:.1f}s ({execution_time / 60:.1f}min)")
