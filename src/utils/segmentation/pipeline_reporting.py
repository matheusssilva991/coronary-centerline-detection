"""Relatórios de console para o pipeline de segmentação."""

from __future__ import annotations

import pandas as pd

from ..project.results import summarize_results_df


def print_statistics(train_ids, val_ids, test_ids, all_ids):
    """Imprime estatísticas dos conjuntos de dados."""
    print("\n" + "=" * 50)
    print("ESTATÍSTICAS DOS CONJUNTOS")
    print("=" * 50)
    print(
        f"Treino:    {len(train_ids):3d} imagens "
        f"({len(train_ids) / len(all_ids) * 100:5.1f}%)"
    )
    print(
        f"Validação: {len(val_ids):3d} imagens "
        f"({len(val_ids) / len(all_ids) * 100:5.1f}%)"
    )
    print(
        f"Teste:     {len(test_ids):3d} imagens "
        f"({len(test_ids) / len(all_ids) * 100:5.1f}%)"
    )
    print(f"Total:     {len(all_ids):3d} imagens")
    print("=" * 50 + "\n")


def _format_duration(seconds):
    """Formata uma duração em segundos, minutos e horas."""
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

    # Centraliza os cálculos no mesmo agregador usado pelos metadados.
    summary = summarize_results_df(df)
    tolerance_mm = config["OSTIA_VALIDATION"]["distance_threshold_mm"]

    print(f"\n📊 Estatísticas do conjunto {split_name}:")
    print(
        f"   - Óstios encontrados:         {summary['ostia_found']:3d} "
        f"({summary['ostia_found_percent']:5.1f}%)"
    )
    print(
        f"   - Óstios não encontrados:     {summary['ostia_status_not_found']:3d} "
        f"({summary['ostia_status_not_found_percent']:5.1f}%)"
    )
    print(
        f"   - Ambos corretos (estrito): {summary['both_correct']:3d} "
        f"({summary['both_correct_percent']:5.1f}%)"
    )
    print(
        f"   - Tolerável apenas:         {summary['both_tolerable']:3d} "
        f"({summary['both_tolerable_percent']:5.1f}%)"
    )
    print(
        f"   - Segmentação tentada:      {summary['segmentation_attempted']:3d} "
        f"({summary['segmentation_attempted_percent']:5.1f}%)"
    )
    print(
        f"   - Prosseguiu com óstio ruim:{summary['proceeded_with_bad_ostia']:3d} "
        f"({summary['proceeded_with_bad_ostia_percent']:5.1f}%)"
    )
    print(
        f"   - Total sucesso (<= {tolerance_mm}mm): {summary['total_success']:3d} "
        f"({summary['total_success_percent']:5.1f}%)"
    )
    # Exibe o efeito da morfologia somente em runs que salvaram a métrica prévia.
    if summary["dice_artery_mean"] is not None:
        if summary["dice_artery_before_morphology_mean"] is not None:
            print(
                "   - Dice médio antes da morfologia: "
                f"{summary['dice_artery_before_morphology_mean']:.4f}"
            )
            print(
                "   - Dice médio após a morfologia:   "
                f"{summary['dice_artery_mean']:.4f}"
            )
            if summary["dice_artery_morphology_delta_mean"] is not None:
                print(
                    "   - Ganho médio da morfologia:      "
                    f"{summary['dice_artery_morphology_delta_mean']:+.4f}"
                )
        else:
            print(f"   - Dice médio:       {summary['dice_artery_mean']:.4f}")
    if execution_time:
        print(f"   - Tempo total conhecido: {_format_duration(execution_time)}")
    if current_run_execution_time and current_run_execution_time != execution_time:
        print(
            f"   - Tempo desta execução: {_format_duration(current_run_execution_time)}"
        )
    # Em retomadas, destaca lotes antigos que não tinham manifest de duração.
    if timing_summary:
        missing_batches = timing_summary.get("missing_timing_batches") or []
        if missing_batches:
            missing_text = ", ".join(str(batch) for batch in missing_batches)
            print(f"   - Lotes sem tempo salvo: {missing_text}")
