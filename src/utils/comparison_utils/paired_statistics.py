"""Estatisticas pareadas de Dice, sem excluir falhas com Dice zero."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


def compare_paired_dice(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    id_column: str = "IMG_ID",
    dice_column: str = "artery_dice",
) -> dict:
    """Compara IDs comuns; Wilcoxon bilateral exclui deltas zero dos ranks.

    IDs repetidos e Dice finito fora de [0, 1] sao erros. Valores ausentes
    sao reportados como pares excluidos, nunca convertidos em Dice zero.
    O efeito rank-biserial positivo favorece o candidato. Diferencas sao
    arredondadas a 12 casas para evitar desempates numericos artificiais.
    """
    frames = []
    for frame in (baseline, candidate):
        data = frame[[id_column, dice_column]].copy()
        data[id_column] = pd.to_numeric(data[id_column], errors="raise")
        if data[id_column].isna().any() or data[id_column].duplicated().any():
            raise ValueError("IDs ausentes ou repetidos na comparacao pareada.")
        data[dice_column] = pd.to_numeric(data[dice_column], errors="raise")
        present = data[dice_column].dropna()
        if not present.between(0, 1).all():
            raise ValueError("Dice deve ser finito e estar entre 0 e 1.")
        frames.append(data.set_index(id_column)[dice_column])

    aligned = pd.concat(frames, axis=1, keys=["baseline", "candidate"])
    paired = aligned.dropna()
    if paired.empty:
        raise ValueError("Sem pares validos para comparar Dice.")
    delta = paired["candidate"] - paired["baseline"]
    ranked_delta = delta.round(12)
    nonzero = ranked_delta[ranked_delta.ne(0)]
    if nonzero.empty:
        statistic, p_value, effect = 0.0, 1.0, 0.0
    else:
        statistic, p_value = wilcoxon(
            ranked_delta, alternative="two-sided", zero_method="wilcox", method="auto"
        )
        ranks = rankdata(nonzero.abs())
        effect = float(np.sum(ranks * np.sign(nonzero)) / ranks.sum())
    return {
        "paired_images": len(paired),
        "excluded_pairs": len(aligned) - len(paired),
        "mean_delta_dice": float(delta.mean()),
        "median_delta_dice": float(delta.median()),
        "improved_images": int(ranked_delta.gt(0).sum()),
        "worse_images": int(ranked_delta.lt(0).sum()),
        "unchanged_images": int(ranked_delta.eq(0).sum()),
        "rank_biserial_effect": effect,
        "wilcoxon_statistic": float(statistic),
        "p_value": float(p_value),
    }


def adjust_holm(p_values: pd.Series) -> pd.Series:
    """Controla multiplas comparacoes e preserva os indices da tabela."""
    ordered = p_values.sort_values()
    factors = np.arange(len(ordered), 0, -1)
    return (ordered * factors).cummax().clip(upper=1).reindex(p_values.index)
