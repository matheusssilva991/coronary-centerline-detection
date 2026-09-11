"""Estatisticas pareadas de Dice, sem excluir falhas com Dice zero."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


def _bootstrap_mean_confidence_interval(
    values: np.ndarray,
    *,
    confidence_level: float,
    samples: int,
    random_state: int,
) -> tuple[float, float]:
    """Estima o IC percentil da média preservando o pareamento dos deltas."""
    if samples <= 0:
        raise ValueError("bootstrap_samples deve ser maior que zero.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level deve estar entre zero e um.")
    if np.all(values == values[0]):
        value = float(values[0])
        return value, value

    rng = np.random.default_rng(random_state)
    bootstrap_means = np.empty(samples, dtype=float)
    # Processa em blocos para limitar memória nas coortes maiores.
    block_size = min(samples, 1_000)
    for start in range(0, samples, block_size):
        stop = min(start + block_size, samples)
        sampled = rng.choice(values, size=(stop - start, len(values)), replace=True)
        bootstrap_means[start:stop] = sampled.mean(axis=1)

    alpha = 1 - confidence_level
    lower, upper = np.quantile(
        bootstrap_means,
        [alpha / 2, 1 - alpha / 2],
    )
    return float(lower), float(upper)


def compare_paired_dice(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    id_column: str = "IMG_ID",
    dice_column: str = "artery_dice",
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    random_state: int = 42,
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
    ci_low, ci_high = _bootstrap_mean_confidence_interval(
        delta.to_numpy(dtype=float),
        confidence_level=confidence_level,
        samples=bootstrap_samples,
        random_state=random_state,
    )
    return {
        "paired_images": len(paired),
        "excluded_pairs": len(aligned) - len(paired),
        "baseline_mean_dice": float(paired["baseline"].mean()),
        "baseline_std_dice": float(paired["baseline"].std()),
        "baseline_median_dice": float(paired["baseline"].median()),
        "baseline_q1_dice": float(paired["baseline"].quantile(0.25)),
        "baseline_q3_dice": float(paired["baseline"].quantile(0.75)),
        "candidate_mean_dice": float(paired["candidate"].mean()),
        "candidate_std_dice": float(paired["candidate"].std()),
        "candidate_median_dice": float(paired["candidate"].median()),
        "candidate_q1_dice": float(paired["candidate"].quantile(0.25)),
        "candidate_q3_dice": float(paired["candidate"].quantile(0.75)),
        "mean_delta_dice": float(delta.mean()),
        "median_delta_dice": float(delta.median()),
        "mean_delta_ci_95_low": ci_low,
        "mean_delta_ci_95_high": ci_high,
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
