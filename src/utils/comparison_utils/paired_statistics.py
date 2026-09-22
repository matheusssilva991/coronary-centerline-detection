"""Calcula estatísticas pareadas de Dice sem excluir falhas com Dice zero."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from ..project.dataframe import require_series_column, to_numeric_series


def _scalar_float(value: object) -> float:
    """Converte um resultado numérico escalar e rejeita vetores inesperados."""
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError("Era esperado um resultado estatístico escalar.")
    return float(array.item())


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
) -> dict[str, float | int]:
    """Compara IDs comuns; Wilcoxon bilateral exclui deltas zero dos ranks.

    IDs repetidos e Dice finito fora de [0, 1] são erros. Valores ausentes
    são reportados como pares excluídos, nunca convertidos em Dice zero.
    O efeito rank-biserial positivo favorece o candidato. Diferenças são
    arredondadas a 12 casas para evitar desempates numéricos artificiais.
    """
    frames = []
    for frame in (baseline, candidate):
        data = frame.copy()
        ids = to_numeric_series(require_series_column(data, id_column), errors="raise")
        data[id_column] = ids
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError("IDs ausentes ou repetidos na comparacao pareada.")
        dice = to_numeric_series(
            require_series_column(data, dice_column), errors="raise"
        )
        data[dice_column] = dice
        present = dice.dropna()
        if not present.between(0, 1).all():
            raise ValueError("Dice deve ser finito e estar entre 0 e 1.")
        indexed = data.set_index(id_column)
        if not isinstance(indexed, pd.DataFrame):
            raise TypeError("A indexação dos resultados não retornou um DataFrame.")
        frames.append(require_series_column(indexed, dice_column))

    aligned = pd.concat(frames, axis=1, keys=["baseline", "candidate"])
    paired = aligned.dropna()
    if paired.empty:
        raise ValueError("Sem pares validos para comparar Dice.")
    baseline_values = require_series_column(paired, "baseline")
    candidate_values = require_series_column(paired, "candidate")
    delta = candidate_values - baseline_values
    ranked_delta = delta.round(12)
    nonzero = ranked_delta[ranked_delta.ne(0)]
    if nonzero.empty:
        statistic, p_value, effect = 0.0, 1.0, 0.0
    else:
        statistic_value, p_value_value = wilcoxon(
            ranked_delta, alternative="two-sided", zero_method="wilcox", method="auto"
        )
        statistic = _scalar_float(statistic_value)
        p_value = _scalar_float(p_value_value)
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
        "baseline_mean_dice": _scalar_float(baseline_values.mean()),
        "baseline_std_dice": _scalar_float(baseline_values.std()),
        "baseline_median_dice": _scalar_float(baseline_values.median()),
        "baseline_q1_dice": _scalar_float(baseline_values.quantile(0.25)),
        "baseline_q3_dice": _scalar_float(baseline_values.quantile(0.75)),
        "candidate_mean_dice": _scalar_float(candidate_values.mean()),
        "candidate_std_dice": _scalar_float(candidate_values.std()),
        "candidate_median_dice": _scalar_float(candidate_values.median()),
        "candidate_q1_dice": _scalar_float(candidate_values.quantile(0.25)),
        "candidate_q3_dice": _scalar_float(candidate_values.quantile(0.75)),
        "mean_delta_dice": _scalar_float(delta.mean()),
        "median_delta_dice": _scalar_float(delta.median()),
        "mean_delta_ci_95_low": ci_low,
        "mean_delta_ci_95_high": ci_high,
        "improved_images": int(ranked_delta.gt(0).sum()),
        "worse_images": int(ranked_delta.lt(0).sum()),
        "unchanged_images": int(ranked_delta.eq(0).sum()),
        "rank_biserial_effect": effect,
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
    }


def adjust_holm(p_values: pd.Series) -> pd.Series:
    """Controla múltiplas comparações e preserva os índices da tabela."""
    ordered = p_values.sort_values()
    factors = np.arange(len(ordered), 0, -1)
    return (ordered * factors).cummax().clip(upper=1).reindex(p_values.index)
