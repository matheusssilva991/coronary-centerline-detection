"""Reúne carregamento e agregação para EDAs de comparação entre runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ..project.dataframe import require_series_column, to_numeric_series
from utils.project.results.schema import (
    add_internal_result_aliases,
    normalize_ostia_status,
    summarize_results_df,
)
from .io import load_split_results
from .paired_statistics import adjust_holm, compare_paired_dice


OSTIA_SUCCESS_STATUSES = frozenset({"both_correct", "both_tolerable"})


def _as_bool_series(series: pd.Series) -> pd.Series:
    """Converte flags persistidas sem tratar qualquer texto como verdadeiro."""
    return series.map(
        lambda value: (
            value
            if isinstance(value, bool)
            else str(value).strip().casefold() in {"true", "1", "yes", "sim", "s"}
        )
    )


def ostia_success_mask(results: pd.DataFrame) -> pd.Series:
    """Retorna um booleano por exame usando a regra canônica dos óstios."""
    normalized = add_internal_result_aliases(results)
    if "ostia_status" in normalized.columns:
        status = require_series_column(normalized, "ostia_status")
        if status.notna().any():
            return status.map(normalize_ostia_status).isin(
                tuple(OSTIA_SUCCESS_STATUSES)
            )

    if {"both_correct", "both_tolerable"}.issubset(normalized.columns):
        return _as_bool_series(
            require_series_column(normalized, "both_correct")
        ) | _as_bool_series(require_series_column(normalized, "both_tolerable"))
    raise ValueError(
        "Resultados sem status ou flags suficientes para avaliar os óstios."
    )


def load_validated_comparison_runs(
    split_paths_by_variant: Mapping[str, Mapping[str, str | Path]],
    expected_images: Mapping[str, int],
    *,
    valid_splits: Sequence[str] = ("train", "val", "test"),
    require_matching_ids: bool = True,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Carrega e valida a matriz de variantes e splits usada nas comparações."""
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for variant, split_paths in split_paths_by_variant.items():
        for split in valid_splits:
            if split not in split_paths:
                raise ValueError(f"Split ausente em {variant!r}: {split!r}")
            frame = load_split_results(split_paths_by_variant, variant, split)
            if frame is None:
                raise FileNotFoundError(f"Resultados ausentes: {variant}/{split}")
            frame = frame.copy()
            image_ids = to_numeric_series(
                require_series_column(frame, "IMG_ID"), errors="raise"
            ).astype(int)
            artery_dice = to_numeric_series(
                require_series_column(frame, "artery_dice"), errors="raise"
            )
            frame["IMG_ID"] = image_ids
            frame["artery_dice"] = artery_dice
            expected_count = expected_images.get(split)
            if expected_count is not None and len(frame) != expected_count:
                raise ValueError(
                    f"Coorte incompleta: {variant}/{split}; "
                    f"esperado={expected_count}, observado={len(frame)}"
                )
            if image_ids.duplicated().any():
                duplicates = sorted(image_ids.loc[image_ids.duplicated()].unique())
                raise ValueError(f"IDs duplicados em {variant}/{split}: {duplicates}")
            if artery_dice.isna().any() or not artery_dice.between(0, 1).all():
                raise ValueError(f"Dice inválido em {variant}/{split}")
            frame["ostia_success"] = ostia_success_mask(frame)
            frames[variant, split] = frame

    if require_matching_ids:
        for split in valid_splits:
            split_frames = [
                (variant, frame)
                for (variant, frame_split), frame in frames.items()
                if frame_split == split
            ]
            reference_variant, reference_frame = split_frames[0]
            reference_ids = set(require_series_column(reference_frame, "IMG_ID"))
            for variant, frame in split_frames[1:]:
                if set(require_series_column(frame, "IMG_ID")) != reference_ids:
                    raise ValueError(
                        f"IDs diferentes em {split}: "
                        f"{reference_variant!r} vs {variant!r}"
                    )
    return frames


def build_dice_ostia_overview(
    frames: Mapping[tuple[str, str], pd.DataFrame],
    *,
    variant_labels: Mapping[str, str] | None = None,
    split_labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Monta uma linha compacta de Dice e óstios por variante e split."""
    variant_names = variant_labels or {}
    split_names = split_labels or {}
    rows = []
    for (variant, split), frame in frames.items():
        summary = summarize_results_df(frame)
        success = ostia_success_mask(frame)
        rows.append(
            {
                "variant": variant,
                "variant_label": variant_names.get(variant, variant),
                "split": split,
                "split_label": split_names.get(split, split),
                "num_images": summary["total_processed"],
                "mean_dice": summary["dice_artery_mean"],
                "ostia_success_count": int(success.sum()),
                "ostia_success_percent": float(success.mean() * 100),
            }
        )
    return pd.DataFrame(rows)


def compare_paired_run_matrix(
    frames: Mapping[tuple[str, str], pd.DataFrame],
    comparisons: Sequence[tuple[str, str]],
    *,
    valid_splits: Sequence[str] = ("train", "val", "test"),
    split_labels: Mapping[str, str] | None = None,
    alpha: float = 0.05,
    comparison_kwargs: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Compara pares de runs e aplica Holm à família completa de testes."""
    split_names = split_labels or {}
    kwargs = dict(comparison_kwargs or {})
    rows = []
    for split in valid_splits:
        for reference, candidate in comparisons:
            try:
                reference_frame = frames[reference, split]
                candidate_frame = frames[candidate, split]
            except KeyError as error:
                raise KeyError(
                    f"Par ausente para {split}: {reference!r} vs {candidate!r}"
                ) from error
            rows.append(
                {
                    "split": split,
                    "split_label": split_names.get(split, split),
                    "reference": reference,
                    "candidate": candidate,
                    **compare_paired_dice(
                        reference_frame,
                        candidate_frame,
                        **kwargs,
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["p_holm"] = adjust_holm(require_series_column(result, "p_value"))
    result["significant"] = require_series_column(result, "p_holm").lt(alpha)
    return result
