"""Shared loading and aggregation helpers for run-comparison EDAs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ..project.results_schema import (
    add_internal_result_aliases,
    normalize_ostia_status,
    summarize_results_df,
)
from .io import load_split_results
from .paired_statistics import adjust_holm, compare_paired_dice


OSTIA_SUCCESS_STATUSES = frozenset({"both_correct", "both_tolerable"})


def _as_bool_series(series: pd.Series) -> pd.Series:
    """Convert persisted bool-like values without treating non-empty strings as true."""
    return series.map(
        lambda value: (
            value
            if isinstance(value, bool)
            else str(value).strip().casefold() in {"true", "1", "yes", "sim", "s"}
        )
    )


def ostia_success_mask(results: pd.DataFrame) -> pd.Series:
    """Return one boolean per exam using the canonical ostia success rule."""
    normalized = add_internal_result_aliases(results)
    status = normalized.get("ostia_status")
    if status is not None and status.notna().any():
        return status.map(normalize_ostia_status).isin(OSTIA_SUCCESS_STATUSES)

    if {"both_correct", "both_tolerable"}.issubset(normalized.columns):
        return _as_bool_series(normalized["both_correct"]) | _as_bool_series(
            normalized["both_tolerable"]
        )
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
    """Load and validate a variant/split matrix used by comparison notebooks."""
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for variant, split_paths in split_paths_by_variant.items():
        for split in valid_splits:
            if split not in split_paths:
                raise ValueError(f"Split ausente em {variant!r}: {split!r}")
            frame = load_split_results(split_paths_by_variant, variant, split)
            if frame is None:
                raise FileNotFoundError(f"Resultados ausentes: {variant}/{split}")
            frame = frame.copy()
            frame["IMG_ID"] = pd.to_numeric(frame["IMG_ID"], errors="raise").astype(int)
            frame["artery_dice"] = pd.to_numeric(frame["artery_dice"], errors="raise")
            expected_count = expected_images.get(split)
            if expected_count is not None and len(frame) != expected_count:
                raise ValueError(
                    f"Coorte incompleta: {variant}/{split}; "
                    f"esperado={expected_count}, observado={len(frame)}"
                )
            if frame["IMG_ID"].duplicated().any():
                duplicates = sorted(
                    frame.loc[frame["IMG_ID"].duplicated(), "IMG_ID"].unique()
                )
                raise ValueError(f"IDs duplicados em {variant}/{split}: {duplicates}")
            if (
                frame["artery_dice"].isna().any()
                or not frame["artery_dice"].between(0, 1).all()
            ):
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
            reference_ids = set(reference_frame["IMG_ID"])
            for variant, frame in split_frames[1:]:
                if set(frame["IMG_ID"]) != reference_ids:
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
    """Build one compact Dice/ostia summary row per variant and split."""
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
    """Compare declared run pairs and apply Holm over the complete test family."""
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
    result["p_holm"] = adjust_holm(result["p_value"])
    result["significant"] = result["p_holm"].lt(alpha)
    return result
