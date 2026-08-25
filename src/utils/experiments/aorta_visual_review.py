"""Load and validate manual reviews used by the aorta EDA notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Collection

import numpy as np
import pandas as pd

from ..comparison_utils.io import load_split_summary


AORTA_REVIEW_ID_FIELDS = (
    "aorta_good_ids",
    "aorta_bad_ids",
)
OSTIA_REVIEW_ID_FIELDS = (
    "ostia_good_ids",
    "ostia_bad_ids",
)


def load_aorta_visual_reviews(path: str | Path) -> dict[str, Any]:
    """Load the visual-review catalog and validate every variant and split."""
    review_path = Path(path)
    data = json.loads(review_path.read_text(encoding="utf-8"))
    variants = data.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise ValueError("The visual-review catalog must contain variants.")

    for variant, split_reviews in variants.items():
        if not isinstance(split_reviews, dict):
            raise ValueError(f"Invalid review groups for variant {variant!r}.")
        for split, review in split_reviews.items():
            _validate_review(review, variant, split)
    return data


def get_aorta_visual_review(
    catalog: dict[str, Any],
    variant: str,
    split: str,
) -> dict[str, Any]:
    """Return one review with ID lists converted to sets and note keys to integers."""
    try:
        raw_review = catalog["variants"][variant][split]
    except KeyError as exc:
        raise KeyError(f"Review not found for variant={variant!r}, split={split!r}.") from exc

    review = dict(raw_review)
    for field in (*AORTA_REVIEW_ID_FIELDS, *OSTIA_REVIEW_ID_FIELDS):
        if field in raw_review:
            review[field] = {int(img_id) for img_id in raw_review[field]}
    review["notes"] = {
        int(img_id): note for img_id, note in raw_review.get("notes", {}).items()
    }
    return review


def resolve_aorta_review_summary_path(
    repo_root: str | Path,
    review: dict[str, Any],
    split: str,
) -> Path:
    """Resolve the summary CSV associated with a catalog entry."""
    return Path(repo_root) / review["run_dir"] / "numeric" / f"ostios_{split}_summary.csv"


def add_aorta_extent_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Derive comparable axial-extent metrics from circles and the aorta mask.

    Positive ``segmented_minus_circle_slices`` values indicate that the final
    mask occupies more slices than the circle trajectory. Negative values
    indicate axial retraction after segmentation and post-processing.
    """
    df = dataframe.copy()
    required = {
        "image_slice_count",
        "aorta_circle_count",
        "aorta_segmented_slice_count",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing aorta extent columns: {sorted(missing)}")

    image_slices = pd.to_numeric(df["image_slice_count"], errors="coerce")
    circle_slices = pd.to_numeric(df["aorta_circle_count"], errors="coerce")
    segmented_slices = pd.to_numeric(
        df["aorta_segmented_slice_count"], errors="coerce"
    )
    valid_image_slices = image_slices.where(image_slices.gt(0))
    valid_circle_slices = circle_slices.where(circle_slices.gt(0))

    df["circle_slice_fraction"] = circle_slices / valid_image_slices
    df["segmented_slice_fraction"] = segmented_slices / valid_image_slices
    df["segmented_minus_circle_slices"] = segmented_slices - circle_slices
    df["segmented_vs_circle_change_fraction"] = (
        segmented_slices - circle_slices
    ) / valid_circle_slices

    if "aorta_volume_fraction" in df.columns:
        df["aorta_volume_percentage"] = (
            pd.to_numeric(df["aorta_volume_fraction"], errors="coerce") * 100.0
        )
    if {
        "aorta_circle_first_slice",
        "aorta_circle_last_slice",
    }.issubset(df.columns):
        first = pd.to_numeric(df["aorta_circle_first_slice"], errors="coerce")
        last = pd.to_numeric(df["aorta_circle_last_slice"], errors="coerce")
        df["circle_first_position"] = first / valid_image_slices
        df["circle_last_position"] = last / valid_image_slices
        df["circle_center_position"] = (first + last) / (2.0 * valid_image_slices)
    return df


def load_aorta_review_cohort(
    repo_root: str | Path,
    review: dict[str, Any],
    split: str,
    *,
    cohort_name: str | None = None,
    required_columns: Collection[str] = (),
    use_reviewed_ostia_labels: bool = False,
) -> pd.DataFrame:
    """Load a reviewed run and add visual, ostia, and axial-extent labels."""
    summary_path = resolve_aorta_review_summary_path(repo_root, review, split)
    numeric_dir = summary_path.parent
    dataframe = load_split_summary({"mid_res": {split: numeric_dir}}, "mid_res", split)
    if dataframe is None:
        raise RuntimeError(f"Could not load the {split!r} summary.")

    missing = set(required_columns).difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing summary columns for {split!r}: {sorted(missing)}")

    df = dataframe.copy()
    df["IMG_ID"] = pd.to_numeric(df["IMG_ID"], errors="raise").astype(int)
    good_ids = {int(img_id) for img_id in review["aorta_good_ids"]}
    bad_ids = {int(img_id) for img_id in review["aorta_bad_ids"]}
    expected_ids = good_ids | bad_ids
    observed_ids = set(df["IMG_ID"])
    if expected_ids != observed_ids:
        raise ValueError(
            f"Incompatible IDs for {split!r}. "
            f"Missing={sorted(expected_ids - observed_ids)}; "
            f"unclassified={sorted(observed_ids - expected_ids)}"
        )

    df["visual_aorta_quality"] = np.where(
        df["IMG_ID"].isin(good_ids), "boa", "ruim"
    )
    df["visual_review_note"] = df["IMG_ID"].map(review.get("notes", {})).fillna("")
    normalized_status = (
        df["ostia_detection_status"]
        .astype(str)
        .str.lower()
        .str.replace("_", " ", regex=False)
    )
    csv_success = normalized_status.isin(
        {"both correct", "both tolerable", "both ostia correct", "both ostia tolerable"}
    )
    if use_reviewed_ostia_labels:
        bad_ostia_ids = {int(img_id) for img_id in review.get("ostia_bad_ids", ())}
        df["ostia_success"] = ~df["IMG_ID"].isin(bad_ostia_ids)
    else:
        df["ostia_success"] = csv_success
    df["ostia_outcome"] = np.where(df["ostia_success"], "sucesso", "falha")
    df["coorte"] = cohort_name or split
    extent_columns = {
        "image_slice_count",
        "aorta_circle_count",
        "aorta_segmented_slice_count",
    }
    if extent_columns.issubset(df.columns):
        df = add_aorta_extent_metrics(df)
    return df.sort_values("IMG_ID").reset_index(drop=True)


def _validate_review(review: Any, variant: str, split: str) -> None:
    """Reject incomplete or contradictory manual classifications."""
    if not isinstance(review, dict) or not review.get("run_dir"):
        raise ValueError(f"Missing run_dir for variant={variant!r}, split={split!r}.")
    missing = [field for field in AORTA_REVIEW_ID_FIELDS if field not in review]
    if missing:
        raise ValueError(
            f"Missing review fields for variant={variant!r}, split={split!r}: {missing}"
        )

    groups = {
        field: {int(img_id) for img_id in review[field]}
        for field in (*AORTA_REVIEW_ID_FIELDS, *OSTIA_REVIEW_ID_FIELDS)
        if field in review
    }
    ostia_fields_present = [field in review for field in OSTIA_REVIEW_ID_FIELDS]
    if any(ostia_fields_present) and not all(ostia_fields_present):
        raise ValueError(
            f"Incomplete ostia labels for variant={variant!r}, split={split!r}."
        )

    subjects = ["aorta"]
    if all(ostia_fields_present):
        subjects.append("ostia")
    for subject in subjects:
        good = groups[f"{subject}_good_ids"]
        bad = groups[f"{subject}_bad_ids"]
        overlap = good & bad
        if overlap:
            raise ValueError(
                f"Contradictory {subject} labels for variant={variant!r}, "
                f"split={split!r}: {sorted(overlap)}"
            )
        if good | bad != groups["aorta_good_ids"] | groups["aorta_bad_ids"]:
            raise ValueError(
                f"The {subject} labels do not cover the same cohort for "
                f"variant={variant!r}, split={split!r}."
            )


__all__ = [
    "add_aorta_extent_metrics",
    "get_aorta_visual_review",
    "load_aorta_review_cohort",
    "load_aorta_visual_reviews",
    "resolve_aorta_review_summary_path",
]
