"""Load and validate manual reviews used by the aorta EDA notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
    "get_aorta_visual_review",
    "load_aorta_visual_reviews",
    "resolve_aorta_review_summary_path",
]
