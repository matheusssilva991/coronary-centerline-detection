"""Tests for the visual-review catalog used by aorta EDA notebooks."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from utils.experiments.aorta_visual_review import (
    add_aorta_extent_metrics,
    get_aorta_visual_review,
    load_aorta_review_cohort,
    load_aorta_visual_reviews,
    resolve_aorta_review_summary_path,
)


def _write_catalog(path: Path, review: dict) -> Path:
    path.write_text(
        json.dumps({"variants": {"normal": {"train": review}}}),
        encoding="utf-8",
    )
    return path


class AortaVisualReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_review_converts_ids_and_notes(self) -> None:
        catalog_path = _write_catalog(
            self.root / "reviews.json",
            {
                "run_dir": "output/example",
                "aorta_good_ids": [1, 2],
                "aorta_bad_ids": [3],
                "ostia_good_ids": [1, 3],
                "ostia_bad_ids": [2],
                "notes": {"3": "reviewed"},
            },
        )

        catalog = load_aorta_visual_reviews(catalog_path)
        review = get_aorta_visual_review(catalog, "normal", "train")

        self.assertEqual(review["aorta_good_ids"], {1, 2})
        self.assertEqual(review["ostia_bad_ids"], {2})
        self.assertEqual(review["notes"], {3: "reviewed"})
        self.assertEqual(
            resolve_aorta_review_summary_path(self.root, review, "train"),
            self.root / "output/example/numeric/ostios_train_summary.csv",
        )

    def test_load_review_rejects_overlapping_labels(self) -> None:
        catalog_path = _write_catalog(
            self.root / "reviews.json",
            {
                "run_dir": "output/example",
                "aorta_good_ids": [1, 2],
                "aorta_bad_ids": [2],
                "ostia_good_ids": [1],
                "ostia_bad_ids": [2],
            },
        )

        with self.assertRaisesRegex(ValueError, "Contradictory aorta labels"):
            load_aorta_visual_reviews(catalog_path)

    def test_load_review_rejects_different_cohort_coverage(self) -> None:
        catalog_path = _write_catalog(
            self.root / "reviews.json",
            {
                "run_dir": "output/example",
                "aorta_good_ids": [1],
                "aorta_bad_ids": [2],
                "ostia_good_ids": [1],
                "ostia_bad_ids": [3],
            },
        )

        with self.assertRaisesRegex(ValueError, "do not cover the same cohort"):
            load_aorta_visual_reviews(catalog_path)

    def test_load_review_accepts_aorta_only_labels(self) -> None:
        catalog_path = _write_catalog(
            self.root / "reviews.json",
            {
                "run_dir": "output/example",
                "aorta_good_ids": [1, 2],
                "aorta_bad_ids": [3],
                "notes": {"3": "leak"},
            },
        )

        catalog = load_aorta_visual_reviews(catalog_path)
        review = get_aorta_visual_review(catalog, "normal", "train")

        self.assertEqual(review["aorta_good_ids"], {1, 2})
        self.assertEqual(review["aorta_bad_ids"], {3})
        self.assertNotIn("ostia_good_ids", review)

    def test_add_extent_metrics_measures_expansion_and_retraction(self) -> None:
        dataframe = pd.DataFrame(
            {
                "image_slice_count": [100, 100],
                "aorta_circle_count": [40, 50],
                "aorta_segmented_slice_count": [44, 45],
            }
        )

        result = add_aorta_extent_metrics(dataframe)

        self.assertEqual(result["segmented_minus_circle_slices"].tolist(), [4, -5])
        self.assertEqual(
            result["segmented_vs_circle_change_fraction"].tolist(),
            [0.1, -0.1],
        )

    def test_load_review_cohort_adds_visual_and_ostia_labels(self) -> None:
        review = {
            "run_dir": "output/example",
            "aorta_good_ids": {1},
            "aorta_bad_ids": {2},
            "ostia_good_ids": {1},
            "ostia_bad_ids": {2},
            "notes": {2: "leak"},
        }
        numeric_dir = self.root / review["run_dir"] / "numeric"
        numeric_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "IMG_ID": [1, 2],
                "ostia_detection_status": ["both correct", "both correct"],
                "image_slice_count": [100, 100],
                "aorta_circle_count": [40, 50],
                "aorta_segmented_slice_count": [44, 45],
            }
        ).to_csv(numeric_dir / "ostios_train_summary.csv", index=False)

        result = load_aorta_review_cohort(
            self.root,
            review,
            "train",
            cohort_name="treino",
            use_reviewed_ostia_labels=True,
        )

        self.assertEqual(result["visual_aorta_quality"].tolist(), ["boa", "ruim"])
        self.assertEqual(result["ostia_success"].tolist(), [True, False])
        self.assertEqual(result["segmented_minus_circle_slices"].tolist(), [4, -5])


if __name__ == "__main__":
    unittest.main()
