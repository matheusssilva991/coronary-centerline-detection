"""Tests for the visual-review catalog used by aorta EDA notebooks."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from utils.experiments.aorta_visual_review import (
    get_aorta_visual_review,
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


if __name__ == "__main__":
    unittest.main()
