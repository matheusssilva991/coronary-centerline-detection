"""Tests for shared run-comparison helpers used by EDA notebooks."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from utils.comparison_utils.run_comparison import (
    build_dice_ostia_overview,
    compare_paired_run_matrix,
    load_validated_comparison_runs,
    ostia_success_mask,
)


def _frame(dice: list[float], statuses: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "IMG_ID": [10, 20],
            "artery_dice": dice,
            "ostia_detection_status": statuses,
        }
    )


class RunComparisonTests(unittest.TestCase):
    def test_ostia_success_mask_normalizes_legacy_labels(self) -> None:
        frame = pd.DataFrame(
            {
                "ostia_detection_status": [
                    "ambos corretos",
                    "both tolerable",
                    "found but incorrect",
                    "not found",
                ]
            }
        )

        self.assertEqual(
            ostia_success_mask(frame).tolist(),
            [True, True, False, False],
        )

    def test_loads_validates_and_pairs_run_matrix(self) -> None:
        frames = {
            ("baseline", "train"): _frame([0.2, 0.6], ["both_correct", "not_found"]),
            ("candidate", "train"): _frame(
                [0.4, 0.8], ["both_tolerable", "found_but_wrong"]
            ),
        }

        def load_results(_paths, variant, split):
            return frames[variant, split]

        paths = {"baseline": {"train": "a"}, "candidate": {"train": "b"}}
        with patch(
            "utils.comparison_utils.run_comparison.load_split_results",
            side_effect=load_results,
        ):
            result = load_validated_comparison_runs(
                paths,
                {"train": 2},
                valid_splits=("train",),
            )

        self.assertEqual(set(result), set(frames))
        self.assertEqual(
            result["baseline", "train"]["ostia_success"].tolist(), [True, False]
        )

    def test_rejects_mismatched_ids(self) -> None:
        baseline = _frame([0.2, 0.6], ["both_correct", "not_found"])
        candidate = _frame([0.4, 0.8], ["both_correct", "not_found"])
        candidate["IMG_ID"] = [10, 30]

        def load_results(_paths, variant, _split):
            return baseline if variant == "baseline" else candidate

        paths = {"baseline": {"train": "a"}, "candidate": {"train": "b"}}
        with patch(
            "utils.comparison_utils.run_comparison.load_split_results",
            side_effect=load_results,
        ):
            with self.assertRaisesRegex(ValueError, "IDs diferentes"):
                load_validated_comparison_runs(
                    paths,
                    {"train": 2},
                    valid_splits=("train",),
                )

    def test_builds_overview_from_canonical_summary(self) -> None:
        frames = {
            ("baseline", "train"): _frame([0.2, 0.6], ["both_correct", "not_found"])
        }

        result = build_dice_ostia_overview(
            frames,
            variant_labels={"baseline": "Anterior"},
            split_labels={"train": "Treino"},
        ).iloc[0]

        self.assertEqual(result["variant_label"], "Anterior")
        self.assertEqual(result["split_label"], "Treino")
        self.assertEqual(result["num_images"], 2)
        self.assertAlmostEqual(result["mean_dice"], 0.4)
        self.assertEqual(result["ostia_success_count"], 1)
        self.assertEqual(result["ostia_success_percent"], 50.0)

    def test_compares_matrix_and_applies_holm(self) -> None:
        frames = {
            ("baseline", "train"): _frame([0.2, 0.6], ["both_correct", "not_found"]),
            ("candidate", "train"): _frame([0.4, 0.8], ["both_correct", "not_found"]),
        }

        result = compare_paired_run_matrix(
            frames,
            [("baseline", "candidate")],
            valid_splits=("train",),
            comparison_kwargs={"bootstrap_samples": 100},
        ).iloc[0]

        self.assertEqual(result["paired_images"], 2)
        self.assertAlmostEqual(result["mean_delta_dice"], 0.2)
        self.assertEqual(result["p_holm"], result["p_value"])
        self.assertFalse(bool(result["significant"]))


if __name__ == "__main__":
    unittest.main()
