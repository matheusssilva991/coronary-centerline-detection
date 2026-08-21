"""Tests for shared helpers extracted from EDA notebooks."""

from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from utils.comparison_utils.bad_cases import (
    prepare_bad_case_qualitative_comparison,
)
from utils.comparison_utils.metadata import build_split_resolution_summary


class SplitResolutionSummaryTest(TestCase):
    def test_supports_status_schemas_and_marks_missing_results(self):
        summaries = {
            "train": pd.DataFrame(
                {
                    "dice_artery": [0.8, 0.2],
                    "both_correct": [True, False],
                    "both_tolerable": [False, False],
                }
            ),
            "val": pd.DataFrame(
                {
                    "dice_artery": [0.4, 0.6],
                    "ostia_status": ["both_tolerable", "found_but_wrong"],
                }
            ),
            "test": pd.DataFrame(
                {
                    "dice_artery": [0.9, 0.3],
                    "status": ["ambos corretos", "nenhum correto"],
                }
            ),
        }
        metadata = {
            split: {
                "execution_info": {
                    "execution_time_seconds": 120,
                    "num_images": 2,
                },
                "results_summary": {"total_success_percent": 50.0},
            }
            for split in summaries
        }

        def load_summary(_, resolution, subset):
            return summaries[subset] if resolution == "mid_res" else None

        def load_metadata(_, resolution, subset):
            return metadata[subset] if resolution == "mid_res" else None

        split_paths = {"mid_res": {}, "high_res": {}}
        with (
            patch(
                "utils.comparison_utils.io.load_split_summary",
                side_effect=load_summary,
            ),
            patch(
                "utils.comparison_utils.io.load_split_metadata",
                side_effect=load_metadata,
            ),
        ):
            result = build_split_resolution_summary(split_paths)

        mid = result[result["resolution"].eq("mid_res")].set_index("subset")
        self.assertAlmostEqual(float(mid.loc["train", "mean_dice_correct"]), 0.8)
        self.assertAlmostEqual(float(mid.loc["val", "mean_dice_correct"]), 0.4)
        self.assertAlmostEqual(float(mid.loc["test", "mean_dice_correct"]), 0.9)
        self.assertAlmostEqual(float(mid.loc["train", "mean_dice_all"]), 0.5)
        self.assertAlmostEqual(float(mid.loc["train", "execution_time_min"]), 2.0)
        self.assertTrue(bool(mid.loc["train", "disponivel"]))

        high = result[result["resolution"].eq("high_res")]
        self.assertEqual(len(high), 3)
        self.assertFalse(high["is_available"].any())


class BadCaseQualitativeComparisonTest(TestCase):
    @staticmethod
    def _summary() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "IMG_ID": [1, 2, 3, 4],
                "status": ["erro", "erro", "erro", "erro"],
                "ostia_status": [
                    "both_correct",
                    "both_tolerable",
                    "not_found",
                    "both_correct",
                ],
                "ostia_found": [True, True, False, True],
                "left_intersects": [True, False, False, True],
                "right_intersects": [False, False, False, False],
                "both_correct": [True, False, False, True],
                "both_tolerable": [False, True, False, False],
                "dice_artery": [0.1, 0.2, 0.0, 0.15],
            }
        )

    def test_selection_is_reproducible_for_the_same_seed(self):
        split_paths = {
            "high_res": {"test": "unused"},
            "mid_res": {"test": "unused"},
        }
        exported = pd.DataFrame(
            {
                "image_id": [1, 2, 3, 4],
                "bad_case_status": ["low_dice", "low_dice", "error", "low_dice"],
                "subset": ["test"] * 4,
            }
        )

        with TemporaryDirectory() as tmp_dir:
            for resolution in ("high", "mid"):
                exported.to_csv(
                    f"{tmp_dir}/bad_cases_test_{resolution}_res.csv",
                    index=False,
                )

            with patch(
                "utils.comparison_utils.io.load_split_summary",
                return_value=self._summary(),
            ):
                first = prepare_bad_case_qualitative_comparison(
                    split_paths,
                    tmp_dir,
                    samples_per_group=1,
                    random_seed=7,
                )
                second = prepare_bad_case_qualitative_comparison(
                    split_paths,
                    tmp_dir,
                    samples_per_group=1,
                    random_seed=7,
                )

        pd.testing.assert_frame_equal(
            first["selected_image_plan"].reset_index(drop=True),
            second["selected_image_plan"].reset_index(drop=True),
        )
        self.assertEqual(first["resolutions"], ("high", "mid"))
        self.assertEqual(
            len(first["selected_cases"]),
            2 * len(first["selected_image_ids"]),
        )
        self.assertIn("intersection_group", first["selected_cases"].columns)

