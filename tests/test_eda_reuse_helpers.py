"""Tests for shared helpers extracted from EDA notebooks."""

from unittest import TestCase
from unittest.mock import patch

import pandas as pd

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
