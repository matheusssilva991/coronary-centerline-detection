"""Tests for shared helpers extracted from EDA notebooks."""

from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from utils.comparison_utils.metadata import (
    build_split_resolution_summary,
    get_execution_time_seconds,
    get_num_images,
    get_total_success_percent,
)


class SplitResolutionSummaryTest(TestCase):
    def test_metadata_helpers_read_schema_v3(self):
        metadata = {
            "results": {
                "execution_time": {"seconds": 90},
                "ostia": {
                    "processed_exam_count": 4,
                    "success": {"count": 2, "percent": 50.0},
                },
            }
        }

        self.assertEqual(get_execution_time_seconds(metadata), 90)
        self.assertEqual(get_num_images(metadata), 4)
        self.assertEqual(get_total_success_percent(metadata), 50.0)

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

        def load_results(_, resolution, subset):
            return summaries[subset] if resolution == "mid_res" else None

        def load_timings(_, resolution, _subset):
            if resolution != "mid_res":
                return None
            return pd.DataFrame({"batch_number": [1], "duration_seconds": [120]})

        split_paths = {"mid_res": {}, "high_res": {}}
        with (
            patch(
                "utils.comparison_utils.io.load_split_results",
                side_effect=load_results,
            ),
            patch(
                "utils.comparison_utils.io.load_split_batch_timings",
                side_effect=load_timings,
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
