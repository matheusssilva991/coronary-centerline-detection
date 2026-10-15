"""Verifica coortes e tratamento de falhas nos CSVs de sweep."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.comparison_utils.io import load_experiment_variant_results
from utils.comparison_utils.ia_math import load_math_results_for_comparison
from utils.comparison_utils.ostia_scenarios import load_math_results_for_ostia_scenario


class ExperimentResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "results").mkdir()
        self.frame = pd.DataFrame(
            {
                "variant": ["normal_rg", "normal_rg", "other"],
                "split": ["test", "test", "train"],
                "IMG_ID": [1, 2, 3],
                "dice_artery": [0.7, None, 0.9],
                "ostia_status": ["both_tolerable", "error", "both_correct"],
                "error": [None, "aorta vazia", None],
                "segmentation_attempted": [True, False, True],
            }
        )

    def load(self):
        self.frame.to_csv(self.root / "results/image_results.csv", index=False)
        return load_experiment_variant_results(
            self.root, variant="normal_rg", split="test"
        )

    def test_failure_is_zero_only_in_general_scenario(self):
        result = self.load()
        self.assertEqual(result["IMG_ID"].tolist(), [1, 2])
        self.assertEqual(result["artery_dice"].tolist(), [0.7, 0.0])
        self.assertEqual(result["dice_artery"].tolist(), [0.7, 0.0])
        self.assertTrue(pd.isna(result.iloc[1]["evaluable_dice"]))
        self.assertEqual(result.iloc[0]["status"], "both_tolerable")

    def test_existing_failure_zero_is_not_an_evaluable_dice(self):
        self.frame.loc[1, "dice_artery"] = 0.0
        self.assertTrue(pd.isna(self.load().iloc[1]["evaluable_dice"]))

    def test_missing_dice_without_failure_is_rejected(self):
        self.frame.loc[1, "error"] = None
        with self.assertRaisesRegex(ValueError, "sem falha"):
            self.load()

    def test_duplicate_ids_are_rejected(self):
        self.frame.loc[1, "IMG_ID"] = 1
        with self.assertRaisesRegex(ValueError, "repetidos"):
            self.load()

    def test_absent_variant_is_rejected(self):
        self.frame["variant"] = "other"
        with self.assertRaisesRegex(ValueError, "ausente"):
            self.load()

    def test_dataframe_consumers_preserve_failures_and_success_scenarios(self):
        source = self.load()
        original = source.copy(deep=True)
        sources = {"mid_res": {"test": source}}
        general, missing = load_math_results_for_comparison(sources)
        self.assertFalse(missing)
        self.assertEqual(general["dice"].tolist(), [0.7, 0.0])
        correct, _ = load_math_results_for_ostia_scenario(sources, "correct")
        incorrect, _ = load_math_results_for_ostia_scenario(sources, "incorrect")
        self.assertEqual(correct["img_id"].tolist(), [1])
        self.assertEqual(incorrect["img_id"].tolist(), [2])
        pd.testing.assert_frame_equal(source, original)

    def test_wrong_ostia_detail_uses_exact_intersection_not_tolerance(self):
        self.frame.loc[0, "ostia_status"] = "found_but_wrong"
        self.frame["left_dist_mm"] = [0.0, None, 0.0]
        self.frame["right_dist_mm"] = [9.0, None, 0.0]
        self.assertEqual(self.load().iloc[0]["status"], "one_correct")
        self.frame.loc[0, "left_dist_mm"] = 1.0
        self.assertEqual(self.load().iloc[0]["status"], "none_correct")
