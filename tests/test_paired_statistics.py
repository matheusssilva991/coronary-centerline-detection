import json
import unittest
from pathlib import Path

import pandas as pd

from utils.comparison_utils.paired_statistics import adjust_holm, compare_paired_dice
from utils.project.config import deep_update_dict


def frame(ids, values):
    return pd.DataFrame({"IMG_ID": ids, "artery_dice": values})


class PairedStatisticsTests(unittest.TestCase):
    def test_pairs_align_by_id_and_keep_zero_dice(self):
        result = compare_paired_dice(
            frame([1, 2, 3], [0, 0.3, 0.4]), frame([2, 1, 4], [0.4, 0, 0.8])
        )
        self.assertEqual(result["paired_images"], 2)
        self.assertEqual(result["excluded_pairs"], 2)
        self.assertAlmostEqual(result["mean_delta_dice"], 0.05)
        self.assertAlmostEqual(result["baseline_median_dice"], 0.15)
        self.assertAlmostEqual(result["candidate_median_dice"], 0.2)
        self.assertLessEqual(result["mean_delta_ci_95_low"], 0.05)
        self.assertGreaterEqual(result["mean_delta_ci_95_high"], 0.05)
        self.assertEqual(result["unchanged_images"], 1)
        self.assertEqual(result["rank_biserial_effect"], 1)

    def test_all_ties_and_roundoff(self):
        result = compare_paired_dice(
            frame([1, 2], [0, 0.5]), frame([1, 2], [0, 0.5 + 1e-15])
        )
        self.assertEqual(result["p_value"], 1)
        self.assertEqual(result["unchanged_images"], 2)
        self.assertAlmostEqual(result["mean_delta_ci_95_low"], 0)
        self.assertAlmostEqual(result["mean_delta_ci_95_high"], 0)

    def test_bootstrap_is_reproducible(self):
        baseline = frame([1, 2, 3, 4], [0.1, 0.2, 0.4, 0.8])
        candidate = frame([1, 2, 3, 4], [0.2, 0.15, 0.6, 0.7])
        first = compare_paired_dice(
            baseline,
            candidate,
            bootstrap_samples=500,
            random_state=7,
        )
        second = compare_paired_dice(
            baseline,
            candidate,
            bootstrap_samples=500,
            random_state=7,
        )
        self.assertEqual(
            first["mean_delta_ci_95_low"], second["mean_delta_ci_95_low"]
        )
        self.assertEqual(
            first["mean_delta_ci_95_high"], second["mean_delta_ci_95_high"]
        )

    def test_invalid_inputs(self):
        baseline = frame([1], [0.5])
        for other in (
            frame([1, 1], [0.5, 0.6]),
            frame([2], [0.5]),
            frame([1], [float("inf")]),
        ):
            with self.assertRaises(ValueError):
                compare_paired_dice(baseline, other)

    def test_holm_preserves_index(self):
        adjusted = adjust_holm(pd.Series([0.04, 0.001, 0.03], index=[4, 1, 9]))
        self.assertEqual(list(adjusted.index), [4, 1, 9])
        for actual, expected in zip(adjusted, [0.06, 0.003, 0.06]):
            self.assertAlmostEqual(actual, expected)


class PromotedConfigTests(unittest.TestCase):
    def test_defaults_and_article_isolation(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "config/pipeline_config.json").read_text())
        overrides = json.loads(
            (root / "config/aorta_filter_envelope_generalization.json").read_text()
        )
        merged = deep_update_dict(json.loads(json.dumps(config)), overrides)
        self.assertEqual(config, merged)
        self.assertEqual(config["OSTIA_DETECTION"]["lower_fraction"], 1)
        self.assertEqual(config["OSTIA_DETECTION"]["surface_padding_radius"], 2)
        article = deep_update_dict(
            config,
            json.loads((root / "config/article_cbeb_sensitivity.json").read_text()),
        )
        self.assertEqual(
            article["CIRCLE_DETECTION"]["trajectory_filter"]["method"], "none"
        )
        self.assertIsNone(article["LEVEL_SET"]["trajectory_radius_factor"])
        self.assertEqual(article["LEVEL_SET"]["num_iter"], 31)
        self.assertEqual(article["OSTIA_DETECTION"]["surface_padding_radius"], 0)
