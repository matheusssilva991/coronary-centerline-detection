import os
import subprocess
import sys
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.comparison_utils.variant_comparison import (
    build_dice_stats_by_variant,
    largest_pair_changes,
    make_pair_delta,
)
from utils.visualization.results.variant_comparison import (
    prepare_variant_for_plot,
)


class AnalysisPresentationSeparationTests(unittest.TestCase):
    def setUp(self):
        self.results = pd.DataFrame(
            {
                "folder_variant": ["baseline"] * 3 + ["best"] * 3,
                "variant_label": ["Baseline"] * 3 + ["Melhor"] * 3,
                "IMG_ID": [3, 1, 2, 3, 1, 2],
                "artery_dice": [0.3, 0.1, 0.2, 0.5, 0.2, 0.1],
            }
        )

    def tearDown(self):
        plt.close("all")

    def test_analysis_imports_without_presentation_dependencies(self):
        root = Path(__file__).resolve().parents[2]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(root / "src")
        code = """
import importlib.abc
import sys

class BlockPresentation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'matplotlib', 'IPython'} or fullname.startswith('utils.visualization'):
            raise AssertionError(fullname)

sys.meta_path.insert(0, BlockPresentation())
from utils.comparison_utils.segmentation_eda import build_success_status_summary_by_subset
from utils.comparison_utils.variant_comparison import make_pair_delta
assert build_success_status_summary_by_subset(None, 'test', [])['quantidade'].tolist() == [0, 0]
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_variant_statistics_keep_values_types_order_and_input(self):
        original = self.results.copy(deep=True)
        actual = build_dice_stats_by_variant(
            self.results, ["best", "baseline"]
        ).reset_index(drop=True)
        expected = pd.DataFrame(
            {
                "folder_variant": ["best", "baseline"],
                "variant_label": ["Melhor", "Baseline"],
                "mean_dice": [np.mean([0.5, 0.2, 0.1]), 0.2],
                "max_dice": [0.5, 0.3],
                "min_dice": [0.1, 0.1],
                "std_dice": [np.std([0.5, 0.2, 0.1], ddof=1), 0.1],
                "median_dice": [0.2, 0.2],
            }
        )
        pd.testing.assert_frame_equal(actual, expected)
        pd.testing.assert_frame_equal(self.results, original)

    def test_pairing_deltas_and_largest_changes_keep_exam_order(self):
        pair = make_pair_delta(self.results, "baseline", "best")
        self.assertEqual(pair["IMG_ID"].tolist(), [3, 1, 2])
        np.testing.assert_allclose(pair["dice_delta"], [0.2, 0.1, -0.1])
        changes = largest_pair_changes(self.results, "baseline", "best", top_n=2)
        self.assertEqual(changes["IMG_ID"].tolist(), [3, 1])

    def test_visual_preparation_preserves_labels_and_categorical_order(self):
        source = pd.DataFrame({"folder_variant": ["baseline", "best"]})
        prepared = prepare_variant_for_plot(
            source, ["best", "baseline"], {"best": "Melhor", "baseline": "Base"}
        )
        self.assertEqual(prepared["folder_variant"].tolist(), ["best", "baseline"])
        self.assertEqual(prepared["variant_label"].tolist(), ["Melhor", "Base"])
        self.assertIsInstance(prepared["variant_label"].dtype, pd.CategoricalDtype)
        self.assertNotIn("variant_label", source)

    def test_historical_facades_point_to_the_same_analysis_functions(self):
        import utils.comparison_utils as comparison
        import utils.visualization as visualization

        self.assertIs(visualization.make_pair_delta, make_pair_delta)
        self.assertIs(visualization.largest_pair_changes, largest_pair_changes)
        self.assertIs(comparison.make_pair_delta, make_pair_delta)
