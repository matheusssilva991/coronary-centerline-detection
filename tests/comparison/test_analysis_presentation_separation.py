import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd

from utils.comparison_utils.segmentation_eda import build_dice_summary_by_subset
from utils.comparison_utils.variant_comparison import (
    build_dice_stats_by_variant,
    build_pair_curve_auc,
    largest_pair_changes,
    load_variant_results,
    make_pair_delta,
)
from utils.visualization.results.variant_comparison import (
    plot_largest_pair_changes,
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
from utils.comparison_utils.segmentation_eda import build_dice_summary_by_subset
from utils.comparison_utils.variant_comparison import make_pair_delta
assert build_dice_summary_by_subset(None, 'test')['count'].tolist() == [0, 0]
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

    def test_dice_summary_preserves_missing_values_and_sample_deviation(self):
        data = {
            "high": {"test": pd.DataFrame({"dice_artery": [0.4, "invalid", None, 0.8]})}
        }
        summary = build_dice_summary_by_subset(data, "test")
        self.assertEqual(summary["resolution"].tolist(), ["HIGH", "MID"])
        self.assertEqual(summary["count"].tolist(), [2, 0])
        self.assertAlmostEqual(summary.iloc[0]["mean"], 0.6)
        self.assertAlmostEqual(summary.iloc[0]["std"], np.std([0.4, 0.8], ddof=1))
        self.assertTrue(pd.isna(summary.iloc[1]["mean"]))

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

    def test_pairing_deltas_largest_changes_and_auc_keep_exam_order(self):
        pair = make_pair_delta(self.results, "baseline", "best")
        self.assertEqual(pair["IMG_ID"].tolist(), [3, 1, 2])
        np.testing.assert_allclose(pair["dice_delta"], [0.2, 0.1, -0.1])
        changes = largest_pair_changes(self.results, "baseline", "best", top_n=2)
        self.assertEqual(changes["IMG_ID"].tolist(), [3, 1])
        auc = build_pair_curve_auc(self.results, "baseline", "best").set_index("curve")
        self.assertAlmostEqual(auc.loc["reference", "normalized_auc"], 0.2)
        self.assertAlmostEqual(auc.loc["comparison", "normalized_auc"], 0.225)
        self.assertAlmostEqual(auc.loc["delta", "normalized_auc"], 0.025)

    def test_current_and_legacy_results_keep_metadata_labels(self):
        import json

        for filename in ("results_test.csv", "ostios_test_summary.csv"):
            with (
                self.subTest(filename=filename),
                tempfile.TemporaryDirectory() as directory,
            ):
                numeric = Path(directory) / "test/baseline/run/numeric"
                numeric.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "IMG_ID": [3, 1],
                        "dice_artery": [0.2, 0.6],
                        "ostia_detected": ["yes", "no"],
                        "ostia_detection_status": ["both correct", "ostia not found"],
                    }
                ).to_csv(numeric / filename, index=False)
                (numeric / "metadata_test.json").write_text(
                    json.dumps(
                        {
                            "configuration": {
                                "threshold_method": "fuzzy",
                                "artery_segmentation_method": "fc",
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                frame, summary = load_variant_results(Path(directory), split="test")
                self.assertEqual(frame["IMG_ID"].tolist(), [3, 1])
                self.assertEqual(summary.iloc[0]["threshold_mode"], "fuzzy")
                self.assertEqual(summary.iloc[0]["artery_method"], "fc")
                self.assertAlmostEqual(summary.iloc[0]["mean_dice"], 0.4)
                self.assertEqual(summary.iloc[0]["both_correct_n"], 1)
                self.assertNotIn("threshold_mode", frame.columns)

    @patch("utils.visualization.results.variant_comparison.largest_pair_changes")
    def test_plot_uses_analysis_output_without_recalculating_pairs(self, changes):
        changes.return_value = pd.DataFrame(
            {"IMG_ID": [9, 2], "dice_delta": [0.2, -0.3]}
        )
        axis = plot_largest_pair_changes(
            self.results, "baseline", "best", title="Delta", top_n=2
        )
        changes.assert_called_once_with(self.results, "baseline", "best", top_n=2)
        self.assertEqual(
            [label.get_text() for label in axis.get_yticklabels()], ["2", "9"]
        )
        np.testing.assert_allclose(
            [bar.get_width() for bar in axis.patches], [-0.3, 0.2]
        )

    def test_plot_preserves_colors_labels_axes_return_and_save(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "figures/delta.png"
            _, existing_axis = plt.subplots()
            axis = plot_largest_pair_changes(
                self.results,
                "baseline",
                "best",
                title="Maiores mudanças",
                ax=existing_axis,
                save_path=path,
            )
            self.assertIs(axis, existing_axis)
            self.assertTrue(path.is_file())
            self.assertEqual(axis.get_title(), "Maiores mudanças")
            self.assertEqual(axis.get_xlabel(), "Delta Dice: comparação - referência")
            self.assertEqual(axis.get_ylabel(), "IMG_ID")
            self.assertEqual(
                [text.get_text() for text in axis.texts], ["-0.100", "+0.100", "+0.200"]
            )
            self.assertEqual(
                [bar.get_facecolor() for bar in axis.patches],
                [to_rgba("#d62728"), to_rgba("#2ca02c"), to_rgba("#2ca02c")],
            )

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
        import utils
        import utils.comparison_utils as comparison
        import utils.visualization as visualization

        self.assertIs(visualization.make_pair_delta, make_pair_delta)
        self.assertIs(
            visualization.build_dice_summary_by_subset, build_dice_summary_by_subset
        )
        self.assertIs(
            visualization.plot_largest_pair_changes, plot_largest_pair_changes
        )
        self.assertIs(comparison.make_pair_delta, make_pair_delta)
        self.assertIs(
            utils.add_pair_ostia_status_groups, comparison.add_pair_ostia_status_groups
        )
