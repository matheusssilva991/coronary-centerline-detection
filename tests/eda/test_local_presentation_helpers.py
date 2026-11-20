import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from eda.comparisons.view.ia_vs_pipeline_views import plot_comparison_bar_by_resolution
from eda.comparisons.view.segmentation_method_views import (
    OSTIA_STATUS_COLUMNS,
    plot_ostia_status_by_variant,
)
from eda.results.view.split_resolution_views import plot_subset_metric_by_resolution
from tests.notebook_helpers import load_presentation_helpers


class LocalPresentationTest(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_notebooks_load_without_dataset_access(self):
        for name in (
            "split_resolution_analysis",
            "segmentation_results_eda",
            "ia_vs_pipeline_analysis",
            "segmentation_method_comparison",
            "aorta_hu_threshold_comparison",
            "aorta_automatic_quality_assessment",
            "aorta_volume_quality_analysis",
        ):
            with (
                self.subTest(notebook=name),
                patch(
                    "pandas.read_csv", side_effect=AssertionError("acesso ao dataset")
                ),
            ):
                namespace = load_presentation_helpers(name)
                self.assertTrue(
                    any(
                        key.startswith(("plot_", "show_", "build_"))
                        for key in namespace
                    )
                )

    @patch("matplotlib.pyplot.show")
    def test_subset_graph_preserves_values_and_labels(self, _show):
        frame = pd.DataFrame(
            {
                "disponivel": [True, True],
                "subset": ["train", "test"],
                "resolucao": ["Mid", "Mid"],
                "score": [0.3, 0.8],
            }
        )
        axis = plot_subset_metric_by_resolution(
            frame, "score", "Dice", "Score", ["blue"]
        )
        self.assertEqual(
            [bar.get_height() for bar in axis.patches if bar.get_width() > 0],
            [0.3, 0.8],
        )
        self.assertEqual(
            [tick.get_text() for tick in axis.get_xticklabels()], ["Treino", "Teste"]
        )

    @patch("matplotlib.pyplot.show")
    def test_ia_comparison_keeps_means_by_method(self, _show):
        frame = pd.DataFrame(
            {
                "target_resolution": ["mid", "mid"],
                "source": ["ia", "math"],
                "method": ["model", "pipeline"],
                "mean_dice": [0.8, 0.4],
                "std_dice": [0.1, 0.2],
            }
        )
        plot_comparison_bar_by_resolution(frame, "mid")
        axis = plt.gca()
        heights = sorted(
            bar.get_height()
            for bar in axis.patches
            if isinstance(bar, Rectangle) and bar.get_width() > 0
        )
        self.assertEqual(heights, [0.4, 0.8])
        self.assertEqual(axis.get_ylim(), (0.0, 1.05))

    def test_variant_status_plot_keeps_counts(self):
        row: dict[str, str | int] = {"folder_variant": "baseline"}
        for index, column in enumerate(OSTIA_STATUS_COLUMNS):
            row[column] = index + 1
        axis = plot_ostia_status_by_variant(pd.DataFrame([row]))
        self.assertEqual(
            [bar.get_height() for bar in axis.patches],
            list(range(1, len(OSTIA_STATUS_COLUMNS) + 1)),
        )
