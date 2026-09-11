import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib.pyplot as plt
import pandas as pd

from utils.visualization.variant_comparison import (
    build_pair_curve_auc,
    load_variant_run,
    plot_pair_delta_by_image,
    plot_pair_dice_by_image,
)


class VariantComparisonCurveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pair_results = pd.DataFrame(
            {
                "folder_variant": ["baseline"] * 3 + ["best"] * 3,
                "IMG_ID": [3, 1, 2, 3, 1, 2],
                "artery_dice": [0.3, 0.1, 0.2, 0.5, 0.2, 0.4],
            }
        )

    def test_build_pair_curve_auc_orders_images_and_normalizes(self) -> None:
        auc_df = build_pair_curve_auc(
            self.pair_results,
            "baseline",
            "best",
        ).set_index("curve")

        self.assertAlmostEqual(auc_df.loc["reference", "auc"], 0.4)
        self.assertAlmostEqual(auc_df.loc["reference", "normalized_auc"], 0.2)
        self.assertAlmostEqual(auc_df.loc["comparison", "auc"], 0.75)
        self.assertAlmostEqual(auc_df.loc["comparison", "normalized_auc"], 0.375)
        self.assertAlmostEqual(auc_df.loc["delta", "auc"], 0.35)
        self.assertAlmostEqual(auc_df.loc["delta", "normalized_auc"], 0.175)

    def test_pair_curve_plots_save_figures(self) -> None:
        plots = [
            (plot_pair_dice_by_image, "dice.png"),
            (plot_pair_delta_by_image, "delta.png"),
        ]
        with TemporaryDirectory() as tmp:
            for plot_function, filename in plots:
                with self.subTest(filename=filename):
                    save_path = Path(tmp) / filename
                    plot_result = plot_function(
                        self.pair_results,
                        "baseline",
                        "best",
                        save_path=save_path,
                    )

                    self.assertTrue(save_path.is_file())
                    if plot_function is plot_pair_dice_by_image:
                        axes = plot_result
                        self.assertEqual(len(axes), 2)
                        self.assertEqual(axes[0].get_ylim(), (0.0, 1.0))
                        self.assertEqual(axes[1].get_ylim(), (0.0, 1.0))
                        self.assertEqual(
                            axes[1].get_xlabel(),
                            "Exames",
                        )
                        plt.close(axes[0].figure)
                    else:
                        ax = plot_result
                        self.assertEqual(ax.get_ylim(), (-1.0, 1.0))
                        self.assertEqual(
                            ax.get_xlabel(),
                            "Exames",
                        )
                        plt.close(ax.figure)

    def test_variant_labels_come_from_metadata_not_result_rows(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            numeric_dir = Path(temporary_dir) / "test/baseline/2026-01-01/numeric"
            numeric_dir.mkdir(parents=True)
            results_path = numeric_dir / "results_test.csv"
            pd.DataFrame(
                {
                    "IMG_ID": [1],
                    "artery_dice": [0.5],
                    "ostia_detected": ["yes"],
                    "ostia_detection_status": ["both correct"],
                }
            ).to_csv(results_path, index=False)
            (numeric_dir / "metadata_test.json").write_text(
                json.dumps(
                    {
                        "configuration": {
                            "threshold_method": "fuzzy",
                            "artery_segmentation_method": "fc",
                        },
                    }
                ),
                encoding="utf-8",
            )

            results, summary = load_variant_run(results_path)

        self.assertNotIn("threshold_mode", results.columns)
        self.assertNotIn("artery_segmentation_method", results.columns)
        self.assertEqual(summary["threshold_mode"], "fuzzy")
        self.assertEqual(summary["artery_method"], "fc")


if __name__ == "__main__":
    unittest.main()
