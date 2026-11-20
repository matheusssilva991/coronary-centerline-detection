import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import pandas as pd

from eda.comparisons.view.segmentation_method_views import (
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


if __name__ == "__main__":
    unittest.main()
