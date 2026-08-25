import unittest

import matplotlib.pyplot as plt
import pandas as pd

from utils.visualization.intensity import (
    calculate_binned_intensity_mean_median,
    plot_binned_intensity_histogram,
)


class IntensityVisualizationTests(unittest.TestCase):
    def test_calculates_weighted_binned_mean_and_median(self) -> None:
        histogram = pd.DataFrame(
            {"bin_center_hu": [0.0, 100.0, 200.0], "count": [1, 2, 1]}
        )

        mean_hu, median_hu = calculate_binned_intensity_mean_median(histogram)

        self.assertAlmostEqual(mean_hu, 100.0)
        self.assertAlmostEqual(median_hu, 100.0)

    def test_plots_distribution_with_numeric_mean_and_median_labels(self) -> None:
        histogram = pd.DataFrame(
            {"bin_center_hu": [0.0, 100.0], "count": [2, 1]}
        )
        fig, ax = plt.subplots()
        try:
            plot_binned_intensity_histogram(
                ax,
                histogram,
                mean_hu=33.3,
                median_hu=0.0,
            )
            labels = ax.get_legend_handles_labels()[1]
        finally:
            plt.close(fig)

        self.assertIn("Média: 33.3 HU", labels)
        self.assertIn("Mediana: 0.0 HU", labels)


if __name__ == "__main__":
    unittest.main()
