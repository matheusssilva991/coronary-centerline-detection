import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd

from utils.visualization.segmentation_eda import (
    build_success_status_summary_by_subset,
    plot_status_distribution_by_subset,
    plot_success_error_by_subset,
)


class SegmentationEdaSuccessTests(unittest.TestCase):
    def setUp(self):
        self.success_status = ["both_tolerable", "both_correct"]

    def test_summary_normalizes_mid_readable_english_statuses(self):
        data = {
            "mid": {
                "test": pd.DataFrame(
                    {
                        "ostia_status": [
                            "both_tolerable",
                            "both_correct",
                            "found_but_wrong",
                        ],
                    }
                )
            }
        }

        summary = build_success_status_summary_by_subset(
            data, "test", self.success_status
        )
        mid_total = summary.query(
            "resolution == 'mid' and status == 'total acertos'"
        ).iloc[0]

        self.assertEqual(mid_total["quantidade"], 2)
        self.assertEqual(mid_total["percentual_do_total"], 66.67)
        self.assertEqual(
            summary.query("resolution == 'mid'")["status"].tolist(),
            ["ambos toleráveis", "ambos corretos", "total acertos"],
        )

    def test_summary_keeps_legacy_portuguese_statuses_compatible(self):
        data = {
            "high": {
                "test": pd.DataFrame(
                    {
                        "status": [
                            "ambos toleráveis",
                            "ambos corretos",
                            "nenhum correto",
                        ]
                    }
                )
            }
        }

        summary = build_success_status_summary_by_subset(
            data, "test", self.success_status
        )
        high_total = summary.query(
            "resolution == 'high' and status == 'total acertos'"
        ).iloc[0]

        self.assertEqual(high_total["quantidade"], 2)
        self.assertEqual(high_total["percentual_do_total"], 66.67)

    @patch("matplotlib.pyplot.show")
    def test_status_distribution_translates_codes_to_portuguese(self, show_mock):
        data = {
            "mid": {
                "train": pd.DataFrame({"status": ["both_tolerable", "none_correct"]})
            }
        }

        plot_status_distribution_by_subset(data, "train")
        labels = [label.get_text() for label in plt.gcf().axes[1].get_xticklabels()]

        show_mock.assert_called_once()
        self.assertEqual(labels, ["ambos toleráveis", "nenhum correto"])
        plt.close("all")

    @patch("matplotlib.pyplot.show")
    def test_success_plot_uses_the_same_normalized_rule(self, show_mock):
        data = {
            "mid": {
                "train": pd.DataFrame(
                    {
                        "status": ["both ostia tolerable", "no ostium correct"],
                        "ostia_status": ["both_tolerable", "found_but_wrong"],
                    }
                )
            }
        }

        plot_success_error_by_subset(data, "train", self.success_status)
        mid_axis = plt.gcf().axes[1]
        bar_heights = [bar.get_height() for bar in mid_axis.patches]

        show_mock.assert_called_once()
        self.assertEqual(bar_heights, [1, 1])
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
