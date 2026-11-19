import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

from utils.comparison_utils.segmentation_eda import (
    artery_surface_distance_cases,
    build_success_status_summary_by_subset,
    pair_artery_surface_distance_cases,
    summarize_artery_surface_distance_cases,
    summarize_surface_distances_with_successful_ostia,
)


from tests.notebook_helpers import load_presentation_helpers

_presentation = load_presentation_helpers("segmentation_results_eda")
plot_status_distribution_by_subset = _presentation["plot_status_distribution_by_subset"]
plot_success_error_by_subset = _presentation["plot_success_error_by_subset"]


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
        bar_heights = [
            bar.get_height() for bar in mid_axis.patches if isinstance(bar, Rectangle)
        ]

        show_mock.assert_called_once()
        self.assertEqual(bar_heights, [1, 1])
        plt.close("all")


class ArterySurfaceComparisonTests(unittest.TestCase):
    @staticmethod
    def _results(ids, statuses, hd, ahd):
        return pd.DataFrame(
            {
                "IMG_ID": ids,
                "artery_distance_metrics_status": statuses,
                "artery_hd_mm": hd,
                "artery_ahd_mm": ahd,
            }
        )

    def test_comparison_pairs_by_id_using_both_valid_distances(self):
        source = self._results(
            [1, 2, 3, 4, 5],
            ["ok", "ok", "prediction_empty", "ok", "ok"],
            [8, 10, 0, 7, 5],
            [1, 2, 0, None, 6],
        ).assign(dice_artery=[0.4, 0.6, 0.0, 0.1, 0.2])
        mid = artery_surface_distance_cases(source)
        self.assertAlmostEqual(
            summarize_artery_surface_distance_cases(source, mid)["Dice médio"], 0.5
        )
        high = artery_surface_distance_cases(
            self._results([2, 6, 1], ["ok"] * 3, [20, 11, 18], [3, 7, 2])
        )

        paired = pair_artery_surface_distance_cases(mid, high)
        self.assertEqual(paired.index.tolist(), [1, 2])
        self.assertEqual(paired["artery_hd_mm_high"].tolist(), [18, 20])
        self.assertEqual(mid.index.tolist(), [1, 2])

    def test_legacy_reference_keeps_exam_count_without_inventing_distances(self):
        legacy = pd.DataFrame({"IMG_ID": [1, 2]})
        cases = artery_surface_distance_cases(legacy)

        summary = summarize_artery_surface_distance_cases(legacy, cases)
        self.assertEqual(summary["Exames no CSV"], 2)
        self.assertEqual(summary["Exames válidos"], 0)
        self.assertEqual(summary["Cobertura"], "Indisponível")
        self.assertTrue(pd.isna(summary["HD média (mm)"]))

    def test_distances_with_ostia_success_include_tolerable_and_exclude_missing_metrics(
        self,
    ):
        source = self._results(
            [1, 2, 3, 4, 5],
            ["ok", "ok", "not_evaluated", "ok", "ok"],
            [10, 90, None, 70, 6],
            [2, 1, None, 3, 1],
        ).assign(
            ostia_status=[
                "both_correct",
                "both_tolerable",
                "both_correct",
                "found_but_wrong",
                "ambos corretos",
            ]
        )
        cases = artery_surface_distance_cases(source)

        summary = summarize_surface_distances_with_successful_ostia(source, cases)

        self.assertEqual(summary["Ambos corretos ou toleráveis"], 4)
        self.assertEqual(summary["Exames com HD e AHD válidos"], 3)
        self.assertAlmostEqual(summary["HD média (mm)"], 106 / 3)
        self.assertAlmostEqual(summary["AHD média (mm)"], 4 / 3)
        self.assertEqual(summary["AHD mediana (mm)"], 1)


if __name__ == "__main__":
    unittest.main()
