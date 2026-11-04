"""Valida os estados Plotly sem kernel interativo nem dados volumétricos."""

from pathlib import Path
import unittest

import nbformat
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde


class VolumeDicePlotlyTest(unittest.TestCase):
    def setUp(self):
        notebook = nbformat.read(
            Path(__file__).resolve().parents[2]
            / "src/eda/images/artery_volume_analysis.ipynb",
            as_version=4,
        )
        self.scope = {
            "go": go,
            "make_subplots": make_subplots,
            "volume_plot_xlim": (0, 15),
            "np": np,
            "pd": pd,
            "gaussian_kde": gaussian_kde,
            "PLOT_SPLIT": "test",
        }
        cell = next(c for c in notebook.cells if c.id == "volume-dice-presentation")
        exec(cell.source, self.scope)
        self.data = pd.DataFrame(
            {
                "IMG_ID": [1, 2, 3, 1, 2, 4],
                "resolution": ["mid_res"] * 3 + ["high_res"] * 3,
                "volume_ml": [2, 4, 12, 2, 4, 8],
                "dice": [0.8, 0, 0.1, 0.7, 0.2, 0.9],
                "ostia_success": [True, True, False, True, False, True],
                "is_volume_outlier": [False, False, True, False, False, False],
                "volume_outlier_group": ["in_range", "in_range", "high"]
                + ["in_range"] * 3,
            }
        )
        self.pairs = pd.DataFrame(
            {
                "IMG_ID": [1, 2],
                "ostia_success_both": [True, False],
                "dice_mid": [0.8, 0],
                "dice_high": [0.7, 0.2],
                "delta_dice": [-0.1, 0.2],
                "volume_ml": [2, 4],
                "ostia_success_mid": [True, True],
                "ostia_success_high": [True, False],
                "is_volume_outlier": [False, False],
                "volume_outlier_group": ["in_range"] * 2,
            }
        )

    def _plot(self, data=None, pairs=None):
        return self.scope["plot_volume_dice_panels"](
            self.data if data is None else data,
            self.pairs if pairs is None else pairs,
            "Volume × Dice",
        )

    def test_cohort_menu_preserves_ids_counts_and_limits(self):
        original = self.data.copy(deep=True)
        fig = self._plot()
        buttons = fig.layout.updatemenus[0].buttons
        self.assertEqual(
            [button.label for button in buttons], ["Todos", "Óstios válidos em ambas"]
        )
        for button, ids, count in ((buttons[0], {1, 2, 3, 4}, 6), (buttons[1], {1}, 2)):
            visible = button.args[0]["visible"]
            traces = [trace for trace, shown in zip(fig.data, visible) if shown]
            self.assertEqual(sum(len(trace.x) for trace in traces), count)
            self.assertEqual(
                {item[0] for trace in traces for item in trace.customdata}, ids
            )
            n = 3 if button is buttons[0] else 1
            titles = [a["text"] for a in button.args[1]["annotations"]]
            self.assertEqual(titles, [f"Mid (N={n})", f"High (N={n})"])
        self.assertEqual(fig.layout.xaxis.range, (0, 15))
        self.assertEqual(fig.layout.xaxis2.range, (0, 15))
        self.assertEqual(fig.layout.yaxis.range, (0, 1))
        self.assertEqual(fig.layout.yaxis2.range, (0, 1))
        pd.testing.assert_frame_equal(self.data, original)

    def test_hover_and_markers_preserve_classes_and_zero_dice(self):
        fig = self._plot()
        traces = [trace for trace in fig.data if trace.visible and len(trace.x)]
        self.assertTrue(any(0 in trace.y for trace in traces))
        self.assertEqual({trace.marker.symbol for trace in traces}, {"circle", "x"})
        self.assertEqual(
            {trace.marker.color for trace in traces}, {"#247A58", "#C44747"}
        )
        self.assertTrue(all("Dice:" in trace.hovertemplate for trace in traces))
        self.assertTrue(
            all(len(item) == 5 for trace in traces for item in trace.customdata)
        )

    def test_empty_conditioned_cohort_keeps_clear_annotation(self):
        pairs = self.pairs.assign(ostia_success_both=False)
        fig = self._plot(pairs=pairs)
        button = fig.layout.updatemenus[0].buttons[1]
        annotations = button.args[1]["annotations"]
        self.assertIn("Nenhum exame", annotations[-1]["text"])
        self.assertEqual(
            sum(
                len(t.x)
                for t, shown in zip(fig.data, button.args[0]["visible"])
                if shown
            ),
            0,
        )
        self.assertIsNone(self._plot(data=self.data.iloc[:0]))

    def test_absent_resolution_has_zero_count_without_imputation(self):
        data = self.data.loc[self.data.resolution.eq("mid_res")]
        fig = self._plot(data=data, pairs=self.pairs.iloc[:0])
        self.assertEqual(fig.layout.annotations[1].text, "High (N=0)")

    def test_histogram_counts_statistics_and_constant_values(self):
        plot = self.scope["plot_volume_histogram"]
        limits = {"exam_count": 4, "lower_limit_ml": 0, "upper_limit_ml": 10}
        values = pd.DataFrame({"volume_ml": [1, 2, 3, 8]})
        fig = plot(values, limits, 5)
        self.assertEqual(sum(fig.data[0].y), 4)
        self.assertEqual(len(fig.data[0].x), 5)
        self.assertEqual(list(fig.data[2].x), [3.5, 3.5])
        self.assertEqual(list(fig.data[3].x), [2.5, 2.5])
        constant = plot(values.assign(volume_ml=2), limits, 5)
        self.assertFalse(
            any(trace.name == "Densidade suavizada" for trace in constant.data)
        )
        self.assertEqual(sum(constant.data[0].y), 4)
        self.assertIsNone(plot(values.iloc[:0], limits, 5))

    def test_delta_menu_preserves_sign_and_detailed_hover(self):
        fig = self.scope["plot_volume_dice_delta"](self.pairs)
        buttons = fig.layout.updatemenus[0].buttons
        for button, expected in ((buttons[0], [-0.1, 0.2]), (buttons[1], [-0.1])):
            traces = [
                t for t, visible in zip(fig.data, button.args[0]["visible"]) if visible
            ]
            self.assertEqual(sorted(y for t in traces for y in t.y), expected)
        self.assertEqual(fig.layout.yaxis.range, (-1, 1))
        self.assertEqual(fig.layout.shapes[0].y0, 0)
        self.assertIn("Dice Mid", fig.data[0].hovertemplate)
        self.assertTrue(all(len(item) == 6 for t in fig.data for item in t.customdata))
        empty = self.pairs.assign(ostia_success_both=False)
        fig = self.scope["plot_volume_dice_delta"](empty)
        self.assertIn(
            "Nenhum exame",
            fig.layout.updatemenus[0].buttons[1].args[1]["annotations"][0]["text"],
        )

    def test_predicted_volume_panels_have_identity_lines_and_equal_scales(self):
        data = self.data.assign(predicted_volume_ml=self.data.volume_ml * 1.2)
        fig = self.scope["plot_volume_dice_panels"](
            data,
            self.pairs,
            "Predição × referência",
            y_column="predicted_volume_ml",
        )
        self.assertEqual(len(fig.layout.shapes), 2)
        for shape in fig.layout.shapes:
            self.assertEqual(shape.x0, shape.y0)
            self.assertEqual(shape.x1, shape.y1)
        self.assertEqual(fig.layout.xaxis.range, fig.layout.yaxis.range)
        self.assertEqual(fig.layout.xaxis.range, fig.layout.xaxis2.range)
        self.assertIn("Predição:", fig.data[0].hovertemplate)

    def test_band_plot_keeps_counts_dispersion_and_success(self):
        from utils.comparison_utils.artery_volume import summarize_artery_volume_bands

        summary, _ = summarize_artery_volume_bands(self.data)
        fig = self.scope["plot_volume_band_performance"](summary)
        self.assertEqual(len(fig.data), 4)
        self.assertIn("DP", fig.layout.annotations[0].text)
        self.assertEqual(fig.layout.yaxis2.range, (0, 100))
        self.assertEqual(
            sum(item[2] for t in fig.data if t.showlegend for item in t.customdata), 6
        )
        self.assertIsNone(self.scope["plot_volume_band_performance"](pd.DataFrame()))


if __name__ == "__main__":
    unittest.main()
