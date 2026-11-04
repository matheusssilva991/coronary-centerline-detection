"""Valida fontes opcionais do notebook sem executar segmentações."""

import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import Mock, mock_open, patch

import nbformat
import numpy as np
import pandas as pd


NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "src/eda/results/external_ccta_visual_assessment.ipynb"
)


class ExternalAssessmentNotebookTest(unittest.TestCase):
    def setUp(self):
        self.cells = nbformat.read(NOTEBOOK, as_version=4).cells
        self.scope = {
            "pd": pd,
            "display": Mock(),
            "print": Mock(),
            "Path": Path,
            "load_assessment_subset_lookup": Mock(),
            "load_external_visual_assessments": Mock(),
        }

    def test_missing_sources_skip_dependent_cells(self):
        import os

        self.scope["os"] = os
        with patch.object(Path, "is_file", return_value=False):
            exec(self.cells[3].source, self.scope)
        self.assertFalse(self.scope["VISUAL_AVAILABLE"])
        self.assertFalse(self.scope["MMWHS_NUMERIC_AVAILABLE"])
        self.scope["load_assessment_subset_lookup"].assert_not_called()
        for index in (5, 18, 20, 22, 24, 25, 26):
            exec(self.cells[index].source, self.scope)
        self.assertEqual(self.scope["QUALITATIVE_SUBSET"], "train")
        self.assertEqual(self.scope["QUALITATIVE_RESOLUTION"], "mid")
        self.assertEqual(self.scope["QUALITATIVE_EXAM_ID"], "ct_train_1007")

    def test_numeric_sources_do_not_require_visual_sheets(self):
        import os

        self.scope["os"] = os
        with patch.object(
            Path,
            "is_file",
            autospec=True,
            side_effect=lambda path: str(path).endswith("results_all.csv"),
        ):
            exec(self.cells[3].source, self.scope)
        self.assertTrue(self.scope["MMWHS_NUMERIC_AVAILABLE"])
        self.assertFalse(self.scope["VISUAL_AVAILABLE"])
        loader = Mock(return_value=pd.Series([0.8], index=["ct_train_1001"]))
        self.scope["load_mmwhs_train_aorta_dice"] = loader
        exec(self.cells[18].source, self.scope)
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(self.scope["dice_by_case"].shape, (1, 2))

    def test_legacy_test_protocol_is_rejected(self):
        series = pd.Series([0.8], index=["ct_test_2001"])
        series.attrs["aorta_evaluation_method"] = "mmwhs_official_1mm_wine"
        self.scope.update(
            MMWHS_NUMERIC_AVAILABLE=True,
            RUN_DIRS={r: {"MM-WHS": Path("unused")} for r in ("mid", "high")},
            load_mmwhs_test_aorta_dice=Mock(return_value=series),
        )
        with self.assertRaisesRegex(ValueError, "label 820"):
            exec(self.cells[24].source, self.scope)

    def test_notebook_validates_and_compiles(self):
        nbformat.validate(nbformat.read(NOTEBOOK, as_version=4))
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in self.cells:
                if cell.cell_type == "code":
                    compile(cell.source, str(NOTEBOOK), "exec")

    def test_valid_and_common_dice_preserve_evaluated_zeros(self):
        ids = [f"ct_test_{2001 + i}" for i in range(40)]
        mid = pd.Series([0.0] + [0.8] * 33 + [float("nan")] * 6, index=ids)
        high = pd.Series([0.0] + [0.9] * 39, index=ids)
        for series in (mid, high):
            series.attrs.update(
                aorta_evaluation_method="mmwhs_aorta_label820_1mm_wine",
                aorta_evaluation_label="Label 820",
            )
        self.scope.update(
            MMWHS_NUMERIC_AVAILABLE=True,
            RUN_DIRS={r: {"MM-WHS": Path("unused")} for r in ("mid", "high")},
            load_mmwhs_test_aorta_dice=Mock(side_effect=[mid, high]),
        )
        exec(self.cells[24].source, self.scope)
        exec(self.cells[25].source, self.scope)
        overall = self.scope["test_dice_overall"]
        common = self.scope["test_dice_common_summary"]
        self.assertEqual(overall.loc["mid", "Exames válidos"], 34)
        self.assertEqual(overall.loc["high", "Exames válidos"], 40)
        self.assertAlmostEqual(overall.loc["mid", "Dice válido médio"], 33 * 0.8 / 34)
        self.assertAlmostEqual(
            overall.loc["mid", "Dice geral (ausências = 0)"], 33 * 0.8 / 40
        )
        self.assertEqual(common["Exames comuns"].tolist(), [34, 34])
        self.assertAlmostEqual(common.loc["high", "Dice comum médio"], 33 * 0.9 / 34)
        self.assertEqual(common.loc["mid", "Mínimo"], 0)

    def test_optional_overlay_does_not_run_when_disabled(self):
        self.scope.update(SHOW_QUALITATIVE=False, predict_aorta=Mock())
        cell = next(c for c in self.cells if c.get("id") == "aorta-3d-overlay")
        exec(cell.source, self.scope)
        self.scope["predict_aorta"].assert_not_called()

    def test_display_exports_existing_plot_without_segmentation(self):
        plot = Mock()
        writer = Mock()
        self.scope.update(
            SHOW_QUALITATIVE=True,
            qual_plot=plot,
            qual_plots={"overlay": plot},
            qual_exam_id="ct_train_1007",
            QUALITATIVE_RESOLUTION="mid",
            qual_run=Path("external/run"),
            save_k3d_plot_html=writer,
            predict_aorta=Mock(),
        )
        cell = next(c for c in self.cells if c.get("id") == "aorta-3d-display")
        exec(cell.source, self.scope)
        writer.assert_called_once_with(
            plot, Path("external/run/visual/ct_train_1007_aorta_overlay.html")
        )
        self.scope["predict_aorta"].assert_not_called()

    def test_overlay_reuses_pipeline_and_native_grid(self):
        import json

        mask = np.ones((4, 5, 3), dtype=bool)
        inventory = pd.DataFrame(
            [
                {
                    "exam_id": "ct_train_1001",
                    "subset": "train",
                    "spacing_x_mm": 0.5,
                    "spacing_y_mm": 0.6,
                    "spacing_z_mm": 1.0,
                }
            ]
        )
        plotter = Mock()
        predictor = Mock(return_value="prediction")
        self.scope.update(
            SHOW_QUALITATIVE=True,
            MMWHS_NUMERIC_AVAILABLE=True,
            QUALITATIVE_RESOLUTION="mid",
            QUALITATIVE_EXAM_ID=None,
            QUALITATIVE_SUBSET="train",
            QUALITATIVE_REFERENCE_PATH=None,
            MESH_SMOOTHING_ITERATIONS=2,
            MMWHS_DATA_ROOT=None,
            EXTERNAL_RESULTS_ROOT=Path("disk/Results_dataset_ccta"),
            resolve_existing_path=Mock(return_value=Path("dataset")),
            np=np,
            json=json,
            RUN_DIRS={"mid": {"MM-WHS": Path("run")}},
            dice_by_case=pd.DataFrame({"mid": [0.8]}, index=["ct_train_1001"]),
            discover_ccta_dataset=Mock(return_value=inventory),
            load_ccta_aorta_ground_truth=Mock(return_value=mask),
            predict_aorta=predictor,
            restore_native_mask=Mock(return_value=mask),
            mask_bounding_box_slices=Mock(return_value=(slice(None),) * 3),
            visualize_binary_masks_comparison=plotter,
        )
        cell = next(c for c in self.cells if c.get("id") == "aorta-3d-overlay")
        with (
            patch.object(Path, "is_dir", return_value=True),
            patch.object(
                Path,
                "open",
                mock_open(read_data='{"orcascore_visual_alignment": false}'),
            ),
        ):
            exec(cell.source, self.scope)
        self.assertFalse(predictor.call_args.kwargs["align_volume"])
        self.scope["resolve_existing_path"].assert_called_once_with(
            "MMWHS_BASE_PATH",
            [Path("disk/DatasetsCCTA/MM-WHS-2017-Dataset")],
            "MM-WHS",
        )
        self.assertEqual(plotter.call_args.args[0].shape, (5, 4, 3))
        self.assertEqual(plotter.call_args.kwargs["spacing"], (0.6, 0.5, 1.0))
        self.assertFalse(plotter.call_args.kwargs["display_plot"])
        self.assertEqual(plotter.call_count, 1)
        self.assertEqual(list(self.scope["qual_plots"]), ["overlay"])
        self.assertEqual(plotter.call_args.kwargs["reference_opacity"], 0.35)
        self.assertEqual(plotter.call_args.kwargs["predicted_opacity"], 0.25)

        # A ausência da referência de teste nunca gera uma referência fictícia visível.
        self.scope["QUALITATIVE_SUBSET"] = "test"
        self.scope["test_dice_by_case"] = self.scope["dice_by_case"]
        inventory["subset"] = "test"
        self.scope["load_ccta_aorta_ground_truth"].return_value = None
        plotter.reset_mock()
        with (
            patch.object(Path, "is_dir", return_value=True),
            patch.object(
                Path,
                "open",
                mock_open(read_data='{"orcascore_visual_alignment": false}'),
            ),
        ):
            exec(cell.source, self.scope)
        self.assertEqual(list(self.scope["qual_plots"]), ["prediction"])
        self.assertEqual(plotter.call_args.kwargs["reference_opacity"], 0.0)


if __name__ == "__main__":
    unittest.main()
