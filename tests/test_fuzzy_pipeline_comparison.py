"""Regression tests for the fuzzy pipeline comparison experiment."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.fuzzy_pipeline_comparison import make_diagnostics
from utils.experiments.fuzzy_pipeline_comparison import run_image


def _diagnostic_row(variant: str) -> dict:
    return {
        "IMG_ID": 10,
        "variant": variant,
        "dice_artery": np.nan,
        "dice_artery_before_morphology": np.nan,
        "dice_artery_after_morphology": np.nan,
        "dice_artery_morphology_delta": np.nan,
        "ostia_status": "error",
        "ostia_success": False,
        "artery_voxels": 0,
        "left_dist_mm": np.nan,
        "right_dist_mm": np.nan,
    }


class DiagnosticsTest(TestCase):
    def test_keeps_pair_columns_when_one_variant_has_only_nan_dice(self):
        rows = [
            _diagnostic_row("normal_rg"),
            _diagnostic_row("normal_threshold_fc"),
        ]

        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            make_diagnostics(run_dir, rows)
            output = pd.read_csv(run_dir / "diagnostics/fc_vs_rg_deltas.csv")

        self.assertIn("normal_rg", output.columns)
        self.assertIn("normal_threshold_fc", output.columns)
        self.assertTrue(output["fc_minus_rg"].isna().all())


class RunImageTest(TestCase):
    @patch("utils.experiments.fuzzy_pipeline_comparison.postprocess_artery_mask")
    @patch("utils.experiments.fuzzy_pipeline_comparison.normal_region_growing_from_ostia")
    @patch("utils.experiments.fuzzy_pipeline_comparison.detect_and_evaluate_ostia")
    @patch("utils.experiments.fuzzy_pipeline_comparison.segment_aorta")
    @patch("utils.experiments.fuzzy_pipeline_comparison.locate_aorta_circles")
    @patch("utils.experiments.fuzzy_pipeline_comparison.compute_vesselness")
    @patch("utils.experiments.fuzzy_pipeline_comparison.build_preprocessed_inputs")
    @patch("utils.experiments.fuzzy_pipeline_comparison.load_downsampled_case")
    def test_passes_detected_circles_to_bilateral_ostia_selection(
        self,
        load_case,
        build_inputs,
        compute_vesselness,
        detect_circles,
        segment_aorta,
        detect_ostia,
        region_growing,
        postprocess,
    ):
        volume = np.ones((2, 2, 2), dtype=np.float32)
        label = np.ones_like(volume, dtype=np.uint8)
        circles = [{"center_x": 1, "center_y": 1, "radius": 1, "slice_index": 1}]
        load_case.return_value = {
            "down_image": volume,
            "down_label": label,
            "scaled_spacing": (1.0, 1.0, 1.0),
            "downscale_factors": (2, 2, 1),
        }
        build_inputs.return_value = (volume, volume.astype(bool), {})
        compute_vesselness.side_effect = [volume, volume]
        detect_circles.return_value = circles
        segment_aorta.return_value = label
        detect_ostia.return_value = {
            "both_correct": True,
            "both_tolerable": True,
            "left_info": {"physical_dist": 0.0},
            "right_info": {"physical_dist": 0.0},
            "label_artery": label,
            "ostia_left": (1, 1, 1),
            "ostia_right": (1, 1, 1),
        }
        region_growing.return_value = label
        postprocess.return_value = label
        config = {
            "USE_GPU": False,
            "VESSELNESS_AORTA": {},
            "VESSELNESS_ARTERY": {},
            "CIRCLE_DETECTION": {},
            "LEVEL_SET": {},
        }

        result = run_image(
            10,
            "normal_rg",
            "val",
            Path("/unused"),
            config,
            {"threshold_mode": "normal", "artery_method": "region_growing"},
        )

        self.assertIsNone(result["error"])
        self.assertEqual(
            detect_ostia.call_args.kwargs["detected_circles"],
            circles,
        )

    @patch("utils.experiments.fuzzy_pipeline_comparison.postprocess_artery_mask")
    @patch("utils.experiments.fuzzy_pipeline_comparison.normal_region_growing_from_ostia")
    @patch("utils.experiments.fuzzy_pipeline_comparison.detect_and_evaluate_ostia")
    @patch("utils.experiments.fuzzy_pipeline_comparison.segment_aorta")
    @patch("utils.experiments.fuzzy_pipeline_comparison.locate_aorta_circles")
    @patch("utils.experiments.fuzzy_pipeline_comparison.compute_vesselness")
    @patch("utils.experiments.fuzzy_pipeline_comparison.build_preprocessed_inputs")
    @patch("utils.experiments.fuzzy_pipeline_comparison.load_downsampled_case")
    def test_ostia_only_skips_arterial_vesselness_and_segmentation(
        self,
        load_case,
        build_inputs,
        compute_vesselness,
        detect_circles,
        segment_aorta,
        detect_ostia,
        region_growing,
        postprocess,
    ):
        volume = np.ones((2, 2, 2), dtype=np.float32)
        label = np.ones_like(volume, dtype=np.uint8)
        load_case.return_value = {
            "down_image": volume,
            "down_label": label,
            "scaled_spacing": (1.0, 1.0, 1.0),
            "downscale_factors": (1, 1, 1),
        }
        build_inputs.return_value = (volume, volume.astype(bool), {})
        compute_vesselness.return_value = volume
        detect_circles.return_value = []
        segment_aorta.return_value = label
        detect_ostia.return_value = {
            "both_correct": True,
            "both_tolerable": True,
            "left_info": {"physical_dist": 0.0},
            "right_info": {"physical_dist": 0.0},
            "label_artery": label,
            "ostia_left": (1, 1, 1),
            "ostia_right": (1, 1, 1),
        }
        config = {
            "USE_GPU": False,
            "VESSELNESS_AORTA": {},
            "VESSELNESS_ARTERY": {},
            "CIRCLE_DETECTION": {},
            "LEVEL_SET": {},
        }

        result = run_image(
            10,
            "ostia_only",
            "val",
            Path("/unused"),
            config,
            {"threshold_mode": "normal", "ostia_only": True},
        )

        self.assertIsNone(result["error"])
        self.assertTrue(result["ostia_success"])
        self.assertFalse(result["segmentation_attempted"])
        self.assertEqual(compute_vesselness.call_count, 1)
        region_growing.assert_not_called()
        postprocess.assert_not_called()
