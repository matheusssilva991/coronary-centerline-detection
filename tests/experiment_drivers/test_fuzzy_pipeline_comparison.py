"""Regression tests for the fuzzy pipeline comparison experiment."""

from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.fuzzy_pipeline_comparison import make_diagnostics
from utils.experiments.fuzzy_pipeline_comparison import (
    evaluate_prepared_image,
    make_image_result_row,
    prepare_image_context,
    run_image,
    save_outputs,
    summarize_variant,
)
from utils.segmentation.pipeline import arteries
from utils.segmentation.pipeline.detection import segment_aorta
from utils.segmentation.pipeline.arteries import (
    evaluate_artery_postprocessing_stages,
    get_artery_postprocessing_stages,
)


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
        "error": None,
        "artery_voxels": 0,
        "left_dist_mm": np.nan,
        "right_dist_mm": np.nan,
    }


class DiagnosticsTest(TestCase):
    def test_new_diagnostics_preserve_masks_and_use_configured_classifier(self):
        volume = np.zeros((24, 24, 9), dtype=np.float32)
        circles = [
            {"center_x": 10, "center_y": 11, "radius": 6, "slice_index": z}
            for z in range(2, 7)
        ]
        config = {
            "VESSELNESS_AORTA": {},
            "VESSELNESS_ARTERY": {},
            "CIRCLE_DETECTION": {},
            "LEVEL_SET": {
                "num_iter": 2,
                "radius_reduction_factor": 0.5,
                "balloon": 0.6,
                "smoothing": 1,
                "leak_removal_radius": 1,
                "quality_feedback": {"oversegmentation_area_ratio_p90": -1},
            },
        }
        case = {
            "down_image": volume,
            "down_label": volume,
            "scaled_spacing": (1, 1, 1),
            "downscale_factors": (2, 2, 1),
        }
        expected = segment_aorta(volume, circles, config["LEVEL_SET"])
        target = "utils.experiments.fuzzy_pipeline_comparison."
        with (
            patch(
                target + "build_preprocessed_inputs",
                return_value=(volume, volume > 0, {}),
            ),
            patch(target + "compute_vesselness", return_value=volume),
            patch(target + "locate_aorta_circles", return_value=circles),
            patch(
                target + "filter_located_aorta_circles",
                return_value=(circles, {"aorta_circle_used_count": 5}),
            ),
        ):
            context = prepare_image_context(case, config, {})
        np.testing.assert_array_equal(context["aorta_mask"], expected)
        metrics = context["aorta_diagnostics"]
        self.assertEqual(metrics["aorta_circle_count"], 5)
        self.assertEqual(metrics["aorta_circle_used_count"], 5)
        self.assertEqual(metrics["aorta_mask_voxels"], int(expected.sum()))
        self.assertEqual(
            metrics["aorta_segmentation_feedback"], "suspected_oversegmentation"
        )
        self.assertEqual(metrics["aorta_level_set_iterations_used"], 2)

    def test_csv_preserves_all_available_fields_including_future_metrics(self):
        row = make_image_result_row(1, "baseline", "train", {})
        row.update(
            aorta_segmentation_feedback="adequate",
            aorta_level_set_circle_fill_q25=0.9,
            future_metric=42,
        )
        with TemporaryDirectory() as directory:
            save_outputs(Path(directory), [], [row], [])
            frame = pd.read_csv(Path(directory) / "results/image_results.csv")
        self.assertEqual(frame["aorta_segmentation_feedback"].item(), "adequate")
        self.assertEqual(frame["aorta_level_set_circle_fill_q25"].item(), 0.9)
        self.assertEqual(frame["future_metric"].item(), 42)

    def test_diagnostics_export_available_intermediate_metrics(self):
        row = {
            **_diagnostic_row("normal_rg"),
            "dice_artery": 0.6,
            "dice_artery_after_closing": 0.8,
            "dice_artery_closing_delta": 0.3,
            "dice_artery_dilation_delta": -0.2,
            "artery_voxels_after_closing": 100,
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            make_diagnostics(root, [row])
            saved = pd.read_csv(root / "diagnostics/best_by_image.csv")
        self.assertEqual(saved.loc[0, "dice_artery_after_closing"], 0.8)
        self.assertEqual(saved.loc[0, "dice_artery_dilation_delta"], -0.2)
        self.assertEqual(saved.loc[0, "artery_voxels_after_closing"], 100)

    def test_summary_counts_only_available_intermediate_measurements(self):
        legacy = _diagnostic_row("normal_rg")
        measured = {
            **legacy,
            "IMG_ID": 11,
            "dice_artery_after_closing": 0.8,
            "dice_artery_closing_delta": 0.3,
            "dice_artery_dilation_delta": -0.2,
            "artery_voxels_after_closing": 100,
            "ostia_found": True,
            "both_correct": True,
            "both_tolerable": False,
        }
        legacy.update(ostia_found=False, both_correct=False, both_tolerable=False)
        summary = summarize_variant("normal_rg", [legacy, measured], 1)
        self.assertEqual(summary["dice_artery_after_closing_count"], 1)
        self.assertEqual(summary["dice_artery_after_closing_mean"], 0.8)
        self.assertEqual(summary["dice_artery_dilation_delta_mean"], -0.2)
        missing = summarize_variant("normal_rg", [legacy], 1)
        self.assertIsNone(missing["dice_artery_after_closing_mean"])
        self.assertEqual(missing["dice_artery_after_closing_count"], 0)
        failed = make_image_result_row(1, "normal_rg", "train", {})
        self.assertTrue(pd.isna(failed["artery_voxels_after_closing"]))
        self.assertTrue(pd.isna(failed["dice_artery_after_closing"]))

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
    def test_prepared_rg_and_fc_measure_all_stages_with_one_morphology_pass(self):
        raw = np.zeros((15, 15, 15), dtype=np.uint8)
        raw[5:10, 5:10, 5:10] = 1
        label = raw.copy()
        raw[7, 7, 7] = 0
        config = {"POSTPROCESSING": {"closing_radius": 1, "dilation_radius": 1}}
        stages = get_artery_postprocessing_stages(raw, config)
        expected = evaluate_artery_postprocessing_stages(
            stages["raw_mask"],
            stages["closed_mask"],
            stages["final_mask"],
            label_artery=label,
        )
        context = {
            "prep_details": {},
            "volume_voxels": raw.size,
            "volume_slices": 15,
            "detected_circles": [],
            "aorta_voxels": 0,
            "aorta_volume_fraction": 0,
            "aorta_mask": raw,
            "vesselness_ostios": raw,
            "vesselness_artery": raw,
            "down_label": label,
            "spacing": (1, 1, 1),
            "lcc_image": raw,
            "lcc_mask": raw > 0,
        }
        ostia = {
            "both_correct": True,
            "both_tolerable": False,
            "left_info": {"physical_dist": 0},
            "right_info": {"physical_dist": 0},
            "label_artery": label,
            "ostia_left": (7, 7, 7),
            "ostia_right": (7, 7, 8),
        }
        for method in ("region_growing", "fuzzy_connectedness"):
            with (
                self.subTest(method=method),
                patch(
                    "utils.experiments.fuzzy_pipeline_comparison.detect_and_evaluate_ostia",
                    return_value=ostia,
                ),
                patch(
                    "utils.experiments.fuzzy_pipeline_comparison.segment_artery_fuzzy_connectedness",
                    return_value={"raw_mask": raw, "details": {}},
                ) as fc,
                patch(
                    "utils.segmentation.pipeline.arteries._segment_with_region_growing",
                    return_value=(raw, {}),
                ),
                patch(
                    "utils.segmentation.pipeline.arteries.binary_closing",
                    wraps=arteries.binary_closing,
                ) as closing,
                patch(
                    "utils.segmentation.pipeline.arteries.binary_dilation",
                    wraps=arteries.binary_dilation,
                ) as dilation,
                patch(
                    "utils.experiments.fuzzy_pipeline_comparison.save_segmentation_visual"
                ) as visual,
            ):
                experiment = {
                    "artery_method": method,
                    "fc": {"alpha": 0.27, "sigma_hu": 55},
                }
                row = make_image_result_row(1, method, "train", experiment)
                result = evaluate_prepared_image(context, row, config, experiment)
                closing.assert_called_once()
                dilation.assert_called_once()
                visual.assert_not_called()
                for column, value in expected.items():
                    self.assertEqual(result[column], value)
                if method == "fuzzy_connectedness":
                    self.assertFalse(fc.call_args.kwargs["apply_postprocessing"])
                    self.assertEqual(fc.call_args.kwargs["params"]["alpha"], 0.27)
                    self.assertEqual(fc.call_args.kwargs["params"]["sigma_hu"], 55)

    @patch(
        "utils.experiments.fuzzy_pipeline_comparison.segment_arteries_from_vesselness"
    )
    @patch("utils.experiments.fuzzy_pipeline_comparison.detect_and_evaluate_ostia")
    @patch("utils.experiments.fuzzy_pipeline_comparison.segment_aorta_with_diagnostics")
    @patch("utils.experiments.fuzzy_pipeline_comparison.locate_aorta_circles")
    @patch("utils.experiments.fuzzy_pipeline_comparison.compute_vesselness")
    @patch("utils.experiments.fuzzy_pipeline_comparison.build_preprocessed_inputs")
    @patch("utils.experiments.fuzzy_pipeline_comparison.load_downsampled_case")
    def test_detects_ostia_with_standard_selection_interface(
        self,
        load_case,
        build_inputs,
        compute_vesselness,
        detect_circles,
        segment_aorta,
        detect_ostia,
        segment_arteries,
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
        segment_aorta.return_value = SimpleNamespace(mask=label, diagnostics={})
        detect_ostia.return_value = {
            "both_correct": True,
            "both_tolerable": True,
            "left_info": {"physical_dist": 0.0},
            "right_info": {"physical_dist": 0.0},
            "label_artery": label,
            "ostia_left": (1, 1, 1),
            "ostia_right": (1, 1, 1),
        }
        segment_arteries.return_value = {
            "artery_mask": label,
            "raw_artery_mask": label,
            **evaluate_artery_postprocessing_stages(
                label, label, label, label_artery=label
            ),
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
            "normal_rg",
            "val",
            Path("/unused"),
            config,
            {"threshold_mode": "normal", "artery_method": "region_growing"},
        )

        self.assertIsNone(result["error"])
        self.assertNotIn("detected_circles", detect_ostia.call_args.kwargs)

    @patch(
        "utils.experiments.fuzzy_pipeline_comparison.segment_arteries_from_vesselness"
    )
    @patch("utils.experiments.fuzzy_pipeline_comparison.detect_and_evaluate_ostia")
    @patch("utils.experiments.fuzzy_pipeline_comparison.segment_aorta_with_diagnostics")
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
        segment_arteries,
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
        segment_aorta.return_value = SimpleNamespace(mask=label, diagnostics={})
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
        segment_arteries.assert_not_called()
