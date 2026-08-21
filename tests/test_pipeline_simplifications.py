"""Regression tests for behavior-preserving pipeline simplifications."""

from contextlib import redirect_stdout
from io import StringIO
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import segmentation_pipeline
from utils.segmentation.pipeline_orchestration import (
    _circle_result_fields,
    _ostia_result_fields,
    _preprocessing_result_fields,
    _resolve_batch_plan,
    run_pipeline,
    summarize_aorta_circles,
    summarize_aorta_volume,
)
from utils.segmentation.pipeline_preprocessing import (
    compute_vesselness,
    load_and_preprocess_image,
)
from utils.segmentation.pipeline_reporting import print_split_summary


class _FakeNifti:
    def __init__(self, data):
        self._data = np.asarray(data)
        self.header = Mock()
        self.header.get_zooms.return_value = (0.5, 0.5, 1.0)

    def get_fdata(self):
        return self._data


def _preprocessing_config(method):
    return {
        "DOWNSCALE_FACTORS": [2, 2, 1],
        "DOWNSCALE_METHOD": "opencv",
        "OPENCV_INTERPOLATION": "linear",
        "MAX_THRESHOLD_PERCENTILE": 99.8,
        "MIN_THRESHOLD": -300,
        "LOWER_THRESHOLD": {
            "method": "fixed",
            "fixed_hu": -300,
            "percentile": 10.5,
        },
        "THRESHOLDING": {"method": method, "fuzzy": {}},
    }


class PipelineSimplificationTests(TestCase):
    @patch("utils.segmentation.pipeline_preprocessing.downscale_image_ndi")
    @patch("utils.segmentation.pipeline_preprocessing.largest_connected_component")
    @patch("utils.segmentation.pipeline_preprocessing.threshold_image_with_offset")
    @patch("utils.segmentation.pipeline_preprocessing.resolve_lower_threshold")
    @patch("utils.segmentation.pipeline_preprocessing.downscale_image_opencv")
    @patch("utils.segmentation.pipeline_preprocessing.load_raw_img_and_label")
    def test_normal_threshold_downscales_intensity_once(
        self,
        load_raw,
        downscale_opencv,
        resolve_lower,
        threshold,
        largest_component,
        downscale_ndi,
    ):
        image = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        label = np.ones_like(image, dtype=np.uint8)
        load_raw.return_value = (_FakeNifti(image), _FakeNifti(label))
        downscale_opencv.return_value = image
        downscale_ndi.return_value = label
        resolve_lower.return_value = (-300.0, {"lower_threshold_method": "fixed"})
        threshold.return_value = (image + 300.0, np.ones_like(image, bool), 300.0)
        largest_component.side_effect = lambda image_slice, mask_slice: (
            image_slice,
            mask_slice,
        )

        result = load_and_preprocess_image(
            "1",
            "/dataset",
            _preprocessing_config("normal"),
        )

        self.assertEqual(downscale_opencv.call_count, 1)
        self.assertEqual(downscale_ndi.call_count, 1)
        np.testing.assert_array_equal(result["lcc_image"], image)

    @patch("utils.segmentation.pipeline_preprocessing.downscale_image_ndi")
    @patch("utils.segmentation.pipeline_preprocessing.build_lcc_image_from_mask")
    @patch("utils.segmentation.pipeline_preprocessing.fuzzy_threshold_from_config")
    @patch("utils.segmentation.pipeline_preprocessing.resolve_lower_threshold")
    @patch("utils.segmentation.pipeline_preprocessing.downscale_image_opencv")
    @patch("utils.segmentation.pipeline_preprocessing.load_raw_img_and_label")
    def test_fuzzy_threshold_downscales_intensity_once(
        self,
        load_raw,
        downscale_opencv,
        resolve_lower,
        fuzzy_threshold,
        build_lcc,
        downscale_ndi,
    ):
        image = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        label = np.ones_like(image, dtype=np.uint8)
        mask = image > 5
        load_raw.return_value = (_FakeNifti(image), _FakeNifti(label))
        downscale_opencv.return_value = image
        downscale_ndi.return_value = label
        resolve_lower.return_value = (-300.0, {"lower_threshold_method": "fixed"})
        fuzzy_threshold.return_value = (mask, {"fuzzy_mask_strategy": "object_argmax"})
        build_lcc.return_value = (image, mask)

        result = load_and_preprocess_image(
            "1",
            "/dataset",
            _preprocessing_config("fuzzy"),
        )

        self.assertEqual(downscale_opencv.call_count, 1)
        self.assertEqual(downscale_ndi.call_count, 1)
        np.testing.assert_array_equal(result["lcc_image"], image)

    def test_result_field_helpers_preserve_pipeline_schema(self):
        preprocessing = _preprocessing_result_fields(
            {"threshold_mode": "fuzzy", "min_threshold": -280.0},
            120,
            256 * 256 * 120,
        )
        self.assertEqual(preprocessing["threshold_mode"], "fuzzy")
        self.assertEqual(preprocessing["image_slice_count"], 120)
        self.assertEqual(preprocessing["image_voxels"], 256 * 256 * 120)

        aorta_volume = summarize_aorta_volume(
            np.array([[[1, 0], [1, 0]], [[1, 0], [0, 0]]], dtype=np.uint8),
            8,
        )
        self.assertEqual(aorta_volume["aorta_mask_voxels"], 3)
        self.assertEqual(aorta_volume["aorta_volume_fraction"], 3 / 8)

        circles = _circle_result_fields(
            [
                {"slice_index": 9},
                {"slice_index": 8, "interpolated": True},
                {"slice_index": 7, "recovered_initialization": True},
            ],
            10,
        )
        self.assertEqual(circles["aorta_circle_count"], 3)
        self.assertEqual(circles["aorta_detected_circle_count"], 2)
        self.assertEqual(circles["aorta_circle_coverage"], 0.3)
        self.assertTrue(circles["aorta_recovered_initialization"])
        self.assertEqual(
            circles,
            summarize_aorta_circles(
                [
                    {"slice_index": 9},
                    {"slice_index": 8, "interpolated": True},
                    {"slice_index": 7, "recovered_initialization": True},
                ],
                10,
            ),
        )

        ostia = _ostia_result_fields(
            {
                "ostia_left": (1, 2, 3),
                "ostia_right": (4, 5, 6),
                "left_info": {
                    "intersects": True,
                    "euclidean_dist": 0.0,
                    "physical_dist": 0.0,
                },
                "right_info": {
                    "intersects": False,
                    "euclidean_dist": 2.0,
                    "physical_dist": 1.5,
                },
                "both_correct": False,
                "both_tolerable": True,
            }
        )
        self.assertEqual(ostia["ostia_status"], "both_tolerable")
        self.assertFalse(ostia["proceeded_with_bad_ostia"])

    @patch("builtins.print")
    def test_cli_override_helpers_keep_experimental_parameters(self, _print):
        config = {
            "LOAD_CACHE": True,
            "SAVE_CACHE": True,
            "ARTERY_SEGMENTATION": {"method": "region_growing"},
            "REGION_GROWING": {"comparison_window": 1},
            "THRESHOLDING": {"method": "normal", "fuzzy": {}},
            "LOWER_THRESHOLD": {"method": "fixed"},
        }
        args = SimpleNamespace(
            downscale_method="opencv",
            opencv_interpolation="area",
            use_gpu=False,
            artery_segmentation_method="fc",
            rg_comparison_window=-1,
            threshold_method="fuzzy",
            upper_threshold_percentile=99.9,
            lower_threshold_method="percentile",
            lower_threshold_percentile=10.5,
            lower_threshold_clip_min=-650.0,
            lower_threshold_clip_max=450.0,
        )

        segmentation_pipeline._apply_execution_overrides(config, args)
        segmentation_pipeline._apply_threshold_overrides(config, args)

        self.assertNotIn("LOAD_CACHE", config)
        self.assertNotIn("SAVE_CACHE", config)
        self.assertEqual(config["DOWNSCALE_METHOD"], "opencv")
        self.assertEqual(config["OPENCV_INTERPOLATION"], "area")
        self.assertEqual(config["ARTERY_SEGMENTATION"]["method"], "fc")
        self.assertEqual(config["REGION_GROWING"]["comparison_window"], -1)
        self.assertEqual(config["THRESHOLDING"]["method"], "fuzzy")
        self.assertEqual(config["MAX_THRESHOLD_PERCENTILE"], 99.9)
        self.assertEqual(
            config["THRESHOLDING"]["fuzzy"]["lower_percentile"],
            10.5,
        )

    @patch("utils.segmentation.pipeline_preprocessing.get_vesselness")
    def test_vesselness_computation_forwards_config(self, get_vesselness):
        expected = np.ones((2, 2, 2), dtype=np.float32)
        get_vesselness.return_value = expected
        vesselness_config = {
            "sigmas": [1.0],
            "alpha": 0.5,
            "beta": 0.5,
            "gamma": 5.0,
        }

        result = compute_vesselness(expected, vesselness_config)

        np.testing.assert_array_equal(result, expected)
        get_vesselness.assert_called_once()

    def test_batch_plan_keeps_one_based_resume_semantics(self):
        plan = _resolve_batch_plan(
            list(range(20)),
            {"NUM_BATCHES": 5},
            resume_from_batch=3,
        )
        self.assertEqual(plan, (5, 4, 2))

    def test_console_summary_uses_canonical_aggregate_metrics(self):
        results = pd.DataFrame(
            {
                "ostia_detected": ["yes", "yes"],
                "ostia_detection_status": ["both correct", "both tolerable"],
                "both_ostia_correct": ["yes", "no"],
                "both_ostia_tolerable": ["no", "yes"],
                "artery_segmentation_run": ["yes", "yes"],
                "segmented_with_incorrect_ostia": ["no", "no"],
                "artery_dice": [0.6, 0.8],
                "artery_dice_before_morphology": [0.5, 0.7],
                "artery_dice_morphology_delta": [0.1, 0.1],
            }
        )
        output = StringIO()

        with redirect_stdout(output):
            print_split_summary(
                results,
                "val",
                {"OSTIA_VALIDATION": {"distance_threshold_mm": 7.0}},
            )

        text = output.getvalue()
        self.assertIn("Total sucesso (<= 7.0mm):   2 (100.0%)", text)
        self.assertIn("Dice médio após a morfologia:   0.7000", text)

    @patch("utils.segmentation.pipeline_orchestration.summarize_batch_timing_records")
    @patch("utils.segmentation.pipeline_orchestration.load_batch_timing_records")
    @patch("utils.segmentation.pipeline_orchestration._process_and_save_batch")
    @patch("utils.segmentation.pipeline_orchestration._load_previous_batches")
    def test_resume_processes_requested_batch_and_later_batches(
        self,
        load_previous,
        process_batch,
        load_timings,
        summarize_timings,
    ):
        load_previous.return_value = ([{"IMG_ID": 1}], [1, 2])
        process_batch.side_effect = lambda _, batch_number, *args: [
            {"IMG_ID": batch_number}
        ]
        load_timings.return_value = []
        summarize_timings.return_value = {}

        with TemporaryDirectory() as output_dir:
            result = run_pipeline(
                list(range(20)),
                "train",
                {"NUM_BATCHES": 5},
                "/dataset",
                output_dir=output_dir,
                resume_from_batch=3,
            )

        processed_numbers = [call.args[1] for call in process_batch.call_args_list]
        self.assertEqual(processed_numbers, [3, 4, 5])
        self.assertEqual(result["batches_processed"], [1, 2, 3, 4, 5])
