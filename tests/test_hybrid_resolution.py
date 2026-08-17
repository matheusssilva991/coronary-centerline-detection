"""Testes do pipeline híbrido mid-óstios/high-artérias."""

from __future__ import annotations

import unittest
from unittest.mock import ANY, patch

import numpy as np

from experiments.hybrid_resolution_pipeline import build_high_variants
from utils.experiments.hybrid_resolution import (
    HybridResolutionPreparedImage,
    evaluate_ostia_coordinates,
    process_hybrid_resolution_image,
    process_hybrid_resolution_variants,
    rescale_voxel_coordinate,
)
from utils.segmentation.pipeline_arteries import segment_arteries_from_ostia


class HybridResolutionTest(unittest.TestCase):
    @patch("utils.segmentation.pipeline_arteries.segment_arteries_from_vesselness")
    @patch("utils.segmentation.pipeline_arteries.compute_vesselness")
    def test_regular_pipeline_delegates_with_precomputed_vesselness(
        self,
        compute_vesselness,
        segment_from_vesselness,
    ):
        image = np.zeros((2, 2, 2), dtype=np.float32)
        vesselness = np.ones_like(image)
        compute_vesselness.return_value = vesselness
        segment_from_vesselness.return_value = {"dice_artery": 0.7}
        config = {"VESSELNESS_ARTERY": {"sigmas": [1.5]}, "USE_GPU": False}

        result = segment_arteries_from_ostia(
            image,
            np.zeros_like(image),
            (0, 0, 0),
            (1, 1, 1),
            config,
        )

        self.assertEqual(result["dice_artery"], 0.7)
        segment_from_vesselness.assert_called_once_with(
            image,
            ANY,
            vesselness,
            (0, 0, 0),
            (1, 1, 1),
            config,
        )

    def test_builds_recommended_high_variants_without_mutating_baseline(self):
        mid = {
            "VESSELNESS_ARTERY": {"sigmas": [1.5, 2.0]},
            "REGION_GROWING": {
                "threshold_divisor": 7,
                "min_vesselness_fraction": 0.078,
            },
            "POSTPROCESSING": {"closing_radius": 3, "dilation_radius": 2},
        }
        high = {
            "VESSELNESS_ARTERY": {"sigmas": [1.5, 2.0]},
            "REGION_GROWING": {
                "threshold_divisor": 12,
                "min_vesselness_fraction": 0.05,
            },
            "POSTPROCESSING": {"closing_radius": 6, "dilation_radius": 4},
        }

        variants = build_high_variants(
            mid,
            high,
            [
                "baseline_high_scaled",
                "morphology_mid_radii",
                "rg_mid_thresholds",
                "artery_sigmas_physical_x2",
                "rg_mid_thresholds_morphology_mid",
            ],
        )

        self.assertEqual(
            variants["baseline_high_scaled"]["POSTPROCESSING"]["closing_radius"],
            6,
        )
        self.assertEqual(
            variants["morphology_mid_radii"]["POSTPROCESSING"],
            mid["POSTPROCESSING"],
        )
        self.assertEqual(
            variants["rg_mid_thresholds"]["REGION_GROWING"]["threshold_divisor"],
            7,
        )
        self.assertEqual(
            variants["artery_sigmas_physical_x2"]["VESSELNESS_ARTERY"]["sigmas"],
            [3.0, 4.0],
        )
        self.assertEqual(high["POSTPROCESSING"]["closing_radius"], 6)

    def test_rescales_mid_coordinate_to_high_and_clips_bounds(self):
        self.assertEqual(
            rescale_voxel_coordinate(
                (10, 15, 3),
                source_factors=(2, 2, 1),
                target_factors=(1, 1, 1),
                target_shape=(32, 32, 4),
            ),
            (20, 30, 3),
        )
        self.assertEqual(
            rescale_voxel_coordinate(
                (20, 20, 8),
                source_factors=(2, 2, 1),
                target_factors=(1, 1, 1),
                target_shape=(32, 32, 4),
            ),
            (31, 31, 3),
        )

    def test_evaluates_rescaled_ostia_on_high_label(self):
        label = np.zeros((16, 16, 2), dtype=np.uint8)
        label[4, 6, 1] = 1
        label[8, 10, 1] = 1

        result = evaluate_ostia_coordinates(
            label,
            (4, 6, 1),
            (8, 10, 1),
            scaled_spacing=(0.5, 0.5, 1.0),
            tolerance_mm=7.0,
        )

        self.assertTrue(result["both_correct"])
        self.assertTrue(result["ostia_success"])
        self.assertEqual(result["ostia_status"], "both_correct")

    @patch("utils.experiments.hybrid_resolution.segment_arteries_from_vesselness")
    @patch("utils.experiments.hybrid_resolution.detect_and_evaluate_ostia")
    @patch("utils.experiments.hybrid_resolution.segment_aorta")
    @patch("utils.experiments.hybrid_resolution.locate_aorta_circles")
    @patch("utils.experiments.hybrid_resolution.compute_vesselness")
    @patch("utils.experiments.hybrid_resolution.load_and_preprocess_image")
    def test_processes_mid_detection_then_high_segmentation(
        self,
        load_image,
        compute_vesselness,
        locate_circles,
        segment_aorta,
        detect_ostia,
        segment_arteries,
    ):
        mid_label = np.zeros((8, 8, 2), dtype=np.uint8)
        high_label = np.zeros((16, 16, 2), dtype=np.uint8)
        high_label[4, 6, 1] = 1
        high_label[8, 10, 1] = 1
        load_image.side_effect = [
            {
                "lcc_image": np.zeros((8, 8, 2), dtype=np.float32),
                "label": mid_label,
                "scaled_spacing": (1.0, 1.0, 1.0),
                "downscale_factors": (2, 2, 1),
                "preprocessing_details": {},
            },
            {
                "lcc_image": np.zeros((16, 16, 2), dtype=np.float32),
                "label": high_label,
                "scaled_spacing": (0.5, 0.5, 1.0),
                "downscale_factors": (1, 1, 1),
                "preprocessing_details": {},
            },
        ]
        compute_vesselness.return_value = np.zeros((8, 8, 2), dtype=np.float32)
        locate_circles.return_value = [
            {
                "slice_index": 1,
                "center_x": 4,
                "center_y": 4,
                "radius": 2,
            }
        ]
        segment_aorta.return_value = np.ones((8, 8, 2), dtype=np.uint8)
        exact_info = {
            "intersects": True,
            "euclidean_dist": 0.0,
            "physical_dist": 0.0,
            "nearest_voxel": (0, 0, 0),
            "is_acceptable": True,
        }
        detect_ostia.return_value = {
            "ostia_left": np.array((2, 3, 1)),
            "ostia_right": np.array((4, 5, 1)),
            "label_artery": (mid_label == 1).astype(np.uint8),
            "left_info": exact_info,
            "right_info": exact_info,
            "both_correct": True,
            "both_tolerable": False,
        }
        segment_arteries.return_value = {
            "artery_mask": np.zeros_like(high_label),
            "raw_artery_mask": np.zeros_like(high_label),
            "dice_artery": 0.75,
            "dice_artery_before_morphology": 0.50,
            "dice_artery_after_morphology": 0.75,
            "dice_artery_morphology_delta": 0.25,
            "artery_voxels": 100,
        }
        mid_config = {
            "DOWNSCALE_FACTORS": (2, 2, 1),
            "VESSELNESS_AORTA": {},
            "CIRCLE_DETECTION": {},
            "LEVEL_SET": {},
            "OSTIA_VALIDATION": {"distance_threshold_mm": 7.0},
            "ARTERY_SEGMENTATION": {"method": "region_growing"},
            "USE_GPU": False,
        }
        high_config = {
            "DOWNSCALE_FACTORS": (1, 1, 1),
            "OSTIA_VALIDATION": {"distance_threshold_mm": 7.0},
            "ARTERY_SEGMENTATION": {"method": "region_growing"},
            "USE_GPU": False,
        }

        result = process_hybrid_resolution_image(
            10,
            mid_config,
            high_config,
            "/dataset",
        )

        self.assertIsNone(result["error"])
        self.assertEqual(result["high_ostia_left"], (4, 6, 1))
        self.assertEqual(result["high_ostia_right"], (8, 10, 1))
        self.assertTrue(result["high_ostia_success"])
        self.assertEqual(result["dice_artery"], 0.75)
        passed_args = segment_arteries.call_args.args
        self.assertEqual(passed_args[3], (4, 6, 1))
        self.assertEqual(passed_args[4], (8, 10, 1))

    @patch("utils.experiments.hybrid_resolution.segment_arteries_from_vesselness")
    @patch("utils.experiments.hybrid_resolution.compute_vesselness")
    @patch("utils.experiments.hybrid_resolution._prepare_hybrid_resolution_image")
    def test_reuses_vesselness_between_compatible_variants(
        self,
        prepare_image,
        compute_vesselness,
        segment_arteries,
    ):
        high_lcc = np.zeros((4, 4, 2), dtype=np.float32)
        high_label = np.zeros_like(high_lcc, dtype=np.uint8)
        prepare_image.return_value = HybridResolutionPreparedImage(
            result={
                "IMG_ID": 10,
                "mid_ostia_success": True,
                "high_ostia_success": True,
                "error": None,
            },
            high_lcc=high_lcc,
            high_label_artery=high_label,
            high_ostia_left=(1, 1, 0),
            high_ostia_right=(2, 2, 0),
        )
        compute_vesselness.return_value = high_lcc
        segment_arteries.return_value = {
            "dice_artery": 0.5,
            "dice_artery_before_morphology": 0.4,
            "dice_artery_after_morphology": 0.5,
            "dice_artery_morphology_delta": 0.1,
            "artery_voxels": 10,
        }
        common = {
            "VESSELNESS_ARTERY": {"sigmas": [1.5, 2.0]},
            "REGION_GROWING": {
                "threshold_divisor": 7,
                "min_vesselness_fraction": 0.05,
            },
            "POSTPROCESSING": {"closing_radius": 3, "dilation_radius": 2},
            "USE_GPU": False,
        }
        variants = {
            "baseline_high_scaled": common,
            "morphology_mid_radii": {
                **common,
                "POSTPROCESSING": {"closing_radius": 2, "dilation_radius": 1},
            },
        }

        rows = process_hybrid_resolution_variants(10, {}, variants, "/dataset")

        self.assertEqual(len(rows), 2)
        self.assertEqual(compute_vesselness.call_count, 1)
        self.assertEqual(segment_arteries.call_count, 2)
        self.assertFalse(rows[0]["high_vesselness_reused"])
        self.assertTrue(rows[1]["high_vesselness_reused"])


if __name__ == "__main__":
    unittest.main()
