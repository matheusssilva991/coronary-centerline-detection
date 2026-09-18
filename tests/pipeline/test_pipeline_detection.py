"""Tests for aorta detection helpers."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np

from utils.segmentation.pipeline_detection import detect_ostia, locate_aorta_circles


class AortaCircleDetectionTest(TestCase):
    @patch("utils.segmentation.pipeline_detection.detect_aorta_circles")
    def test_locates_circles_with_scaled_spacing(self, detect_circles):
        expected = [{"slice_index": 2, "center": (4, 4), "radius": 2}]
        detect_circles.return_value = expected
        config = {
            "radii_start_px": 2,
            "radii_end_px": 5,
            "radius_step_px": 1,
            "tol_radius_mm": 2.0,
            "tol_distance_mm": 5.0,
            "quadrant_offset": (0, 0),
            "max_slice_miss_threshold": 2,
            "neighbor_distance_threshold": 5,
            "total_num_peaks_initial": 5,
            "total_num_peaks": 3,
            "canny_sigma": 2.0,
        }

        result = locate_aorta_circles(
            np.zeros((8, 8, 3), dtype=np.float32),
            (2, 2, 1),
            (1.0, 1.0, 1.0),
            config,
        )

        self.assertEqual(result, expected)
        detect_circles.assert_called_once()

    @patch("utils.segmentation.pipeline_detection.find_ostia")
    def test_detects_ostia_without_reference_label(self, find_ostia):
        find_ostia.return_value = ((1, 2, 3), (4, 5, 6))
        config = {
            "OSTIA_DETECTION": {
                "top_n": 50,
                "max_z_diff_mm": 20.0,
                "lower_fraction": 1.0,
                "min_center_distance_factor": 0.8,
                "min_lateral_factor": 0.3,
                "erosion_radius": 2,
                "surface_padding_radius": 1,
            }
        }

        result = detect_ostia(
            np.zeros((4, 4, 4)),
            np.zeros((4, 4, 4)),
            (0.5, 0.75, 1.0),
            config,
        )

        self.assertEqual(result, ((1, 2, 3), (4, 5, 6)))
        self.assertEqual(find_ostia.call_args.kwargs["spacing"], (0.75, 0.5, 1.0))
