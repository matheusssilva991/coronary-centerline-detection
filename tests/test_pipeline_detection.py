"""Tests for aorta detection helpers."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np

from utils.segmentation.pipeline_detection import locate_aorta_circles


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
