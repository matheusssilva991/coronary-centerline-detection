"""Tests for aorta detection helpers."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np
from skimage.draw import disk

from utils.segmentation.aorta import localization as aorta_localization
from utils.segmentation.pipeline.detection import (
    detect_ostia,
    locate_and_filter_aorta_circles,
    locate_aorta_circles,
)


class AortaCircleDetectionTest(TestCase):
    @patch("utils.segmentation.pipeline.detection.filter_located_aorta_circles")
    @patch("utils.segmentation.pipeline.detection.locate_aorta_circles")
    def test_locates_and_filters_circles_in_one_shared_stage(
        self,
        locate_circles,
        filter_circles,
    ):
        original = [{"slice_index": 1}, {"slice_index": 2}]
        filtered = [{"slice_index": 1}]
        diagnostics = {"aorta_circle_filter_applied": True}
        locate_circles.return_value = original
        filter_circles.return_value = filtered, diagnostics
        image = np.zeros((8, 8, 3), dtype=np.float32)

        result = locate_and_filter_aorta_circles(
            image,
            (2, 2, 1),
            (1.0, 1.0, 1.5),
            {"trajectory_filter": {}},
        )

        self.assertEqual(result.original_circles, original)
        self.assertEqual(result.filtered_circles, filtered)
        self.assertEqual(result.filter_diagnostics, diagnostics)
        filter_circles.assert_called_once_with(
            original,
            (1.0, 1.0, 1.5),
            3,
            {"trajectory_filter": {}},
        )

    @patch("utils.segmentation.pipeline.detection.detect_aorta_circles")
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

    @patch("utils.segmentation.pipeline.detection.find_ostia")
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

    def test_detect_aorta_circles_runs_initial_slice_hough_once_with_identical_result(
        self,
    ):
        volume = np.zeros((64, 64, 3), dtype=np.float32)
        for z in range(3):
            rr, cc = disk((16, 46), 8, shape=(64, 64))
            volume[rr, cc, z] = 500.0

        hough_radii = np.arange(7, 10, 1)
        expected_initial = aorta_localization.detect_initial_circle(
            volume[:, :, 2],
            hough_radii,
            quadrant_offset=(0, 0),
            total_num_peaks=5,
            canny_sigma=1.5,
        )
        self.assertIsNotNone(expected_initial)

        with patch(
            "utils.segmentation.aorta.localization._detect_circles_in_slice",
            wraps=aorta_localization._detect_circles_in_slice,
        ) as detect_slice:
            circles = aorta_localization.detect_aorta_circles(
                volume,
                hough_radii,
                pixel_spacing=1.0,
                tol_radius_mm=4.0,
                tol_distance_mm=6.0,
                quadrant_offset=(0, 0),
                total_num_peaks_initial=5,
                total_num_peaks=5,
                canny_sigma=1.5,
                verbose=False,
            )

        self.assertEqual(len(circles), 3)
        # 1 chamada para a fatia inicial (z=2) + 1 para z=1 + 1 para z=0.
        self.assertEqual(detect_slice.call_count, 3)
        self.assertAlmostEqual(circles[0]["accum"], expected_initial["accum"])
