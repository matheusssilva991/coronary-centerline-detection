"""Testes do filtro experimental da trajetória circular da aorta."""

import unittest
from pathlib import Path

import numpy as np
from skimage.draw import disk

from utils.segmentation.aorta_correction import find_mask_guided_tail_start
from utils.segmentation.aorta_localization import (
    filter_aorta_circle_trajectory,
)
from utils.segmentation.aorta_segmentation import (
    build_circle_trajectory_envelope,
    calculate_circle_mask_profile,
)
from utils.segmentation.pipeline_cli import build_parser


def _circle(slice_index, *, center_x=20.0, center_y=20.0, radius=20.0, accum=0.6):
    return {
        "slice_index": slice_index,
        "center_x": center_x,
        "center_y": center_y,
        "radius": radius,
        "accum": accum,
        "interpolated": False,
    }


class AortaCircleTrajectoryFilterTests(unittest.TestCase):
    def test_envelope_extends_endpoint_circles_with_axial_margin(self):
        circles = [
            _circle(4, center_x=10.0, center_y=10.0, radius=2.0),
            _circle(6, center_x=14.0, center_y=10.0, radius=2.0),
        ]

        envelope = build_circle_trajectory_envelope(
            (24, 24, 12),
            circles,
            radius_factor=1.0,
            axial_margin_slices=2,
        )

        segmented_slices = np.flatnonzero(envelope.any(axis=(0, 1))).tolist()
        self.assertEqual(segmented_slices, [2, 3, 4, 5, 6, 7, 8])
        self.assertTrue(envelope[10, 10, 2])
        self.assertTrue(envelope[10, 14, 8])

    def test_envelope_axial_margin_is_clipped_to_volume(self):
        circles = [
            _circle(1, center_x=10.0, center_y=10.0, radius=2.0),
            _circle(10, center_x=10.0, center_y=10.0, radius=2.0),
        ]

        envelope = build_circle_trajectory_envelope(
            (24, 24, 12),
            circles,
            axial_margin_slices=5,
        )

        segmented_slices = np.flatnonzero(envelope.any(axis=(0, 1))).tolist()
        self.assertEqual(segmented_slices, list(range(12)))

    def test_envelope_rejects_negative_axial_margin(self):
        with self.assertRaisesRegex(ValueError, "axial_margin_slices"):
            build_circle_trajectory_envelope(
                (24, 24, 12),
                [_circle(4)],
                axial_margin_slices=-1,
            )

    def test_cli_accepts_robust_circle_filter(self):
        parser = build_parser(Path("/dataset"), Path("/output"))

        args = parser.parse_args(["--aorta-circle-filter", "robust"])

        self.assertEqual(args.aorta_circle_filter, "robust")

    def test_cli_accepts_trajectory_axial_margin(self):
        parser = build_parser(Path("/dataset"), Path("/output"))

        args = parser.parse_args(["--aorta-trajectory-axial-margin-slices", "5"])

        self.assertEqual(args.aorta_trajectory_axial_margin_slices, 5)

    def test_none_preserves_original_trajectory(self):
        circles = [_circle(z) for z in range(20, 10, -1)]

        filtered, diagnostics = filter_aorta_circle_trajectory(
            circles,
            pixel_spacing=1.0,
            image_slice_count=30,
            filter_config={"method": "none"},
        )

        self.assertEqual(filtered, circles)
        self.assertFalse(diagnostics["aorta_circle_filter_applied"])
        self.assertEqual(diagnostics["aorta_circle_filter_reason"], "disabled")

    def test_trims_persistently_incompatible_tracking_tail(self):
        stable = [_circle(z) for z in range(100, 60, -1)]
        incompatible_tail = [
            _circle(z, center_x=35.0, radius=30.0, accum=0.3) for z in range(60, 40, -1)
        ]

        filtered, diagnostics = filter_aorta_circle_trajectory(
            stable + incompatible_tail,
            pixel_spacing=1.0,
            image_slice_count=120,
            filter_config={"method": "robust", "min_tail_coverage": 0.4},
        )

        self.assertEqual(filtered, stable)
        self.assertEqual(diagnostics["aorta_circle_filter_trimmed_tail_count"], 20)
        self.assertEqual(diagnostics["aorta_circle_filter_trim_start_slice"], 60)
        self.assertEqual(
            diagnostics["aorta_circle_filter_reason"],
            "persistent_tail_trimmed",
        )

    def test_does_not_trim_tail_below_minimum_coverage(self):
        stable = [_circle(z) for z in range(100, 60, -1)]
        incompatible_tail = [
            _circle(z, center_x=35.0, radius=30.0, accum=0.3) for z in range(60, 40, -1)
        ]

        filtered, diagnostics = filter_aorta_circle_trajectory(
            stable + incompatible_tail,
            pixel_spacing=1.0,
            image_slice_count=120,
            filter_config={"method": "robust", "min_tail_coverage": 0.8},
        )

        self.assertEqual(filtered, stable + incompatible_tail)
        self.assertFalse(diagnostics["aorta_circle_filter_applied"])
        self.assertEqual(
            diagnostics["aorta_circle_filter_reason"],
            "coverage_below_tail_threshold",
        )

    def test_rejects_tail_that_exceeds_maximum_trim_fraction(self):
        stable = [_circle(z) for z in range(100, 60, -1)]
        incompatible_tail = [
            _circle(z, center_x=35.0, radius=30.0, accum=0.3) for z in range(60, 20, -1)
        ]
        original = stable + incompatible_tail

        filtered, diagnostics = filter_aorta_circle_trajectory(
            original,
            pixel_spacing=1.0,
            image_slice_count=100,
            filter_config={
                "method": "robust",
                "min_tail_coverage": 0.4,
                "max_tail_trim_fraction": 0.4,
            },
        )

        self.assertEqual(filtered, original)
        self.assertFalse(diagnostics["aorta_circle_filter_applied"])
        self.assertEqual(
            diagnostics["aorta_circle_filter_reason"],
            "tail_trim_fraction_exceeded",
        )

    def test_cli_accepts_maximum_tail_trim_fraction(self):
        parser = build_parser(Path("/dataset"), Path("/output"))

        args = parser.parse_args(["--aorta-circle-filter-max-trim-fraction", "0.4"])

        self.assertEqual(args.aorta_circle_filter_max_trim_fraction, 0.4)

    def test_cli_accepts_hough_radius_interval(self):
        parser = build_parser(Path("/dataset"), Path("/output"))

        args = parser.parse_args(
            [
                "--aorta-hough-radii-start-px",
                "19",
                "--aorta-hough-radii-end-px",
                "31",
            ]
        )

        self.assertEqual(args.aorta_hough_radii_start_px, 19)
        self.assertEqual(args.aorta_hough_radii_end_px, 31)

    def test_extrapolates_short_synthetic_tail_from_stable_circles(self):
        stable = [_circle(z) for z in range(100, 60, -1)]
        incompatible_tail = [
            _circle(z, center_x=35.0, radius=30.0, accum=0.3) for z in range(60, 40, -1)
        ]

        filtered, diagnostics = filter_aorta_circle_trajectory(
            stable + incompatible_tail,
            pixel_spacing=1.0,
            image_slice_count=120,
            filter_config={
                "method": "robust",
                "min_tail_coverage": 0.4,
                "synthetic_tail_slices": 3,
            },
        )

        synthetic = filtered[-3:]
        self.assertEqual([circle["slice_index"] for circle in synthetic], [60, 59, 58])
        self.assertTrue(
            all(
                circle["trajectory_filter_action"] == "extrapolated_stable_tail"
                for circle in synthetic
            )
        )
        self.assertTrue(all(circle["center_x"] == 20.0 for circle in synthetic))
        self.assertTrue(all(circle["radius"] == 20.0 for circle in synthetic))
        self.assertEqual(diagnostics["aorta_circle_filter_synthetic_tail_count"], 3)
        self.assertEqual(diagnostics["aorta_circle_used_count"], len(stable) + 3)
        self.assertEqual(
            diagnostics["aorta_circle_filter_reason"],
            "persistent_tail_trimmed+stable_tail_extrapolated",
        )

    def test_cli_accepts_synthetic_tail_slice_count(self):
        parser = build_parser(Path("/dataset"), Path("/output"))

        args = parser.parse_args(["--aorta-circle-filter-synthetic-tail-slices", "5"])

        self.assertEqual(args.aorta_circle_filter_synthetic_tail_slices, 5)

    def test_mask_profile_reports_fill_and_area_ratio_per_slice(self):
        mask = np.zeros((48, 48, 3), dtype=np.uint8)
        mask[10:30, 10:30, 1] = 1

        profile = calculate_circle_mask_profile(mask, [_circle(1, radius=10.0)])

        self.assertEqual(len(profile), 1)
        self.assertEqual(profile[0]["slice_index"], 1)
        self.assertGreater(profile[0]["circle_fill_ratio"], 0.9)
        self.assertGreater(profile[0]["circle_area_ratio"], 1.2)

    def test_cli_rejects_removed_conditional_aorta_correction(self):
        parser = build_parser(Path("/dataset"), Path("/output"))

        with self.assertRaises(SystemExit):
            parser.parse_args(["--aorta-correction", "conditional"])

    def test_mask_guided_fallback_finds_persistent_high_area_tail(self):
        circles = [
            _circle(z, center_x=24.0, center_y=24.0, radius=5.0)
            for z in range(39, -1, -1)
        ]
        mask = np.zeros((48, 48, 40), dtype=np.uint8)
        for circle in circles:
            z = int(circle["slice_index"])
            radius = 11.0 if z <= 9 else 5.0
            rr, cc = disk((24.0, 24.0), radius, shape=mask.shape[:2])
            mask[rr, cc, z] = 1

        tail_start = find_mask_guided_tail_start(
            mask,
            circles,
            {
                "tail_search_start_fraction": 0.35,
                "persistence_window": 5,
                "persistence_required": 4,
                "min_tail_circles": 8,
                "min_remaining_circles": 30,
                "max_tail_trim_fraction": 0.4,
                "slice_area_ratio_threshold": 2.5,
            },
        )

        self.assertEqual(tail_start, 30)

    def test_cli_accepts_mask_guided_circle_filter(self):
        parser = build_parser(Path("/dataset"), Path("/output"))

        args = parser.parse_args(["--aorta-circle-filter-mask-guided"])

        self.assertTrue(args.aorta_circle_filter_mask_guided)


if __name__ == "__main__":
    unittest.main()
