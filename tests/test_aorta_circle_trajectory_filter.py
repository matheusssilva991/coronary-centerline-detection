"""Testes do filtro experimental da trajetória circular da aorta."""

import unittest
from pathlib import Path

from utils.segmentation.aorta_localization import filter_aorta_circle_trajectory
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
    def test_cli_accepts_robust_circle_filter(self):
        parser = build_parser(Path("/dataset"), Path("/output"))

        args = parser.parse_args(["--aorta-circle-filter", "robust"])

        self.assertEqual(args.aorta_circle_filter, "robust")

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

    def test_interpolates_isolated_geometric_outlier(self):
        circles = [_circle(z) for z in range(20, 10, -1)]
        circles[4] = _circle(16, center_x=35.0, radius=30.0, accum=0.2)

        filtered, diagnostics = filter_aorta_circle_trajectory(
            circles,
            pixel_spacing=1.0,
            image_slice_count=30,
            filter_config={
                "method": "robust",
                "min_remaining_circles": 5,
                "interpolate_isolated_outliers": True,
            },
        )

        self.assertEqual(len(filtered), len(circles))
        self.assertAlmostEqual(filtered[4]["center_x"], 20.0)
        self.assertAlmostEqual(filtered[4]["radius"], 20.0)
        self.assertTrue(filtered[4]["trajectory_filtered"])
        self.assertEqual(diagnostics["aorta_circle_filter_interpolated_count"], 1)
        self.assertEqual(diagnostics["aorta_circle_filter_trimmed_tail_count"], 0)

    def test_trims_persistently_incompatible_tracking_tail(self):
        stable = [_circle(z) for z in range(100, 60, -1)]
        incompatible_tail = [
            _circle(z, center_x=35.0, radius=30.0, accum=0.3)
            for z in range(60, 40, -1)
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
            _circle(z, center_x=35.0, radius=30.0, accum=0.3)
            for z in range(60, 40, -1)
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


if __name__ == "__main__":
    unittest.main()
