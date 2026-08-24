import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from utils.segmentation.pipeline_cli import build_parser
from utils.segmentation.pipeline_visuals import save_segmentation_visual


class PipelineVisualTests(unittest.TestCase):
    def test_cli_accepts_conservative_circle_filter_and_run_group(self):
        parser = build_parser(Path("/dataset"), Path("/output"))

        args = parser.parse_args(
            [
                "--aorta-circle-filter",
                "robust",
                "--aorta-circle-filter-min-coverage",
                "0.8",
                "--no-aorta-circle-filter-interpolate",
                "--aorta-circle-filter-reject-oversegmented",
                "--ostia-surface-mode",
                "physical_distance",
                "--ostia-surface-thickness-mm",
                "2.0",
                "--ostia-candidate-score-mode",
                "voxel",
                "--ostia-pair-selection-mode",
                "greedy",
                "--run-group",
                "aorta_segmentation_experiments/val/circle_filter_conservative",
            ]
        )

        self.assertEqual(args.aorta_circle_filter_min_coverage, 0.8)
        self.assertFalse(args.aorta_circle_filter_interpolate)
        self.assertTrue(args.aorta_circle_filter_reject_oversegmented)
        self.assertEqual(args.ostia_surface_mode, "physical_distance")
        self.assertEqual(args.ostia_candidate_score_mode, "voxel")
        self.assertEqual(args.ostia_pair_selection_mode, "greedy")
        self.assertEqual(
            args.run_group,
            "aorta_segmentation_experiments/val/circle_filter_conservative",
        )

    def test_cli_selects_adaptive_aorta_level_set(self):
        parser = build_parser(Path("/dataset"), Path("/output"))

        default_args = parser.parse_args([])
        adaptive_args = parser.parse_args(["--aorta-level-set-mode", "adaptive"])

        self.assertIsNone(default_args.aorta_level_set_mode)
        self.assertEqual(adaptive_args.aorta_level_set_mode, "adaptive")

    def test_cli_selects_experimental_aorta_leak_correction(self):
        parser = build_parser(Path("/dataset"), Path("/output"))

        default_args = parser.parse_args([])
        pruning_args = parser.parse_args(
            [
                "--aorta-leak-correction",
                "circle_seeded_neck_pruning",
                "--aorta-neck-pruning-erosion-radius",
                "3",
                "--aorta-neck-pruning-core-radius-factor",
                "0.85",
                "--aorta-neck-pruning-max-volume-loss",
                "0.15",
            ]
        )

        self.assertIsNone(default_args.aorta_leak_correction)
        self.assertEqual(
            pruning_args.aorta_leak_correction,
            "circle_seeded_neck_pruning",
        )
        self.assertEqual(pruning_args.aorta_neck_pruning_erosion_radius, 3)
        self.assertEqual(pruning_args.aorta_neck_pruning_core_radius_factor, 0.85)
        self.assertEqual(pruning_args.aorta_neck_pruning_max_volume_loss, 0.15)

        area_jump_args = parser.parse_args(
            ["--aorta-leak-correction", "circle_area_jump_pruning"]
        )
        self.assertEqual(
            area_jump_args.aorta_leak_correction,
            "circle_area_jump_pruning",
        )

    def test_cli_enables_segmentation_visuals_explicitly(self):
        parser = build_parser(Path("/dataset"), Path("/output"))

        default_args = parser.parse_args([])
        enabled_args = parser.parse_args(["--save-segmentation-visuals"])

        self.assertFalse(default_args.save_segmentation_visuals)
        self.assertTrue(enabled_args.save_segmentation_visuals)

    @patch("utils.segmentation.pipeline_visuals.visualize_aorta_ostia_artery")
    def test_saves_combined_visual_in_requested_directory(self, visualize):
        def write_snapshot(*args, **kwargs):
            Path(kwargs["save_html_path"]).write_text("<html></html>")

        visualize.side_effect = write_snapshot
        mask = np.ones((2, 2, 2), dtype=np.uint8)

        with TemporaryDirectory() as temp_dir:
            output = save_segmentation_visual(
                temp_dir,
                28,
                aorta_mask=mask,
                ostia_left=(0, 0, 0),
                ostia_right=(1, 1, 1),
                artery_mask=mask,
                label_artery=mask,
                spacing=(1.0, 1.0, 1.5),
            )

            self.assertEqual(
                output,
                Path(temp_dir) / "img_28_aorta_ostia_artery.html",
            )
            self.assertTrue(output.exists())

        visualize.assert_called_once()
        self.assertFalse(visualize.call_args.kwargs["display_plot"])

    @patch("utils.segmentation.pipeline_visuals.visualize_aorta_ostia_artery")
    def test_visualization_failure_does_not_raise(self, visualize):
        visualize.side_effect = RuntimeError("mesh failure")

        with TemporaryDirectory() as temp_dir:
            output = save_segmentation_visual(
                temp_dir,
                28,
                aorta_mask=np.ones((2, 2, 2), dtype=np.uint8),
                ostia_left=None,
                ostia_right=None,
                artery_mask=None,
                label_artery=np.zeros((2, 2, 2), dtype=np.uint8),
                spacing=(1.0, 1.0, 1.0),
            )

        self.assertIsNone(output)


if __name__ == "__main__":
    unittest.main()
