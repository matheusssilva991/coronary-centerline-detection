import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from segmentation_pipeline import resolve_visual_output_dir
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

    def test_cli_rejects_removed_aorta_correction_flags(self):
        parser = build_parser(Path("/dataset"), Path("/output"))
        with self.assertRaises(SystemExit):
            parser.parse_args(["--aorta-leak-correction", "circle_area_jump_pruning"])

    def test_cli_rejects_removed_aorta_ostia_profile(self):
        parser = build_parser(Path("/dataset"), Path("/output"))
        with self.assertRaises(SystemExit):
            parser.parse_args(["--aorta-ostia-method", "bilateral_thin"])

    def test_cli_enables_segmentation_visuals_explicitly(self):
        parser = build_parser(Path("/dataset"), Path("/output"))

        default_args = parser.parse_args([])
        enabled_args = parser.parse_args(["--save-segmentation-visuals"])

        self.assertFalse(default_args.save_segmentation_visuals)
        self.assertTrue(enabled_args.save_segmentation_visuals)

    def test_cli_accepts_external_visual_output_directory(self):
        parser = build_parser(Path("/dataset"), Path("/output"))

        args = parser.parse_args(
            [
                "--save-segmentation-visuals",
                "--visual-output-dir",
                "/media/results",
            ]
        )

        self.assertEqual(args.visual_output_dir, "/media/results")

    def test_external_visual_directory_preserves_run_structure(self):
        args = SimpleNamespace(
            visual_output_dir=Path("/media/results"),
            resolution="mid",
        )
        output_root = Path("/repo/output")
        run_dir = output_root / "segmentation/runs/mid_res/group/timestamp"

        visual_dir = resolve_visual_output_dir(args, run_dir, output_root)

        self.assertEqual(
            visual_dir,
            Path("/media/results/segmentation/runs/mid_res/group/timestamp/visual"),
        )

    def test_default_visual_directory_remains_inside_run(self):
        args = SimpleNamespace(visual_output_dir=None, resolution="mid")
        run_dir = Path("/repo/output/segmentation/runs/mid_res/timestamp")

        visual_dir = resolve_visual_output_dir(args, run_dir, Path("/repo/output"))

        self.assertEqual(visual_dir, run_dir / "visual")

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
