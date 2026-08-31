import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from segmentation_pipeline import _apply_execution_overrides, resolve_visual_output_dir
from utils.segmentation.pipeline_cli import build_parser
from utils.segmentation.pipeline_visuals import save_segmentation_visual


class PipelineVisualTests(unittest.TestCase):
    def test_cli_accepts_robust_circle_filter_and_run_group(self):
        parser = build_parser(Path("/dataset"), Path("/output"))

        args = parser.parse_args(
            [
                "--aorta-circle-filter",
                "robust",
                "--aorta-circle-filter-min-coverage",
                "0.8",
                "--run-group",
                "aorta_segmentation_experiments/val/circle_filter_conservative",
            ]
        )

        self.assertEqual(args.aorta_circle_filter_min_coverage, 0.8)
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

    def test_cli_overrides_aorta_level_set_and_mask_guided_parameters(self):
        parser = build_parser(Path("/dataset"), Path("/output"))
        args = parser.parse_args(
            [
                "--aorta-level-set-iterations",
                "26",
                "--aorta-level-set-radius-reduction-factor",
                "0.2",
                "--aorta-level-set-balloon",
                "0.7",
                "--aorta-level-set-alpha",
                "1250",
                "--aorta-opening-radius",
                "1",
                "--aorta-conservative-balloon",
                "0.4",
                "--aorta-conservative-alpha",
                "1750",
                "--aorta-conservative-threshold-percentile",
                "60",
                "--aorta-conservative-min-ratio-improvement",
                "0.1",
                "--aorta-localization-leak-override",
                "--aorta-localization-leak-min-area-ratio-p90",
                "2.0",
                "--aorta-localization-leak-min-circle-fill-q25",
                "0.8",
                "--aorta-localization-leak-min-volume-fraction",
                "0.015",
                "--aorta-mask-guided-area-ratio-p90",
                "2.3",
                "--aorta-mask-guided-max-fill-loss",
                "0.025",
                "--aorta-mask-guided-min-ratio-improvement",
                "0.05",
            ]
        )
        config = {}

        _apply_execution_overrides(config, args)

        self.assertEqual(config["LEVEL_SET"]["num_iter"], 26)
        self.assertEqual(config["LEVEL_SET"]["radius_reduction_factor"], 0.2)
        self.assertEqual(config["LEVEL_SET"]["balloon"], 0.7)
        self.assertEqual(config["LEVEL_SET"]["alpha"], 1250.0)
        self.assertEqual(config["LEVEL_SET"]["leak_removal_radius"], 1)
        adaptive = config["LEVEL_SET"]["adaptive"]
        self.assertEqual(adaptive["conservative"]["balloon"], 0.4)
        self.assertEqual(adaptive["conservative"]["alpha"], 1750.0)
        self.assertEqual(
            adaptive["conservative"]["threshold_percentile"],
            60.0,
        )
        self.assertEqual(adaptive["min_area_ratio_improvement_fraction"], 0.1)
        localization_override = adaptive["localization_leak_override"]
        self.assertTrue(localization_override["enabled"])
        self.assertEqual(localization_override["min_area_ratio_p90"], 2.0)
        self.assertEqual(localization_override["min_circle_fill_q25"], 0.8)
        self.assertEqual(localization_override["min_volume_fraction"], 0.015)
        mask_guided = config["CIRCLE_DETECTION"]["trajectory_filter"][
            "mask_guided_fallback"
        ]
        self.assertEqual(mask_guided["min_area_ratio_p90"], 2.3)
        self.assertEqual(mask_guided["slice_area_ratio_threshold"], 2.3)
        self.assertEqual(mask_guided["max_fill_loss"], 0.025)
        self.assertEqual(mask_guided["min_ratio_improvement"], 0.05)

    def test_cli_rejects_removed_recovery_flags(self):
        parser = build_parser(Path("/dataset"), Path("/output"))
        with self.assertRaises(SystemExit):
            parser.parse_args(["--aorta-recovery-max-extra-slices", "15"])

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
