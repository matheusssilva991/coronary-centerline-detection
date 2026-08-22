import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from utils.segmentation.pipeline_cli import build_parser
from utils.segmentation.pipeline_visuals import save_segmentation_visual


class PipelineVisualTests(unittest.TestCase):
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
