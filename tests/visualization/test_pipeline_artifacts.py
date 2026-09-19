import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.visualization.pipeline_artifacts import (
    STAGE_VIEW_FILENAMES,
    save_detected_circles_figure,
    save_stage_views,
)


class PipelineArtifactsTest(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_saves_mip_and_three_axial_slices(self):
        volume = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)

        with tempfile.TemporaryDirectory() as temporary_dir:
            saved = save_stage_views(volume, temporary_dir, title="Etapa")

            self.assertSetEqual(set(saved), set(STAGE_VIEW_FILENAMES))
            for path in saved.values():
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

    def test_saves_detected_circle_samples(self):
        volume = np.ones((8, 8, 4), dtype=np.float32)
        circles = [
            {
                "slice_index": index,
                "center_x": 4,
                "center_y": 4,
                "radius": 2,
            }
            for index in range(4)
        ]

        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "circles.png"
            saved = save_detected_circles_figure(volume, circles, output)

            self.assertEqual(saved, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
