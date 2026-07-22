"""Tests for complete 2D slice export."""

from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import matplotlib.image as mpimg
import numpy as np

from utils.visualization.image_slices import save_volume_slice_figure
from utils.visualization.preprocessing_views import plot_preprocessing_grid
from utils.visualization.vesselness import plot_vesselness_mip_grid


class SaveVolumeSliceFigureTest(TestCase):
    def test_preserves_slice_aspect_ratio_without_cropping(self):
        volume = np.arange(20 * 10 * 3, dtype=np.float32).reshape(20, 10, 3)

        with TemporaryDirectory() as tmp_dir:
            output = save_volume_slice_figure(
                volume,
                2,
                f"{tmp_dir}/slice.png",
                dpi=100,
                figure_width=4,
            )
            saved = mpimg.imread(output)

        self.assertEqual(saved.shape[:2], (800, 400))

    def test_rejects_invalid_slice_index(self):
        volume = np.zeros((4, 5, 6), dtype=np.float32)

        with self.assertRaises(IndexError):
            save_volume_slice_figure(volume, 6, "unused.png")

    def test_can_render_an_exact_preview_size(self):
        volume = np.zeros((32, 32, 2), dtype=np.float32)

        with TemporaryDirectory() as tmp_dir:
            output = save_volume_slice_figure(
                volume,
                1,
                f"{tmp_dir}/preview.png",
                dpi=100,
                output_size_px=(256, 256),
            )
            saved = mpimg.imread(output)

        self.assertEqual(saved.shape[:2], (256, 256))

    def test_uses_image_array_orientation_by_default(self):
        volume = np.zeros((2, 2, 1), dtype=np.float32)
        volume[1, :, 0] = 1.0

        with TemporaryDirectory() as tmp_dir:
            output = save_volume_slice_figure(
                volume,
                0,
                f"{tmp_dir}/orientation.png",
                vmin=0,
                vmax=1,
                dpi=100,
                output_size_px=(100, 100),
            )
            saved = mpimg.imread(output)

        top_brightness = saved[:40, :, :3].mean()
        bottom_brightness = saved[-40:, :, :3].mean()
        self.assertLess(top_brightness, bottom_brightness)

    def test_accepts_numpy_id_arrays_in_visualization_grids(self):
        volume = np.zeros((4, 4, 3), dtype=np.float32)
        preprocessed = {
            image_id: {
                "down_image": volume,
                "thresh_image": volume,
                "lcc_image": volume,
                "center_slice": 1,
            }
            for image_id in (10, 20)
        }
        vessel_maps = {
            image_id: {"vesselness_artery": volume}
            for image_id in (10, 20)
        }

        with patch("matplotlib.pyplot.show"):
            plot_preprocessing_grid(
                preprocessed,
                ids_to_plot=np.array([10, 20]),
                show_title=False,
            )
            plot_vesselness_mip_grid(
                vessel_maps,
                ids_to_plot=np.array([10, 20]),
            )
