"""Tests for 2D slice plotting and visualization grids."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np

from utils.visualization.images.image_slices import plot_volume_slice
from utils.visualization.pipeline.preprocessing_views import plot_preprocessing_grid


class SaveVolumeSliceFigureTest(TestCase):
    def test_plot_volume_slice_returns_the_complete_selected_slice(self):
        volume = np.arange(6 * 4 * 3, dtype=np.float32).reshape(6, 4, 3)

        fig, ax = plot_volume_slice(
            volume,
            1,
            padding_fraction=0.1,
            show=False,
        )
        rendered = np.asarray(ax.images[0].get_array())

        np.testing.assert_array_equal(rendered, volume[:, :, 1])
        self.assertLess(ax.get_xlim()[0], -0.5)
        self.assertGreater(ax.get_xlim()[1], volume.shape[1] - 0.5)
        fig.clear()

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

        with patch("matplotlib.pyplot.show"):
            plot_preprocessing_grid(
                preprocessed,
                ids_to_plot=np.array([10, 20]),
                show_title=False,
            )
