import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from utils.visualization.volume import (
    visualize_arteries_comparison,
    visualize_binary_masks_comparison,
    visualize_label_map_3d,
)


class _FakePlot:
    def __init__(self):
        self.display_count = 0

    def display(self):
        self.display_count += 1


class LabelMapVisualizationTests(unittest.TestCase):
    @patch("utils.visualization.volume._add_mask_mesh")
    @patch("utils.visualization.volume.create_plot")
    def test_adds_only_present_labels_with_physical_spacing(
        self,
        create_plot,
        add_mask_mesh,
    ):
        plot = _FakePlot()
        create_plot.return_value = plot
        label_map = np.zeros((3, 3, 3), dtype=np.uint16)
        label_map[1, 1, 1] = 10
        label_map[2, 2, 2] = 20

        result = visualize_label_map_3d(
            label_map,
            {
                10: ("Estrutura A", 0x112233),
                20: ("Estrutura B", 0x445566),
                30: ("Ausente", 0x778899),
            },
            spacing=(0.5, 0.75, 1.5),
            opacity=0.6,
            plot_name="Mapa cardíaco",
        )

        self.assertIs(result, plot)
        self.assertEqual(add_mask_mesh.call_count, 2)
        self.assertEqual(
            [call.kwargs["name"] for call in add_mask_mesh.call_args_list],
            ["Estrutura A", "Estrutura B"],
        )
        for call in add_mask_mesh.call_args_list:
            self.assertEqual(call.kwargs["spacing"], (0.5, 0.75, 1.5))
            self.assertEqual(call.kwargs["opacity"], 0.6)
        create_plot.assert_called_once_with(
            name="Mapa cardíaco",
            height=800,
            grid_visible=True,
            axes=["X (mm)", "Y (mm)", "Z (mm)"],
        )
        self.assertEqual(plot.display_count, 1)

    @patch("utils.visualization.volume.save_k3d_plot_html")
    @patch("utils.visualization.volume._add_mask_mesh")
    @patch("utils.visualization.volume.create_plot")
    def test_saves_html_optionally_without_displaying(
        self,
        create_plot,
        _add_mask_mesh,
        save_k3d_plot_html,
    ):
        plot = _FakePlot()
        create_plot.return_value = plot
        label_map = np.ones((2, 2, 2), dtype=np.uint8)

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "labels.html"
            visualize_label_map_3d(
                label_map,
                {1: ("Estrutura", 0xFFFFFF)},
                save_html_path=output_path,
                display_plot=False,
            )

        save_k3d_plot_html.assert_called_once_with(plot, output_path)
        self.assertEqual(plot.display_count, 0)

    @patch("utils.visualization.volume.create_plot")
    def test_rejects_selection_without_present_labels(self, create_plot):
        label_map = np.ones((2, 2, 2), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "Nenhum dos rótulos"):
            visualize_label_map_3d(
                label_map,
                {2: ("Ausente", 0xFFFFFF)},
                display_plot=False,
            )

        create_plot.assert_not_called()

    def test_rejects_invalid_volume_spacing_and_opacity(self):
        label_map = np.ones((2, 2, 2), dtype=np.uint8)
        specs = {1: ("Estrutura", 0xFFFFFF)}

        with self.assertRaisesRegex(ValueError, "tridimensional"):
            visualize_label_map_3d(
                label_map[0],
                specs,
                display_plot=False,
            )
        with self.assertRaisesRegex(ValueError, "espaçamento"):
            visualize_label_map_3d(
                label_map,
                specs,
                spacing=(1.0, 0.0, 1.0),
                display_plot=False,
            )
        with self.assertRaisesRegex(ValueError, "opacidade"):
            visualize_label_map_3d(
                label_map,
                specs,
                opacity=1.1,
                display_plot=False,
            )


class BinaryMaskComparisonTests(unittest.TestCase):
    @patch("utils.visualization.volume._add_mask_mesh")
    @patch("utils.visualization.volume.create_plot")
    def test_builds_named_reference_and_prediction_meshes(
        self,
        create_plot,
        add_mask_mesh,
    ):
        plot = _FakePlot()
        create_plot.return_value = plot
        reference = np.zeros((3, 3, 3), dtype=np.uint8)
        prediction = np.zeros_like(reference)
        reference[1, 1, 1] = 1
        prediction[1:, 1, 1] = 1

        result = visualize_binary_masks_comparison(
            reference,
            prediction,
            spacing=(0.7, 0.8, 1.2),
            reference_name="Aorta ground truth",
            predicted_name="Aorta predita",
            display_plot=False,
        )

        self.assertIs(result, plot)
        self.assertEqual(add_mask_mesh.call_count, 2)
        self.assertEqual(
            [call.kwargs["name"] for call in add_mask_mesh.call_args_list],
            ["Aorta ground truth", "Aorta predita"],
        )
        self.assertEqual(
            add_mask_mesh.call_args_list[0].kwargs["spacing"],
            (0.7, 0.8, 1.2),
        )

    def test_rejects_incompatible_or_empty_masks(self):
        populated = np.ones((2, 2, 2), dtype=np.uint8)
        empty = np.zeros_like(populated)

        with self.assertRaisesRegex(ValueError, "mesmo shape"):
            visualize_binary_masks_comparison(
                populated,
                np.ones((3, 2, 2), dtype=np.uint8),
                display_plot=False,
            )
        with self.assertRaisesRegex(ValueError, "não podem estar vazias"):
            visualize_binary_masks_comparison(
                populated,
                empty,
                display_plot=False,
            )

    @patch("utils.visualization.volume.visualize_binary_masks_comparison")
    def test_artery_wrapper_preserves_legacy_labels(self, compare_masks):
        mask = np.ones((2, 2, 2), dtype=np.uint8)
        expected_plot = _FakePlot()
        compare_masks.return_value = expected_plot

        result = visualize_arteries_comparison(
            mask,
            mask,
            display_plot=False,
        )

        self.assertIs(result, expected_plot)
        self.assertEqual(compare_masks.call_args.kwargs["reference_name"], "Label")
        self.assertEqual(compare_masks.call_args.kwargs["predicted_name"], "Predita")


if __name__ == "__main__":
    unittest.main()
