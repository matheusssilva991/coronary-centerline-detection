"""Tests for reusable helpers extracted from the interactive pipeline notebook."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from utils.project.notebook_env import load_notebook_pipeline_config
from utils.utils.metrics import (
    binary_segmentation_metrics,
    print_segmentation_metrics,
)


class NotebookPipelineHelperTests(TestCase):
    def test_binary_metrics_match_notebook_calculation(self) -> None:
        prediction = np.array([1, 1, 0, 0])
        ground_truth = np.array([1, 0, 1, 0])

        metrics = binary_segmentation_metrics(prediction, ground_truth)

        self.assertEqual(metrics["predicted_voxels"], 2)
        self.assertEqual(metrics["ground_truth_voxels"], 2)
        self.assertEqual(metrics["true_positives"], 1)
        self.assertAlmostEqual(float(metrics["dice"]), 0.5)
        self.assertAlmostEqual(float(metrics["sensitivity"]), 0.5)
        self.assertAlmostEqual(float(metrics["precision"]), 0.5)

    def test_print_metrics_preserves_interactive_format(self) -> None:
        stream = StringIO()
        with redirect_stdout(stream):
            print_segmentation_metrics(
                "Resultado",
                {
                    "dice": 0.5,
                    "predicted_voxels": 2,
                    "ground_truth_voxels": 2,
                    "true_positives": 1,
                    "sensitivity": 0.5,
                    "precision": 0.5,
                },
            )

        output = stream.getvalue()
        self.assertIn("Resultado", output)
        self.assertIn("Dice: 0.5000", output)
        self.assertIn("Valor preditivo positivo: 0.5000", output)

    @patch("utils.project.notebook_env.scale_config_to_resolution")
    @patch("utils.project.notebook_env.apply_aorta_ostia_method")
    @patch("utils.project.notebook_env.load_config_json")
    def test_loads_high_resolution_config_in_original_order(
        self,
        load_config,
        apply_method,
        scale_config,
    ) -> None:
        with TemporaryDirectory() as temporary_dir:
            config_path = Path(temporary_dir) / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            loaded = {
                "DOWNSCALE_FACTORS": [2, 2, 1],
                "AORTA_OSTIA_METHOD": {"method": "bilateral_thin"},
            }
            load_config.return_value = loaded
            apply_method.side_effect = lambda config, method: config
            scale_config.side_effect = lambda config: config

            result = load_notebook_pipeline_config(config_path, "high")

        self.assertEqual(result["DOWNSCALE_FACTORS"], [1, 1, 1])
        apply_method.assert_called_once_with(loaded, method="bilateral_thin")
        scale_config.assert_called_once()
