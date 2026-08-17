"""Tests for the qualitative pipeline helper."""

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from utils.experiments.qualitative_pipeline import (
    display_qualitative_pipeline_case,
    run_qualitative_pipeline_case,
)


class QualitativePipelineTest(TestCase):
    @patch("utils.experiments.qualitative_pipeline.visualize_aorta_ostia_artery")
    @patch("utils.experiments.qualitative_pipeline.run_qualitative_pipeline_case")
    def test_display_reuses_cached_result(self, run_case, visualize):
        volume = np.zeros((4, 4, 3), dtype=np.uint8)
        cached_result = {
            "aorta_mask": volume,
            "ostia_left": (1, 1, 1),
            "ostia_right": (2, 2, 1),
            "artery_mask": volume,
            "label_artery": volume,
            "scaled_spacing": (0.6, 0.7, 0.8),
        }
        cache = {(90, "baseline"): cached_result}

        result = display_qualitative_pipeline_case(
            90,
            {},
            Path("dataset"),
            variant_label="Baseline",
            case_label="Dice próximo da média",
            cache=cache,
            cache_key=(90, "baseline"),
        )

        self.assertIs(result, cached_result)
        run_case.assert_not_called()
        visualize.assert_called_once_with(
            volume,
            (1, 1, 1),
            (2, 2, 1),
            artery_mask=volume,
            label_artery=volume,
            spacing=(0.6, 0.7, 0.8),
            use_physical_coords=True,
            save_html_path=None,
            display_plot=True,
            plot_name="Baseline | IMG 90 | Dice próximo da média",
        )

    @patch("utils.experiments.qualitative_pipeline.segment_arteries_from_ostia")
    @patch("utils.experiments.qualitative_pipeline.detect_and_evaluate_ostia")
    @patch("utils.experiments.qualitative_pipeline.segment_aorta")
    @patch("utils.experiments.qualitative_pipeline.locate_aorta_circles")
    @patch("utils.experiments.qualitative_pipeline.compute_vesselness")
    @patch("utils.experiments.qualitative_pipeline.load_and_preprocess_image")
    def test_returns_intermediate_masks(
        self,
        load_image,
        compute_vesselness,
        detect_circles,
        segment_aorta,
        detect_ostia,
        segment_arteries,
    ):
        volume = np.zeros((4, 4, 3), dtype=np.uint8)
        load_image.return_value = {
            "lcc_image": volume,
            "label": volume,
            "scaled_spacing": (0.6, 0.7, 0.8),
            "downscale_factors": [2, 2, 1],
        }
        compute_vesselness.return_value = volume
        detect_circles.return_value = [{"slice_index": 2}]
        segment_aorta.return_value = volume
        detect_ostia.return_value = {
            "label_artery": volume,
            "ostia_left": (1, 1, 1),
            "ostia_right": (2, 2, 1),
        }
        segment_arteries.return_value = {
            "artery_mask": volume,
            "dice_artery": 0.5,
        }
        config = {
            "USE_GPU": False,
            "VESSELNESS_AORTA": {},
            "CIRCLE_DETECTION": {},
            "LEVEL_SET": {},
        }

        result = run_qualitative_pipeline_case(
            90,
            config,
            Path("dataset"),
        )

        self.assertEqual(result["img_id"], 90)
        self.assertIs(result["aorta_mask"], volume)
        self.assertIs(result["artery_mask"], volume)
        self.assertEqual(result["ostia_left"], (1, 1, 1))
        self.assertEqual(result["scaled_spacing"], (0.6, 0.7, 0.8))
        self.assertTrue(load_image.call_args.kwargs["include_intermediates"])
        compute_vesselness.assert_called_once()
        segment_arteries.assert_called_once()
