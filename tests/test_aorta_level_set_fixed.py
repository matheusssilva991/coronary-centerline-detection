"""Testes do fluxo fixo de segmentação da aorta."""

import unittest

import numpy as np

from utils.segmentation.aorta_segmentation import (
    classify_aorta_segmentation_feedback,
)
from utils.segmentation.pipeline_detection import segment_aorta_with_diagnostics


class FixedAortaLevelSetTests(unittest.TestCase):
    def test_classifies_aorta_quality_feedback(self):
        self.assertEqual(
            classify_aorta_segmentation_feedback(0.9, 1.6, 0.012),
            "adequate",
        )
        self.assertEqual(
            classify_aorta_segmentation_feedback(0.9, 3.1, 0.012),
            "suspected_oversegmentation",
        )
        self.assertEqual(
            classify_aorta_segmentation_feedback(0.7, 1.2, 0.010),
            "suspected_undersegmentation",
        )
        self.assertEqual(
            classify_aorta_segmentation_feedback(None, 1.2, 0.010),
            "insufficient_data",
        )

    def test_empty_circle_trajectory_returns_empty_mask_and_fixed_diagnostics(self):
        volume = np.zeros((8, 8, 4), dtype=np.float32)
        config = {
            "num_iter": 26,
            "radius_reduction_factor": 0.1,
            "balloon": 0.6,
            "smoothing": 2,
            "threshold": "auto",
            "alpha": 1000,
            "sigma": 2,
            "leak_removal_radius": 0,
        }

        result = segment_aorta_with_diagnostics(volume, [], config)

        self.assertEqual(int(result.mask.sum()), 0)
        self.assertEqual(result.diagnostics["aorta_level_set_iterations_used"], 26)
        self.assertIn("aorta_slice_area_jump_p95", result.diagnostics)


if __name__ == "__main__":
    unittest.main()
