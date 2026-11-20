"""Testes do fluxo fixo de segmentação da aorta."""

import unittest

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import ball

from utils.segmentation.aorta.segmentation import (
    classify_aorta_segmentation_feedback,
    remove_leaks_morphology,
)
from utils.segmentation.pipeline.detection import segment_aorta_with_diagnostics


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

    def test_remove_leaks_morphology_bbox_matches_full_volume_and_empty(self):
        empty = np.zeros((10, 10, 10), dtype=np.uint8)
        np.testing.assert_array_equal(
            remove_leaks_morphology(empty, radius=2, use_gpu=False),
            empty,
        )

        mask = np.zeros((15, 16, 17), dtype=np.uint8)
        mask[0:6, 0:6, 0:6] = 1
        mask[8:15, 10:16, 11:17] = 1
        mask[7, 7, 7] = 1  # Vazamento fino isolado removido pela abertura

        expected = ndi.binary_opening(mask, structure=ball(2)).astype(np.uint8)
        actual = remove_leaks_morphology(mask, radius=2, use_gpu=False)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
