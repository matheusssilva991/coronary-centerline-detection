import unittest

import numpy as np

from utils.segmentation.artery_segmentation import expand_region_from_mask_mean


class SecondRegionGrowingTests(unittest.TestCase):
    def test_expands_from_mask_using_running_mean(self) -> None:
        vesselness = np.zeros((5, 5, 1), dtype=float)
        vesselness[2, 1:4, 0] = [0.75, 0.80, 0.78]
        initial_mask = np.zeros_like(vesselness, dtype=np.uint8)
        initial_mask[2, 2, 0] = 1

        result, details = expand_region_from_mask_mean(
            vesselness,
            initial_mask,
            tolerance=0.1,
            min_vesselness=0.7,
            max_new_voxels=10,
            neighborhood=6,
        )

        self.assertEqual(int(result.sum()), 3)
        self.assertEqual(details["initial_voxels"], 1)
        self.assertEqual(details["added_voxels"], 2)
        self.assertAlmostEqual(details["initial_mean_vesselness"], 0.8)

    def test_preserves_initial_mask_and_respects_growth_limit(self) -> None:
        vesselness = np.full((5, 5, 1), 0.8, dtype=float)
        initial_mask = np.zeros_like(vesselness, dtype=np.uint8)
        initial_mask[2, 2, 0] = 1

        result, details = expand_region_from_mask_mean(
            vesselness,
            initial_mask,
            tolerance=0.01,
            min_vesselness=0.5,
            max_new_voxels=2,
            neighborhood=6,
        )

        self.assertEqual(int(result.sum()), 3)
        self.assertEqual(details["added_voxels"], 2)
        self.assertEqual(result[2, 2, 0], 1)


if __name__ == "__main__":
    unittest.main()
