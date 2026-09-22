import unittest

import numpy as np

from utils.utils.roi import mask_bounding_box_slices


class MaskBoundingBoxSlicesTests(unittest.TestCase):
    def test_applies_physical_margin_per_axis(self):
        mask = np.zeros((6, 7, 8), dtype=bool)
        mask[2:4, 3:5, 4:6] = True

        slices = mask_bounding_box_slices(
            mask,
            spacing=(2.0, 1.0, 0.5),
            margin_mm=2.0,
        )

        self.assertEqual(
            slices,
            (slice(1, 5), slice(1, 7), slice(0, 8)),
        )

    def test_limits_margin_to_volume_bounds(self):
        mask = np.zeros((4, 5, 6), dtype=np.uint8)
        mask[0, 0, 0] = 1
        mask[-1, -1, -1] = 1

        slices = mask_bounding_box_slices(mask, (1.0, 1.0, 1.0), 10.0)

        self.assertEqual(slices, (slice(0, 4), slice(0, 5), slice(0, 6)))

    def test_rejects_empty_or_non_3d_mask(self):
        with self.assertRaisesRegex(ValueError, "não pode estar vazia"):
            mask_bounding_box_slices(
                np.zeros((2, 2, 2), dtype=bool),
                (1.0, 1.0, 1.0),
                0.0,
            )
        with self.assertRaisesRegex(ValueError, "tridimensional"):
            mask_bounding_box_slices(
                np.ones((2, 2), dtype=bool),
                (1.0, 1.0, 1.0),
                0.0,
            )

    def test_rejects_invalid_spacing(self):
        mask = np.ones((2, 2, 2), dtype=bool)
        for spacing in (
            (1.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, float("nan"), 1.0),
        ):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                mask_bounding_box_slices(mask, spacing, 0.0)

    def test_rejects_invalid_margin(self):
        mask = np.ones((2, 2, 2), dtype=bool)
        for margin in (-1.0, float("inf"), float("nan")):
            with (
                self.subTest(margin=margin),
                self.assertRaisesRegex(ValueError, "margem"),
            ):
                mask_bounding_box_slices(mask, (1.0, 1.0, 1.0), margin)


if __name__ == "__main__":
    unittest.main()
