"""Regressoes da superficie candidata e do intervalo axial dos ostios."""

from unittest import TestCase

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import ball

from utils.segmentation.ostia_detection import _extract_lower_region, find_aorta_surface


class OstiaSurfaceTests(TestCase):
    def test_padding_matches_dilated_shell_without_changing_aorta(self):
        mask = np.zeros((25, 25, 25), dtype=np.uint8)
        mask[5:20, 5:20, 5:20] = 1
        original = mask.copy()
        for padding in (0, 1, 2, 3):
            with self.subTest(padding=padding):
                expanded = ndi.binary_dilation(mask, structure=ball(padding))
                expected = expanded & ~ndi.binary_erosion(expanded, structure=ball(4))
                actual = find_aorta_surface(mask, 4, padding)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(mask, original)

    def test_full_fraction_includes_last_occupied_slice(self):
        mask = np.zeros((4, 4, 10), dtype=np.uint8)
        mask[1, 1, 2:8] = 1
        selected, start, end = _extract_lower_region(mask, 1.0)
        np.testing.assert_array_equal(selected, mask)
        self.assertEqual((start, end), (2, 7))

    def test_partial_fraction_preserves_historical_interval(self):
        mask = np.ones((3, 3, 10), dtype=np.uint8)
        selected, _, _ = _extract_lower_region(mask, 0.85)
        self.assertEqual(np.flatnonzero(selected.any(axis=(0, 1))).tolist(), list(range(7)))

    def test_invalid_padding_is_rejected(self):
        for radius in (-1, 1.5):
            with self.assertRaises(ValueError):
                find_aorta_surface(np.ones((3, 3, 3)), surface_padding_radius=radius)
