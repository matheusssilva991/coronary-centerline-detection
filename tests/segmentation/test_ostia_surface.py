"""Regressoes da superficie candidata e do intervalo axial dos ostios."""

from typing import cast
from unittest import TestCase

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import ball

from utils.segmentation.ostia_detection import (
    _extract_lower_region,
    check_ostium_intersection,
    find_aorta_surface,
)


class OstiaSurfaceTests(TestCase):
    def test_padding_matches_dilated_shell_without_changing_aorta(self):
        mask = np.zeros((25, 25, 25), dtype=np.uint8)
        mask[5:20, 5:20, 5:20] = 1
        original = mask.copy()
        for padding in (0, 1, 2, 3):
            with self.subTest(padding=padding):
                expanded = ndi.binary_dilation(mask, structure=ball(padding))
                eroded = np.asarray(
                    ndi.binary_erosion(expanded, structure=ball(4)), dtype=bool
                )
                expected = np.asarray(expanded, dtype=bool) & ~eroded
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
        self.assertEqual(
            np.flatnonzero(selected.any(axis=(0, 1))).tolist(), list(range(7))
        )

    def test_invalid_padding_is_rejected(self):
        for radius in (-1, cast(int, 1.5)):
            with self.assertRaises(ValueError):
                find_aorta_surface(np.ones((3, 3, 3)), surface_padding_radius=radius)

    def test_surface_bbox_matches_full_volume_on_edges_and_empty_masks(self):
        empty = np.zeros((10, 11, 12), dtype=np.uint8)
        np.testing.assert_array_equal(
            find_aorta_surface(empty, erosion_radius=2, surface_padding_radius=1),
            empty,
        )

        edge_mask = np.zeros((14, 15, 16), dtype=np.uint8)
        edge_mask[0:6, 0:6, 0:6] = 1
        edge_mask[9:14, 10:15, 11:16] = 1
        for padding in (0, 2):
            with self.subTest(padding=padding):
                expanded = (
                    ndi.binary_dilation(edge_mask > 0, structure=ball(padding))
                    if padding
                    else edge_mask > 0
                )
                eroded = np.asarray(
                    ndi.binary_erosion(expanded, structure=ball(2)), dtype=bool
                )
                expected = (np.asarray(expanded, dtype=bool) & ~eroded).astype(np.uint8)
                actual = find_aorta_surface(
                    edge_mask, erosion_radius=2, surface_padding_radius=padding
                )
                np.testing.assert_array_equal(actual, expected)

    def test_check_ostium_intersection_matches_edt_on_anisotropic_volumes(self):
        label = np.zeros((10, 11, 12), dtype=np.uint8)
        label[2:5, 7:10, 3:6] = 1
        spacing = (0.4, 0.8, 1.5)

        dist_mm, indices = cast(
            tuple[np.ndarray, np.ndarray],
            ndi.distance_transform_edt(
                label == 0, sampling=spacing, return_indices=True
            ),
        )

        for point in ((2, 7, 3), (8, 1, 10), (0, 0, 0), (9, 10, 11)):
            with self.subTest(point=point):
                info = check_ostium_intersection(
                    point, label, spacing=spacing, distance_threshold_mm=5.0
                )
                y, x, z = point
                if label[y, x, z] == 1:
                    self.assertTrue(info["intersects"])
                    self.assertEqual(info["physical_dist"], 0.0)
                    self.assertEqual(info["nearest_voxel"], point)
                else:
                    self.assertFalse(info["intersects"])
                    expected_nearest = (
                        int(indices[0, y, x, z]),
                        int(indices[1, y, x, z]),
                        int(indices[2, y, x, z]),
                    )
                    expected_eucl = float(
                        np.linalg.norm(
                            np.array(point, dtype=float)
                            - np.array(expected_nearest, dtype=float)
                        )
                    )
                    self.assertAlmostEqual(
                        info["physical_dist"], float(dist_mm[y, x, z])
                    )
                    self.assertAlmostEqual(info["euclidean_dist"], expected_eucl)
                    self.assertEqual(info["nearest_voxel"], expected_nearest)
