"""Valida distâncias físicas entre superfícies com exemplos sintéticos."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np
from utils.utils.metrics import cKDTree, surface_distance_metrics


class SurfaceDistanceMetricsTests(TestCase):
    def test_identical_nonempty_masks_have_zero_distance_and_keep_inputs(self):
        mask = np.ones((3, 3, 3), dtype=np.uint8)
        original = mask.copy()
        with patch("utils.utils.metrics.cKDTree", wraps=cKDTree) as tree:
            result = surface_distance_metrics(mask, mask, (0.5, 1, 2))
        self.assertEqual(result, {"hd_mm": 0.0, "ahd_mm": 0.0, "status": "ok"})
        # O centro não faz parte da superfície; a borda do volume é preservada.
        self.assertEqual(tree.call_args_list[0].args[0].shape, (26, 3))
        np.testing.assert_array_equal(mask, original)

    def test_known_anisotropic_displacement_and_symmetry(self):
        first = np.zeros((5, 6, 7), dtype=np.uint8)
        second = np.zeros_like(first)
        first[0, 0, 0] = 1
        second[1, 2, 3] = 1
        expected = np.sqrt((1 * 0.5) ** 2 + (2 * 2) ** 2 + (3 * 3) ** 2)
        forward = surface_distance_metrics(first, second, (0.5, 2, 3))
        backward = surface_distance_metrics(second, first, (0.5, 2, 3))
        self.assertEqual(forward, backward)
        self.assertAlmostEqual(forward["hd_mm"], expected)
        self.assertAlmostEqual(forward["ahd_mm"], expected)

    def test_ahd_weights_directions_equally_not_points_or_maximum_mean(self):
        first = np.zeros((5, 3, 3), dtype=np.uint8)
        second = np.zeros_like(first)
        first[:3, 0, 0] = 1
        second[0, 0, 0] = 1
        result = surface_distance_metrics(first, second, (1, 1, 1))
        self.assertEqual(result["hd_mm"], 2)
        # A->B = [0, 1, 2], B->A = [0]: (1 + 0)/2, não 3/4 ou 1.
        self.assertEqual(result["ahd_mm"], 0.5)
        self.assertEqual(result, surface_distance_metrics(second, first, (1, 1, 1)))

    def test_distant_component_is_not_removed(self):
        reference = np.zeros((12, 3, 3), dtype=np.uint8)
        reference[0, 1, 1] = 1
        prediction = reference.copy()
        prediction[10, 1, 1] = 1
        result = surface_distance_metrics(prediction, reference, (2, 1, 1))
        self.assertEqual(result["hd_mm"], 20)
        self.assertEqual(result["ahd_mm"], 5)

    def test_empty_masks_are_unavailable_without_building_trees(self):
        empty = np.zeros((3, 3, 3), dtype=np.uint8)
        full = np.ones_like(empty)
        cases = (
            (empty, full, "prediction_empty"),
            (full, empty, "reference_empty"),
            (empty, empty, "both_empty"),
        )
        with patch("utils.utils.metrics.cKDTree") as tree:
            for first, second, status in cases:
                with self.subTest(status=status):
                    result = surface_distance_metrics(first, second, (1, 1, 1))
                    self.assertEqual(
                        result, {"hd_mm": None, "ahd_mm": None, "status": status}
                    )
        tree.assert_not_called()

    def test_invalid_geometry_and_spacing_raise_clear_errors(self):
        mask = np.ones((3, 3, 3), dtype=np.uint8)
        for other in (np.ones((2, 3, 3)), np.ones((3, 3)), np.ones((0, 3, 3))):
            with self.subTest(shape=other.shape), self.assertRaises(ValueError):
                surface_distance_metrics(mask, other, (1, 1, 1))
        with self.assertRaises(ValueError):
            surface_distance_metrics(np.ones((3, 3)), np.ones((3, 3)), (1, 1, 1))
        for spacing in ((1, 2), (1, 0, 1), (-1, 1, 1), (1, np.nan, 1), (1, np.inf, 1)):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                surface_distance_metrics(mask, mask, spacing)
