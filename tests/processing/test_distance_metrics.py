"""Valida distâncias físicas arteriais com exemplos sintéticos."""

from dataclasses import replace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from utils.project.analysis.artery_distance_diagnostics import (
    sample_foreground_distances,
)
from utils.project.evaluation.artery_distances import evaluate_artery_distances
from utils.project.evaluation.surface_distance_validation import (
    compare_surface_distance_samples,
    reference_surface_distances,
)
from utils.utils.metrics import (
    cKDTree,
    hausdorff_distances,
    sample_surface_distances,
    surface_hd95,
)


class SurfaceDistanceMetricsTests(TestCase):
    def test_foreground_samples_follow_mask_index_order_and_reproduce_hausdorff(self):
        prediction = np.zeros((9, 10, 11), dtype=bool)
        prediction[2:5, 3:6, 1:4] = True
        prediction[5, 4, 7] = True
        reference = prediction.copy()
        reference[3, 4, 2] = False
        reference[5, 4, 7] = False
        reference[6, 5, 1] = True
        spacing = (0.5, 1.25, 2.0)

        sampled = sample_foreground_distances(prediction, reference, spacing)
        assert sampled is not None
        forward, backward = sampled
        exact = cdist(
            np.argwhere(prediction) * spacing, np.argwhere(reference) * spacing
        )
        np.testing.assert_allclose(forward, exact.min(axis=1))
        np.testing.assert_allclose(backward, exact.min(axis=0))
        np.testing.assert_allclose(
            (
                max(forward.max(), backward.max()),
                (forward.mean() + backward.mean()) / 2,
            ),
            hausdorff_distances(prediction, reference, spacing),
        )
        self.assertIsNone(
            sample_foreground_distances(np.zeros_like(prediction), reference, spacing)
        )

    def test_edt_matches_directional_distances_including_volume_boundaries(self):
        prediction = np.zeros((12, 13, 14), dtype=bool)
        prediction[:4, :4, :4] = True
        prediction[8:12, 7:10, 10:14] = True
        reference = np.zeros_like(prediction)
        reference[2:6, 1:5, :4] = True
        spacing = (0.52, 1.17, 2.35)
        calculated = sample_surface_distances(prediction, reference, spacing)
        expected = reference_surface_distances(prediction, reference, spacing)
        self.assertIsNotNone(expected)
        assert expected is not None
        hausdorff = calculated.hausdorff_distances()
        assert hausdorff is not None

        comparison = compare_surface_distance_samples(calculated, expected)
        self.assertLess(comparison.max_directional_error_mm, 1e-12)
        np.testing.assert_allclose(
            hausdorff,
            (comparison.reference_hd_mm, comparison.reference_ahd_mm),
            rtol=0,
            atol=1e-12,
        )

    def test_edt_comparison_detects_per_point_error_and_different_coordinates(self):
        mask = np.ones((2, 3, 4), dtype=bool)
        expected = reference_surface_distances(mask, mask, (0.5, 1.0, 2.0))
        self.assertIsNotNone(expected)
        assert expected is not None
        incorrect_distances = replace(
            expected,
            prediction_to_reference_mm=np.ones(len(expected.prediction_points_mm)),
        )
        comparison = compare_surface_distance_samples(incorrect_distances, expected)
        self.assertEqual(comparison.max_directional_error_mm, 1.0)
        self.assertEqual(
            (comparison.reference_hd_mm, comparison.reference_ahd_mm), (0, 0)
        )

        incorrect_points = replace(
            expected, reference_points_mm=expected.reference_points_mm + 1
        )
        comparison = compare_surface_distance_samples(incorrect_points, expected)
        self.assertEqual(comparison.max_directional_error_mm, np.inf)

    def test_surface_samples_match_anisotropic_coordinates_in_both_directions(self):
        prediction = np.zeros((5, 6, 7), dtype=bool)
        prediction[0, 1, 2] = True
        prediction[4, 5, 6] = True
        reference = np.zeros_like(prediction)
        reference[0, 1, 2] = True

        samples = sample_surface_distances(prediction, reference, (0.5, 1.5, 2.0))

        np.testing.assert_array_equal(
            samples.prediction_points_mm, [[0.0, 1.5, 4.0], [2.0, 7.5, 12.0]]
        )
        np.testing.assert_array_equal(samples.reference_points_mm, [[0.0, 1.5, 4.0]])
        expected_hd = np.sqrt(2**2 + 6**2 + 8**2)
        forward = samples.prediction_to_reference_mm
        hausdorff = samples.hausdorff_distances()
        assert forward is not None
        assert hausdorff is not None
        np.testing.assert_allclose(forward, [0, expected_hd])
        np.testing.assert_array_equal(samples.reference_to_prediction_mm, [0])
        np.testing.assert_allclose(hausdorff, (expected_hd, expected_hd / 4))

    def test_identical_surface_samples_exclude_interior_and_have_zero_distances(self):
        mask = np.ones((3, 3, 3), dtype=bool)
        samples = sample_surface_distances(mask, mask, (0.5, 1, 2))

        self.assertEqual(len(samples.prediction_points_mm), 26)
        np.testing.assert_array_equal(
            samples.prediction_points_mm, samples.reference_points_mm
        )
        np.testing.assert_array_equal(samples.prediction_to_reference_mm, np.zeros(26))
        np.testing.assert_array_equal(samples.reference_to_prediction_mm, np.zeros(26))
        self.assertEqual(samples.hausdorff_distances(), (0.0, 0.0))
        self.assertTrue(mask.all())

    def test_empty_surface_samples_keep_available_coordinates_without_distances(self):
        empty = np.zeros((3, 3, 3), dtype=bool)
        reference = empty.copy()
        reference[2, 1, 0] = True

        with patch("utils.utils.metrics.cKDTree") as tree:
            for prediction, target in (
                (empty, reference),
                (reference, empty),
                (empty, empty),
            ):
                samples = sample_surface_distances(prediction, target, (2, 1, 0.5))
                self.assertEqual(len(samples.prediction_points_mm), prediction.sum())
                self.assertEqual(len(samples.reference_points_mm), target.sum())
                self.assertIsNone(samples.prediction_to_reference_mm)
                self.assertIsNone(samples.reference_to_prediction_mm)
                self.assertIsNone(samples.hausdorff_distances())
            tree.assert_not_called()

    def test_identical_nonempty_masks_have_zero_distance_and_keep_inputs(self):
        mask = np.ones((3, 3, 3), dtype=np.uint8)
        original = mask.copy()
        with patch("utils.utils.metrics.cKDTree", wraps=cKDTree) as tree:
            self.assertEqual(surface_hd95(mask > 0, mask > 0, (0.5, 1, 2)), 0)
        self.assertEqual(hausdorff_distances(mask, mask, (0.5, 1, 2)), (0, 0))
        # O centro não faz parte da superfície; a borda do volume é preservada.
        self.assertEqual(tree.call_args_list[0].args[0].shape, (26, 3))
        np.testing.assert_array_equal(mask, original)

    def test_known_anisotropic_displacement_and_symmetry(self):
        first = np.zeros((5, 6, 7), dtype=np.uint8)
        second = np.zeros_like(first)
        first[0, 0, 0] = 1
        second[1, 2, 3] = 1
        expected = np.sqrt((1 * 0.5) ** 2 + (2 * 2) ** 2 + (3 * 3) ** 2)
        forward = hausdorff_distances(first, second, (0.5, 2, 3))
        backward = hausdorff_distances(second, first, (0.5, 2, 3))
        self.assertEqual(forward, backward)
        self.assertAlmostEqual(forward[0], expected)
        self.assertAlmostEqual(forward[1], expected)

    def test_ahd_weights_directions_equally_not_points_or_maximum_mean(self):
        first = np.zeros((5, 3, 3), dtype=np.uint8)
        second = np.zeros_like(first)
        first[:3, 0, 0] = 1
        second[0, 0, 0] = 1
        result = hausdorff_distances(first, second, (1, 1, 1))
        self.assertEqual(result[0], 2)
        # A->B = [0, 1, 2], B->A = [0]: (1 + 0)/2, não 3/4 ou 1.
        self.assertEqual(result[1], 0.5)
        self.assertEqual(result, hausdorff_distances(second, first, (1, 1, 1)))

    def test_distant_component_is_not_removed(self):
        reference = np.zeros((12, 3, 3), dtype=np.uint8)
        reference[0, 1, 1] = 1
        prediction = reference.copy()
        prediction[10, 1, 1] = 1
        result = hausdorff_distances(prediction, reference, (2, 1, 1))
        self.assertEqual(result[0], 20)
        self.assertEqual(result[1], 5)

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
                    result = evaluate_artery_distances(
                        first,
                        second,
                        (1, 1, 1),
                        {"EVALUATION": {"distance_metrics": True}},
                    )
                    self.assertEqual(result["artery_distance_metrics_status"], status)
                    for column in (
                        "artery_hd_foreground_mm",
                        "artery_avghd_foreground_mm",
                        "artery_hd95_surface_mm",
                    ):
                        self.assertIsNone(result[column])
        tree.assert_not_called()

    def test_invalid_geometry_and_spacing_raise_clear_errors(self):
        mask = np.ones((3, 3, 3), dtype=np.uint8)
        invalid: list[
            tuple[NDArray[np.generic], NDArray[np.generic], tuple[float, ...]]
        ] = [
            (mask, other, (1, 1, 1))
            for other in (np.ones((2, 3, 3)), np.ones((3, 3)), np.ones((0, 3, 3)))
        ]
        invalid.append((np.ones((3, 3)), np.ones((3, 3)), (1, 1, 1)))
        invalid.extend(
            (mask, mask, spacing)
            for spacing in (
                (1, 2),
                (1, 0, 1),
                (-1, 1, 1),
                (1, np.nan, 1),
                (1, np.inf, 1),
            )
        )
        for prediction, reference, spacing in invalid:
            result = evaluate_artery_distances(
                prediction,
                reference,
                spacing,
                {"EVALUATION": {"distance_metrics": True}},
            )
            self.assertEqual(result["artery_distance_metrics_status"], "error")
            self.assertIn("ValueError", result["artery_distance_metrics_error"])

    def test_surface_hd95_bbox_matches_full_volume_erosion_on_borders_and_interior(
        self,
    ):
        pred = np.zeros((16, 17, 18), dtype=bool)
        ref = np.zeros_like(pred)
        pred[0:4, 0:4, 0:4] = True
        pred[7:11, 8:12, 9:13] = True
        ref[1:5, 0:4, 1:5] = True
        ref[12:16, 13:17, 14:18] = True
        spacing = (0.5, 0.75, 1.25)

        structure = generate_binary_structure(3, 1)
        pred_surf = pred & ~np.asarray(
            binary_erosion(pred, structure=structure, border_value=0), dtype=bool
        )
        ref_surf = ref & ~np.asarray(
            binary_erosion(ref, structure=structure, border_value=0), dtype=bool
        )
        pred_pts = np.argwhere(pred_surf) * spacing
        ref_pts = np.argwhere(ref_surf) * spacing
        fwd = KDTree(ref_pts).query(pred_pts, k=1, eps=0)[0]
        bwd = KDTree(pred_pts).query(ref_pts, k=1, eps=0)[0]
        expected = float(
            max(
                np.percentile(fwd, 95, method="linear"),
                np.percentile(bwd, 95, method="linear"),
            )
        )

        self.assertAlmostEqual(surface_hd95(pred, ref, spacing), expected)
