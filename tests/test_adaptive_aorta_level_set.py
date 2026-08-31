"""Tests for the checkpoint-based adaptive aorta level-set controller."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np
from skimage.segmentation import morphsnakes

from utils.processing.binary_operations import keep_largest_component
from utils.segmentation.aorta_segmentation import (
    calculate_circle_mask_metrics,
    calculate_mask_change_fraction,
    calculate_slice_area_jump_p95,
    iter_level_set_checkpoints,
    level_set_segmentation,
    prepare_level_set_evolution,
    remove_leaks_morphology,
)
from utils.segmentation.pipeline_detection import (
    _AdaptiveCheckpoint,
    _accept_conservative_candidate,
    _adaptive_checkpoint_iterations,
    _circle_confidence_signal_count,
    _classify_adaptive_state,
    _should_override_localization_for_leak,
    segment_aorta_with_diagnostics,
)


def _adaptive_config(mode="adaptive"):
    return {
        "iteration_mode": mode,
        "radius_reduction_factor": 0.4,
        "num_iter": 31,
        "balloon": 0.8,
        "smoothing": 1,
        "threshold": "auto",
        "roi_margin": 2,
        "use_roi": True,
        "alpha": 1000,
        "sigma": 1,
        "leak_removal_radius": 0,
        "adaptive": {
            "min_iter": 16,
            "check_interval": 5,
            "early_stop_iteration": 26,
            "convergence_tolerance": 0.01,
            "convergence_patience": 2,
            "adequate_min_circle_fill_q25": 0.8,
            "adequate_min_circle_area_ratio_p90": 1.3,
            "adequate_max_circle_area_ratio_p90": 2.7,
            "oversegmented_area_ratio_p90": 2.7,
            "localization_min_circle_fill_q25": 0.25,
            "localization_signal_threshold": 4,
            "localization_min_radius_median_mm": 13.9,
            "localization_max_radius_step_mm": 4.8,
            "localization_max_radius_p90_step_mm": 1.85,
            "localization_min_hough_accumulator": 0.408,
            "localization_max_lower_bound_fraction": 0.05,
            "localization_leak_override": {
                "enabled": False,
                "min_area_ratio_p90": 2.0,
                "min_circle_fill_q25": 0.8,
                "min_volume_fraction": 0.015,
            },
            "max_fill_loss": 0.01,
            "max_axial_jump_increase_fraction": 0.1,
            "min_area_ratio_improvement_fraction": 0.05,
            "conservative": {
                "balloon": 0.5,
                "alpha": 1500,
                "threshold_percentile": 55,
                "smoothing": 2,
            },
        },
    }


def _mask(voxels, shape=(20, 20, 10)):
    mask = np.zeros(shape, dtype=np.uint8)
    mask.reshape(-1)[:voxels] = 1
    return mask


def _checkpoint(
    *, iterations=31, voxels=50, volume=0.012, fill=0.9, area=1.8,
    change=0.05, mask=None, voxels_per_slice=2500.0,
):
    mask = _mask(voxels) if mask is None else mask
    return _AdaptiveCheckpoint(
        iterations=iterations,
        mask=mask,
        voxel_count=int(mask.sum()),
        voxels_per_segmented_slice=voxels_per_slice,
        volume_fraction=volume,
        relative_growth=0.05,
        mask_change_fraction=change,
        circle_fill_q25=fill,
        circle_area_ratio_p90=area,
        leak_signal=area > 2.7,
    )


def _circle():
    return {"slice_index": 5, "center_x": 10.0, "center_y": 10.0, "radius": 4.0}


def _reset_morphgac_curvature_cycle():
    cycle_factory = getattr(morphsnakes, "_fcycle")
    setattr(
        morphsnakes,
        "_curvop",
        cycle_factory(
            [
                lambda mask: morphsnakes.sup_inf(morphsnakes.inf_sup(mask)),
                lambda mask: morphsnakes.inf_sup(morphsnakes.sup_inf(mask)),
            ]
        ),
    )


class AortaLevelSetRegressionTests(TestCase):
    def test_checkpoint_schedule_contains_nominal(self):
        self.assertEqual(_adaptive_checkpoint_iterations(16, 31, 5), [16, 21, 26, 31])
        self.assertEqual(
            _adaptive_checkpoint_iterations(36, 70, 11), [36, 47, 58, 69, 70]
        )

    def test_chunked_evolution_matches_continuous_execution(self):
        rng = np.random.default_rng(42)
        volume = rng.normal(size=(20, 20, 8)).astype(np.float32)
        circles = [{**_circle(), "slice_index": 3}]
        kwargs = {
            "num_iter": 31, "smoothing": 1, "balloon": 0.8,
            "threshold": "auto", "radius_reduction_factor": 0.4,
            "roi_margin": 2, "use_roi": True, "alpha": 1000,
            "sigma": 1, "use_gpu": False,
        }
        _reset_morphgac_curvature_cycle()
        continuous = level_set_segmentation(volume, circles, **kwargs)
        _reset_morphgac_curvature_cycle()
        chunked = list(
            iter_level_set_checkpoints(
                volume, circles, [16, 21, 26, 31],
                **{key: value for key, value in kwargs.items() if key != "num_iter"},
            )
        )[-1][1]
        np.testing.assert_array_equal(chunked, continuous)

    def test_fixed_mode_matches_historical_flow(self):
        rng = np.random.default_rng(7)
        volume = rng.normal(size=(20, 20, 8)).astype(np.float32)
        circles = [{**_circle(), "slice_index": 3}]
        config = _adaptive_config(mode="fixed")
        _reset_morphgac_curvature_cycle()
        raw = level_set_segmentation(
            volume, circles,
            radius_reduction_factor=config["radius_reduction_factor"],
            num_iter=config["num_iter"], balloon=config["balloon"],
            smoothing=config["smoothing"], threshold=config["threshold"],
            roi_margin=config["roi_margin"], use_roi=config["use_roi"],
            alpha=config["alpha"], sigma=config["sigma"], use_gpu=False,
        )
        expected = keep_largest_component(
            remove_leaks_morphology(raw, radius=0, use_gpu=False), gpu=False
        )
        _reset_morphgac_curvature_cycle()
        result = segment_aorta_with_diagnostics(volume, circles, config)
        np.testing.assert_array_equal(result.mask, expected)

        context = prepare_level_set_evolution(
            volume,
            circles,
            radius_reduction_factor=config["radius_reduction_factor"],
            roi_margin=config["roi_margin"],
            use_roi=config["use_roi"],
            alpha=config["alpha"],
            sigma=config["sigma"],
            use_gpu=False,
        )
        initial_voxels = int(np.count_nonzero(context.current_mask))
        self.assertEqual(
            result.diagnostics["aorta_level_set_initial_voxel_count"],
            initial_voxels,
        )
        self.assertEqual(
            result.diagnostics["aorta_level_set_raw_voxel_count"],
            int(np.count_nonzero(raw)),
        )
        self.assertAlmostEqual(
            result.diagnostics["aorta_level_set_initial_volume_fraction"],
            initial_voxels / volume.size,
        )
        self.assertAlmostEqual(
            result.diagnostics["aorta_level_set_raw_volume_fraction"],
            int(np.count_nonzero(raw)) / volume.size,
        )

    def test_metric_helpers(self):
        previous = _mask(3)
        current = np.roll(previous, 1, axis=0)
        self.assertGreater(calculate_mask_change_fraction(previous, current), 0)
        self.assertGreaterEqual(calculate_slice_area_jump_p95(current), 0)
        self.assertIn("circle_fill_q25", calculate_circle_mask_metrics(current, [_circle()]))

    def test_fixed_mode_without_circles_preserves_empty_mask(self):
        volume = np.zeros((20, 20, 8), dtype=np.float32)
        result = segment_aorta_with_diagnostics(
            volume,
            [],
            _adaptive_config(mode="fixed"),
        )

        self.assertEqual(int(result.mask.sum()), 0)
        self.assertEqual(
            result.diagnostics["aorta_level_set_initial_voxel_count"], 0
        )
        self.assertEqual(result.diagnostics["aorta_level_set_raw_voxel_count"], 0)


class AdaptiveStateTests(TestCase):
    def setUp(self):
        self.adaptive = _adaptive_config()["adaptive"]

    def test_classifies_active_controller_states(self):
        self.assertEqual(_classify_adaptive_state(_checkpoint(), 0, self.adaptive), "adequate")
        self.assertEqual(
            _classify_adaptive_state(_checkpoint(volume=0.025, area=2.8), 0, self.adaptive),
            "oversegmented",
        )
        self.assertEqual(
            _classify_adaptive_state(
                _checkpoint(volume=0.006, fill=0.7, area=1.1), 0, self.adaptive
            ),
            "adequate",
        )
        self.assertEqual(
            _classify_adaptive_state(
                _checkpoint(volume=0.03, area=2.6, voxels_per_slice=4000),
                0,
                self.adaptive,
            ),
            "adequate",
        )
        self.assertEqual(
            _classify_adaptive_state(_checkpoint(fill=0.2), 0, self.adaptive),
            "localization_suspected",
        )
        self.assertEqual(
            _classify_adaptive_state(_checkpoint(), 4, self.adaptive),
            "localization_suspected",
        )

    def test_oversegmentation_threshold_is_strict(self):
        self.assertEqual(
            _classify_adaptive_state(
                _checkpoint(area=2.7),
                0,
                self.adaptive,
            ),
            "adequate",
        )
        self.assertEqual(
            _classify_adaptive_state(
                _checkpoint(area=2.7001),
                0,
                self.adaptive,
            ),
            "oversegmented",
        )

    def test_circle_confidence_uses_five_independent_signals(self):
        summary = {
            "aorta_circle_radius_median_mm": 13.0,
            "aorta_circle_radius_max_step_change_mm": 5.0,
            "aorta_circle_radius_p90_step_change_mm": 2.0,
            "aorta_circle_mean_hough_accumulator": 0.40,
            "aorta_circle_lower_radius_bound_fraction": 0.06,
        }
        self.assertEqual(_circle_confidence_signal_count(summary, self.adaptive), 5)
        self.assertEqual(_circle_confidence_signal_count(None, self.adaptive), 0)

    def test_localization_leak_override_requires_all_strong_signals(self):
        override = self.adaptive["localization_leak_override"]
        override["enabled"] = True

        self.assertTrue(
            _should_override_localization_for_leak(
                _checkpoint(area=2.1, fill=0.8, volume=0.015),
                "localization_suspected",
                self.adaptive,
            )
        )
        self.assertFalse(
            _should_override_localization_for_leak(
                _checkpoint(area=2.0, fill=0.8, volume=0.015),
                "localization_suspected",
                self.adaptive,
            )
        )
        self.assertFalse(
            _should_override_localization_for_leak(
                _checkpoint(area=2.1, fill=0.79, volume=0.015),
                "localization_suspected",
                self.adaptive,
            )
        )
        self.assertFalse(
            _should_override_localization_for_leak(
                _checkpoint(area=2.1, fill=0.8, volume=0.0149),
                "localization_suspected",
                self.adaptive,
            )
        )

    def test_localization_leak_override_is_disabled_by_default(self):
        self.assertFalse(
            _should_override_localization_for_leak(
                _checkpoint(area=4.0, fill=0.95, volume=0.04),
                "localization_suspected",
                self.adaptive,
            )
        )


class AlternativeAcceptanceTests(TestCase):
    def setUp(self):
        self.adaptive = _adaptive_config()["adaptive"]

    @patch(
        "utils.segmentation.pipeline_detection.calculate_slice_area_jump_p95",
        side_effect=[0.2, 0.21],
    )
    def test_accepts_conservative_candidate(self, _):
        nominal = _checkpoint(volume=0.025, fill=0.9, area=4.0)
        candidate = _checkpoint(volume=0.018, fill=0.895, area=2.7)
        self.assertEqual(
            _accept_conservative_candidate(nominal, candidate, self.adaptive),
            (True, "accepted"),
        )

    def test_rejects_conservative_fill_loss(self):
        nominal = _checkpoint(volume=0.025, fill=0.9, area=4.0)
        candidate = _checkpoint(volume=0.018, fill=0.85, area=2.7)
        self.assertEqual(
            _accept_conservative_candidate(nominal, candidate, self.adaptive)[1],
            "circle_fill_loss",
        )

    def test_rejects_conservative_candidate_below_minimum_improvement(self):
        nominal = _checkpoint(volume=0.025, fill=0.9, area=4.0)
        candidate = _checkpoint(volume=0.018, fill=0.895, area=3.84)

        self.assertEqual(
            _accept_conservative_candidate(nominal, candidate, self.adaptive),
            (False, "area_reduction_below_minimum"),
        )

class AdaptiveControllerIntegrationTests(TestCase):
    def _run(self, checkpoints, alternative=None, circle_summary=None):
        config = _adaptive_config()
        raw = [(item.iterations, item.mask) for item in checkpoints]
        circle_metrics = [
            {"circle_fill_q25": item.circle_fill_q25,
             "circle_area_ratio_p90": item.circle_area_ratio_p90}
            for item in checkpoints
        ]
        with (
            patch("utils.segmentation.pipeline_detection.prepare_level_set_evolution"),
            patch(
                "utils.segmentation.pipeline_detection.iter_level_set_checkpoints",
                return_value=iter(raw),
            ),
            patch(
                "utils.segmentation.pipeline_detection._postprocess_aorta_mask",
                side_effect=lambda mask, *_: mask,
            ),
            patch(
                "utils.segmentation.pipeline_detection.calculate_circle_mask_metrics",
                side_effect=circle_metrics,
            ),
            patch(
                "utils.segmentation.pipeline_detection._run_alternative_evolution",
                return_value=alternative,
            ) as branch,
        ):
            result = segment_aorta_with_diagnostics(
                np.zeros((20, 20, 10), dtype=np.float32), [_circle()], config,
                circle_summary=circle_summary,
            )
        return result, branch

    def test_safe_early_stop_at_iteration_26(self):
        checkpoints = [
            _checkpoint(iterations=iteration, mask=_mask(voxels), change=change)
            for iteration, voxels, change in (
                (16, 50, 0.05), (21, 50, 0.0),
                (26, 50, 0.0), (31, 50, 0.0),
            )
        ]
        result, branch = self._run(checkpoints)
        self.assertEqual(result.diagnostics["aorta_level_set_stop_reason"], "early_stable")
        self.assertEqual(result.diagnostics["aorta_level_set_iterations_used"], 26)
        branch.assert_not_called()

    def test_localization_suspected_keeps_nominal(self):
        checkpoints = [
            _checkpoint(iterations=iteration, mask=_mask(40 + index), change=0.05)
            for index, iteration in enumerate((16, 21, 26, 31))
        ]
        summary = {
            "aorta_circle_radius_median_mm": 13.0,
            "aorta_circle_radius_max_step_change_mm": 5.0,
            "aorta_circle_radius_p90_step_change_mm": 2.0,
            "aorta_circle_mean_hough_accumulator": 0.4,
            "aorta_circle_lower_radius_bound_fraction": 0.0,
        }
        result, branch = self._run(checkpoints, circle_summary=summary)
        np.testing.assert_array_equal(result.mask, checkpoints[-1].mask)
        self.assertEqual(
            result.diagnostics["aorta_level_set_controller_state"],
            "localization_suspected",
        )
        branch.assert_not_called()

    @patch(
        "utils.segmentation.pipeline_detection.calculate_slice_area_jump_p95",
        side_effect=[0.2, 0.21, 0.2, 0.21],
    )
    def test_strong_leak_overrides_localization_block(self, _):
        checkpoints = [
            _checkpoint(iterations=16, mask=_mask(40), volume=0.01, area=1.8),
            _checkpoint(iterations=21, mask=_mask(50), volume=0.012, area=1.9),
            _checkpoint(iterations=26, mask=_mask(70), volume=0.018, area=2.1),
            _checkpoint(
                iterations=31,
                mask=_mask(100),
                volume=0.03,
                fill=0.9,
                area=3.5,
            ),
        ]
        alternative = _checkpoint(
            iterations=31,
            mask=_mask(80),
            volume=0.02,
            fill=0.895,
            area=2.7,
        )
        summary = {
            "aorta_circle_radius_median_mm": 13.0,
            "aorta_circle_radius_max_step_change_mm": 5.0,
            "aorta_circle_radius_p90_step_change_mm": 2.0,
            "aorta_circle_mean_hough_accumulator": 0.4,
            "aorta_circle_lower_radius_bound_fraction": 0.0,
        }

        config = _adaptive_config()
        config["adaptive"]["localization_leak_override"]["enabled"] = True
        raw = [(item.iterations, item.mask) for item in checkpoints]
        circle_metrics = [
            {
                "circle_fill_q25": item.circle_fill_q25,
                "circle_area_ratio_p90": item.circle_area_ratio_p90,
            }
            for item in checkpoints
        ]
        with (
            patch("utils.segmentation.pipeline_detection.prepare_level_set_evolution"),
            patch(
                "utils.segmentation.pipeline_detection.iter_level_set_checkpoints",
                return_value=iter(raw),
            ),
            patch(
                "utils.segmentation.pipeline_detection._postprocess_aorta_mask",
                side_effect=lambda mask, *_: mask,
            ),
            patch(
                "utils.segmentation.pipeline_detection.calculate_circle_mask_metrics",
                side_effect=circle_metrics,
            ),
            patch(
                "utils.segmentation.pipeline_detection._run_alternative_evolution",
                return_value=alternative,
            ) as branch,
        ):
            result = segment_aorta_with_diagnostics(
                np.zeros((20, 20, 10), dtype=np.float32),
                [_circle()],
                config,
                circle_summary=summary,
            )

        branch.assert_called_once()
        self.assertEqual(
            result.diagnostics["aorta_level_set_controller_state"],
            "localization_suspected",
        )
        self.assertTrue(
            result.diagnostics[
                "aorta_level_set_localization_leak_override_triggered"
            ]
        )
        self.assertTrue(result.diagnostics["aorta_level_set_leak_suspected"])

    @patch(
        "utils.segmentation.pipeline_detection.calculate_slice_area_jump_p95",
        side_effect=[0.2, 0.21, 0.2, 0.21],
    )
    def test_accepts_conservative_branch(self, _):
        checkpoints = [
            _checkpoint(iterations=16, mask=_mask(40), volume=0.01, area=2.0),
            _checkpoint(iterations=21, mask=_mask(50), volume=0.012, area=2.4),
            _checkpoint(iterations=26, mask=_mask(70), volume=0.018, area=2.9),
            _checkpoint(iterations=31, mask=_mask(100), volume=0.025, area=4.0),
        ]
        alternative = _checkpoint(
            iterations=31, mask=_mask(80), volume=0.018, fill=0.895, area=2.7
        )
        result, branch = self._run(checkpoints, alternative=alternative)
        np.testing.assert_array_equal(result.mask, alternative.mask)
        self.assertTrue(result.diagnostics["aorta_level_set_conservative_accepted"])
        self.assertEqual(result.diagnostics["aorta_level_set_rollback_iteration"], 21)
        branch.assert_called_once()

    def test_rejected_branch_returns_exact_nominal(self):
        checkpoints = [
            _checkpoint(iterations=iteration, mask=_mask(40 + 10 * index),
                        volume=0.01, change=0.05)
            for index, iteration in enumerate((16, 21, 26, 31))
        ]
        checkpoints[-1] = _checkpoint(
            iterations=31, mask=_mask(100), volume=0.025, area=4.0
        )
        alternative = _checkpoint(
            iterations=31, mask=_mask(80), volume=0.018, fill=0.7, area=2.7
        )
        result, _ = self._run(checkpoints, alternative=alternative)
        np.testing.assert_array_equal(result.mask, checkpoints[-1].mask)
        self.assertFalse(result.diagnostics["aorta_level_set_alternative_accepted"])
        self.assertEqual(result.diagnostics["aorta_level_set_profile_used"], "nominal")
        self.assertEqual(
            result.diagnostics["aorta_level_set_candidate_voxel_count"],
            alternative.voxel_count,
        )
        self.assertEqual(
            result.diagnostics["aorta_level_set_candidate_circle_area_ratio_p90"],
            2.7,
        )
        self.assertAlmostEqual(
            result.diagnostics[
                "aorta_level_set_candidate_area_ratio_improvement_fraction"
            ],
            0.325,
        )
