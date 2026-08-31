import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils.experiments.parameter_validation import (
    build_mean_intensity_histogram,
    build_mean_normalized_intensity_histogram,
    build_normalized_intensity_histograms,
    build_parameter_pairwise_summary,
    build_parameter_sensitivity_summary,
    build_threshold_performance_data,
    compute_intensity_histogram_analysis,
    image_load_cache_key,
    parameter_validation_variants,
    prepared_context_cache_key,
    resolution_scaling_variants,
    select_parameter_validation_cases,
    select_top_threshold_cases,
    summarize_intensity_histograms,
    summarize_top_threshold_cases,
    validate_parameter_validation_append,
)
from utils.project.config import (
    RESOLUTION_SCALING_GROUPS,
    load_config_json,
    scale_config_to_resolution,
)


class ParameterValidationTests(unittest.TestCase):
    def test_builds_mean_histogram_on_common_probability_grid(self) -> None:
        histogram_bins = pd.DataFrame(
            {
                "IMG_ID": [1, 1, 2, 2],
                "histogram": ["full"] * 4,
                "bin_left_hu": [0.0, 1.0, 0.0, 2.0],
                "bin_right_hu": [1.0, 2.0, 2.0, 4.0],
                "count": [1, 1, 2, 2],
            }
        )

        mean_histogram = build_mean_intensity_histogram(
            histogram_bins,
            histogram_name="full",
            bins=4,
        )

        self.assertEqual(len(mean_histogram), 4)
        self.assertEqual(int(mean_histogram["images"].iloc[0]), 2)
        self.assertAlmostEqual(mean_histogram["mean_probability"].sum(), 1.0)
        self.assertTrue((mean_histogram["std_probability"] >= 0).all())

        profiles = build_normalized_intensity_histograms(
            histogram_bins,
            histogram_name="full",
            bins=4,
        )
        probability_by_image = profiles.groupby("IMG_ID")["probability"].sum()
        self.assertEqual(profiles.groupby("IMG_ID")["bin_center_hu"].nunique().nunique(), 1)
        self.assertTrue(np.allclose(probability_by_image, 1.0))

        selected_mean = build_mean_normalized_intensity_histogram(
            profiles,
            image_ids=[1],
        )
        self.assertEqual(int(selected_mean["images"].iloc[0]), 1)
        self.assertAlmostEqual(selected_mean["mean_probability"].sum(), 1.0)

    def test_summarizes_full_and_dense_intensity_histograms(self) -> None:
        values = np.array(
            [-1000.0, -500.0, 0.0, 300.0, 301.0, 500.0, 1000.0, np.nan]
        )

        summary, histogram = summarize_intensity_histograms(
            values,
            image_id=42,
            dense_min_hu=300.0,
            bins=4,
        )

        self.assertEqual(summary["IMG_ID"], 42)
        self.assertEqual(summary["full_voxel_count"], 7)
        self.assertAlmostEqual(summary["full_median_hu"], 300.0)
        self.assertAlmostEqual(summary["full_max_hu"], 1000.0)
        self.assertEqual(summary["dense_voxel_count"], 4)
        self.assertAlmostEqual(
            summary["dense_mean_hu"], 2101.0 / 4.0, places=4
        )
        self.assertAlmostEqual(summary["dense_median_hu"], 400.5)
        self.assertAlmostEqual(summary["dense_max_hu"], 1000.0)
        self.assertAlmostEqual(summary["dense_voxel_percent"], 400.0 / 7.0)
        self.assertEqual(histogram.groupby("histogram")["count"].sum()["full"], 7)
        self.assertEqual(
            histogram.groupby("histogram")["count"].sum()["dense_hu"], 4
        )

    def test_rejects_invalid_histogram_progress_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "progress_every"):
            compute_intensity_histogram_analysis(
                [1],
                "/unused",
                {},
                progress_every=0,
            )

    def test_computes_percentiles_with_the_histogram_pass(self) -> None:
        values = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float32)
        with patch(
            "utils.experiments.parameter_validation.load_downscaled_intensity_values",
            return_value=values,
        ):
            summary, _ = compute_intensity_histogram_analysis(
                [7],
                "/unused",
                {},
                bins=2,
                percentiles=(50.0, 75.0),
            )

        self.assertAlmostEqual(float(summary.loc[0, "p500_hu"]), 15.0)
        self.assertAlmostEqual(float(summary.loc[0, "p750_hu"]), 22.5)

    def test_builds_top_threshold_analysis(self) -> None:
        results = pd.DataFrame(
            {
                "variant": ["baseline", "baseline", "upper_p997", "upper_p997"],
                "IMG_ID": [1, 2, 1, 2],
                "dice_artery": [0.8, 0.6, 0.7, 0.9],
                "threshold_voxels": [80, 70, 75, 72],
                "volume_voxels": [100, 100, 100, 100],
                "ostia_success": [True, True, True, False],
            }
        )
        parameters = pd.DataFrame(
            {
                "variant": ["baseline", "upper_p997"],
                "MAX_THRESHOLD_PERCENTILE": [99.9, 99.7],
            }
        )

        performance = build_threshold_performance_data(results, parameters)
        top = select_top_threshold_cases(performance, top_n=1)
        effective = pd.DataFrame(
            {
                "IMG_ID": [1, 2],
                "upper_percentile": [99.9, 99.7],
                "max_threshold_hu": [900.0, 700.0],
            }
        )
        top = top.merge(effective, on=["IMG_ID", "upper_percentile"])
        summary = summarize_top_threshold_cases(top)

        self.assertEqual(len(top), 2)
        self.assertEqual(set(top["IMG_ID"]), {1, 2})
        self.assertEqual(set(summary["median_threshold_hu"]), {700.0, 900.0})

    def test_builds_compact_sensitivity_and_pairwise_summaries(self) -> None:
        results = pd.DataFrame(
            {
                "variant": ["baseline", "baseline", "changed", "changed"],
                "IMG_ID": [1, 2, 1, 2],
                "dice_artery": [0.5, 0.7, 0.6, 0.6],
                "ostia_success": [True, "false", "true", False],
            }
        )
        parameters = pd.DataFrame(
            {
                "variant": ["baseline", "changed"],
                "parameter_group": ["baseline", "threshold"],
                "description": ["Referência", "Alterado"],
            }
        )

        summary = build_parameter_sensitivity_summary(results, parameters)
        pairwise = build_parameter_pairwise_summary(results)

        baseline = summary.loc[summary["variant"].eq("baseline")].iloc[0]
        self.assertAlmostEqual(float(baseline["mean_dice"]), 0.6)
        self.assertEqual(int(baseline["ostia_success_count"]), 1)
        self.assertAlmostEqual(float(baseline["ostia_success_percent"]), 50.0)
        self.assertAlmostEqual(float(pairwise.iloc[0]["mean_delta_dice"]), 0.0)
        self.assertEqual(int(pairwise.iloc[0]["improved_images"]), 1)
        self.assertEqual(int(pairwise.iloc[0]["worse_images"]), 1)

    def test_reuses_upstream_context_for_ostia_and_rg_variations(self) -> None:
        base = {
            "DOWNSCALE_FACTORS": [2, 2, 1],
            "DOWNSCALE_METHOD": "opencv",
            "OPENCV_INTERPOLATION": "linear",
            "MAX_THRESHOLD_PERCENTILE": 99.9,
            "VESSELNESS_AORTA": {"sigmas": [2.5, 3.0]},
            "VESSELNESS_ARTERY": {"sigmas": [1.5, 2.0]},
            "OSTIA_DETECTION": {"max_z_diff_mm": 40.0},
            "REGION_GROWING": {"min_vesselness_fraction": 0.078},
        }
        changed_z = {
            **base,
            "OSTIA_DETECTION": {"max_z_diff_mm": 30.0},
        }
        changed_rg = {
            **base,
            "REGION_GROWING": {"min_vesselness_fraction": 0.05},
        }
        changed_threshold = {**base, "MAX_THRESHOLD_PERCENTILE": 99.7}
        experiment = {"threshold_mode": "normal", "artery_method": "region_growing"}

        self.assertEqual(
            image_load_cache_key(base), image_load_cache_key(changed_threshold)
        )
        self.assertEqual(
            prepared_context_cache_key(base, experiment),
            prepared_context_cache_key(changed_z, experiment),
        )
        self.assertEqual(
            prepared_context_cache_key(base, experiment),
            prepared_context_cache_key(changed_rg, experiment),
        )
        self.assertNotEqual(
            prepared_context_cache_key(base, experiment),
            prepared_context_cache_key(changed_threshold, experiment),
        )

    def test_append_validation_accepts_matching_run(self) -> None:
        existing = {
            "split": "val",
            "ids": [1, 2, 3],
            "resolution": "mid",
            "config_path": "config/reference.json",
            "use_gpu": True,
        }

        validate_parameter_validation_append(
            existing,
            split="val",
            image_ids=[1, 2, 3],
            resolution="mid",
            config_path=Path("config/reference.json"),
            use_gpu=True,
        )

    def test_append_validation_rejects_different_ids(self) -> None:
        existing = {
            "split": "val",
            "ids": [1, 2, 3],
            "resolution": "mid",
            "config_path": "config/reference.json",
            "use_gpu": True,
        }

        with self.assertRaisesRegex(ValueError, "ids"):
            validate_parameter_validation_append(
                existing,
                split="val",
                image_ids=[1, 2, 4],
                resolution="mid",
                config_path=Path("config/reference.json"),
                use_gpu=True,
            )

    def test_variants_are_unique_and_ofat(self) -> None:
        variants = parameter_validation_variants()
        names = [item["name"] for item in variants]

        self.assertEqual(len(variants), 11)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[0], "baseline")
        self.assertIn("upper_p995", names)
        self.assertIn("upper_p997", names)
        self.assertIn("ostia_lower_70", names)
        self.assertIn("ostia_lower_100", names)
        self.assertIn("rg_vessel_09", names)

        baseline = variants[0]["overrides"]
        self.assertEqual(baseline["MAX_THRESHOLD_PERCENTILE"], 99.9)
        self.assertEqual(baseline["OSTIA_DETECTION.max_z_diff_mm"], 40.0)
        self.assertEqual(baseline["REGION_GROWING.threshold_divisor"], 7.0)

        vessel_variant = next(
            item for item in variants if item["name"] == "rg_vessel_05"
        )
        self.assertEqual(
            vessel_variant["overrides"]["REGION_GROWING.min_vesselness_fraction"],
            0.05,
        )
        self.assertEqual(
            vessel_variant["overrides"]["REGION_GROWING.relaxed_floor_factor"],
            0.98,
        )

        lower_region_variant = next(
            item for item in variants if item["name"] == "ostia_lower_70"
        )
        self.assertEqual(
            lower_region_variant["overrides"]["OSTIA_DETECTION.lower_fraction"],
            0.70,
        )
        self.assertEqual(
            lower_region_variant["overrides"]["OSTIA_DETECTION.max_z_diff_mm"],
            40.0,
        )

    def test_resolution_scaling_groups_can_be_isolated(self) -> None:
        config = load_config_json("config/article_cbeb_sensitivity.json", {})
        config["DOWNSCALE_FACTORS"] = [1, 1, 1]

        fully_scaled = scale_config_to_resolution(config)
        without_surface = scale_config_to_resolution(
            config,
            enabled_groups=RESOLUTION_SCALING_GROUPS.difference(
                {"ostia_surface"}
            ),
        )
        without_candidates = scale_config_to_resolution(
            config,
            enabled_groups=RESOLUTION_SCALING_GROUPS.difference(
                {"ostia_candidates"}
            ),
        )

        self.assertEqual(fully_scaled["OSTIA_DETECTION"]["erosion_radius"], 8)
        self.assertEqual(without_surface["OSTIA_DETECTION"]["erosion_radius"], 4)
        self.assertEqual(fully_scaled["OSTIA_DETECTION"]["top_n"], 8000)
        self.assertEqual(without_candidates["OSTIA_DETECTION"]["top_n"], 2000)
        self.assertEqual(fully_scaled["CIRCLE_DETECTION"]["radii_start_px"], 36)

    def test_default_config_uses_standard_aorta_ostia_values_directly(self) -> None:
        config = load_config_json("config/pipeline_config.json", {})

        self.assertNotIn("AORTA_OSTIA_METHOD", config)
        self.assertEqual(config["OSTIA_DETECTION"]["erosion_radius"], 4)
        self.assertNotIn("pair_selection_mode", config["OSTIA_DETECTION"])
        self.assertNotIn("experimental_leak_correction", config["LEVEL_SET"])

    def test_adaptive_level_set_iterations_follow_high_resolution_scaling(self) -> None:
        config = load_config_json("config/pipeline_config.json", {})
        config["DOWNSCALE_FACTORS"] = [1, 1, 1]

        scaled = scale_config_to_resolution(config)

        self.assertEqual(scaled["LEVEL_SET"]["num_iter"], 70)
        self.assertEqual(scaled["LEVEL_SET"]["adaptive"]["min_iter"], 36)
        self.assertEqual(scaled["LEVEL_SET"]["adaptive"]["check_interval"], 11)
        self.assertEqual(
            scaled["LEVEL_SET"]["adaptive"]["early_stop_iteration"],
            59,
        )
        self.assertNotIn("permissive", scaled["LEVEL_SET"]["adaptive"])
        self.assertNotIn(
            "oversegmented_voxels_per_slice",
            scaled["LEVEL_SET"]["adaptive"],
        )

    def test_resolution_scaling_rejects_unknown_group(self) -> None:
        config = load_config_json("config/article_cbeb_sensitivity.json", {})
        config["DOWNSCALE_FACTORS"] = [1, 1, 1]

        with self.assertRaisesRegex(ValueError, "grupo_inexistente"):
            scale_config_to_resolution(
                config,
                enabled_groups={"grupo_inexistente"},
            )

    def test_resolution_scaling_variants_are_explicit(self) -> None:
        variants = resolution_scaling_variants()
        names = [item["name"] for item in variants]

        self.assertEqual(names[0], "all_scaled")
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("circle_geometry_unscaled", names)
        self.assertIn("morphology_radii_unscaled", names)
        self.assertTrue(
            all("disabled_scaling_groups" in item for item in variants)
        )
        canny_variant = next(
            item for item in variants if item["name"] == "canny_sigma_mid"
        )
        self.assertEqual(
            canny_variant["post_scale_overrides"],
            {"CIRCLE_DETECTION.canny_sigma": 3.0},
        )
        level_set_variant = next(
            item for item in variants if item["name"] == "level_set_iterations_50"
        )
        self.assertEqual(
            level_set_variant["post_scale_overrides"],
            {"LEVEL_SET.num_iter": 50},
        )
        combined_variant = next(
            item
            for item in variants
            if item["name"] == "canny_sigma_4_level_set_50"
        )
        self.assertEqual(
            combined_variant["post_scale_overrides"],
            {
                "CIRCLE_DETECTION.canny_sigma": 4.0,
                "LEVEL_SET.num_iter": 50,
            },
        )

    def test_selects_distinct_qualitative_cases_when_available(self) -> None:
        frame = pd.DataFrame(
            {
                "variant": ["best"] * 6,
                "IMG_ID": [1, 2, 3, 4, 5, 6],
                "dice_artery": [0.9, 0.58, 0.2, 0.4, 0.1, 0.3],
                "ostia_success": [True, True, False, True, True, True],
                "ostia_found": [True, True, False, True, True, True],
                "aorta_voxels": [10, 20, 30, 100, 40, 50],
                "aorta_volume_fraction": [0.01, 0.02, 0.03, 0.10, 0.04, 0.05],
                "artery_volume_ratio": [1.0, 1.1, 0.5, 1.0, 3.0, 2.0],
            }
        )

        selected = select_parameter_validation_cases(frame, "best")

        self.assertEqual(len(selected), 5)
        self.assertEqual(selected["IMG_ID"].nunique(), 5)
        self.assertEqual(
            set(selected["case_type"]),
            {
                "high_dice",
                "near_target_mean",
                "ostia_failure",
                "suspected_aorta_leak",
                "segmentation_failure",
            },
        )
        near_mean = selected.loc[
            selected["case_type"].eq("near_target_mean")
        ].iloc[0]
        self.assertTrue(bool(near_mean["ostia_success"]))

        ostia_failure = selected.loc[
            selected["case_type"].eq("ostia_failure")
        ].iloc[0]
        self.assertGreater(float(ostia_failure["dice_artery"]), 0.0)

    def test_prefers_visible_segmentation_failure_with_accepted_ostia(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "variant": ["baseline"] * 5,
                "IMG_ID": [1, 2, 3, 4, 5],
                "dice_artery": [0.90, 0.58, 0.30, 0.04, 0.52],
                "ostia_success": [True, True, False, True, True],
                "ostia_found": [True] * 5,
                "ostia_status": [
                    "both_correct",
                    "both_tolerable",
                    "found_but_wrong",
                    "both_tolerable",
                    "both_correct",
                ],
                "aorta_voxels": [10, 20, 30, 40, 50],
                "aorta_volume_fraction": [0.01, 0.02, 0.03, 0.10, 0.04],
                "artery_volume_ratio": [1.0, 1.1, 0.5, 0.1, 0.8],
                "artery_voxels": [20_000, 15_000, 8_000, 100, 10_000],
            }
        )

        selected = select_parameter_validation_cases(frame, "baseline")
        failure = selected.loc[
            selected["case_type"].eq("segmentation_failure")
        ].iloc[0]

        self.assertEqual(int(failure["IMG_ID"]), 5)
        self.assertEqual(failure["ostia_status"], "both_correct")
