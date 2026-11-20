import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils.experiments.parameter_validation import (
    build_mean_normalized_intensity_histogram,
    build_normalized_intensity_histograms,
    build_parameter_pairwise_summary,
    build_parameter_sensitivity_summary,
    build_threshold_performance_data,
    compute_intensity_histogram_analysis,
    summarize_intensity_histograms,
)
from utils.project.config import (
    RESOLUTION_SCALING_GROUPS,
    load_config_json,
    scale_config_to_resolution,
)
from utils.project.dataframe import require_series_column


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

        mean_histogram = build_mean_normalized_intensity_histogram(
            build_normalized_intensity_histograms(
                histogram_bins,
                histogram_name="full",
                bins=4,
            )
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
        if not isinstance(probability_by_image, pd.Series):
            self.fail("O agrupamento deveria produzir uma Series.")
        self.assertEqual(
            profiles.groupby("IMG_ID")["bin_center_hu"].nunique().nunique(), 1
        )
        self.assertTrue(
            np.allclose(probability_by_image.to_numpy(dtype=np.float64), 1.0)
        )

        selected_mean = build_mean_normalized_intensity_histogram(
            profiles,
            image_ids=[1],
        )
        self.assertEqual(int(selected_mean["images"].iloc[0]), 1)
        self.assertAlmostEqual(selected_mean["mean_probability"].sum(), 1.0)

    def test_summarizes_full_and_dense_intensity_histograms(self) -> None:
        values = np.array([-1000.0, -500.0, 0.0, 300.0, 301.0, 500.0, 1000.0, np.nan])

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
        self.assertAlmostEqual(summary["dense_mean_hu"], 2101.0 / 4.0, places=4)
        self.assertAlmostEqual(summary["dense_median_hu"], 400.5)
        self.assertAlmostEqual(summary["dense_max_hu"], 1000.0)
        self.assertAlmostEqual(summary["dense_voxel_percent"], 400.0 / 7.0)
        histogram_names = require_series_column(histogram, "histogram")
        counts = require_series_column(histogram, "count")
        self.assertEqual(int(counts.loc[histogram_names.eq("full")].sum()), 7)
        self.assertEqual(int(counts.loc[histogram_names.eq("dense_hu")].sum()), 4)

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

    def test_builds_threshold_performance_data(self) -> None:
        results = pd.DataFrame(
            {
                "variant": ["baseline", "upper_p997"],
                "IMG_ID": [1, 2],
                "dice_artery": [0.8, 0.9],
                "threshold_voxels": [80, 72],
                "volume_voxels": [100, 100],
                "ostia_success": [True, False],
            }
        )
        parameters = pd.DataFrame(
            {
                "variant": ["baseline", "upper_p997"],
                "MAX_THRESHOLD_PERCENTILE": [99.9, 99.7],
            }
        )

        performance = build_threshold_performance_data(results, parameters)

        self.assertEqual(len(performance), 2)
        self.assertAlmostEqual(
            float(performance.iloc[0]["threshold_volume_fraction"]), 0.72
        )
        self.assertAlmostEqual(
            float(performance.iloc[1]["threshold_volume_fraction"]), 0.80
        )

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

    def test_resolution_scaling_groups_can_be_isolated(self) -> None:
        config = load_config_json("config/pipeline_config.json", {})
        config["DOWNSCALE_FACTORS"] = [1, 1, 1]

        fully_scaled = scale_config_to_resolution(config)
        without_surface = scale_config_to_resolution(
            config,
            enabled_groups=RESOLUTION_SCALING_GROUPS.difference({"ostia_surface"}),
        )
        without_candidates = scale_config_to_resolution(
            config,
            enabled_groups=RESOLUTION_SCALING_GROUPS.difference({"ostia_candidates"}),
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

    def test_level_set_iterations_follow_high_resolution_scaling(self) -> None:
        config = load_config_json("config/pipeline_config.json", {})
        config["DOWNSCALE_FACTORS"] = [1, 1, 1]

        scaled = scale_config_to_resolution(config)

        self.assertEqual(scaled["LEVEL_SET"]["num_iter"], 70)
        self.assertNotIn("adaptive", scaled["LEVEL_SET"])

    def test_resolution_scaling_rejects_unknown_group(self) -> None:
        config = load_config_json("config/pipeline_config.json", {})
        config["DOWNSCALE_FACTORS"] = [1, 1, 1]

        with self.assertRaisesRegex(ValueError, "grupo_inexistente"):
            scale_config_to_resolution(
                config,
                enabled_groups={"grupo_inexistente"},
            )
