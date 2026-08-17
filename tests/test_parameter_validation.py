import unittest
from pathlib import Path

import pandas as pd

from utils.experiments.parameter_validation import (
    build_parameter_pairwise_summary,
    build_parameter_sensitivity_summary,
    build_threshold_performance_data,
    image_load_cache_key,
    parameter_validation_variants,
    prepared_context_cache_key,
    select_parameter_validation_cases,
    select_top_threshold_cases,
    summarize_top_threshold_cases,
    validate_parameter_validation_append,
)


class ParameterValidationTests(unittest.TestCase):
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
            "aorta_ostia_method": "standard",
            "config_path": "config/reference.json",
            "use_gpu": True,
        }

        validate_parameter_validation_append(
            existing,
            split="val",
            image_ids=[1, 2, 3],
            resolution="mid",
            aorta_ostia_method="standard",
            config_path=Path("config/reference.json"),
            use_gpu=True,
        )

    def test_append_validation_rejects_different_ids(self) -> None:
        existing = {
            "split": "val",
            "ids": [1, 2, 3],
            "resolution": "mid",
            "aorta_ostia_method": "standard",
            "config_path": "config/reference.json",
            "use_gpu": True,
        }

        with self.assertRaisesRegex(ValueError, "ids"):
            validate_parameter_validation_append(
                existing,
                split="val",
                image_ids=[1, 2, 4],
                resolution="mid",
                aorta_ostia_method="standard",
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
