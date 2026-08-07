import unittest
from pathlib import Path

import pandas as pd

from utils.experiments.parameter_validation import (
    parameter_validation_variants,
    select_parameter_validation_cases,
    validate_parameter_validation_append,
)


class ParameterValidationTests(unittest.TestCase):
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
        self.assertIn("upper_p999", names)
        self.assertIn("ostia_lower_70", names)
        self.assertIn("ostia_lower_100", names)
        self.assertIn("rg_vessel_09", names)

        baseline = variants[0]["overrides"]
        self.assertEqual(baseline["MAX_THRESHOLD_PERCENTILE"], 99.7)
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
