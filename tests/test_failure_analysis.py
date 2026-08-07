"""Tests for pipeline failure classification."""

from unittest import TestCase

import pandas as pd

from utils.comparison_utils.failure_analysis import (
    build_failure_case_catalog,
    compact_focused_failure_cohort,
    select_focused_failure_cohort,
)


def _row(image_id, variant, dice, status, voxels, left, right):
    return {
        "IMG_ID": image_id,
        "folder_variant": variant,
        "artery_dice": dice,
        "artery_voxel_count": voxels,
        "ostia_detection_status": status,
        "left_ostium": left,
        "right_ostium": right,
    }


class FailureAnalysisTest(TestCase):
    def test_selects_failures_and_stable_controls(self):
        success = "both tolerable"
        wrong = "found but incorrect"
        rows = [
            _row(1, variant, 0.7, success, 20_000, "a", "b")
            for variant in ("normal_rg", "th_fuzzy_rg", "normal_fc", "th_fuzzy_fc")
        ]
        rows.extend(
            [
                _row(2, "normal_rg", 0.0, wrong, 200, "a", "b"),
                _row(2, "th_fuzzy_rg", 0.7, success, 20_000, "c", "d"),
                _row(2, "normal_fc", 0.0, wrong, 150, "a", "b"),
                _row(2, "th_fuzzy_fc", 0.68, success, 18_000, "c", "d"),
            ]
        )
        catalog = build_failure_case_catalog(pd.DataFrame(rows))
        cohort = select_focused_failure_cohort(
            catalog,
            max_per_category=2,
            shared_ostia_cases=0,
            stable_controls=1,
        ).set_index("IMG_ID")
        self.assertEqual(set(cohort.index), {1, 2})
        self.assertEqual(cohort.loc[1, "cohort_kind"], "control")
        self.assertIn("fuzzy_threshold_rescue", cohort.loc[2, "cohort_roles"])

    def test_compacts_cohort_to_runner_contract(self):
        cohort = pd.DataFrame(
            {
                "IMG_ID": [10],
                "cohort_kind": ["failure"],
                "cohort_roles": ["shared_ostia_failure"],
                "normal_rg_dice": [0.2],
            }
        )

        compact = compact_focused_failure_cohort(cohort)

        self.assertEqual(
            compact.columns.tolist(),
            ["IMG_ID", "cohort_kind", "cohort_roles"],
        )
        self.assertEqual(compact.iloc[0].to_dict()["IMG_ID"], 10)
