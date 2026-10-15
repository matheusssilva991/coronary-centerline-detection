"""Verifica classificação, pareamento e exportação opcional de casos ruins."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.comparison_utils.bad_cases import (
    analyze_bad_case_groups,
    export_bad_case_groups,
)


class BadCasesAnalysisTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(
            {
                "IMG_ID": [1, 2, 3, 4, 5],
                "dice_artery": [0.30, 0.29, 0.10, None, None],
                "status": [
                    "both_correct",
                    "both_tolerable",
                    "one_correct",
                    "both_correct",
                    "not_found",
                ],
                "source_run": ["run/reference"] * 5,
                "variant": ["normal_rg"] * 5,
            }
        )

    def test_boundary_missing_dice_and_exclusive_reasons(self):
        analysis = analyze_bad_case_groups({("mid_res", "train"): self.frame()})
        self.assertEqual(analysis["bad_cases"]["IMG_ID"].tolist(), [2, 3, 5])
        self.assertEqual(
            analysis["bad_cases"]["reason"].tolist(),
            ["low_dice", "ostia_failure", "ostia_failure"],
        )
        self.assertEqual(analysis["overview"].iloc[0]["missing_dice"], 2)
        self.assertEqual(analysis["reasons"]["count"].sum(), 3)

    def test_ostia_status_and_boolean_schemas(self):
        original = self.frame()
        for frame in (
            original.rename(columns={"status": "ostia_status"}),
            original.drop(columns="status").assign(
                both_correct=[True, False, False, True, False],
                both_tolerable=[False, True, False, False, False],
            ),
        ):
            result = analyze_bad_case_groups({("mid_res", "train"): frame})
            self.assertEqual(result["bad_cases"]["IMG_ID"].tolist(), [2, 3, 5])

    def test_failure_types_preserve_detail_and_percentages(self):
        frame = pd.DataFrame(
            {
                "IMG_ID": range(1, 8),
                "status": [
                    "not_found",
                    "one_correct",
                    "none_correct",
                    "error",
                    "both_tolerable",
                    "unexpected_status",
                    "both_correct",
                ],
                "dice_artery": [0.1, 0.1, 0.1, None, 0.2, 0.1, 0.8],
            }
        )
        result = analyze_bad_case_groups({("mid_res", "train"): frame})
        details = result["failure_types"].set_index("failure_type")
        self.assertEqual(details["count"].tolist(), [1] * 6)
        self.assertAlmostEqual(details.loc["not_found", "percent_of_exams"], 100 / 7)
        self.assertEqual(
            details["count"].sum(), result["overview"].iloc[0]["bad_cases"]
        )
        self.assertEqual(
            result["bad_cases"].iloc[-1]["bad_case_status"], "unexpected_status"
        )
        self.assertEqual(result["bad_cases"]["IMG_ID"].tolist(), [1, 2, 3, 4, 5, 6])

    def test_absent_failure_types_are_zero(self):
        result = analyze_bad_case_groups({("mid_res", "train"): self.frame().iloc[:1]})
        self.assertEqual(len(result["failure_types"]), 6)
        self.assertTrue(result["failure_types"]["count"].eq(0).all())
        self.assertTrue(result["failure_types"]["percent_of_exams"].eq(0).all())

    def test_missing_status_is_unclassified(self):
        frame = self.frame().iloc[:1].assign(status=None)
        result = analyze_bad_case_groups({("mid_res", "train"): frame})
        details = result["failure_types"].set_index("failure_type")
        self.assertEqual(details.loc["unknown", "count"], 1)

    def test_empty_group_failure_percentages_are_zero(self):
        result = analyze_bad_case_groups({("mid_res", "train"): self.frame().iloc[:0]})
        self.assertTrue(result["failure_types"]["percent_of_exams"].eq(0).all())

    def test_pairing_uses_only_common_ids(self):
        mid = self.frame().iloc[:3].copy()
        high = mid.copy()
        high["IMG_ID"] = [1, 2, 6]
        high["dice_artery"] = [0.1, 0.8, 0.2]
        result = analyze_bad_case_groups(
            {("mid_res", "train"): mid, ("high_res", "train"): high}
        )
        row = result["paired"].iloc[0]
        self.assertEqual(row["common_ids"], 2)
        self.assertEqual(row["bad_both"], 0)
        self.assertEqual(row["bad_mid_only"], 1)
        self.assertEqual(row["bad_high_only"], 1)
        self.assertEqual(row["unpaired_mid"], 1)
        self.assertEqual(row["unpaired_high"], 1)

    def test_missing_and_empty_groups(self):
        self.assertTrue(analyze_bad_case_groups({})["overview"].empty)
        result = analyze_bad_case_groups({("mid_res", "train"): self.frame().iloc[:0]})
        self.assertTrue(result["paired"].empty)
        self.assertEqual(result["overview"].iloc[0]["total_cases"], 0)

    def test_invalid_ids_are_rejected(self):
        for ids in ([1, 1, 3, 4, 5], [None, 2, 3, 4, 5], [1.5, 2, 3, 4, 5]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                analyze_bad_case_groups(
                    {("mid_res", "train"): self.frame().assign(IMG_ID=ids)}
                )

    def test_analysis_preserves_input(self):
        frame = self.frame()
        original = frame.copy(deep=True)
        analyze_bad_case_groups({("mid_res", "train"): frame})
        pd.testing.assert_frame_equal(frame, original)

    def test_export_disabled_does_not_create_directory(self):
        analysis = analyze_bad_case_groups({("mid_res", "train"): self.frame()})
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "absent"
            self.assertEqual(export_bad_case_groups(analysis, destination), [])
            self.assertFalse(destination.exists())

    def test_export_preserves_names_and_adds_provenance(self):
        analysis = analyze_bad_case_groups({("mid_res", "train"): self.frame()})
        with tempfile.TemporaryDirectory() as directory:
            export_bad_case_groups(analysis, directory, enabled=True)
            path = Path(directory) / "bad_cases_train_mid_res.csv"
            result = pd.read_csv(path)
            self.assertEqual(result["image_id"].tolist(), [2, 3, 5])
            self.assertEqual(result["source_run"].tolist(), ["run/reference"] * 3)
            self.assertIn("dice_artery", result)
            self.assertTrue(path.with_suffix(".json").is_file())

    def test_export_replaces_stale_catalog_for_group_with_no_failures(self):
        analysis = analyze_bad_case_groups(
            {("mid_res", "train"): self.frame().iloc[:1]}
        )
        with tempfile.TemporaryDirectory() as directory:
            export_bad_case_groups(analysis, directory, enabled=True)
            self.assertTrue(
                pd.read_csv(Path(directory) / "bad_cases_train_mid_res.csv").empty
            )


if __name__ == "__main__":
    unittest.main()
