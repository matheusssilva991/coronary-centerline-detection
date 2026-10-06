"""Tests for external-CCTA visual-assessment loading and summaries."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.project.analysis.external_visual_assessment import (
    AORTA_ARTERY_STATUS_ORDER,
    attach_assessment_subsets,
    load_assessment_subset_lookup,
    load_external_visual_assessments,
    summarize_visual_status,
    summarize_visual_overview,
)
from utils.project.dataframe import require_series_column


class ExternalVisualAssessmentTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    @staticmethod
    def _valid_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "image_id": ["case_1", "case_2"],
                "aorta_result": ["Adequada", "Parcial"],
                "ostia_result": ["Ambos adequados", "Um adequado"],
                "artery_result": ["Adequada", "Não avaliável"],
                "visual_assessment_notes": [None, "Achatado"],
                "Guia de avaliação": ["texto", None],
            }
        )

    def _write_workbook(self, frame: pd.DataFrame, name: str = "review.xlsx") -> Path:
        path = self.root / name
        frame.to_excel(path, sheet_name="Avaliação visual", index=False)
        return path

    def test_loads_only_assessment_columns_and_preserves_optional_notes(self):
        path = self._write_workbook(self._valid_frame())

        result = load_external_visual_assessments({"Dataset": path})

        self.assertEqual(len(result), 2)
        self.assertListEqual(
            result.columns.tolist(),
            [
                "dataset",
                "image_id",
                "aorta_result",
                "ostia_result",
                "artery_result",
                "visual_assessment_notes",
            ],
        )
        self.assertEqual(result.loc[0, "visual_assessment_notes"], "")
        self.assertEqual(result.loc[1, "visual_assessment_notes"], "Achatado")

    def test_rejects_missing_file(self):
        with self.assertRaisesRegex(FileNotFoundError, "não encontrado"):
            load_external_visual_assessments({"Dataset": self.root / "missing.xlsx"})

    def test_rejects_missing_column(self):
        frame = self._valid_frame().drop(columns="artery_result")
        path = self._write_workbook(frame)

        with self.assertRaisesRegex(ValueError, "colunas obrigatórias ausentes"):
            load_external_visual_assessments({"Dataset": path})

    def test_rejects_missing_value(self):
        frame = self._valid_frame()
        frame.loc[1, "aorta_result"] = None
        path = self._write_workbook(frame)

        with self.assertRaisesRegex(ValueError, r"valor\(es\) ausente"):
            load_external_visual_assessments({"Dataset": path})

    def test_rejects_duplicate_ids(self):
        frame = self._valid_frame()
        frame.loc[1, "image_id"] = "case_1"
        path = self._write_workbook(frame)

        with self.assertRaisesRegex(ValueError, "IDs duplicados"):
            load_external_visual_assessments({"Dataset": path})

    def test_rejects_unknown_status(self):
        frame = self._valid_frame()
        frame.loc[1, "artery_result"] = "Incerto"
        path = self._write_workbook(frame)

        with self.assertRaisesRegex(ValueError, "status desconhecido"):
            load_external_visual_assessments({"Dataset": path})

    def test_summary_uses_all_exams_as_percentage_denominator(self):
        first = self._write_workbook(self._valid_frame(), "first.xlsx")
        second = self._write_workbook(self._valid_frame().iloc[:1], "second.xlsx")
        assessments = load_external_visual_assessments(
            {"First": first, "Second": second}
        )

        summary = summarize_visual_status(
            assessments,
            "aorta_result",
            AORTA_ARTERY_STATUS_ORDER,
            ["First", "Second"],
        )

        totals_frame = summary.groupby("dataset", as_index=False)["percent"].sum()
        if not isinstance(totals_frame, pd.DataFrame):
            self.fail("O agrupamento deveria produzir um DataFrame.")
        totals = require_series_column(totals_frame, "percent")
        self.assertTrue(totals.round(10).eq(100.0).all())
        general_adequate = summary.loc[
            summary["dataset"].eq("Geral") & summary["status"].eq("Adequada")
        ].iloc[0]
        self.assertEqual(general_adequate["count"], 2)
        self.assertAlmostEqual(general_adequate["percent"], 200 / 3)

    def _write_results(self, name: str, records: list[dict[str, str]]) -> Path:
        path = self.root / name
        pd.DataFrame.from_records(records).to_csv(path, index=False)
        return path

    def test_split_lookup_uses_other_run_for_missing_result(self):
        mid = self._write_results(
            "mid.csv",
            [
                {"dataset": "Dataset", "exam_id": "case_1", "subset": "train"},
                {"dataset": "Dataset", "exam_id": "case_2", "subset": "test"},
            ],
        )
        high = self._write_results(
            "high.csv",
            [{"dataset": "Dataset", "exam_id": "case_1", "subset": "train"}],
        )
        assessments = load_external_visual_assessments(
            {"Dataset": self._write_workbook(self._valid_frame())}
        )

        lookup = load_assessment_subset_lookup({"Dataset": [mid, high]})
        result = attach_assessment_subsets(assessments, lookup)

        self.assertEqual(result["subset"].tolist(), ["train", "test"])
        self.assertEqual(len(result), 2)

    def test_split_lookup_rejects_conflicts(self):
        mid = self._write_results(
            "mid.csv",
            [{"dataset": "Dataset", "exam_id": "case_1", "subset": "train"}],
        )
        high = self._write_results(
            "high.csv",
            [{"dataset": "Dataset", "exam_id": "case_1", "subset": "test"}],
        )

        with self.assertRaisesRegex(ValueError, "Splits contraditórios"):
            load_assessment_subset_lookup({"Dataset": [mid, high]})

    def test_split_join_rejects_visual_id_without_result(self):
        result_path = self._write_results(
            "result.csv",
            [{"dataset": "Dataset", "exam_id": "case_1", "subset": "train"}],
        )
        assessments = load_external_visual_assessments(
            {"Dataset": self._write_workbook(self._valid_frame())}
        )

        with self.assertRaisesRegex(ValueError, "sem split"):
            attach_assessment_subsets(
                assessments,
                load_assessment_subset_lookup({"Dataset": [result_path]}),
            )

    def test_overview_weights_totals_by_exam_count(self):
        frame = pd.DataFrame(
            {
                "dataset": ["A", "A", "A", "B", "B", "B"],
                "subset": ["train", "train", "test", "train", "test", "test"],
                "aorta_result": [
                    "Adequada",
                    "Adequada",
                    "Parcial",
                    "Não avaliável",
                    "Adequada",
                    "Parcial",
                ],
                "ostia_result": [
                    "Ambos adequados",
                    "Um adequado",
                    "Inadequados",
                    "Não avaliável",
                    "Ambos adequados",
                    "Não avaliável",
                ],
                "artery_result": [
                    "Adequada",
                    "Parcial",
                    "Inadequada",
                    "Não avaliável",
                    "Adequada",
                    "Parcial",
                ],
            }
        )

        result = summarize_visual_overview(frame, ["A", "B"])

        self.assertEqual(len(result), 7)
        train = result.loc[
            result["dataset"].eq("Geral") & result["subset"].eq("train")
        ].iloc[0]
        self.assertEqual(train["exam_count"], 3)
        self.assertEqual(train["aorta_adequate_count"], 2)
        self.assertAlmostEqual(train["aorta_adequate_percent"], 200 / 3)
        total = result.loc[
            result["dataset"].eq("Geral") & result["subset"].eq("total")
        ].iloc[0]
        self.assertEqual(total["exam_count"], 6)
        self.assertEqual(total["aorta_adequate_count"], 3)
        self.assertEqual(total["aorta_adequate_percent"], 50.0)
