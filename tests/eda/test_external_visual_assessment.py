"""Tests for external-CCTA visual-assessment loading and summaries."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.project.external_visual_assessment import (
    AORTA_ARTERY_STATUS_ORDER,
    load_external_visual_assessments,
    summarize_visual_status,
)


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

        totals = summary.groupby("dataset")["percent"].sum()
        self.assertTrue(totals.round(10).eq(100.0).all())
        general_adequate = summary.loc[
            summary["dataset"].eq("Geral") & summary["status"].eq("Adequada")
        ].iloc[0]
        self.assertEqual(general_adequate["count"], 2)
        self.assertAlmostEqual(general_adequate["percent"], 200 / 3)
