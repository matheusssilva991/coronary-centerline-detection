"""Testes do pareamento entre feedback da aorta e avaliação visual."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.project.external_aorta_assessment import attach_aorta_feedback
from utils.project.external_visual_assessment import summarize_visual_status
from utils.project.external_aorta_assessment import AORTA_FEEDBACK_ORDER


class ExternalAortaAssessmentTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.visual = pd.DataFrame(
            {
                "dataset": ["MM-WHS", "MM-WHS", "OrCaScore"],
                "image_id": ["case_1", "case_2", "case_3"],
                "subset": ["train", "test", "test"],
                "aorta_result": ["Adequada", "Parcial", "Não avaliável"],
            }
        )

    def write_results(self, dataset, rows):
        path = self.root / f"{dataset}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_preserves_unmatched_visual_exam_as_no_feedback(self):
        mmwhs = self.write_results(
            "MM-WHS",
            [
                {
                    "dataset": "MM-WHS",
                    "exam_id": "case_1",
                    "aorta_segmentation_feedback": "adequate",
                }
            ],
        )
        orca = self.write_results(
            "OrCaScore",
            [
                {
                    "dataset": "OrCaScore",
                    "exam_id": "case_3",
                    "aorta_segmentation_feedback": "suspected_undersegmentation",
                }
            ],
        )

        paired = attach_aorta_feedback(
            self.visual, {"MM-WHS": mmwhs, "OrCaScore": orca}
        )

        self.assertEqual(
            paired["feedback"].tolist(),
            ["Adequada", "Sem feedback", "Suspeita de subsegmentação"],
        )
        self.assertEqual(paired["image_id"].tolist(), self.visual["image_id"].tolist())
        self.assertEqual(paired["subset"].tolist(), self.visual["subset"].tolist())
        summary = summarize_visual_status(
            paired, "feedback", AORTA_FEEDBACK_ORDER, ["MM-WHS", "OrCaScore"]
        )
        general = summary.loc[summary["dataset"].eq("Geral")]
        self.assertEqual(general["count"].sum(), 3)
        self.assertAlmostEqual(general["percent"].sum(), 100.0)

    def test_rejects_duplicate_result_ids(self):
        path = self.write_results(
            "MM-WHS",
            [
                {
                    "dataset": "MM-WHS",
                    "exam_id": "case_1",
                    "aorta_segmentation_feedback": "adequate",
                },
                {
                    "dataset": "MM-WHS",
                    "exam_id": "case_1",
                    "aorta_segmentation_feedback": "adequate",
                },
            ],
        )
        with self.assertRaisesRegex(ValueError, "IDs inválidos"):
            attach_aorta_feedback(
                self.visual.loc[self.visual["dataset"].eq("MM-WHS")],
                {"MM-WHS": path},
            )

    def test_rejects_unknown_feedback(self):
        path = self.write_results(
            "MM-WHS",
            [
                {
                    "dataset": "MM-WHS",
                    "exam_id": "case_1",
                    "aorta_segmentation_feedback": "unknown",
                }
            ],
        )
        with self.assertRaisesRegex(ValueError, "feedback desconhecido"):
            attach_aorta_feedback(
                self.visual.loc[self.visual["dataset"].eq("MM-WHS")],
                {"MM-WHS": path},
            )

    def test_rejects_result_without_visual_exam(self):
        path = self.write_results(
            "MM-WHS",
            [
                {
                    "dataset": "MM-WHS",
                    "exam_id": "other",
                    "aorta_segmentation_feedback": "adequate",
                }
            ],
        )
        with self.assertRaisesRegex(ValueError, "sem avaliação visual"):
            attach_aorta_feedback(
                self.visual.loc[self.visual["dataset"].eq("MM-WHS")],
                {"MM-WHS": path},
            )

    def test_rejects_missing_file_and_columns(self):
        only_mmwhs = self.visual.loc[self.visual["dataset"].eq("MM-WHS")]
        with self.assertRaisesRegex(FileNotFoundError, "não encontrado"):
            attach_aorta_feedback(only_mmwhs, {"MM-WHS": self.root / "missing.csv"})
        path = self.write_results(
            "MM-WHS", [{"dataset": "MM-WHS", "exam_id": "case_1"}]
        )
        with self.assertRaisesRegex(ValueError, "colunas obrigatórias ausentes"):
            attach_aorta_feedback(only_mmwhs, {"MM-WHS": path})


if __name__ == "__main__":
    unittest.main()
