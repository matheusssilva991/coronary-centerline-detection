"""Valida diagnósticos por coorte sem executar o pipeline de segmentação."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat
import pandas as pd

from utils.project.analysis.aorta_automatic_quality import (
    AortaFeedbackSource,
    compare_aorta_resolutions,
    default_aorta_feedback_sources,
    load_aorta_feedback_sources,
    summarize_aorta_feedback,
)


class AortaAutomaticQualityTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def source(
        self, rows, dataset="ImageCAS", resolution="mid", subset="train", **kwargs
    ):
        path = self.root / f"{dataset}_{resolution}_{subset}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return AortaFeedbackSource(dataset, resolution, subset, "Atual", path, **kwargs)

    def test_counts_include_empty_feedback_but_not_missing_column(self):
        sources = [
            self.source(
                [
                    {"IMG_ID": "001", "aorta_segmentation_feedback": "adequate"},
                    {
                        "IMG_ID": "002",
                        "aorta_segmentation_feedback": "insufficient_data",
                    },
                    {"IMG_ID": "003", "aorta_segmentation_feedback": None},
                ]
            ),
            self.source([{"IMG_ID": "001"}], resolution="high"),
        ]
        records, coverage = load_aorta_feedback_sources(sources)
        self.assertEqual(records.iloc[0]["image_id"], "001")
        self.assertEqual(
            coverage["status"].tolist(), ["Disponível", "Diagnóstico não persistido"]
        )
        self.assertEqual(
            records["feedback"].tolist(),
            [
                "Adequada",
                "Dados insuficientes",
                "Sem feedback",
                "Diagnóstico não persistido",
            ],
        )
        summary = summarize_aorta_feedback(records)
        self.assertEqual(len(summary), 5)
        self.assertEqual(summary["count"].sum(), 3)
        self.assertTrue(summary["total"].eq(3).all())
        self.assertAlmostEqual(summary["percent"].sum(), 100)
        self.assertEqual(
            summary.loc[
                summary["status"].eq("Suspeita de sobresegmentação"), "count"
            ].item(),
            0,
        )

    def test_external_subsets_and_expected_coverage_are_separate(self):
        source = self.source(
            [
                {
                    "dataset": "OrCaScore",
                    "exam_id": "train_1",
                    "subset": "train",
                    "aorta_segmentation_feedback": "adequate",
                },
                {
                    "dataset": "OrCaScore",
                    "exam_id": "test_1",
                    "subset": "test",
                    "aorta_segmentation_feedback": "suspected_undersegmentation",
                },
            ],
            dataset="OrCaScore",
            subset="test",
            expected_count=2,
        )
        records, coverage = load_aorta_feedback_sources([source])
        self.assertEqual(records["image_id"].tolist(), ["test_1"])
        self.assertEqual(coverage["exam_count"].item(), 1)
        self.assertEqual(coverage["missing_count"].item(), 1)
        self.assertEqual(summarize_aorta_feedback(records)["count"].sum(), 1)

    def test_absent_sources_do_not_block_available_cohorts(self):
        available = self.source(
            [{"IMG_ID": 1, "aorta_segmentation_feedback": "adequate"}]
        )
        absent = AortaFeedbackSource(
            "MM-WHS", "mid", "test", "Atual", self.root / "missing.csv", 40
        )
        not_applicable = AortaFeedbackSource(
            "MM-WHS", "mid", "val", "Atual", None, applicable=False
        )
        records, coverage = load_aorta_feedback_sources(
            [available, absent, not_applicable]
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(
            coverage["status"].tolist(),
            ["Disponível", "Arquivo ausente", "Não se aplica"],
        )
        self.assertTrue(pd.isna(coverage.iloc[1]["exam_count"]))
        records, _ = load_aorta_feedback_sources([absent])
        self.assertTrue(summarize_aorta_feedback(records).empty)
        paired, coverage = compare_aorta_resolutions(records)
        self.assertTrue(paired.empty)
        self.assertTrue(coverage.empty)

    def test_invalid_records_are_rejected(self):
        cases = [
            ([{"IMG_ID": 1}, {"IMG_ID": 1}], "IDs ausentes ou duplicados"),
            ([{"IMG_ID": None}], "IDs ausentes ou duplicados"),
            (
                [{"IMG_ID": 1, "aorta_segmentation_feedback": "unknown"}],
                "feedback desconhecido",
            ),
            ([{"IMG_ID": 1, "dataset": "Other"}], "banco incompatível"),
            ([{"IMG_ID": 1, "split": "other"}], "subconjunto inválido"),
            ([{"other": 1}], "coluna de ID ausente"),
        ]
        for rows, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                load_aorta_feedback_sources([self.source(rows)])
        source = self.source([{"IMG_ID": 1}])
        with self.assertRaisesRegex(ValueError, "Coorte duplicada"):
            load_aorta_feedback_sources([source, source])
        with self.assertRaisesRegex(ValueError, "excede"):
            load_aorta_feedback_sources(
                [self.source([{"IMG_ID": 1}], expected_count=0)]
            )
        with self.assertRaisesRegex(ValueError, "subconjunto ausente"):
            load_aorta_feedback_sources(
                [self.source([{"exam_id": "x"}], dataset="MM-WHS")]
            )

    def test_pairing_uses_ids_and_reports_exclusive_cases(self):
        mid = self.source(
            [
                {"IMG_ID": "1", "aorta_segmentation_feedback": "adequate"},
                {"IMG_ID": "2", "aorta_segmentation_feedback": None},
            ]
        )
        high = self.source(
            [
                {"IMG_ID": "3", "aorta_segmentation_feedback": "adequate"},
                {
                    "IMG_ID": "1",
                    "aorta_segmentation_feedback": "suspected_undersegmentation",
                },
            ],
            resolution="high",
        )
        records, _ = load_aorta_feedback_sources([mid, high])
        paired, coverage = compare_aorta_resolutions(records)
        self.assertEqual(paired["image_id"].tolist(), ["1"])
        self.assertEqual(paired["feedback_mid"].item(), "Adequada")
        self.assertEqual(paired["feedback_high"].item(), "Suspeita de subsegmentação")
        self.assertEqual(coverage["mid_only_ids"].item(), ["2"])
        self.assertEqual(coverage["high_only_ids"].item(), ["3"])
        self.assertEqual(coverage["paired_diagnostics"].item(), 1)
        records.loc[records["resolution"].eq("high"), "diagnostic_available"] = False
        _, coverage = compare_aorta_resolutions(records)
        self.assertEqual(coverage["common_count"].item(), 1)
        self.assertEqual(coverage["paired_diagnostics"].item(), 0)

    def test_references_and_subsets_do_not_mix(self):
        mid = self.source(
            [{"IMG_ID": 1, "split": "train", "aorta_segmentation_feedback": "adequate"}]
        )
        high = self.source(
            [{"IMG_ID": 1, "split": "test", "aorta_segmentation_feedback": "adequate"}],
            resolution="high",
            subset="test",
        )
        history = AortaFeedbackSource("ImageCAS", "mid", "train", "Histórico", mid.path)
        records, _ = load_aorta_feedback_sources([mid, high, history])
        self.assertEqual(len(summarize_aorta_feedback(records)), 15)
        paired, _ = compare_aorta_resolutions(records)
        self.assertTrue(paired.empty)

    def test_orcascore_missing_high_exam_is_not_an_algorithm_class(self):
        rows = [
            {
                "dataset": "OrCaScore",
                "exam_id": exam_id,
                "subset": "test",
                "aorta_segmentation_feedback": "adequate",
            }
            for exam_id in ("TEV2P6", "common")
        ]
        mid = self.source(rows, dataset="OrCaScore", subset="test", expected_count=2)
        high = self.source(
            rows[1:],
            dataset="OrCaScore",
            resolution="high",
            subset="test",
            expected_count=2,
        )
        records, coverage = load_aorta_feedback_sources([mid, high])
        paired, pairing = compare_aorta_resolutions(records)
        self.assertEqual(coverage["missing_count"].tolist(), [0, 1])
        self.assertEqual(pairing["mid_only_ids"].item(), ["TEV2P6"])
        self.assertEqual(paired["image_id"].tolist(), ["common"])
        summary = summarize_aorta_feedback(records)
        high_summary = summary.loc[summary["resolution"].eq("high")]
        self.assertEqual(high_summary["count"].sum(), 1)
        self.assertEqual(
            high_summary.loc[high_summary["status"].eq("Adequada"), "percent"].item(),
            100,
        )

    def test_canonical_provenance_and_empty_cohort_remain_visible(self):
        numeric = self.root / "current/numeric"
        numeric.mkdir(parents=True)
        path = numeric / "results_train.csv"
        pd.DataFrame(columns=["IMG_ID", "aorta_segmentation_feedback"]).to_csv(
            path, index=False
        )
        (numeric.parent / "source_manifest.json").write_text(
            json.dumps({"source_run": "runs/original"})
        )
        records, coverage = load_aorta_feedback_sources(
            [AortaFeedbackSource("ImageCAS", "mid", "train", "Atual", path, 30)]
        )
        self.assertTrue(records.empty)
        self.assertTrue(summarize_aorta_feedback(records).empty)
        self.assertEqual(coverage["source_run"].item(), "runs/original")
        self.assertEqual(coverage["status"].item(), "Coorte vazia")
        self.assertEqual(coverage["missing_count"].item(), 30)

    def test_registry_keeps_canonical_gaps_and_selects_full_validation(self):
        (self.root / "config").mkdir()
        (self.root / "config/imagecas_splits.json").write_text(
            json.dumps({"splits": {"train": [1], "val": [2, 3], "test": [4]}})
        )
        with patch(
            "utils.project.analysis.aorta_automatic_quality.get_default_split_paths",
            return_value={"mid_res": {}, "high_res": {}},
        ):
            sources = default_aorta_feedback_sources(self.root, self.root / "external")
        self.assertEqual(len(sources), 21)
        self.assertTrue(all(s.path is None for s in sources[:6]))
        history = next(
            s for s in sources if s.reference == "Histórico" and s.subset == "val"
        )
        self.assertIn("ostia_validation270", str(history.path))
        self.assertEqual(history.expected_count, 2)
        self.assertEqual(sum(not s.applicable for s in sources), 4)
        mmwhs_mid = next(
            s
            for s in sources
            if s.dataset == "MM-WHS" and s.resolution == "mid" and s.subset == "test"
        )
        self.assertIn("2026-10-06_07-53-03", str(mmwhs_mid.path))


class AortaAutomaticNotebookTest(unittest.TestCase):
    def test_notebooks_validate_compile_and_have_no_outputs(self):
        root = Path(__file__).resolve().parents[2]
        for relative in (
            "aorta/aorta_automatic_quality_assessment.ipynb",
            "results/external_aorta_assessment.ipynb",
        ):
            path = root / "src/eda" / relative
            book = nbformat.read(path, as_version=4)
            nbformat.validate(book)
            for cell in book.cells:
                if cell.cell_type == "code":
                    compile(cell.source, str(path), "exec")
                    self.assertEqual(cell.outputs, [])
                    self.assertIsNone(cell.execution_count)


if __name__ == "__main__":
    unittest.main()
