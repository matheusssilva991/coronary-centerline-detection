"""Valida diagnósticos por coorte sem executar o pipeline de segmentação."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat
import pandas as pd

from utils.project.analysis.aorta_automatic_quality import (
    AortaFeedbackSource,
    compare_aorta_resolutions,
    compare_mmwhs_aorta_dice,
    default_aorta_feedback_sources,
    load_aorta_feedback_sources,
    summarize_aorta_feedback,
    summarize_aorta_feedback_comparison,
    summarize_imagecas_aorta_ostia,
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

    def test_registry_uses_only_canonical_imagecas_and_keeps_gaps(self):
        (self.root / "config").mkdir()
        (self.root / "config/imagecas_splits.json").write_text(
            json.dumps({"splits": {"train": [1], "val": [2, 3], "test": [4]}})
        )
        with patch(
            "utils.project.analysis.aorta_automatic_quality.get_default_split_paths",
            return_value={"mid_res": {}, "high_res": {}},
        ):
            sources = default_aorta_feedback_sources(self.root, self.root / "external")
            local_sources = default_aorta_feedback_sources(self.root)
        self.assertEqual(len(sources), 18)
        self.assertTrue(all(s.path is None for s in sources[:6]))
        self.assertTrue(all(s.reference == "Atual" for s in sources))
        validation = next(
            s for s in sources if s.dataset == "ImageCAS" and s.subset == "val"
        )
        self.assertEqual(validation.expected_count, 2)
        self.assertEqual(sum(not s.applicable for s in sources), 4)
        mmwhs_mid = next(
            s
            for s in sources
            if s.dataset == "MM-WHS" and s.resolution == "mid" and s.subset == "test"
        )
        self.assertIn("2026-10-06_07-53-03", str(mmwhs_mid.path))
        local_paths = {
            (source.dataset, source.resolution): source.path
            for source in local_sources
            if source.dataset != "ImageCAS" and source.subset == "train"
        }
        expected_runs = {
            ("MM-WHS", "mid"): "mid_res/external_ccta/mmwhs/2026-09-22_18-51-28",
            ("MM-WHS", "high"): "high_res/external_ccta/mmwhs/2026-09-22_17-47-45",
            ("OrCaScore", "mid"): "mid_res/external_ccta/orcascore/2026-09-19_10-23-50",
            (
                "OrCaScore",
                "high",
            ): "high_res/external_ccta/orcascore/2026-09-22_19-12-41",
        }
        for key, run in expected_runs.items():
            self.assertEqual(
                local_paths[key],
                self.root
                / "output/segmentation/runs"
                / run
                / "numeric/results_all.csv",
            )
        # A seleção explícita continua externa, mesmo com o caminho local definido.
        self.assertTrue(mmwhs_mid.path.is_relative_to(self.root / "external"))

    def test_overall_weights_exams_and_keeps_empty_feedback_in_denominator(self):
        imagecas = self.source(
            [
                {
                    "IMG_ID": i,
                    "aorta_segmentation_feedback": "adequate" if i < 3 else None,
                }
                for i in range(4)
            ],
            expected_count=4,
        )
        mmwhs = self.source(
            [
                {
                    "exam_id": str(i),
                    "subset": "train",
                    "aorta_segmentation_feedback": status,
                }
                for i, status in enumerate(
                    ["suspected_undersegmentation", "insufficient_data"]
                )
            ],
            dataset="MM-WHS",
            expected_count=2,
        )
        records, coverage = load_aorta_feedback_sources([imagecas, mmwhs])
        summary, coverage = summarize_aorta_feedback_comparison(records, coverage)
        overall = summary.loc[summary["dataset"].eq("Geral")]
        self.assertEqual(overall["total"].unique().tolist(), [6])
        self.assertEqual(overall["count"].sum(), 6)
        self.assertAlmostEqual(overall["percent"].sum(), 100)
        self.assertEqual(
            overall.loc[overall["status"].eq("Adequada"), "percent"].item(), 50
        )
        self.assertEqual(
            overall.loc[overall["status"].eq("Sem feedback"), "count"].item(), 1
        )
        self.assertEqual(
            overall.loc[
                overall["status"].eq("Suspeita de sobresegmentação"), "count"
            ].item(),
            0,
        )
        general = coverage.loc[coverage["dataset"].eq("Geral")].iloc[0]
        self.assertEqual(general["status"], "Disponível")
        self.assertEqual(general["included_datasets"], "ImageCAS, MM-WHS")

    def test_overall_marks_missing_column_and_file_as_partial(self):
        sources = [
            self.source([{"IMG_ID": 1}]),
            self.source(
                [
                    {
                        "exam_id": "x",
                        "subset": "train",
                        "aorta_segmentation_feedback": "adequate",
                    }
                ],
                dataset="MM-WHS",
                expected_count=2,
            ),
            AortaFeedbackSource(
                "OrCaScore", "mid", "train", "Atual", None, expected_count=32
            ),
        ]
        records, coverage = load_aorta_feedback_sources(sources)
        summary, coverage = summarize_aorta_feedback_comparison(records, coverage)
        general = coverage.loc[coverage["dataset"].eq("Geral")].iloc[0]
        self.assertEqual(general["status"], "Parcial")
        self.assertEqual(general["included_datasets"], "MM-WHS")
        self.assertEqual(general["exam_count"], 1)
        self.assertTrue(pd.isna(general["expected_count"]))
        self.assertEqual(summary.loc[summary["dataset"].eq("Geral"), "count"].sum(), 1)

    def test_overall_unavailable_has_no_fabricated_counts(self):
        records, coverage = load_aorta_feedback_sources(
            [
                self.source([{"IMG_ID": 1}]),
                AortaFeedbackSource("MM-WHS", "mid", "train", "Atual", None),
            ]
        )
        summary, coverage = summarize_aorta_feedback_comparison(records, coverage)
        self.assertTrue(summary.empty)
        general = coverage.loc[coverage["dataset"].eq("Geral")].iloc[0]
        self.assertEqual(general["status"], "Indisponível")
        self.assertTrue(pd.isna(general["exam_count"]))

    def test_overall_keeps_resolution_subset_and_not_applicable_separate(self):
        sources = [
            self.source(
                [{"IMG_ID": 1, "aorta_segmentation_feedback": "adequate"}],
                subset="val",
                expected_count=1,
            ),
            self.source(
                [
                    {
                        "IMG_ID": 1,
                        "aorta_segmentation_feedback": "suspected_undersegmentation",
                    }
                ],
                resolution="high",
                subset="val",
                expected_count=1,
            ),
            self.source(
                [{"IMG_ID": 1, "aorta_segmentation_feedback": "insufficient_data"}],
                subset="test",
                expected_count=1,
            ),
            AortaFeedbackSource(
                "MM-WHS", "mid", "val", "Atual", None, applicable=False
            ),
            AortaFeedbackSource(
                "OrCaScore", "mid", "val", "Atual", None, applicable=False
            ),
        ]
        records, coverage = load_aorta_feedback_sources(sources)
        summary, coverage = summarize_aorta_feedback_comparison(records, coverage)
        general = summary.loc[summary["dataset"].eq("Geral")]
        self.assertEqual(general["count"].sum(), 3)
        adequate = general.loc[general["status"].eq("Adequada")]
        self.assertEqual(adequate["count"].tolist(), [1, 0, 0])
        general_coverage = coverage.loc[coverage["dataset"].eq("Geral")]
        self.assertTrue(general_coverage["status"].eq("Disponível").all())
        self.assertTrue(general_coverage["expected_count"].eq(1).all())

    def test_overall_rejects_duplicate_coverage(self):
        records, coverage = load_aorta_feedback_sources([self.source([{"IMG_ID": 1}])])
        with self.assertRaisesRegex(ValueError, "Coortes duplicadas"):
            summarize_aorta_feedback_comparison(
                records, pd.concat([coverage, coverage])
            )

    def test_ostia_comparison_preserves_denominators_and_exclusive_categories(self):
        rows = [
            {
                "IMG_ID": str(i),
                "aorta_segmentation_feedback": feedback,
                "ostia_detection_status": status,
            }
            for i, (feedback, status) in enumerate(
                [
                    ("adequate", "both correct"),
                    ("adequate", "both_tolerable"),
                    ("adequate", "found_but_wrong"),
                    ("adequate", "not_found"),
                    ("adequate", None),
                    ("suspected_undersegmentation", "both_correct"),
                    ("suspected_oversegmentation", "both_tolerable"),
                    ("insufficient_data", "not_evaluated"),
                    (None, "pipeline_error"),
                ]
            )
        ]
        records, _ = load_aorta_feedback_sources([self.source(rows)])
        summary = summarize_imagecas_aorta_ostia(records).set_index("aorta_group")
        adequate = summary.loc["Adequada"]
        self.assertEqual(adequate["total"], 5)
        self.assertEqual(adequate["accepted"], 2)
        self.assertEqual(adequate["accepted_percent"], 40)
        self.assertEqual(adequate["not_evaluated"], 1)
        self.assertEqual(adequate["both_correct"], 1)
        self.assertEqual(adequate["both_tolerable"], 1)
        self.assertEqual(summary.loc["Com alertas", "accepted"], 2)
        self.assertEqual(summary.loc["Indeterminada", "not_evaluated"], 2)
        self.assertEqual(summary["total"].sum(), 9)

    def test_ostia_groups_without_cases_have_no_rate_and_splits_do_not_mix(self):
        row = {
            "IMG_ID": "001",
            "aorta_segmentation_feedback": "adequate",
            "ostia_detection_status": "both_correct",
        }
        records, _ = load_aorta_feedback_sources(
            [
                self.source([row]),
                self.source(
                    [{**row, "ostia_detection_status": "not_found"}], subset="test"
                ),
            ]
        )
        summary = summarize_imagecas_aorta_ostia(records)
        empty = summary.loc[summary["total"].eq(0)]
        self.assertTrue(empty["accepted_percent"].isna().all())
        train = summary.loc[summary["subset"].eq("train")]
        test = summary.loc[summary["subset"].eq("test")]
        self.assertEqual(train["accepted"].sum(), 1)
        self.assertEqual(test["accepted"].sum(), 0)
        records["ostia_status"] = "typo"
        with self.assertRaisesRegex(ValueError, "Status de óstios desconhecido"):
            summarize_imagecas_aorta_ostia(records)

    def mmwhs_source(self, rows, subset="train", resolution="mid"):
        return self.source(
            [
                {
                    "exam_id": str(i),
                    "dataset": "MM-WHS",
                    "subset": subset,
                    "aorta_segmentation_feedback": "adequate",
                    **row,
                }
                for i, row in enumerate(rows)
            ],
            dataset="MM-WHS",
            subset=subset,
            resolution=resolution,
        )

    def test_dice_threshold_is_inclusive_and_excludes_missing_invalid_unevaluated(self):
        rows = [
            {"aorta_dice": value, "aorta_ground_truth_evaluated": True}
            for value in (0, 0.699, 0.70, 1, None, "bad", float("inf"), -1, 1.1)
        ]
        rows.append({"aorta_dice": 0.9, "aorta_ground_truth_evaluated": False})
        records, _ = load_aorta_feedback_sources([self.mmwhs_source(rows)])
        cases, summary, coverage = compare_mmwhs_aorta_dice(records)
        self.assertEqual(cases["dice_eligible"].sum(), 4)
        row = summary.iloc[0]
        self.assertEqual(row["exam_count"], 10)
        self.assertEqual(row["above_threshold"], 2)
        self.assertEqual(row["below_threshold"], 2)
        self.assertEqual(row["above_percent"], 50)
        self.assertEqual(row["excluded_count"], 6)
        self.assertEqual(coverage["status"].item(), "Parcial")
        for threshold in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                compare_mmwhs_aorta_dice(records, threshold)

    def test_test_dice_rejects_legacy_and_mixed_protocols(self):
        for methods in [
            ["mmwhs_official_1mm_wine"],
            [None],
            ["mmwhs_aorta_label820_1mm_wine", "mmwhs_official_1mm_wine"],
        ]:
            with self.subTest(methods=methods):
                source = self.mmwhs_source(
                    [
                        {
                            "aorta_dice": 0.9,
                            "aorta_evaluation_status": "success",
                            "aorta_evaluation_method": method,
                        }
                        for method in methods
                    ],
                    subset="test",
                )
                records, _ = load_aorta_feedback_sources([source])
                cases, summary, coverage = compare_mmwhs_aorta_dice(records)
                self.assertFalse(cases["dice_eligible"].any())
                self.assertTrue(summary["above_percent"].isna().all())
                self.assertTrue(coverage["status"].str.startswith("Indisponível").all())
                self.assertNotEqual(
                    coverage["protocol"].item(), "Aorta isolada · label 820 · 1 mm"
                )

    def test_corrected_test_keeps_evaluated_zero_and_diagnostic_missingness(self):
        source = self.mmwhs_source(
            [
                {"aorta_dice": 0, "aorta_evaluation_status": "success"},
                {
                    "aorta_dice": 0.7,
                    "aorta_evaluation_status": "success",
                    "aorta_segmentation_feedback": "insufficient_data",
                },
                {
                    "aorta_dice": 0.8,
                    "aorta_evaluation_status": "success",
                    "aorta_segmentation_feedback": None,
                },
                {"aorta_dice": None, "aorta_evaluation_status": "unavailable"},
                {"aorta_dice": 0.99, "aorta_evaluation_status": "error"},
            ],
            subset="test",
        )
        raw = pd.read_csv(source.path)
        raw["aorta_evaluation_method"] = "mmwhs_aorta_label820_1mm_wine"
        raw.to_csv(source.path, index=False)
        records, _ = load_aorta_feedback_sources([source])
        cases, summary, coverage = compare_mmwhs_aorta_dice(records)
        self.assertEqual(
            cases["dice_eligible"].tolist(), [True, True, True, False, False]
        )
        self.assertEqual(coverage["evaluated_count"].item(), 3)
        self.assertEqual(coverage["excluded_count"].item(), 2)
        summary = summary.set_index("feedback")
        self.assertEqual(summary.loc["Adequada", "median_dice"], 0)
        self.assertEqual(summary.loc["Dados insuficientes", "above_percent"], 100)
        self.assertEqual(summary.loc["Sem feedback", "above_percent"], 100)

    def test_outcome_comparisons_handle_absent_sources_without_fabricating_rows(self):
        records, _ = load_aorta_feedback_sources([])
        self.assertTrue(summarize_imagecas_aorta_ostia(records).empty)
        cases, summary, coverage = compare_mmwhs_aorta_dice(records)
        self.assertTrue(cases.empty)
        self.assertTrue(summary.empty)
        self.assertTrue(coverage.empty)


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
