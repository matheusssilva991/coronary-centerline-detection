"""Testa a integração da avaliação oficial ao batch MM-WHS."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from external_ccta_batch_pipeline import (
    build_parser,
    evaluate_test_aorta_result,
    run,
)
from utils.project.mmwhs_official_aorta import OFFICIAL_AORTA_METHOD


class MmwhsOfficialAortaBatchTest(unittest.TestCase):
    def test_short_flag_and_legacy_alias_enable_official_evaluation(self):
        parser = build_parser()
        required = ["--dataset", "mmwhs", "--resolution", "mid"]
        for flag in ("--test-aorta-dice", "--evaluate-mmwhs-test-aorta"):
            with self.subTest(flag=flag):
                args = parser.parse_args([*required, flag])
                self.assertTrue(args.evaluate_mmwhs_test_aorta)
        self.assertIn("--test-aorta-dice", parser.format_help())
        self.assertNotIn("--evaluate-mmwhs-test-aorta", parser.format_help())

    def test_wine_failure_does_not_change_scientific_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            exam_id = "ct_test_2001"
            output_dir = run_dir / "evaluation/aorta/test" / exam_id
            output_dir.mkdir(parents=True)
            (output_dir / f"{exam_id}_label.nii.gz").touch()
            row = {
                "subset": "test",
                "exam_id": exam_id,
                "status": "success",
                "error": None,
                "aorta_mask_voxels": 10,
                "aorta_circle_count": 2,
            }
            record = pd.Series({"exam_id": exam_id, "path": "/dataset/image.nii.gz"})
            with (
                patch("external_ccta_batch_pipeline.validate_saved_prediction"),
                patch(
                    "external_ccta_batch_pipeline.evaluate_with_wine",
                    side_effect=RuntimeError("Wine indisponível"),
                ),
            ):
                result = evaluate_test_aorta_result(
                    row, record, {}, run_dir, run_dir, rebuild_mask=False
                )

        self.assertEqual(result["status"], "success")
        self.assertIsNone(result["error"])
        self.assertEqual(result["aorta_mask_voxels"], 10)
        self.assertEqual(result["aorta_evaluation_status"], "error")
        self.assertIsNone(result["aorta_dice"])
        self.assertIn("Wine indisponível", result["aorta_evaluation_error"])

    @patch("external_ccta_batch_pipeline.process_external_exam")
    @patch("external_ccta_batch_pipeline.evaluate_test_aorta_result")
    @patch("external_ccta_batch_pipeline.preflight_evaluator")
    @patch("external_ccta_batch_pipeline._configure_logging")
    @patch("external_ccta_batch_pipeline.discover_ccta_dataset")
    @patch("external_ccta_batch_pipeline.resolve_dataset_path")
    def test_evaluation_only_updates_existing_csv_without_full_pipeline(
        self,
        resolve_path,
        discover,
        _logging,
        _preflight,
        evaluate,
        process,
    ):
        exam_id = "ct_test_2001"
        resolve_path.return_value = Path("/dataset")
        discover.return_value = pd.DataFrame(
            {"dataset": ["MM-WHS"], "subset": ["test"], "exam_id": [exam_id]}
        )

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            (run_dir / "config").mkdir(parents=True)
            (run_dir / "numeric").mkdir()
            (run_dir / "config/effective_pipeline_config.json").write_text(
                json.dumps({"USE_GPU": False}), encoding="utf-8"
            )
            (run_dir / "config/run_manifest.json").write_text(
                json.dumps(
                    {
                        "dataset": "mmwhs",
                        "resolution": "high",
                        "selected_subset": "test",
                        "selected_exams": [{"subset": "test", "exam_id": exam_id}],
                        "visuals_enabled": False,
                        "orcascore_visual_alignment": True,
                    }
                ),
                encoding="utf-8",
            )
            row = {
                "dataset": "MM-WHS",
                "subset": "test",
                "exam_id": exam_id,
                "resolution": "high",
                "status": "success",
                "aorta_mask_voxels": 10,
                "aorta_circle_count": 2,
                "execution_time_seconds": 1.0,
            }
            pd.DataFrame([row]).to_csv(run_dir / "numeric/results_all.csv", index=False)
            evaluate.return_value = {
                **row,
                "aorta_evaluation_method": OFFICIAL_AORTA_METHOD,
                "aorta_evaluation_status": "success",
                "aorta_ground_truth_available": True,
                "aorta_ground_truth_evaluated": True,
                "aorta_dice": 0.632263,
            }
            args = build_parser().parse_args(
                [
                    "--dataset",
                    "mmwhs",
                    "--resolution",
                    "high",
                    "--subset",
                    "test",
                    "--resume-dir",
                    str(run_dir),
                    "--no-visuals",
                    "--test-aorta-dice",
                    "--aorta-eval-only",
                ]
            )

            run(args)
            results = pd.read_csv(run_dir / "numeric/results_all.csv")
            metadata = json.loads(
                (run_dir / "metadata.json").read_text(encoding="utf-8")
            )

        process.assert_not_called()
        evaluate.assert_called_once()
        self.assertTrue(evaluate.call_args.kwargs["rebuild_mask"])
        self.assertEqual(results.loc[0, "aorta_dice"], 0.632263)
        self.assertEqual(results.loc[0, "status"], "success")
        self.assertEqual(
            metadata["ground_truth_metrics"]["aorta"]["test_official"][
                "evaluated_exam_count"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
