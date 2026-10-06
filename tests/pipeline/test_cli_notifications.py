"""Verifica avisos dos entry points sem executar exames reais."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

import external_ccta_batch_pipeline as external
import segmentation_pipeline as imagecas
from utils.segmentation.pipeline.cli import build_parser as build_imagecas_parser


class ImagecasNotificationTests(TestCase):
    @staticmethod
    def _args(*, notify=True):
        return SimpleNamespace(
            notify=notify,
            split="train",
            resolution="mid",
            resume_dir=None,
        )

    def test_flag_is_optional(self):
        parser = build_imagecas_parser(Path("/dataset"), Path("/output"))
        self.assertFalse(parser.parse_args([]).notify)
        self.assertTrue(parser.parse_args(["--notify"]).notify)

    def test_error_count_ignores_empty_values(self):
        results = pd.DataFrame({"error": [None, "", "  ", "Falha na imagem"]})
        self.assertEqual(imagecas._pipeline_error_count(results), 1)

    @patch("segmentation_pipeline.notify_run_completion")
    @patch("segmentation_pipeline._run_from_args")
    @patch("segmentation_pipeline.parse_pipeline_args")
    def test_complete_and_partial_runs_notify_once(self, parse_args, run, notify):
        parse_args.return_value = self._args()
        for errors, expected_status in (
            (0, "complete"),
            (2, "complete_with_errors"),
        ):
            with self.subTest(errors=errors):
                notify.reset_mock()
                run.return_value = (Path("/runs/imagecas"), "train", errors)

                imagecas.main()

                notify.assert_called_once()
                self.assertEqual(notify.call_args.kwargs["status"], expected_status)
                self.assertEqual(
                    notify.call_args.kwargs["run_dir"], Path("/runs/imagecas")
                )

    @patch("segmentation_pipeline.notify_run_completion")
    @patch("segmentation_pipeline._run_from_args")
    @patch("segmentation_pipeline.parse_pipeline_args")
    def test_fatal_error_and_interrupt_keep_original_exception(
        self, parse_args, run, notify
    ):
        parse_args.return_value = self._args()
        for error, expected_status in (
            (RuntimeError("run falhou"), "failed"),
            (KeyboardInterrupt(), "interrupted"),
        ):
            with self.subTest(expected_status=expected_status):
                notify.reset_mock()
                run.side_effect = error

                with self.assertRaises(type(error)):
                    imagecas.main()

                notify.assert_called_once()
                self.assertEqual(notify.call_args.kwargs["status"], expected_status)

    @patch("segmentation_pipeline.notify_run_completion")
    @patch("segmentation_pipeline._run_from_args")
    @patch("segmentation_pipeline.parse_pipeline_args")
    def test_disabled_flag_never_notifies(self, parse_args, run, notify):
        parse_args.return_value = self._args(notify=False)
        run.return_value = (Path("/runs/imagecas"), "train", 0)

        imagecas.main()

        notify.assert_not_called()

    @patch("segmentation_pipeline.run_processing_split", return_value=2)
    @patch("segmentation_pipeline.run_merge_only_split", return_value=3)
    def test_merge_and_processing_forward_error_counts(self, merge, process):
        args = SimpleNamespace(merge_only=True, resolution="mid")
        count = imagecas.run_requested_split(
            args, "train", [1], Path("/tmp"), {}, "/data"
        )
        self.assertEqual(count, 3)
        merge.assert_called_once()

        args.merge_only = False
        count = imagecas.run_requested_split(
            args, "train", [1], Path("/tmp"), {}, "/data"
        )
        self.assertEqual(count, 2)
        process.assert_called_once()


class ExternalNotificationTests(TestCase):
    @staticmethod
    def _argv(*, notify=True):
        argv = ["--dataset", "mmwhs", "--resolution", "mid"]
        return [*argv, "--notify"] if notify else argv

    def test_flag_is_optional(self):
        parser = external.build_parser()
        self.assertFalse(parser.parse_args(self._argv(notify=False)).notify)
        self.assertTrue(parser.parse_args(self._argv()).notify)

    @patch("external_ccta_batch_pipeline.notify_run_completion")
    @patch("external_ccta_batch_pipeline.run")
    def test_metadata_state_and_evaluation_errors_notify_once(self, run, notify):
        with TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            run.return_value = external.RunPaths(
                run_dir,
                run_dir / "numeric",
                run_dir / "config",
                run_dir / "visual",
                run_dir / "logs",
            )
            metadata_path = run_dir / "metadata.json"
            metadata = {
                "state": "complete_with_errors",
                "processed_exam_count": 60,
                "status_counts": {"success": 58, "error": 2},
                "ground_truth_metrics": {
                    "aorta": {"test_official": {"error_exam_count": 3}}
                },
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            self.assertEqual(external.main(self._argv()), 0)

            notify.assert_called_once()
            self.assertEqual(notify.call_args.kwargs["status"], "complete_with_errors")
            self.assertIn("Pipeline: 2 erro(s)", notify.call_args.kwargs["details"])
            self.assertIn(
                "Avaliação oficial: 3 falha(s)", notify.call_args.kwargs["details"]
            )

            notify.reset_mock()
            metadata["state"] = "complete"
            metadata["status_counts"] = {"success": 60}
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            self.assertEqual(external.main(self._argv()), 0)
            self.assertEqual(
                notify.call_args.kwargs["status"], "complete_with_warnings"
            )

    @patch("external_ccta_batch_pipeline.notify_run_completion")
    @patch("external_ccta_batch_pipeline.run")
    def test_complete_and_incomplete_follow_persisted_metadata(self, run, notify):
        with TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            run.return_value = external.RunPaths(
                run_dir,
                run_dir / "numeric",
                run_dir / "config",
                run_dir / "visual",
                run_dir / "logs",
            )
            metadata_path = run_dir / "metadata.json"
            for state in ("complete", "incomplete"):
                with self.subTest(state=state):
                    notify.reset_mock()
                    metadata_path.write_text(
                        json.dumps(
                            {
                                "state": state,
                                "processed_exam_count": 4,
                                "status_counts": {"success": 4},
                            }
                        ),
                        encoding="utf-8",
                    )

                    self.assertEqual(external.main(self._argv()), 0)

                    notify.assert_called_once()
                    self.assertEqual(notify.call_args.kwargs["status"], state)
                    self.assertEqual(notify.call_args.kwargs["split"], "all")
                    self.assertEqual(notify.call_args.kwargs["run_dir"], run_dir)

    @patch("external_ccta_batch_pipeline.notify_run_completion")
    @patch("external_ccta_batch_pipeline.run")
    def test_failure_and_interrupt_keep_exit_behavior(self, run, notify):
        run.side_effect = ValueError("config inválida")

        self.assertEqual(external.main(self._argv()), 1)
        notify.assert_called_once()
        self.assertEqual(notify.call_args.kwargs["status"], "failed")

        notify.reset_mock()
        run.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            external.main(self._argv())
        notify.assert_called_once()
        self.assertEqual(notify.call_args.kwargs["status"], "interrupted")

    @patch("external_ccta_batch_pipeline.notify_run_completion")
    @patch("external_ccta_batch_pipeline.run")
    def test_disabled_flag_does_not_read_metadata_or_notify(self, run, notify):
        run.return_value = external.RunPaths(
            Path("/runs/external"),
            Path("/runs/external/numeric"),
            Path("/runs/external/config"),
            Path("/runs/external/visual"),
            Path("/runs/external/logs"),
        )

        self.assertEqual(external.main(self._argv(notify=False)), 0)

        notify.assert_not_called()
