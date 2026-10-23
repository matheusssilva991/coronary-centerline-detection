"""Testes de amostragem por PID sem processar exames científicos."""

import argparse
import subprocess
import threading
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils.project.results.external import load_existing_results, save_numeric_results
from utils.project.results.io import merge_batch_results, save_results
from utils.project.results.schema import build_result_row
from utils.project.runtime import resource_monitor as resources


class ResourceMonitorTests(TestCase):
    def config(self, gpu: bool = False, interval: float = 100) -> dict[str, Any]:
        return {
            "USE_GPU": gpu,
            "RESOURCE_MONITORING": {
                "enabled": True,
                "sampling_interval_seconds": interval,
            },
        }

    @patch.object(resources.os, "sysconf", return_value=4096)
    @patch.object(Path, "read_text", return_value="200 123 10 0 0 0 0")
    def test_rss_uses_resident_pages_of_requested_pid(self, read, sysconf):
        self.assertEqual(resources.read_process_rss(1234), 123 * 4096)
        read.assert_called_once()
        sysconf.assert_called_once_with("SC_PAGE_SIZE")

    def test_vram_sums_only_requested_pid_across_devices(self):
        output = "42, GPU-b, 12.5\n100, GPU-a, 999\n42, GPU-a, 20\n"
        memory, devices = resources.parse_process_vram(output, 42)
        self.assertEqual(memory, int(32.5 * 1024**2))
        self.assertEqual(devices, ["GPU-a", "GPU-b"])
        self.assertEqual(resources.parse_process_vram(output, 999), (0, []))
        for value in ("N/A", "-1", "nan"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resources.parse_process_vram(f"42, GPU-a, {value}", 42)

    @patch.object(resources.subprocess, "run")
    def test_query_is_bounded_and_does_not_select_cuda_index(self, run):
        run.return_value.stdout = "42, GPU-a, 10"
        self.assertEqual(resources.query_process_vram(42)[0], 10 * 1024**2)
        args, kwargs = run.call_args
        self.assertNotIn("-i", args[0])
        self.assertEqual(kwargs["timeout"], resources.QUERY_TIMEOUT)
        run.side_effect = subprocess.TimeoutExpired("nvidia-smi", 1)
        with self.assertRaises(subprocess.TimeoutExpired):
            resources.query_process_vram(42)

    @patch.object(resources, "query_process_vram")
    @patch.object(resources, "read_process_rss")
    @patch.object(resources.threading, "Thread")
    def test_disabled_has_no_queries_or_thread(self, thread, rss, vram):
        result = {}
        monitor = resources.ResourceMonitor({"USE_GPU": True})
        monitor.start()
        monitor.finish(result)
        thread.assert_not_called()
        rss.assert_not_called()
        vram.assert_not_called()
        self.assertEqual(result["resource_monitor_status"], "disabled")
        self.assertIsNone(result["ram_peak_observed_bytes"])

    @patch.object(resources, "query_process_vram")
    @patch.object(resources, "read_process_rss", side_effect=[100, 200, 30, 20])
    def test_initial_final_samples_and_peaks_are_independent(self, rss, vram):
        for expected in (200, 30):
            monitor = resources.ResourceMonitor(self.config())
            result = {}
            monitor.start()
            monitor.finish(result)
            monitor.finish(result)
            self.assertEqual(result["ram_peak_observed_bytes"], expected)
            self.assertEqual(result["ram_sample_count"], 2)
            self.assertIsNone(result["vram_peak_observed_bytes"])
            self.assertEqual(result["resource_monitor_status"], "ok")
            assert monitor._thread is not None
            self.assertFalse(monitor._thread.is_alive())
        vram.assert_not_called()
        self.assertEqual(rss.call_count, 4)

    @patch.object(resources, "read_process_rss")
    def test_periodic_sampling_stops_before_next_exam(self, rss):
        periodic_sample = threading.Event()

        def sample(_pid):
            if rss.call_count >= 2:
                periodic_sample.set()
            return rss.call_count * 10

        rss.side_effect = sample
        monitor = resources.ResourceMonitor(self.config(interval=0.005))
        result = {}
        monitor.start()
        try:
            self.assertTrue(periodic_sample.wait(2))
        finally:
            monitor.finish(result)
        self.assertGreaterEqual(result["ram_sample_count"], 3)
        assert monitor._thread is not None
        self.assertFalse(monitor._thread.is_alive())

    @patch.object(resources, "read_process_rss", return_value=100)
    @patch.object(resources, "query_process_vram")
    def test_missing_gpu_and_timeout_are_not_scientific_errors(self, query, _rss):
        for error in (
            FileNotFoundError("nvidia-smi"),
            subprocess.TimeoutExpired("nvidia-smi", 1),
        ):
            with self.subTest(error=error):
                query.reset_mock(side_effect=True)
                query.side_effect = error
                monitor = resources.ResourceMonitor(self.config(gpu=True))
                result: dict[str, Any] = {"dice_artery": 0.8}
                monitor.start()
                monitor.finish(result)
                self.assertEqual(result["resource_monitor_status"], "partial")
                self.assertIsNone(result["vram_peak_observed_bytes"])
                self.assertEqual(result["vram_sample_count"], 0)
                self.assertEqual(result["dice_artery"], 0.8)
                self.assertNotIn("error", result)
                self.assertTrue(result["_resource_monitor_details"]["errors"])
                self.assertEqual(
                    query.call_count, 1 if isinstance(error, FileNotFoundError) else 2
                )

    @patch.object(resources, "read_process_rss", side_effect=OSError("proc absent"))
    def test_unavailable_does_not_report_zero_memory(self, _rss):
        monitor = resources.ResourceMonitor(self.config())
        monitor.start()
        result = {}
        monitor.finish(result)
        self.assertEqual(result["resource_monitor_status"], "unavailable")
        self.assertIsNone(result["ram_peak_observed_bytes"])

    @patch.object(
        resources, "read_process_rss", side_effect=[OSError("temporary"), 200]
    )
    @patch.object(
        resources,
        "query_process_vram",
        side_effect=[(300, ["GPU-b"]), (400, ["GPU-a"])],
    )
    def test_valid_samples_survive_intermediate_failure(self, _query, _rss):
        monitor = resources.ResourceMonitor(self.config(gpu=True))
        monitor.start()
        result = {}
        monitor.finish(result)
        self.assertEqual(result["ram_peak_observed_bytes"], 200)
        self.assertEqual(result["vram_peak_observed_bytes"], 400)
        self.assertEqual(result["ram_sample_count"], 1)
        self.assertEqual(result["vram_sample_count"], 2)
        self.assertEqual(result["resource_monitor_status"], "partial")

    def test_options_validate_interval_and_preserve_resume_config(self):
        parser = argparse.ArgumentParser()
        resources.add_resource_monitoring_arguments(parser)
        unchanged = {"USE_GPU": False}
        resources.apply_resource_monitoring_options(unchanged, parser.parse_args([]))
        self.assertEqual(unchanged, {"USE_GPU": False})
        config = self.config()
        resources.apply_resource_monitoring_options(config, parser.parse_args([]))
        self.assertTrue(config["RESOURCE_MONITORING"]["enabled"])
        args = parser.parse_args(
            ["--no-monitor-resources", "--resource-sampling-interval", "0.25"]
        )
        resources.apply_resource_monitoring_options(config, args)
        self.assertFalse(config["RESOURCE_MONITORING"]["enabled"])
        self.assertEqual(
            config["RESOURCE_MONITORING"]["sampling_interval_seconds"], 0.25
        )
        for value in ("0", "-1", "nan", "inf"):
            with (
                self.subTest(value=value),
                self.assertRaises(argparse.ArgumentTypeError),
            ):
                resources.positive_interval(value)

    def test_metadata_updates_only_reexecuted_exam(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "resources.json"
            for exam, pid in (("old", 1), ("new", 2), ("new", 3)):
                result = {
                    "_resource_monitor_details": {
                        "pid": pid,
                        "errors": {},
                        "environment": {"host": "test"},
                    }
                }
                resources.persist_resource_diagnostics(result, path, exam)
                self.assertNotIn("_resource_monitor_details", result)
            diagnostics = resources.read_resource_diagnostics(path)
            self.assertEqual(diagnostics["exams"]["old"]["pid"], 1)
            self.assertEqual(diagnostics["exams"]["new"]["pid"], 3)
            self.assertEqual(len(diagnostics["environments"]), 1)
            self.assertTrue(
                resources.resource_metadata(self.config(), diagnostics)["environments"]
            )

    def test_disabled_diagnostics_do_not_write_and_invalid_metadata_is_preserved(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "resources.json"
            resources.persist_resource_diagnostics({}, path, "1")
            self.assertFalse(path.exists())
            path.write_text("broken json", encoding="utf-8")
            result = {"_resource_monitor_details": {"pid": 1, "environment": {}}}
            with self.assertLogs(resources.LOGGER, level="WARNING"):
                resources.persist_resource_diagnostics(result, path, "1")
            self.assertEqual(path.read_text(), "broken json")

    def test_metadata_keeps_monitor_errors_separate_from_scientific_results(self):
        from utils.project.results.metadata import build_metadata

        diagnostic = {
            "environments": {"abc": {"host": "test"}},
            "exams": {
                "1": {"errors": {"vram: timeout": 2}, "gpu_device_uuids": ["GPU-a"]},
            },
        }
        metadata = build_metadata(
            "train", self.config(), resource_diagnostics=diagnostic
        )
        monitor = metadata["configuration"]["resource_monitoring"]
        self.assertEqual(monitor["errors_by_exam"], {"1": {"vram: timeout": 2}})
        self.assertEqual(monitor["gpu_device_uuids"], ["GPU-a"])
        self.assertNotIn("errors_by_exam", metadata["results"])

    @patch.object(resources, "read_process_rss", return_value=100)
    @patch.object(
        resources.threading.Thread, "start", side_effect=RuntimeError("no thread")
    )
    def test_thread_start_failure_still_records_final_sample(self, _thread, _rss):
        monitor = resources.ResourceMonitor(self.config())
        result = {}
        monitor.start()
        monitor.finish(result)
        self.assertEqual(result["ram_sample_count"], 2)
        self.assertEqual(result["resource_monitor_status"], "partial")

    def test_schema_and_merge_preserve_metrics_without_backfilling_old_rows(self):
        measured = {
            "IMG_ID": 2,
            "ram_peak_observed_bytes": 123456,
            "resource_monitor_status": "ok",
            "ram_sample_count": 2,
        }
        self.assertEqual(build_result_row(measured)["ram_peak_observed_bytes"], 123456)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame([{"IMG_ID": 1}]).to_csv(
                root / "results_train_lote_1.csv", index=False
            )
            save_results([measured], "train_lote_2", root)
            merged_path = merge_batch_results("train", root)
            assert merged_path is not None
            merged = pd.read_csv(merged_path)
            self.assertTrue(pd.isna(merged.iloc[0]["ram_peak_observed_bytes"]))
            self.assertEqual(merged.iloc[1]["ram_peak_observed_bytes"], 123456)
            self.assertEqual(merged.iloc[1]["resource_monitor_status"], "ok")

    def test_external_csv_resume_preserves_measured_and_old_rows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"subset": "train", "exam_id": "old"},
                {
                    "subset": "test",
                    "exam_id": "new",
                    "ram_peak_observed_bytes": 777,
                    "resource_monitor_status": "ok",
                },
            ]
            save_numeric_results(rows, root)
            loaded = load_existing_results(root)
            self.assertTrue(pd.isna(loaded[0]["ram_peak_observed_bytes"]))
            self.assertEqual(loaded[1]["ram_peak_observed_bytes"], 777)
            save_numeric_results(loaded, root)
            self.assertEqual(
                load_existing_results(root)[1]["ram_peak_observed_bytes"], 777
            )


class PipelineResourceIntegrationTests(TestCase):
    def test_both_clis_offer_same_monitor_options(self):
        from external_ccta_batch_pipeline import build_parser as external_parser
        from utils.segmentation.pipeline.cli import build_parser

        for parser, required in (
            (build_parser("/dataset", "/output"), []),
            (external_parser(), ["--dataset", "mmwhs", "--resolution", "mid"]),
        ):
            args = parser.parse_args(
                [
                    *required,
                    "--monitor-resources",
                    "--resource-sampling-interval",
                    "0.75",
                ]
            )
            self.assertTrue(args.monitor_resources)
            self.assertEqual(args.resource_sampling_interval, 0.75)

    @patch.object(resources, "read_process_rss", return_value=123)
    def test_imagecas_success_and_early_return_keep_scientific_results(self, _rss):
        from utils.segmentation.pipeline.orchestration import process_image

        volume = np.ones((4, 4, 3), dtype=np.uint8)
        preprocessing = {
            "lcc_image": volume,
            "label": volume,
            "scaled_spacing": (1, 1, 1),
            "downscale_factors": (1, 1, 1),
        }
        config = {"VESSELNESS_AORTA": {}, "CIRCLE_DETECTION": {}, "LEVEL_SET": {}}
        prefix = "utils.segmentation.pipeline.orchestration."
        with ExitStack() as stack:
            stack.enter_context(
                patch(prefix + "load_and_preprocess_image", return_value=preprocessing)
            )
            stack.enter_context(
                patch(prefix + "compute_vesselness", return_value=volume)
            )
            stack.enter_context(
                patch(
                    prefix + "locate_and_filter_aorta_circles",
                    return_value=SimpleNamespace(
                        original_circles=[], filtered_circles=[], filter_diagnostics={}
                    ),
                )
            )
            stack.enter_context(
                patch(
                    prefix + "segment_aorta_with_diagnostics",
                    return_value=SimpleNamespace(mask=volume, diagnostics={}),
                )
            )
            ostia = stack.enter_context(
                patch(
                    prefix + "detect_and_evaluate_ostia",
                    return_value={
                        "label_artery": volume,
                        "ostia_left": (1, 1, 1),
                        "ostia_right": (2, 2, 1),
                    },
                )
            )
            stack.enter_context(
                patch(
                    prefix + "_ostia_result_fields",
                    return_value={"ostia_status": "both_correct"},
                )
            )
            segment = stack.enter_context(
                patch(
                    prefix + "segment_arteries_from_ostia",
                    side_effect=lambda *args: {
                        "dice_artery": 0.7,
                        "artery_mask": volume.copy(),
                    },
                )
            )
            disabled = process_image(1, config, Path("/dataset"))
            enabled = process_image(
                1, {**config, **ResourceMonitorTests().config()}, Path("/dataset")
            )
            self.assertEqual(disabled["dice_artery"], enabled["dice_artery"])
            self.assertEqual(enabled["resource_monitor_status"], "ok")
            np.testing.assert_array_equal(segment.call_args.args[0], volume)
            ostia.side_effect = ValueError("Nenhum óstio encontrado")
            failed = process_image(
                1, {**config, **ResourceMonitorTests().config()}, Path("/dataset")
            )
            self.assertEqual(failed["ostia_status"], "not_found")
            self.assertEqual(failed["ram_sample_count"], 2)

    @patch.object(resources, "read_process_rss", return_value=123)
    def test_external_loading_failure_still_finishes_monitor(self, _rss):
        from external_ccta_batch_pipeline import process_external_exam

        record = {
            "dataset": "MM-WHS",
            "subset": "test",
            "exam_id": "1",
            "reported_orientation": "RAS",
        }
        with patch(
            "external_ccta_batch_pipeline.load_ccta_volume",
            side_effect=OSError("missing image"),
        ):
            result = process_external_exam(
                record, ResourceMonitorTests().config(), "mid", visual_root=None
            )
        self.assertEqual(result["status"], "error")
        self.assertIn("missing image", result["error"])
        self.assertEqual(result["ram_sample_count"], 2)
        self.assertFalse(
            any(thread.name == "resource-monitor" for thread in threading.enumerate())
        )

    @patch.object(resources, "read_process_rss", return_value=123)
    def test_keyboard_interrupt_does_not_leave_background_monitor(self, _rss):
        from utils.segmentation.pipeline.orchestration import process_image

        with (
            patch(
                "utils.segmentation.pipeline.orchestration.load_and_preprocess_image",
                side_effect=KeyboardInterrupt,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            process_image(1, ResourceMonitorTests().config(), Path("/dataset"))
        self.assertEqual(_rss.call_count, 2)
        self.assertFalse(
            any(thread.name == "resource-monitor" for thread in threading.enumerate())
        )
