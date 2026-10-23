"""Testa a atribuição de memória por PID sem carregar dados científicos."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from experiments import pipeline_resource_benchmark as benchmark


class ResourceBenchmarkTests(TestCase):
    def test_vram_counts_only_worker_pid_and_converts_mib_to_bytes(self):
        output = "123, 2048\n456, 1000\n123, 2\n"
        self.assertEqual(benchmark.process_vram_bytes(output, 123), 2050 * 1024**2)
        self.assertEqual(benchmark.process_vram_bytes(output, 999), 0)
        self.assertEqual(benchmark.process_vram_bytes("", 123), 0)

    def test_unknown_memory_is_not_reported_as_zero(self):
        for value in ["N/A", "[N/A]", "Not Supported", "[Not Supported]"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                benchmark.process_vram_bytes(f"123, {value}", 123)

    def test_dry_run_does_not_launch_process_or_create_output(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            for suffix in ["img", "label"]:
                (base / f"642.{suffix}.nii.gz").touch()
            output = base / "not_created"
            args = benchmark.build_parser().parse_args(
                ["--base-path", str(base), "--output-root", str(output), "--dry-run"]
            )
            with patch.object(benchmark.subprocess, "Popen") as popen:
                self.assertIsNone(benchmark.benchmark(args))
                popen.assert_not_called()
            self.assertFalse(output.exists())

    def test_duplicate_resolutions_and_overfrequent_sampling_are_rejected(self):
        for options in [["--resolutions", "mid,mid"], ["--interval", "0.01"]]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                benchmark.benchmark(benchmark.build_parser().parse_args(options))
