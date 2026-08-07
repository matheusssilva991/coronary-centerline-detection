from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from utils.project.results import (
    batch_result_number,
    get_batch_result_file,
    list_batch_result_files,
)


class ProjectResultsTests(TestCase):
    def test_batch_helpers_accept_only_current_summary_format(self):
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            current_batch_2 = output_dir / "ostios_test_lote_2_summary.csv"
            current_batch_10 = output_dir / "ostios_test_lote_10_summary.csv"
            legacy_batch = output_dir / "ostios_test_lote_1.csv"
            unrelated_split = output_dir / "ostios_val_lote_3_summary.csv"
            for path in (
                current_batch_2,
                current_batch_10,
                legacy_batch,
                unrelated_split,
            ):
                path.touch()

            self.assertEqual(batch_result_number(current_batch_2, "test"), 2)
            self.assertIsNone(batch_result_number(legacy_batch, "test"))
            self.assertEqual(
                list_batch_result_files("test", output_dir),
                [current_batch_2, current_batch_10],
            )
            self.assertEqual(
                get_batch_result_file(output_dir, "test", 2),
                current_batch_2,
            )
            self.assertIsNone(get_batch_result_file(output_dir, "test", 1))
