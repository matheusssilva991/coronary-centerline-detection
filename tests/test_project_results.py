from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from utils.project.results import (
    batch_result_number,
    get_batch_result_file,
    list_batch_result_files,
)


class ProjectResultsTests(TestCase):
    def test_batch_helpers_accept_current_and_legacy_names(self):
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            current_batch_2 = output_dir / "results_test_lote_2.csv"
            legacy_batch_10 = output_dir / "ostios_test_lote_10_summary.csv"
            invalid_batch = output_dir / "ostios_test_lote_1.csv"
            unrelated_split = output_dir / "ostios_val_lote_3_summary.csv"
            for path in (
                current_batch_2,
                legacy_batch_10,
                invalid_batch,
                unrelated_split,
            ):
                path.touch()

            self.assertEqual(batch_result_number(current_batch_2, "test"), 2)
            self.assertEqual(batch_result_number(legacy_batch_10, "test"), 10)
            self.assertIsNone(batch_result_number(invalid_batch, "test"))
            self.assertEqual(
                list_batch_result_files("test", output_dir),
                [current_batch_2, legacy_batch_10],
            )
            self.assertEqual(
                get_batch_result_file(output_dir, "test", 2),
                current_batch_2,
            )
            self.assertEqual(
                get_batch_result_file(output_dir, "test", 10), legacy_batch_10
            )

    def test_duplicate_batch_number_between_formats_is_an_error(self):
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            (output_dir / "results_train_lote_1.csv").touch()
            (output_dir / "ostios_train_lote_1_summary.csv").touch()

            with self.assertRaisesRegex(ValueError, "duplicados entre formatos"):
                list_batch_result_files("train", output_dir)
            with self.assertRaisesRegex(ValueError, "duplicado entre formatos"):
                get_batch_result_file(output_dir, "train", 1)
