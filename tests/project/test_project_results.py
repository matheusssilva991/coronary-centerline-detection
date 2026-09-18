from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from utils.project.results import (
    batch_result_number,
    get_batch_result_file,
    list_batch_result_files,
    merge_batch_results,
    save_results,
)


class ProjectResultsTests(TestCase):
    def test_save_results_persists_canonical_english_status_codes(self):
        with TemporaryDirectory() as temporary_dir:
            path = save_results(
                [
                    {
                        "IMG_ID": 1,
                        "ostia_status": "ambos toleráveis",
                        "both_tolerable": True,
                    }
                ],
                "train",
                temporary_dir,
            )

            saved = pd.read_csv(path)

        self.assertEqual(saved.loc[0, "ostia_detection_status"], "both_tolerable")
        self.assertEqual(saved.loc[0, "status"], "both_tolerable")

    def test_merge_normalizes_legacy_status_values(self):
        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            pd.DataFrame(
                {
                    "IMG_ID": [1],
                    "ostia_detection_status": ["both correct"],
                    "status": ["ambos corretos"],
                }
            ).to_csv(output_dir / "results_val_lote_1.csv", index=False)

            path = merge_batch_results("val", output_dir)
            merged = pd.read_csv(path)

        self.assertEqual(merged.loc[0, "ostia_detection_status"], "both_correct")
        self.assertEqual(merged.loc[0, "status"], "both_correct")

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
