"""Tests for the historical status normalization migration."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.normalize_run_statuses import normalize_runs


class NormalizeRunStatusesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_csv(self, relative_path: str, rows: list[list[str]]) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as csv_file:
            csv.writer(csv_file).writerows(rows)
        return path

    def test_inspection_does_not_modify_files(self) -> None:
        path = self._write_csv(
            "run/numeric/results_train.csv",
            [
                ["IMG_ID", "artery_dice", "ostia_detection_status", "status"],
                ["7", "0.475", "both correct", "ambos corretos"],
            ],
        )
        original = path.read_bytes()

        report = normalize_runs(self.root, apply=False)

        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(report["summary"]["changed_file_count"], 1)
        self.assertEqual(report["summary"]["changed_cell_count"], 2)

    def test_apply_normalizes_only_status_cells_and_is_idempotent(self) -> None:
        path = self._write_csv(
            "run/numeric/results_val_lote_1.csv",
            [
                ["IMG_ID", "artery_dice", "ostia_detection_status", "status"],
                ["10", "0.123456789", "found but incorrect", "um correto"],
                ["11", "", "not found", "óstios não encontrados"],
            ],
        )
        manifest = self.root / "manifest.json"

        report = normalize_runs(self.root, apply=True, manifest_path=manifest)

        with path.open(encoding="utf-8", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        self.assertEqual([row["IMG_ID"] for row in rows], ["10", "11"])
        self.assertEqual(rows[0]["artery_dice"], "0.123456789")
        self.assertEqual(rows[1]["artery_dice"], "")
        self.assertEqual(rows[0]["ostia_detection_status"], "found_but_wrong")
        self.assertEqual(rows[0]["status"], "one_correct")
        self.assertEqual(rows[1]["ostia_detection_status"], "not_found")
        self.assertEqual(rows[1]["status"], "not_found")
        self.assertEqual(report["summary"]["changed_cell_count"], 4)
        self.assertTrue(manifest.exists())
        self.assertEqual(json.loads(manifest.read_text())["mode"], "apply")

        second_report = normalize_runs(self.root, apply=True)
        self.assertEqual(second_report["summary"]["changed_file_count"], 0)

    def test_legacy_results_are_included_but_provenance_is_immutable(self) -> None:
        legacy = self._write_csv(
            "legacy/numeric/ostios_test_summary.csv",
            [["IMG_ID", "status"], ["3", "no ostium correct"]],
        )
        provenance = self._write_csv(
            "reference/provenance/original_results.csv",
            [["IMG_ID", "status"], ["3", "no ostium correct"]],
        )
        provenance_before = provenance.read_bytes()

        normalize_runs(self.root, apply=True)

        self.assertIn("none_correct", legacy.read_text())
        self.assertEqual(provenance.read_bytes(), provenance_before)

    def test_unknown_status_aborts_without_writing(self) -> None:
        path = self._write_csv(
            "run/numeric/results_test.csv",
            [["IMG_ID", "status"], ["1", "unexpected result"]],
        )
        original = path.read_bytes()

        with self.assertRaisesRegex(ValueError, "Unknown normalized status"):
            normalize_runs(self.root, apply=True)

        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
