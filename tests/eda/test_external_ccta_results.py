"""Testa o carregamento de métricas quantitativas dos runs externos."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.project.external_ccta_results import load_mmwhs_train_aorta_dice


class ExternalCctaResultsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.results_path = Path(self.temp_dir.name) / "results_all.csv"

    def _write_results(self, frame: pd.DataFrame) -> None:
        frame.to_csv(self.results_path, index=False)

    @staticmethod
    def _valid_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "dataset": ["MM-WHS", "MM-WHS", "MM-WHS"],
                "subset": ["train", "train", "test"],
                "exam_id": ["ct_train_1002", "ct_train_1001", "ct_test_2001"],
                "aorta_ground_truth_evaluated": [True, True, False],
                "aorta_dice": [0.0, 0.9, None],
            }
        )

    def test_loads_only_evaluated_train_dice_in_id_order(self):
        self._write_results(self._valid_frame())

        result = load_mmwhs_train_aorta_dice(self.results_path)

        self.assertEqual(result.index.tolist(), ["ct_train_1001", "ct_train_1002"])
        self.assertEqual(result.tolist(), [0.9, 0.0])

    def test_rejects_missing_train_dice(self):
        frame = self._valid_frame()
        frame.loc[0, "aorta_dice"] = None
        self._write_results(frame)

        with self.assertRaisesRegex(ValueError, "Dice da aorta ausente"):
            load_mmwhs_train_aorta_dice(self.results_path)

    def test_rejects_duplicate_train_ids(self):
        frame = self._valid_frame()
        frame.loc[1, "exam_id"] = "ct_train_1002"
        self._write_results(frame)

        with self.assertRaisesRegex(ValueError, "IDs de treino ausentes ou duplicados"):
            load_mmwhs_train_aorta_dice(self.results_path)
