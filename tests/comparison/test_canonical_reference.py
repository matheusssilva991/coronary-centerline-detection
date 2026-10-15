"""Testa consolidação sem executar processamento de imagens."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.comparison_utils.canonical import consolidate_experiment_reference
from utils.comparison_utils.io import load_split_results, load_split_batch_timings


class CanonicalReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        (self.source / "results").mkdir(parents=True)
        self.csv = self.source / "results/image_results.csv"
        pd.DataFrame(
            {
                "IMG_ID": [1, 2, 99],
                "variant": ["baseline", "baseline", "other"],
                "split": ["test"] * 3,
                "dice_artery": [0.7, None, 0.9],
                "ostia_status": ["both_correct", "error", "both_tolerable"],
                "error": [None, "surface missing", None],
                "segmentation_attempted": [True, False, True],
            }
        ).to_csv(self.csv, index=False)
        self.record = {
            "effective_base_config": {
                "DOWNSCALE_FACTORS": [2, 2, 1],
                "MAX_THRESHOLD_PERCENTILE": 99.7,
            },
            "variants": [
                {
                    "name": "baseline",
                    "overrides": {
                        "MAX_THRESHOLD_PERCENTILE": 99.9,
                        "threshold_mode": "normal",
                        "artery_method": "region_growing",
                    },
                }
            ],
        }
        (self.source / "run_config.json").write_text(json.dumps(self.record))
        self.destination = self.root / "canonical/test/current"

    def consolidate(self, count=2):
        return consolidate_experiment_reference(
            self.root,
            self.source,
            self.destination,
            variant="baseline",
            split="test",
            expected_count=count,
        )

    def test_roundtrip_preserves_selected_ids_errors_and_source(self):
        before = self.csv.read_bytes()
        manifest = self.consolidate()
        paths = {"mid_res": {"test": self.destination / "numeric"}}
        frame = load_split_results(paths, "mid_res", "test")
        self.assertEqual(frame["IMG_ID"].tolist(), [1, 2])
        self.assertEqual(frame["dice_artery"].tolist(), [0.7, 0.0])
        self.assertEqual(frame["status"].tolist(), ["both_correct", "error"])
        self.assertEqual(frame.iloc[1]["pipeline_error"], "surface missing")
        self.assertTrue(pd.isna(frame.iloc[1]["evaluable_dice"]))
        self.assertEqual(self.csv.read_bytes(), before)
        self.assertEqual(
            manifest["source_csv_sha256"], hashlib.sha256(before).hexdigest()
        )
        self.assertEqual(manifest["source_run"], "source")
        self.assertFalse(frame.columns.duplicated().any())
        self.assertIsNone(load_split_batch_timings(paths, "mid_res", "test"))

    def test_config_comes_from_recorded_snapshot_and_overrides(self):
        self.consolidate()
        config = json.loads(
            (self.destination / "config/effective_pipeline_config.json").read_text()
        )
        self.assertEqual(config["MAX_THRESHOLD_PERCENTILE"], 99.9)
        experiment = json.loads(
            (self.destination / "config/experiment_settings.json").read_text()
        )
        self.assertEqual(experiment["artery_method"], "region_growing")

    def test_incomplete_cohort_does_not_write_destination(self):
        with self.assertRaisesRegex(ValueError, "incompleta"):
            self.consolidate(count=3)
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
