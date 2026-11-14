"""Testa repetição e promoção canônica sem carregar volumes científicos."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from experiments import rebuild_canonical_metrics as driver
from utils.experiments.fuzzy_pipeline_comparison import make_image_result_row


class RebuildCanonicalMetricsTest(TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "config").mkdir()
        (self.root / "config/imagecas_splits.json").write_text(
            json.dumps({"splits": {"train": [1, 2]}})
        )
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        for img_id in (1, 2):
            for kind in ("img", "label"):
                (self.dataset / f"{img_id}.{kind}.nii.gz").touch()
        self.config = {
            "DOWNSCALE_FACTORS": [2, 2, 1],
            "CIRCLE_DETECTION": {"radii_start_px": 12},
        }
        self.registry = (
            self.root / "output/segmentation/canonical/references_manifest.json"
        )
        for resolution in ("mid", "high"):
            current = self.current(resolution)
            (current / "numeric").mkdir(parents=True)
            (current / "config").mkdir()
            pd.DataFrame({"IMG_ID": [1, 2]}).to_csv(
                current / "numeric/results_train.csv", index=False
            )
            (current / "config/effective_pipeline_config.json").write_text(
                json.dumps(self.config)
            )
            (current / "config/experiment_settings.json").write_text(
                json.dumps({"threshold_mode": "normal"})
            )
        self.registry.write_text(
            json.dumps(
                {
                    "current_references": [
                        {
                            "resolution": "mid_res",
                            "split": "train",
                            "source_run": "original",
                        },
                        {
                            "resolution": "high_res",
                            "split": "train",
                            "source_run": "original_high",
                        },
                    ]
                }
            )
        )

    def current(self, resolution):
        return (
            self.root / f"output/segmentation/canonical/{resolution}_res/train/current"
        )

    def candidate(self, resolution):
        return (
            self.root
            / f"output/segmentation/runs/canonical_metrics/new_metrics/{resolution}_res/train"
        )

    def row(self, img_id):
        return {
            **make_image_result_row(img_id, "baseline", "train", {}),
            "dice_artery": 0.8,
            "dice_artery_after_closing": 0.7,
            "segmentation_attempted": True,
            "ostia_status": "both_correct",
            "aorta_segmentation_feedback": "adequate",
            "aorta_level_set_circle_fill_q25": 0.9,
            "aorta_level_set_circle_area_ratio_p90": 1.5,
            "artery_hd_mm": 2.0,
            "artery_ahd_mm": 0.5,
            "artery_distance_metrics_status": "ok",
        }

    def test_plan_uses_saved_config_and_exact_ids_without_rescaling(self):
        plan = driver.build_plan(self.root, "mid", "train")
        self.assertEqual(plan["ids"], [1, 2])
        self.assertEqual(
            plan["effective_config"]["CIRCLE_DETECTION"],
            self.config["CIRCLE_DETECTION"],
        )
        self.assertTrue(plan["effective_config"]["EVALUATION"]["distance_metrics"])
        self.assertNotIn(
            "EVALUATION",
            json.loads(
                (
                    self.current("mid") / "config/effective_pipeline_config.json"
                ).read_text()
            ),
        )
        self.assertEqual(plan["execution_flow"], "shared_experiment")

    def test_rebuild_mid_and_high_preserves_new_metrics_without_touching_current(self):
        for resolution in ("mid", "high"):
            with self.subTest(resolution=resolution):
                plan = driver.build_plan(self.root, resolution, "train")
                original = (
                    self.current(resolution) / "numeric/results_train.csv"
                ).read_bytes()
                with (
                    patch.object(
                        driver,
                        "run_image",
                        side_effect=lambda img_id, *args: self.row(img_id),
                    ) as shared,
                    patch.object(
                        driver,
                        "process_image",
                        side_effect=lambda img_id, *args: self.row(img_id),
                    ) as principal,
                ):
                    driver.rebuild(self.candidate(resolution), plan, self.dataset)
                self.assertEqual(shared.call_count, 2 if resolution == "mid" else 0)
                self.assertEqual(principal.call_count, 2 if resolution == "high" else 0)
                frame = driver.validate_complete(self.candidate(resolution), plan)
                self.assertEqual(
                    frame["artery_dice_after_closing"].tolist(), [0.7, 0.7]
                )
                self.assertEqual(frame["artery_hd_mm"].tolist(), [2, 2])
                self.assertEqual(
                    (
                        self.current(resolution) / "numeric/results_train.csv"
                    ).read_bytes(),
                    original,
                )

    def test_resume_skips_completed_exam_and_rejects_changed_plan(self):
        plan = driver.build_plan(self.root, "mid", "train")
        candidate = self.candidate("mid")
        with patch.object(
            driver, "run_image", side_effect=[self.row(1), KeyboardInterrupt()]
        ):
            with self.assertRaises(KeyboardInterrupt):
                driver.rebuild(candidate, plan, self.dataset)
        with patch.object(driver, "run_image", return_value=self.row(2)) as run:
            driver.rebuild(candidate, plan, self.dataset, resume=True)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0], 2)
        changed = copy.deepcopy(plan)
        changed["effective_config"]["USE_GPU"] = False
        with self.assertRaisesRegex(ValueError, "mudou"):
            driver.rebuild(candidate, changed, self.dataset, resume=True)

    def test_promotion_archives_old_reference_and_updates_registry(self):
        plan = driver.build_plan(self.root, "mid", "train")
        candidate = self.candidate("mid")
        original = (self.current("mid") / "numeric/results_train.csv").read_bytes()
        with patch.object(driver, "run_image", side_effect=[self.row(1), self.row(2)]):
            driver.rebuild(candidate, plan, self.dataset)
        driver.promote(self.root, candidate, plan)
        self.assertEqual(self.current("mid").resolve(), candidate)
        archive = self.current("mid").with_name("before_new_metrics")
        self.assertEqual((archive / "numeric/results_train.csv").read_bytes(), original)
        registry = json.loads(self.registry.read_text())
        self.assertEqual(len(registry["metric_refresh_history"]), 1)
        self.assertEqual(len(registry["current_references"]), 2)
        driver.promote(self.root, candidate, plan)

    def test_missing_volumes_fail_before_creating_a_run(self):
        plan = driver.build_plan(self.root, "mid", "train")
        (self.dataset / "1.img.nii.gz").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Volumes ausentes"):
            driver.rebuild(self.candidate("mid"), plan, self.dataset)
        self.assertFalse(self.candidate("mid").exists())

    def test_promotion_rejects_modified_results_and_incomplete_cohort(self):
        plan = driver.build_plan(self.root, "mid", "train")
        candidate = self.candidate("mid")
        with patch.object(driver, "run_image", side_effect=[self.row(1), self.row(2)]):
            driver.rebuild(candidate, plan, self.dataset)
        csv = candidate / "numeric/results_train.csv"
        original = csv.read_text()
        csv.write_text(original + "\n")
        with self.assertRaisesRegex(ValueError, "CSV candidato mudou"):
            driver.promote(self.root, candidate, plan)
        frame = pd.read_csv(csv)
        frame.iloc[:1].to_csv(csv, index=False)
        with self.assertRaisesRegex(ValueError, "incompleto"):
            driver.promote(self.root, candidate, plan)
        self.assertFalse(self.current("mid").is_symlink())

    def test_registry_write_failure_restores_original_current(self):
        plan = driver.build_plan(self.root, "mid", "train")
        candidate = self.candidate("mid")
        with patch.object(driver, "run_image", side_effect=[self.row(1), self.row(2)]):
            driver.rebuild(candidate, plan, self.dataset)
        with patch.object(driver, "save_json_atomic", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                driver.promote(self.root, candidate, plan)
        self.assertTrue(self.current("mid").is_dir())
        self.assertFalse(self.current("mid").is_symlink())
        self.assertFalse(self.current("mid").with_name("before_new_metrics").exists())

    def test_high_symlink_is_archived_without_modifying_original_run(self):
        current = self.current("high")
        original = current.with_name("original_run")
        current.rename(original)
        current.symlink_to("original_run", target_is_directory=True)
        plan = driver.build_plan(self.root, "high", "train")
        with patch.object(
            driver, "process_image", side_effect=[self.row(1), self.row(2)]
        ):
            driver.rebuild(self.candidate("high"), plan, self.dataset)
        driver.promote(self.root, self.candidate("high"), plan)
        self.assertEqual(current.resolve(), self.candidate("high"))
        self.assertEqual(current.with_name("before_new_metrics").resolve(), original)
        self.assertEqual(
            pd.read_csv(original / "numeric/results_train.csv").columns.tolist(),
            ["IMG_ID"],
        )

    def test_validation_rejects_disabled_distances_or_missing_ok_measurements(self):
        plan = driver.build_plan(self.root, "mid", "train")
        candidate = self.candidate("mid")
        with patch.object(driver, "run_image", side_effect=[self.row(1), self.row(2)]):
            driver.rebuild(candidate, plan, self.dataset)
        path = candidate / "numeric/results_train.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "artery_distance_metrics_status"] = "disabled"
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "desativadas"):
            driver.validate_complete(candidate, plan)
        frame.loc[0, "artery_distance_metrics_status"] = "ok"
        frame.loc[0, "artery_hd_mm"] = float("nan")
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "medida inválida"):
            driver.validate_complete(candidate, plan)
