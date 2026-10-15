"""Testa ablações, coorte fixa e retomada sem executar exames científicos."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments import pipeline_component_ablation as driver
from utils.experiments import fuzzy_pipeline_comparison as helpers
from utils.segmentation.aorta import segmentation
from utils.segmentation.pipeline import detection


class ComponentBehaviorTests(TestCase):
    def setUp(self):
        self.volume = np.zeros((24, 24, 9), dtype=np.float32)
        self.circles = [
            {"center_x": 10, "center_y": 11, "radius": 6, "slice_index": z}
            for z in range(2, 7)
        ]
        self.config = {
            "num_iter": 2,
            "radius_reduction_factor": 0.5,
            "balloon": 0.6,
            "smoothing": 1,
            "threshold": "auto",
            "leak_removal_radius": 1,
            "use_roi": True,
            "roi_margin": 2,
            "trajectory_radius_factor": 2.25,
            "trajectory_axial_margin_slices": 1,
        }

    def test_initial_mask_matches_original_roi_initialization(self):
        for use_roi in [False, True]:
            with self.subTest(use_roi=use_roi):
                context = segmentation.prepare_level_set_evolution(
                    self.volume,
                    self.circles,
                    radius_reduction_factor=0.5,
                    roi_margin=2,
                    use_roi=use_roi,
                )
                expected = segmentation._restore_level_set_mask(
                    context.current_mask, self.volume.shape, context.roi_bounds
                )
                actual = segmentation.initialize_level_set_mask(
                    self.volume.shape,
                    self.circles,
                    radius_reduction_factor=0.5,
                    roi_margin=2,
                    use_roi=use_roi,
                )
                np.testing.assert_array_equal(actual, expected)

    def test_without_level_set_skips_gradients_and_evolution_keeps_postprocessing(self):
        initial = segmentation.initialize_level_set_mask(
            self.volume.shape,
            self.circles,
            radius_reduction_factor=0.5,
            roi_margin=2,
        )
        expected = detection._postprocess_aorta_mask(initial, self.circles, self.config)
        with (
            patch.object(segmentation, "inverse_gaussian_gradient") as gradient,
            patch.object(segmentation, "_evolve_level_set") as evolve,
            patch.object(detection, "prepare_level_set_evolution") as prepare,
        ):
            actual = detection.segment_aorta_without_level_set(
                self.volume, self.circles, self.config
            )
        gradient.assert_not_called()
        evolve.assert_not_called()
        prepare.assert_not_called()
        np.testing.assert_array_equal(actual, expected)
        self.assertGreater(int(actual.sum()), 0)

    def test_full_circles_survive_opening_when_reduced_seeds_disappear(self):
        config = {
            **self.config,
            "radius_reduction_factor": 0.1,
            "leak_removal_radius": 2,
        }
        with (
            patch.object(segmentation, "inverse_gaussian_gradient") as gradient,
            patch.object(segmentation, "_evolve_level_set") as evolve,
        ):
            legacy = detection.segment_aorta_without_level_set(
                self.volume, self.circles, config
            )
            corrected = detection.segment_aorta_without_level_set(
                self.volume, self.circles, config, radius_factor=1.0
            )
        expected_initial = segmentation.initialize_level_set_mask(
            self.volume.shape,
            self.circles,
            radius_reduction_factor=1.0,
            roi_margin=config["roi_margin"],
            use_roi=config["use_roi"],
        )
        expected = detection._postprocess_aorta_mask(
            expected_initial, self.circles, config
        )
        self.assertEqual(int(legacy.sum()), 0)
        self.assertGreater(int(corrected.sum()), 0)
        np.testing.assert_array_equal(corrected, expected)
        self.assertEqual(config["radius_reduction_factor"], 0.1)
        gradient.assert_not_called()
        evolve.assert_not_called()

    def test_invalid_circle_radius_factors_are_rejected(self):
        for factor in [0, -1, 1.1, float("nan"), float("inf")]:
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                detection.segment_aorta_without_level_set(
                    self.volume, self.circles, self.config, radius_factor=factor
                )

    def test_empty_circles_without_level_set_returns_empty_mask(self):
        result = detection.segment_aorta_without_level_set(self.volume, [], self.config)
        self.assertEqual(result.shape, self.volume.shape)
        self.assertEqual(int(result.sum()), 0)

    def context(self, experiment):
        config = {
            "VESSELNESS_AORTA": {"sigmas": [2.5]},
            "VESSELNESS_ARTERY": {"sigmas": [1.5, 2.5]},
            "CIRCLE_DETECTION": {},
            "LEVEL_SET": self.config,
            "USE_GPU": False,
        }
        case = {
            "down_image": self.volume,
            "down_label": self.volume.astype(np.uint8),
            "scaled_spacing": (1, 1, 1),
            "downscale_factors": (2, 2, 1),
        }
        calls = []

        def vesselness(image, vesselness_config, use_gpu):
            calls.append(vesselness_config)
            return np.full_like(image, len(calls), dtype=np.float32)

        with (
            patch.object(
                helpers,
                "build_preprocessed_inputs",
                return_value=(self.volume, self.volume.astype(bool), {}),
            ),
            patch.object(helpers, "compute_vesselness", side_effect=vesselness),
            patch.object(helpers, "locate_aorta_circles", return_value=self.circles),
            patch.object(
                helpers, "filter_located_aorta_circles", return_value=(self.circles, {})
            ),
        ):
            context = helpers.prepare_image_context(case, config, experiment)
        return context, calls, config

    def test_no_level_set_mode_routes_context_without_gradient_or_evolution(self):
        with (
            patch.object(segmentation, "inverse_gaussian_gradient") as gradient,
            patch.object(segmentation, "_evolve_level_set") as evolve,
            patch.object(detection, "prepare_level_set_evolution") as prepare,
        ):
            context, calls, _ = self.context(driver.VARIANTS["no_level_set"])
        gradient.assert_not_called()
        evolve.assert_not_called()
        prepare.assert_not_called()
        self.assertEqual(len(calls), 2)
        self.assertGreater(int(context["aorta_mask"].sum()), 0)

    def test_single_map_computed_once_with_arterial_parameters(self):
        context, calls, config = self.context(driver.VARIANTS["single_vesselness"])
        self.assertEqual(calls, [config["VESSELNESS_ARTERY"]])
        self.assertIs(context["vesselness_ostios"], context["vesselness_artery"])

    def test_full_variant_preserves_default_masks_and_separate_maps(self):
        default, default_calls, _ = self.context({})
        full, full_calls, _ = self.context(driver.VARIANTS["full_pipeline"])
        self.assertEqual(default_calls, full_calls)
        self.assertEqual(len(full_calls), 2)
        self.assertIsNot(full["vesselness_ostios"], full["vesselness_artery"])
        for key in [
            "aorta_mask",
            "lcc_image",
            "vesselness_ostios",
            "vesselness_artery",
        ]:
            np.testing.assert_array_equal(default[key], full[key])
        # A máscara final arterial também permanece idêntica no caminho de controle.
        label = np.ones_like(self.volume, dtype=np.uint8)
        ostia = {
            "both_correct": True,
            "both_tolerable": False,
            "left_info": {"physical_dist": 0.0},
            "right_info": {"physical_dist": 0.0},
            "label_artery": label,
            "ostia_left": (10, 11, 3),
            "ostia_right": (10, 11, 4),
        }
        with (
            patch.object(helpers, "detect_and_evaluate_ostia", return_value=ostia),
            patch.object(
                helpers,
                "segment_arteries_from_vesselness",
                return_value={"raw_artery_mask": label, "artery_mask": label},
            ) as arteries,
        ):
            rows = []
            for context, experiment in [
                ({}, {}),
                (full, driver.VARIANTS["full_pipeline"]),
            ]:
                context = default if not context else context
                row = helpers.make_image_result_row(
                    1, "full_pipeline", "test", experiment
                )
                rows.append(
                    helpers.evaluate_prepared_image(context, row, {}, experiment)
                )
            self.assertEqual(rows[0]["dice_artery"], rows[1]["dice_artery"])
            np.testing.assert_array_equal(
                arteries.call_args_list[0].args[2], arteries.call_args_list[1].args[2]
            )


class DriverTests(TestCase):
    def args(self, *extra):
        return driver.build_parser().parse_args(["--no-gpu", *map(str, extra)])

    def row(self, image_id, dice, success=False, error=None):
        return {
            "variant": "full_pipeline",
            "IMG_ID": image_id,
            "dice_artery": dice,
            "ostia_success": success,
            "error": error,
            "elapsed_seconds": 2.0,
        }

    def test_default_test_cohort_matches_reference_700_ids(self):
        manifest = driver.prepare_manifest(self.args())
        self.assertEqual(len(manifest["ids"]), 700)
        self.assertEqual(len(set(manifest["ids"])), 700)
        expected = json.loads(driver.DEFAULT_SPLITS.read_text())["splits"]["test"]
        self.assertEqual(manifest["ids"], expected)
        self.assertEqual(manifest["effective_config"]["LEVEL_SET"]["num_iter"], 26)
        self.assertEqual(list(manifest["variants"]), list(driver.VARIANTS))

    def test_rejects_duplicate_foreign_and_conflicting_ids(self):
        ids = driver.prepare_manifest(self.args())["ids"]
        train_id = json.loads(driver.DEFAULT_SPLITS.read_text())["splits"]["train"][0]
        for extra in [
            ["--ids", f"{ids[0]},{ids[0]}"],
            ["--ids", str(train_id)],
            ["--ids", str(ids[0]), "--sample-size", "1"],
            ["--variants", "no_level_set,no_level_set"],
            ["--variants", "unknown"],
        ]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                driver.prepare_manifest(self.args(*extra))

    def test_summary_keeps_bad_seeds_dice_and_zeroes_only_explicit_errors(self):
        manifest = driver.prepare_manifest(self.args("--variants", "full_pipeline"))
        rows = [
            self.row(1, 0.8, True),
            self.row(2, 0.4),
            self.row(3, np.nan, error="failure"),
        ]
        summary = driver.summarize_results(rows, manifest).iloc[0]
        self.assertAlmostEqual(summary.mean_dice_end_to_end, 0.4)
        self.assertAlmostEqual(summary.mean_dice_conditional, 0.8)
        self.assertEqual(summary.images, 3)
        self.assertFalse(summary.complete)
        self.assertEqual(summary.valid_dice_n, 2)
        self.assertEqual(summary.error_count, 1)
        self.assertIsNone(summary.peak_ram_bytes)
        with self.assertRaises(ValueError):
            driver.summarize_results([self.row(1, np.nan)], manifest)

    def test_resume_rejects_config_cohort_and_variant_changes(self):
        manifest = driver.prepare_manifest(self.args())
        for field, value in [
            ("ids", [1]),
            ("variants", {}),
            ("config_sha256", "changed"),
        ]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                driver.validate_resume(manifest, {**manifest, field: value})

    def test_corrected_no_level_set_cannot_resume_legacy_run(self):
        manifest = driver.prepare_manifest(self.args("--variants", "no_level_set"))
        self.assertEqual(
            manifest["variants"]["no_level_set"]["aorta_circle_radius_factor"], 1.0
        )
        legacy = json.loads(json.dumps(manifest))
        legacy["variants"]["no_level_set"]["aorta_segmentation_mode"] = "initial_mask"
        del legacy["variants"]["no_level_set"]["aorta_circle_radius_factor"]
        with self.assertRaisesRegex(ValueError, "variants"):
            driver.validate_resume(legacy, manifest)

    def test_dry_run_loads_no_images_and_creates_no_folder(self):
        with TemporaryDirectory() as tmp, patch.object(driver, "run_image") as run:
            output = Path(tmp) / "must_not_exist"
            result = driver.execute(self.args("--output-root", output, "--dry-run"))
            self.assertIsNone(result)
            self.assertFalse(output.exists())
            run.assert_not_called()

    def test_checkpoint_resume_skips_completed_errors_and_rejects_duplicates(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp) / "dataset"
            base.mkdir()
            args = self.args(
                "--sample-size", "2", "--base-path", base, "--output-root", tmp
            )
            manifest = driver.prepare_manifest(args)
            for image_id in manifest["ids"]:
                for kind in ["img", "label"]:
                    (base / f"{image_id}.{kind}.nii.gz").touch()

            def fake_run(image_id, name, split, path, config, experiment):
                return {
                    **self.row(image_id, np.nan, error="synthetic error"),
                    "variant": name,
                }

            with patch.object(driver, "run_image", side_effect=fake_run) as run:
                result = driver.execute(args)
                self.assertEqual(run.call_count, 6)
            resume_args = self.args(
                "--sample-size", "2", "--base-path", base, "--resume-dir", result
            )
            with patch.object(driver, "run_image") as run:
                driver.execute(resume_args)
                run.assert_not_called()
            csv = result / "results/image_results.csv"
            frame = pd.read_csv(csv)
            self.assertEqual(len(frame), 6)
            self.assertTrue(frame.dice_artery.isna().all())
            pd.concat([frame, frame.iloc[:1]]).to_csv(csv, index=False)
            with self.assertRaises(ValueError):
                driver.load_completed_rows(result, manifest)

    def test_partial_resume_runs_only_missing_pairs(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp) / "data"
            base.mkdir()
            run_dir = Path(tmp) / "interrupted"
            run_dir.mkdir()
            args = self.args(
                "--sample-size",
                "2",
                "--variants",
                "no_level_set",
                "--base-path",
                base,
                "--resume-dir",
                run_dir,
            )
            manifest = driver.prepare_manifest(args)
            driver.write_json(run_dir / "run_config.json", manifest)
            first, second = manifest["ids"]
            for image_id in [first, second]:
                for kind in ["img", "label"]:
                    (base / f"{image_id}.{kind}.nii.gz").touch()
            first_row = {
                **self.row(first, np.nan, error="already recorded"),
                "variant": "no_level_set",
            }
            driver.save_checkpoint(run_dir, [first_row], manifest)
            last_row = {**self.row(second, 0.5, True), "variant": "no_level_set"}
            with patch.object(driver, "run_image", return_value=last_row) as run:
                driver.execute(args)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0], second)
            result = pd.read_csv(run_dir / "summary/ablation_summary.csv").iloc[0]
            self.assertEqual(result.images, 2)
            self.assertTrue(result.complete)
            self.assertAlmostEqual(result.mean_dice_end_to_end, 0.25)
            self.assertEqual(result.ostia_success_n, 1)

    def test_existing_named_run_is_never_overwritten(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp) / "data"
            base.mkdir()
            args = self.args(
                "--sample-size",
                "1",
                "--base-path",
                base,
                "--output-root",
                tmp,
                "--run-name",
                "existing",
            )
            manifest = driver.prepare_manifest(args)
            for kind in ["img", "label"]:
                (base / f"{manifest['ids'][0]}.{kind}.nii.gz").touch()
            (Path(tmp) / "existing").mkdir()
            with (
                patch.object(driver, "run_image") as run,
                self.assertRaises(FileExistsError),
            ):
                driver.execute(args)
            run.assert_not_called()
