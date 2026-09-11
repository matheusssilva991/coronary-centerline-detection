import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from experiments.migrate_run_result_layout import migrate_run
from utils.comparison_utils.io import load_split_results, load_split_summary
from utils.comparison_utils.metadata import summarize_split_results
from utils.project.results import (
    EDA_REQUIRED_RESULT_COLUMN_UNION,
    READABLE_COLUMN_NAMES,
    RESULT_COLUMNS,
    build_metadata,
    merge_batch_results,
    select_per_image_result_columns,
)
from utils.project.run_summary import (
    ResultIntegrityError,
    build_run_summary_row,
    effective_config_sha256,
    validate_result_integrity,
)


def _config(method: str = "rg") -> dict:
    return {
        "USE_GPU": True,
        "DOWNSCALE_METHOD": "opencv",
        "OPENCV_INTERPOLATION": "linear",
        "DOWNSCALE_FACTORS": [2, 2, 1],
        "MIN_THRESHOLD": -300,
        "MAX_THRESHOLD_PERCENTILE": 99.9,
        "THRESHOLDING": {"method": "normal"},
        "LOWER_THRESHOLD": {"method": "fixed"},
        "VESSELNESS_AORTA": {
            "sigmas": [2.5, 3.0],
            "alpha": 0.5,
            "beta": 0.5,
            "gamma": 5.0,
        },
        "VESSELNESS_ARTERY": {
            "sigmas": [1.5, 3.0],
            "alpha": 0.5,
            "beta": 0.5,
            "gamma": 5.0,
        },
        "CIRCLE_DETECTION": {
            "radii_start_px": 18,
            "radii_end_px": 31,
            "radius_step_px": 1,
            "canny_sigma": 3,
            "trajectory_filter": {"method": "robust"},
        },
        "LEVEL_SET": {"num_iter": 31, "balloon": 0.8},
        "OSTIA_DETECTION": {"top_n": 2000, "erosion_radius": 4},
        "OSTIA_VALIDATION": {"distance_threshold_mm": 7.0},
        "ARTERY_SEGMENTATION": {"method": method},
        "REGION_GROWING": {
            "comparison_window": 1,
            "min_vesselness_fraction": 0.078,
        },
        "FUZZY_CONNECTEDNESS": {
            "alpha": 0.16,
            "sigma_hu": 100,
            "neighborhood": 26,
        },
        "POSTPROCESSING": {"closing_radius": 3, "dilation_radius": 2},
    }


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "IMG_ID": [10, 20, 30, 40],
            "artery_dice": [0.1, 0.5, 0.7, 0.9],
            "artery_dice_before_morphology": [0.0, 0.4, 0.6, 0.8],
            "artery_dice_morphology_delta": [0.1, 0.1, 0.1, 0.1],
            "ostia_detected": ["yes", "yes", "yes", "no"],
            "both_ostia_correct": ["yes", "no", "no", "no"],
            "both_ostia_tolerable": ["no", "yes", "no", "no"],
            "artery_segmentation_run": ["yes", "yes", "yes", "no"],
            "segmented_with_incorrect_ostia": ["no", "no", "yes", "no"],
            "effective_upper_threshold_hu": [800, 900, 1000, None],
            "image_slice_count": [100, 110, 120, 130],
            "artery_voxel_count": [10, 20, 30, 40],
            "aorta_circle_count": [50, 60, 70, 80],
            "aorta_segmented_slice_count": [45, 55, 65, 75],
            "aorta_circle_coverage": [0.5, 0.6, 0.7, 0.8],
            "aorta_mask_voxel_count": [1000, 2000, 3000, 4000],
            "aorta_volume_fraction": [0.01, 0.02, 0.03, 0.04],
            "aorta_segmentation_feedback": [
                "adequate",
                "suspected_undersegmentation",
                "suspected_oversegmentation",
                "insufficient_data",
            ],
            "pipeline_error": [None, None, "failed", None],
        }
    )


class RunSummaryTests(unittest.TestCase):
    def test_hash_is_deterministic(self) -> None:
        self.assertEqual(
            effective_config_sha256({"b": 2, "a": {"d": 4, "c": 3}}),
            effective_config_sha256({"a": {"c": 3, "d": 4}, "b": 2}),
        )

    def test_summary_identity_supports_every_split(self) -> None:
        for split in ("train", "val", "test", "full"):
            with self.subTest(split=split):
                row = build_run_summary_row(
                    _results(),
                    _config(),
                    run_dir=Path(f"runs/mid_res/group/{split}/run-id"),
                    split_name=split,
                    expected_image_count=4,
                )
                self.assertEqual(row["split"], split)

    def test_summary_is_scalar_one_run_aggregate(self) -> None:
        row = build_run_summary_row(
            _results(),
            _config("rg"),
            run_dir=Path("runs/mid_res/group/train/2026-01-02_03-04-05"),
            split_name="train",
            expected_image_count=4,
        )

        self.assertEqual(row["processed_image_count"], 4)
        self.assertAlmostEqual(row["dice_artery_mean"], 0.55)
        self.assertAlmostEqual(row["dice_artery_median"], 0.6)
        self.assertAlmostEqual(row["dice_artery_q1"], 0.4)
        self.assertAlmostEqual(row["dice_artery_q3"], 0.75)
        self.assertAlmostEqual(row["dice_artery_valid_ostia_mean"], 0.3)
        self.assertAlmostEqual(row["dice_artery_invalid_ostia_mean"], 0.8)
        self.assertEqual(row["pipeline_error_count"], 1)
        self.assertEqual(row["image_slice_count_sum"], 460)
        self.assertEqual(row["image_slice_count_median"], 115)
        self.assertEqual(row["aorta_circle_count_mean"], 65)
        self.assertEqual(row["artery_voxel_count_mean"], 25)
        self.assertEqual(row["summary_schema_version"], 2)
        self.assertNotIn("IMG_ID", row)
        self.assertNotIn("config_sha256", row)
        self.assertNotIn("artery_segmentation_method", row)
        self.assertNotIn("threshold_mode", row)
        self.assertFalse(any("timing" in key or "duration" in key for key in row))
        self.assertTrue(
            all(not isinstance(value, (list, dict, tuple)) for value in row.values())
        )

    def test_summary_never_repeats_method_configuration(self) -> None:
        row = build_run_summary_row(
            _results(),
            _config("fc"),
            run_dir=Path("runs/high_res/group/test/run-id"),
            split_name="test",
            expected_image_count=4,
        )

        self.assertFalse(any(key.startswith(("rg_", "fc_")) for key in row))
        self.assertNotIn("artery_segmentation_method", row)

    def test_result_schema_covers_every_explicit_eda_requirement(self) -> None:
        readable_columns = {
            READABLE_COLUMN_NAMES.get(column, column) for column in RESULT_COLUMNS
        }
        self.assertTrue(EDA_REQUIRED_RESULT_COLUMN_UNION.issubset(readable_columns))

    def test_results_drop_configuration_and_metadata_keeps_it(self) -> None:
        projected = select_per_image_result_columns(
            _results().assign(
                threshold_mode="normal",
                artery_segmentation_method="region_growing",
                lcc_per_slice=True,
            )
        )
        self.assertNotIn("threshold_mode", projected.columns)
        self.assertNotIn("artery_segmentation_method", projected.columns)
        self.assertNotIn("lcc_per_slice", projected.columns)

        metadata = build_metadata("train", _config(), resolution="mid")
        self.assertEqual(metadata["configuration"]["artery_segmentation_method"], "rg")
        self.assertEqual(
            metadata["configuration"]["sha256"],
            effective_config_sha256(_config()),
        )
        self.assertEqual(
            set(metadata), {"metadata_schema_version", "run", "configuration"}
        )
        serialized = json.dumps(metadata)
        self.assertNotIn("IMG_ID", serialized)
        self.assertNotIn("batch_timing", serialized)
        self.assertNotIn("dice_artery_q1", serialized)
        self.assertNotIn("circle_detection", serialized)

    def test_integrity_rejects_missing_unexpected_and_duplicate_ids(self) -> None:
        with self.assertRaises(ResultIntegrityError) as context:
            validate_result_integrity(
                pd.DataFrame({"IMG_ID": [1, 1, 4]}),
                [1, 2, 3],
            )

        report = context.exception.report
        self.assertEqual(report["duplicate_image_ids"], [1])
        self.assertEqual(report["missing_image_ids"], [2, 3])
        self.assertEqual(report["unexpected_image_ids"], [4])

    def test_merge_writes_results_and_derives_summary_on_demand(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            numeric_dir = Path(temporary_dir)
            _results().iloc[:2].to_csv(
                numeric_dir / "results_val_lote_1.csv", index=False
            )
            _results().iloc[2:].to_csv(
                numeric_dir / "ostios_val_lote_2_summary.csv", index=False
            )

            results_path = merge_batch_results("val", numeric_dir)
            paths = {"mid_res": {"val": numeric_dir}}

            self.assertEqual(Path(results_path).name, "results_val.csv")
            self.assertEqual(len(load_split_results(paths, "mid_res", "val")), 4)
            with self.assertRaises(FileNotFoundError):
                load_split_summary(paths, "mid_res", "val")
            self.assertAlmostEqual(
                summarize_split_results(paths, "mid_res", "val")["dice_artery_mean"],
                0.55,
            )

    def test_legacy_results_remain_readable(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            numeric_dir = Path(temporary_dir)
            _results().to_csv(numeric_dir / "ostios_test_summary.csv", index=False)
            paths = {"mid_res": {"test": numeric_dir}}

            self.assertEqual(len(load_split_results(paths, "mid_res", "test")), 4)
            with self.assertRaisesRegex(ValueError, "formato legado"):
                load_split_summary(paths, "mid_res", "test")

    def test_migration_preserves_scientific_rows_and_ignores_partial(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            complete_run = root / "mid_res/group/train/complete"
            complete_numeric = complete_run / "numeric"
            complete_numeric.mkdir(parents=True)
            old_summary = complete_numeric / "ostios_train_summary.csv"
            legacy_results = _results().assign(
                threshold_mode="normal", downscale_method="opencv"
            )
            legacy_results.to_csv(old_summary, index=False)
            expected_scientific = select_per_image_result_columns(legacy_results)
            (complete_run / "config").mkdir()
            (complete_run / "config/split_ids.json").write_text(
                json.dumps({"splits": {"train": [10, 20, 30, 40]}}),
                encoding="utf-8",
            )
            (complete_run / "config/effective_pipeline_config.json").write_text(
                json.dumps(_config()), encoding="utf-8"
            )

            self.assertEqual(migrate_run(old_summary, apply=False), "ready")
            self.assertTrue(old_summary.exists())
            self.assertEqual(migrate_run(old_summary, apply=True), "migrated")
            actual_scientific = pd.read_csv(complete_numeric / "results_train.csv")
            pd.testing.assert_frame_equal(
                actual_scientific.fillna("<NA>").astype(str),
                expected_scientific.fillna("<NA>").astype(str),
            )
            self.assertFalse((complete_numeric / "summary_train.csv").exists())
            metadata = json.loads(
                (complete_numeric / "metadata_train.json").read_text()
            )
            self.assertEqual(metadata["configuration"]["threshold_method"], "normal")
            self.assertNotIn("results_summary", metadata)

            partial_run = root / "high_res/group/train/partial"
            partial_numeric = partial_run / "numeric"
            partial_numeric.mkdir(parents=True)
            partial_summary = partial_numeric / "ostios_train_summary.csv"
            _results().iloc[:2].to_csv(partial_summary, index=False)
            (partial_run / "config").mkdir()
            (partial_run / "config/split_ids.json").write_text(
                json.dumps({"splits": {"train": [10, 20, 30]}}),
                encoding="utf-8",
            )
            (partial_numeric / "ostios_train_metadata.json").write_text(
                "{}", encoding="utf-8"
            )
            (partial_numeric / "ostios_train_integrity.json").write_text(
                json.dumps({"status": "incomplete"}), encoding="utf-8"
            )
            original_partial = partial_summary.read_bytes()

            self.assertEqual(
                migrate_run(partial_summary, apply=True), "partial_preserved"
            )
            self.assertEqual(partial_summary.read_bytes(), original_partial)
            self.assertFalse((partial_numeric / "summary_train.csv").exists())

    def test_migration_uses_legacy_metadata_without_inventing_config_hash(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir) / "mid_res/group/val/legacy"
            numeric_dir = run_dir / "numeric"
            numeric_dir.mkdir(parents=True)
            old_summary = numeric_dir / "ostios_val_summary.csv"
            _results().to_csv(old_summary, index=False)
            metadata = {
                "execution_info": {"image_ids": [10, 20, 30, 40]},
                "runtime_config": {"use_gpu": False},
                "preprocessing_config": {
                    "downscale_factors": [2, 2, 1],
                    "min_threshold": -300,
                },
                "artery_segmentation_config": {"method": "region_growing"},
                "region_growing_config": {"comparison_window": 1},
            }
            (numeric_dir / "ostios_val_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

            self.assertEqual(migrate_run(old_summary, apply=True), "migrated")
            self.assertFalse((numeric_dir / "summary_val.csv").exists())
            migrated_metadata = json.loads(
                (numeric_dir / "metadata_val.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                migrated_metadata["configuration"]["source"],
                "legacy_metadata",
            )
            self.assertEqual(migrated_metadata["configuration"]["sha256"], "")
            self.assertEqual(
                migrated_metadata["configuration"]["artery_segmentation_method"],
                "region_growing",
            )
            split_ids = json.loads(
                (run_dir / "config/split_ids.json").read_text(encoding="utf-8")
            )
            self.assertEqual(split_ids["splits"]["val"], [10, 20, 30, 40])


if __name__ == "__main__":
    unittest.main()
