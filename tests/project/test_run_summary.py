import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from utils.comparison_utils.io import load_split_results
from utils.project.results import (
    EDA_REQUIRED_RESULT_COLUMN_UNION,
    READABLE_COLUMN_NAMES,
    RESULT_COLUMNS,
    ResultIntegrityError,
    build_metadata,
    build_metadata_results,
    effective_config_sha256,
    merge_batch_results,
    select_per_image_result_columns,
    summarize_results_df,
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

        timings = [
            {"batch_number": 1, "total_batches": 2, "duration_seconds": 6},
            {"batch_number": 2, "total_batches": 2, "duration_seconds": 9},
        ]
        metadata = build_metadata(
            "train",
            _config(),
            resolution="mid",
            results=_results(),
            batch_timings=timings,
            expected_batches=[1, 2],
        )
        self.assertEqual(metadata["configuration"]["artery_segmentation_method"], "rg")
        self.assertEqual(
            metadata["configuration"]["sha256"],
            effective_config_sha256(_config()),
        )
        self.assertEqual(
            set(metadata),
            {"metadata_schema_version", "run", "configuration", "results"},
        )
        self.assertEqual(metadata["metadata_schema_version"], 4)
        self.assertEqual(metadata["results"]["execution_time"]["seconds"], 15)
        self.assertAlmostEqual(
            metadata["results"]["dice"]["before_morphology"]["mean"], 0.45
        )
        self.assertAlmostEqual(
            metadata["results"]["dice"]["after_morphology"]["mean"], 0.55
        )
        self.assertEqual(metadata["results"]["ostia"]["success"]["count"], 2)
        self.assertEqual(metadata["results"]["ostia"]["success"]["percent"], 50)
        serialized = json.dumps(metadata)
        self.assertNotIn("IMG_ID", serialized)
        self.assertNotIn("batch_timing", serialized)
        self.assertNotIn("dice_artery_q1", serialized)
        self.assertNotIn("circle_detection", serialized)

    def test_metadata_result_types_are_exclusive_and_sides_independent(self) -> None:
        results = _results().assign(
            ostia_detection_status=[
                "both correct",
                "both tolerable",
                "found but incorrect",
                "not evaluated",
            ],
            left_ostium_correct=["yes", "no", "yes", "no"],
            right_ostium_correct=["yes", "yes", "no", "no"],
        )
        aggregate = build_metadata_results(results, [], expected_batches=[1])
        ostia = aggregate["ostia"]

        self.assertEqual(ostia["processed_exam_count"], 4)
        self.assertEqual(
            {name: entry["count"] for name, entry in ostia["types"].items()},
            {
                "both_correct": 1,
                "both_tolerable": 1,
                "found_but_incorrect": 1,
                "not_found": 0,
                "not_evaluated_or_error": 1,
            },
        )
        self.assertEqual(sum(item["count"] for item in ostia["types"].values()), 4)
        self.assertEqual(ostia["sides"]["left_correct"]["count"], 2)
        self.assertEqual(ostia["sides"]["right_correct"]["count"], 2)
        self.assertIsNone(aggregate["execution_time"]["seconds"])

    def test_metadata_results_handle_zero_denominator_and_legacy_dice(self) -> None:
        empty = build_metadata_results(pd.DataFrame(), [], expected_batches=[])
        self.assertIsNone(empty["ostia"]["success"]["percent"])
        self.assertIsNone(empty["dice"]["before_morphology"]["mean"])
        self.assertEqual(empty["dice"]["before_morphology"]["valid_exam_count"], 0)

        legacy = pd.DataFrame(
            {
                "IMG_ID": [1, 2],
                "artery_dice": [0.25, 0.75],
                "ostia_detection_status": ["not found", "found but incorrect"],
            }
        )
        aggregate = build_metadata_results(legacy, [], expected_batches=[1])
        self.assertIsNone(aggregate["dice"]["before_morphology"]["mean"])
        self.assertEqual(
            aggregate["dice"]["after_morphology"],
            {"mean": 0.5, "valid_exam_count": 2},
        )

    def test_metadata_execution_time_requires_every_valid_batch(self) -> None:
        valid = [
            {"batch_number": 1, "duration_seconds": 30},
            {"batch_number": 2, "duration_seconds": 90},
        ]
        complete = build_metadata_results(_results(), valid, expected_batches=[1, 2])[
            "execution_time"
        ]
        self.assertEqual(complete, {"seconds": 120, "minutes": 2, "hours": 1 / 30})

        missing = build_metadata_results(
            _results(), valid[:1], expected_batches=[1, 2]
        )["execution_time"]
        invalid = build_metadata_results(
            _results(),
            [valid[0], {"batch_number": 2, "duration_seconds": "invalid"}],
            expected_batches=[1, 2],
        )["execution_time"]
        missing_declared_batch = build_metadata_results(
            _results(),
            [
                {"batch_number": 1, "total_batches": 3, "duration_seconds": 30},
                {"batch_number": 3, "total_batches": 3, "duration_seconds": 90},
            ],
            expected_batches=[1, 3],
        )["execution_time"]
        self.assertEqual(missing, {"seconds": None, "minutes": None, "hours": None})
        self.assertEqual(invalid, missing)
        self.assertEqual(missing_declared_batch, missing)

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
            self.assertIsNotNone(results_path)
            if results_path is None:
                self.fail("A consolidação deveria retornar o caminho do CSV.")

            self.assertEqual(Path(results_path).name, "results_val.csv")
            loaded = load_split_results(paths, "mid_res", "val")
            self.assertIsNotNone(loaded)
            if loaded is None:
                self.fail("Os resultados consolidados deveriam estar disponíveis.")
            self.assertEqual(len(loaded), 4)
            summary = summarize_results_df(loaded)
            self.assertAlmostEqual(summary["dice_artery_mean"], 0.55)

    def test_legacy_results_remain_readable(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            numeric_dir = Path(temporary_dir)
            _results().to_csv(numeric_dir / "ostios_test_summary.csv", index=False)
            paths = {"mid_res": {"test": numeric_dir}}

            loaded = load_split_results(paths, "mid_res", "test")
            self.assertIsNotNone(loaded)
            if loaded is None:
                self.fail("Os resultados legados deveriam continuar legíveis.")
            self.assertEqual(len(loaded), 4)


if __name__ == "__main__":
    unittest.main()
