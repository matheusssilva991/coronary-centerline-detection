"""Regressões da persistência dos estágios morfológicos arteriais."""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from utils.project.results.external import (
    build_external_metadata,
    load_existing_results,
    save_numeric_results,
    upsert_external_result,
)
from utils.project.results.io import merge_batch_results, save_results
from utils.project.results.metadata import build_metadata_results
from utils.project.results.schema import (
    add_internal_result_aliases,
    make_result_dataframe,
    summarize_results_df,
)
from utils.project.results.summary import build_run_summary_row
from utils.segmentation.pipeline.orchestration import _load_previous_batches


class MorphologyResultsTests(TestCase):
    def measured(self):
        return {
            "IMG_ID": 2,
            "dice_artery": 0.6,
            "dice_artery_before_morphology": 0.5,
            "dice_artery_after_closing": 0.8,
            "dice_artery_after_morphology": 0.6,
            "dice_artery_closing_delta": 0.3,
            "dice_artery_dilation_delta": -0.2,
            "dice_artery_morphology_delta": 0.1,
            "artery_voxels_after_closing": 100,
        }

    def test_aliases_round_trip_and_absent_legacy_values_remain_missing(self):
        with TemporaryDirectory() as directory:
            path = save_results([{"IMG_ID": 1}, self.measured()], "train", directory)
            saved = pd.read_csv(path)
            for internal, readable in (
                ("dice_artery_after_closing", "artery_dice_after_closing"),
                ("dice_artery_closing_delta", "artery_dice_closing_delta"),
                ("dice_artery_dilation_delta", "artery_dice_dilation_delta"),
                ("artery_voxels_after_closing", "artery_voxel_count_after_closing"),
            ):
                self.assertTrue(pd.isna(saved.loc[0, readable]))
                self.assertEqual(saved.loc[1, readable], self.measured()[internal])
                normalized = add_internal_result_aliases(saved)
                self.assertEqual(normalized.loc[1, internal], self.measured()[internal])
            # A retomada pode fornecer dicionários com os nomes legíveis do CSV.
            again = make_result_dataframe(saved.to_dict("records"))
            self.assertEqual(again.loc[1, "dice_artery_after_closing"], 0.8)
            self.assertTrue(pd.isna(again.loc[0, "dice_artery_after_closing"]))

    def test_merge_and_resume_preserve_completed_batches_without_backfill(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_path = root / "results_train_lote_1.csv"
            pd.DataFrame([{"IMG_ID": 1, "artery_dice": 0.4}]).to_csv(
                legacy_path, index=False
            )
            original_bytes = legacy_path.read_bytes()
            save_results([self.measured()], "train_lote_2", root)
            previous, batches = _load_previous_batches(root, "train", 2)
            self.assertEqual(batches, [1, 2])
            self.assertNotIn("artery_dice_after_closing", previous[0])
            self.assertEqual(previous[1]["artery_dice_after_closing"], 0.8)
            merged_path = merge_batch_results("train", root)
            assert merged_path is not None
            merged = pd.read_csv(merged_path)
            self.assertTrue(pd.isna(merged.loc[0, "artery_dice_after_closing"]))
            self.assertEqual(merged.loc[1, "artery_dice_dilation_delta"], -0.2)
            self.assertEqual(legacy_path.read_bytes(), original_bytes)

    def test_summary_means_counts_and_legacy_metadata_are_not_backfilled(self):
        frame = make_result_dataframe(
            [{"IMG_ID": 1, "dice_artery": 0}, self.measured()]
        )
        summary = summarize_results_df(frame)
        self.assertEqual(summary["dice_artery_after_closing_count"], 1)
        self.assertEqual(summary["dice_artery_after_closing_mean"], 0.8)
        self.assertEqual(summary["dice_artery_dilation_delta_mean"], -0.2)
        self.assertEqual(summary["artery_voxel_count_after_closing_count"], 1)
        row = build_run_summary_row(
            frame,
            run_dir=Path("runs/mid_res/run"),
            split_name="train",
            expected_image_count=2,
        )
        self.assertEqual(row["dice_artery_after_closing_mean"], 0.8)
        self.assertEqual(row["dice_artery_closing_delta_count"], 1)
        self.assertEqual(row["artery_voxel_count_after_closing_mean"], 100)
        metadata = build_metadata_results(frame, [])
        self.assertEqual(
            metadata["dice"]["after_closing"], {"mean": 0.8, "valid_exam_count": 1}
        )
        self.assertEqual(metadata["dice"]["dilation_delta"]["mean"], -0.2)
        for legacy in (
            pd.DataFrame(),
            pd.DataFrame({"IMG_ID": [1], "artery_dice": [0.5]}),
        ):
            missing = build_metadata_results(legacy, [])
            for stage in ("after_closing", "closing_delta", "dilation_delta"):
                self.assertEqual(
                    missing["dice"][stage], {"mean": None, "valid_exam_count": 0}
                )
                aggregate = summarize_results_df(legacy)
                self.assertIsNone(aggregate[f"dice_artery_{stage}_mean"])
                self.assertEqual(aggregate[f"dice_artery_{stage}_count"], 0)

    def test_external_resume_preserves_legacy_rows_and_intermediate_volume(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            old = {"subset": "train", "exam_id": "old", "status": "success"}
            new = {
                "subset": "test",
                "exam_id": "new",
                "status": "success",
                "artery_voxels_after_closing": 20,
                "artery_volume_after_closing_ml": 0.03,
            }
            save_numeric_results([old], root)
            resumed = upsert_external_result(load_existing_results(root), new)
            save_numeric_results(resumed, root)
            loaded = load_existing_results(root)
            self.assertTrue(pd.isna(loaded[0]["artery_volume_after_closing_ml"]))
            self.assertEqual(loaded[1]["artery_volume_after_closing_ml"], 0.03)
            self.assertFalse(any("dice_artery" in row for row in loaded))
            metadata = build_external_metadata(
                dataset="mmwhs",
                resolution="mid",
                subset="all",
                rows=loaded,
                started_at=datetime.now(timezone.utc),
                state="completed",
            )
            self.assertEqual(
                metadata["artery_postprocessing"]["artery_volume_after_closing_ml"],
                {"mean": 0.03, "valid_exam_count": 1},
            )
