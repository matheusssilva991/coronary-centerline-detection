import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from external_ccta_batch_pipeline import (
    _metadata_payload,
    build_parser,
    create_run_paths,
    load_existing_results,
    normalize_dataset_name,
    process_external_exam,
    run,
    select_inventory,
)


class ExternalCctaBatchPipelineTest(unittest.TestCase):
    def test_loads_existing_results_with_string_keys_and_preserved_order(self):
        rows = [
            {
                "subset": "train",
                "exam_id": "TRV1P2",
                "execution_time_seconds": 1.5,
            },
            {
                "subset": "test",
                "exam_id": "TRV2P1",
                "execution_time_seconds": 2.0,
            },
        ]

        with tempfile.TemporaryDirectory() as temporary_dir:
            numeric_dir = Path(temporary_dir)
            pd.DataFrame(rows).to_csv(numeric_dir / "results_all.csv", index=False)

            loaded = load_existing_results(numeric_dir)

        self.assertEqual(loaded, rows)
        self.assertTrue(
            all(isinstance(key, str) for row in loaded for key in row),
        )

    def test_cli_normalizes_supported_dataset_aliases(self):
        parser = build_parser()

        orca = parser.parse_args(["--dataset", "orca", "--resolution", "mid"])
        whs = parser.parse_args(["--dataset", "owhs", "--resolution", "high"])

        self.assertEqual(orca.dataset, "orcascore")
        self.assertEqual(whs.dataset, "mmwhs")
        self.assertEqual(normalize_dataset_name("MM-WHS"), "mmwhs")

    def test_inventory_filters_subset_ids_and_limit(self):
        inventory = pd.DataFrame(
            {
                "subset": ["train", "test", "train"],
                "exam_id": ["B", "C", "A"],
            }
        )

        selected = select_inventory(
            inventory,
            subset="train",
            exam_ids=["A", "B"],
            limit=1,
        )

        self.assertEqual(selected["exam_id"].tolist(), ["A"])

    def test_run_hierarchy_separates_dataset_and_resolution(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            paths = create_run_paths(Path(temporary_dir), "orcascore", "high")

            self.assertEqual(paths.run_dir.parent.name, "high_res")
            self.assertEqual(paths.run_dir.parent.parent.name, "orcascore")
            self.assertTrue(paths.numeric_dir.is_dir())
            self.assertTrue(paths.config_dir.is_dir())
            self.assertTrue(paths.logs_dir.is_dir())

    def test_metadata_aggregates_only_valid_aorta_dice(self):
        metadata = _metadata_payload(
            dataset="mmwhs",
            resolution="mid",
            subset="train",
            rows=[
                {
                    "status": "success",
                    "aorta_ground_truth_available": True,
                    "aorta_dice": 0.6,
                },
                {
                    "status": "ostia_not_found",
                    "aorta_ground_truth_available": True,
                    "aorta_dice": 0.8,
                },
                {
                    "status": "success",
                    "aorta_ground_truth_available": False,
                    "aorta_dice": None,
                },
            ],
            started_at=datetime.now(timezone.utc),
            state="complete",
        )

        aorta = metadata["ground_truth_metrics"]["aorta"]
        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(aorta["available_exam_count"], 2)
        self.assertEqual(aorta["evaluated_exam_count"], 2)
        self.assertAlmostEqual(aorta["dice_mean"], 0.7)

    @patch("external_ccta_batch_pipeline.use_gpu", return_value=False)
    @patch(
        "external_ccta_batch_pipeline.load_notebook_pipeline_config",
        return_value={"USE_GPU": False},
    )
    @patch("external_ccta_batch_pipeline.resolve_dataset_path")
    @patch("external_ccta_batch_pipeline.discover_ccta_dataset")
    @patch("external_ccta_batch_pipeline.process_external_exam")
    def test_resume_uses_manifest_and_skips_completed_exams(
        self,
        process_exam,
        discover_dataset,
        resolve_dataset_path,
        _load_config,
        _use_gpu,
    ):
        inventory = pd.DataFrame(
            {
                "dataset": ["OrCaScore"],
                "subset": ["train"],
                "exam_id": ["TRV1P1"],
            }
        )
        discover_dataset.return_value = inventory
        resolve_dataset_path.return_value = Path("/dataset")
        process_exam.return_value = {
            "dataset": "OrCaScore",
            "subset": "train",
            "exam_id": "TRV1P1",
            "resolution": "mid",
            "status": "success",
            "execution_time_seconds": 1.0,
        }

        with tempfile.TemporaryDirectory() as temporary_dir:
            args = build_parser().parse_args(
                [
                    "--dataset",
                    "orcascore",
                    "--resolution",
                    "mid",
                    "--subset",
                    "train",
                    "--output-root",
                    temporary_dir,
                    "--no-visuals",
                ]
            )
            with patch("external_ccta_batch_pipeline._configure_logging"):
                paths = run(args)
                process_exam.reset_mock()
                resume_args = build_parser().parse_args(
                    [
                        "--dataset",
                        "orcascore",
                        "--resolution",
                        "mid",
                        "--subset",
                        "train",
                        "--resume-dir",
                        str(paths.run_dir),
                        "--no-visuals",
                    ]
                )

                resumed_paths = run(resume_args)
            metadata = json.loads(
                (resumed_paths.run_dir / "metadata.json").read_text(encoding="utf-8")
            )

        self.assertEqual(resumed_paths.run_dir, paths.run_dir)
        self.assertEqual(metadata["state"], "complete")
        process_exam.assert_not_called()

    @patch("external_ccta_batch_pipeline.use_gpu", return_value=False)
    @patch(
        "external_ccta_batch_pipeline.load_notebook_pipeline_config",
        return_value={"USE_GPU": False},
    )
    @patch("external_ccta_batch_pipeline.resolve_dataset_path")
    @patch("external_ccta_batch_pipeline.discover_ccta_dataset")
    @patch("external_ccta_batch_pipeline.process_external_exam")
    def test_resume_reprocesses_labeled_exam_without_aorta_dice(
        self,
        process_exam,
        discover_dataset,
        resolve_dataset_path,
        _load_config,
        _use_gpu,
    ):
        discover_dataset.return_value = pd.DataFrame(
            {
                "dataset": ["MM-WHS"],
                "subset": ["train"],
                "exam_id": ["ct_train_1001"],
                "label_path": ["/dataset/ct_train_1001_label.nii.gz"],
            }
        )
        resolve_dataset_path.return_value = Path("/dataset")
        process_exam.return_value = {
            "dataset": "MM-WHS",
            "subset": "train",
            "exam_id": "ct_train_1001",
            "resolution": "mid",
            "status": "success",
            "aorta_ground_truth_available": True,
            "aorta_ground_truth_evaluated": True,
            "aorta_dice": 0.75,
            "execution_time_seconds": 1.0,
        }

        with tempfile.TemporaryDirectory() as temporary_dir:
            args = build_parser().parse_args(
                [
                    "--dataset",
                    "mmwhs",
                    "--resolution",
                    "mid",
                    "--subset",
                    "train",
                    "--output-root",
                    temporary_dir,
                    "--no-visuals",
                ]
            )
            with patch("external_ccta_batch_pipeline._configure_logging"):
                paths = run(args)
                for filename in ("results_all.csv", "results_train.csv"):
                    result_path = paths.numeric_dir / filename
                    legacy = pd.read_csv(result_path).drop(
                        columns=[
                            "aorta_ground_truth_available",
                            "aorta_ground_truth_evaluated",
                            "aorta_dice",
                        ]
                    )
                    legacy.to_csv(result_path, index=False)

                process_exam.reset_mock()
                resume_args = build_parser().parse_args(
                    [
                        "--dataset",
                        "mmwhs",
                        "--resolution",
                        "mid",
                        "--subset",
                        "train",
                        "--resume-dir",
                        str(paths.run_dir),
                        "--no-visuals",
                    ]
                )
                run(resume_args)

            refreshed = pd.read_csv(paths.numeric_dir / "results_train.csv")

        process_exam.assert_called_once()
        self.assertEqual(refreshed.loc[0, "aorta_dice"], 0.75)

    @patch("external_ccta_batch_pipeline._save_aorta_ground_truth_visual")
    @patch("external_ccta_batch_pipeline._save_combined_visual")
    @patch("external_ccta_batch_pipeline.save_detected_circles_figure")
    @patch("external_ccta_batch_pipeline._save_stage")
    @patch("external_ccta_batch_pipeline.segment_artery_masks_from_vesselness")
    @patch("external_ccta_batch_pipeline.detect_ostia")
    @patch("external_ccta_batch_pipeline.compute_vesselness")
    @patch("external_ccta_batch_pipeline.segment_aorta_with_diagnostics")
    @patch("external_ccta_batch_pipeline.locate_and_filter_aorta_circles")
    @patch("external_ccta_batch_pipeline.preprocess_ccta_volume")
    @patch("external_ccta_batch_pipeline.load_ccta_aorta_ground_truth")
    @patch("external_ccta_batch_pipeline.load_ccta_volume")
    def test_processes_all_notebook_stages_and_returns_external_metrics(
        self,
        load_volume,
        load_ground_truth,
        preprocess,
        track_circles,
        segment_aorta,
        vesselness,
        detect_ostia,
        segment_arteries,
        save_stage,
        save_circles,
        save_combined,
        save_aorta_comparison,
    ):
        image = np.arange(48, dtype=np.float32).reshape(4, 4, 3)
        processed = np.ones((2, 2, 3), dtype=np.float32)
        mask = np.ones_like(processed, dtype=np.uint8)
        circles = [
            {
                "slice_index": 1,
                "center_x": 1,
                "center_y": 1,
                "radius": 1,
                "accum": 0.9,
                "interpolated": False,
            },
            {
                "slice_index": 2,
                "center_x": 1,
                "center_y": 1,
                "radius": 1,
                "accum": 0.8,
                "interpolated": False,
            },
        ]
        load_volume.return_value = image
        load_ground_truth.return_value = np.ones_like(image, dtype=bool)
        preprocess.return_value = {
            "threshold_mask": mask,
            "lcc_image": processed,
            "downscale_factors": (2, 2, 1),
            "scaled_spacing": (1.0, 1.0, 1.5),
            "label": mask,
            "preprocessing_details": {
                "threshold_mode": "normal",
                "threshold_voxels": 12,
                "lcc_voxels": 12,
            },
        }
        track_circles.return_value = SimpleNamespace(
            original_circles=circles,
            filtered_circles=circles[:1],
            filter_diagnostics={"aorta_circle_filter_method": "robust"},
        )
        segment_aorta.return_value = SimpleNamespace(
            mask=mask,
            diagnostics={"aorta_level_set_iterations_used": 26},
        )
        vesselness.side_effect = [processed, processed]
        detect_ostia.return_value = ((0, 0, 1), (1, 1, 1))
        segment_arteries.return_value = SimpleNamespace(
            raw_mask=mask,
            closed_mask=mask,
            final_mask=mask,
        )
        record = pd.Series(
            {
                "dataset": "MM-WHS",
                "subset": "train",
                "exam_id": "ct_train_1001",
                "reported_orientation": "RAS",
                "spacing_x_mm": 0.5,
                "spacing_y_mm": 0.5,
                "spacing_z_mm": 1.5,
                "path": "/dataset/ct_train_1001_image.nii.gz",
                "file_format": "NIfTI",
            }
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            result = process_external_exam(
                record,
                {
                    "CIRCLE_DETECTION": {},
                    "LEVEL_SET": {},
                    "VESSELNESS_AORTA": {},
                    "VESSELNESS_ARTERY": {},
                    "ARTERY_SEGMENTATION": {"method": "fuzzy_connectedness"},
                    "USE_GPU": False,
                },
                "mid",
                visual_root=Path(temporary_dir),
            )
            result_path = (
                Path(temporary_dir) / "train" / "ct_train_1001" / "result.json"
            )
            persisted = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["processed_slice_count"], 3)
        self.assertEqual(result["aorta_circle_count_before_filter"], 2)
        self.assertEqual(result["aorta_circle_count"], 1)
        self.assertEqual(result["ostia_left_z"], 1)
        self.assertGreater(result["artery_volume_after_morphology_ml"], 0)
        self.assertTrue(result["aorta_ground_truth_available"])
        self.assertTrue(result["aorta_ground_truth_evaluated"])
        self.assertEqual(result["aorta_ground_truth_voxels"], int(mask.sum()))
        self.assertEqual(result["aorta_dice"], 1.0)
        self.assertEqual(persisted["status"], "success")
        self.assertEqual(save_stage.call_count, 9)
        save_circles.assert_called_once()
        save_combined.assert_called_once()
        save_aorta_comparison.assert_called_once()
        np.testing.assert_array_equal(
            preprocess.call_args.kwargs["label"],
            load_ground_truth.return_value,
        )
        self.assertEqual(
            segment_arteries.call_args.kwargs["method"],
            "region_growing",
        )

    @patch("external_ccta_batch_pipeline.detect_ostia")
    @patch("external_ccta_batch_pipeline.compute_vesselness")
    @patch("external_ccta_batch_pipeline.segment_aorta_with_diagnostics")
    @patch("external_ccta_batch_pipeline.locate_and_filter_aorta_circles")
    @patch("external_ccta_batch_pipeline.preprocess_ccta_volume")
    @patch("external_ccta_batch_pipeline.load_ccta_aorta_ground_truth")
    @patch("external_ccta_batch_pipeline.load_ccta_volume")
    def test_ostia_failure_skips_arterial_vesselness(
        self,
        load_volume,
        load_ground_truth,
        preprocess,
        track_circles,
        segment_aorta,
        vesselness,
        detect_ostia,
    ):
        image = np.ones((4, 4, 3), dtype=np.float32)
        processed = np.ones((2, 2, 3), dtype=np.float32)
        mask = np.ones_like(processed, dtype=np.uint8)
        circles = [
            {
                "slice_index": 1,
                "center_x": 1,
                "center_y": 1,
                "radius": 1,
                "accum": 0.9,
                "interpolated": False,
            }
        ]
        load_volume.return_value = image
        load_ground_truth.return_value = np.ones_like(image, dtype=bool)
        preprocess.return_value = {
            "threshold_mask": mask,
            "lcc_image": processed,
            "downscale_factors": (2, 2, 1),
            "scaled_spacing": (1.0, 1.0, 1.5),
            "label": mask,
            "preprocessing_details": {},
        }
        track_circles.return_value = SimpleNamespace(
            original_circles=circles,
            filtered_circles=circles,
            filter_diagnostics={},
        )
        segment_aorta.return_value = SimpleNamespace(mask=mask, diagnostics={})
        vesselness.return_value = processed
        detect_ostia.side_effect = ValueError("Nenhum óstio encontrado")
        record = pd.Series(
            {
                "dataset": "MM-WHS",
                "subset": "train",
                "exam_id": "ct_train_1001",
                "reported_orientation": "RAS",
                "spacing_x_mm": 0.5,
                "spacing_y_mm": 0.5,
                "spacing_z_mm": 1.5,
            }
        )

        result = process_external_exam(
            record,
            {
                "CIRCLE_DETECTION": {},
                "LEVEL_SET": {},
                "VESSELNESS_AORTA": {"stage": "ostia"},
                "VESSELNESS_ARTERY": {"stage": "artery"},
                "USE_GPU": False,
            },
            "mid",
            visual_root=None,
        )

        self.assertEqual(result["status"], "ostia_not_found")
        self.assertEqual(result["aorta_dice"], 1.0)
        self.assertEqual(vesselness.call_count, 1)
        self.assertEqual(
            vesselness.call_args.kwargs["vesselness_config"],
            {"stage": "ostia"},
        )


if __name__ == "__main__":
    unittest.main()
