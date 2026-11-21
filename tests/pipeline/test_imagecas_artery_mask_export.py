import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np
import pandas as pd

import segmentation_pipeline as imagecas
from utils.segmentation.pipeline import orchestration
from utils.segmentation.pipeline.cli import parse_pipeline_args


class ImagecasArteryMaskExportTest(unittest.TestCase):
    def test_cli_exports_final_mask_and_keeps_the_existing_scientific_result(self):
        mask = np.zeros((3, 4, 3), dtype=np.uint8)
        mask[1, 1:3, 1] = 1
        raw_mask = np.zeros_like(mask)
        ostium = {"intersects": True, "euclidean_dist": 0, "physical_dist": 0.0}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "17.img.nii.gz"
            nib.save(
                nib.Nifti1Image(np.zeros((6, 8, 3), dtype=np.int16), np.eye(4)), source
            )
            with (
                patch.object(
                    imagecas, "get_data_splits", return_value=([17], [], [], [17])
                ),
                patch.object(imagecas, "setup_file_logging"),
                patch.multiple(
                    orchestration,
                    load_and_preprocess_image=Mock(
                        return_value={
                            "lcc_image": mask,
                            "label": mask,
                            "scaled_spacing": (2, 2, 1),
                            "downscale_factors": (2, 2, 1),
                            "source_path": source,
                        }
                    ),
                    locate_and_segment_aorta=Mock(
                        return_value=SimpleNamespace(aorta_mask=mask, diagnostics={})
                    ),
                    compute_vesselness=Mock(return_value=mask),
                    detect_and_evaluate_ostia=Mock(
                        return_value={
                            "both_correct": True,
                            "both_tolerable": True,
                            "ostia_left": (1, 1, 1),
                            "ostia_right": (1, 2, 1),
                            "left_info": ostium,
                            "right_info": ostium,
                            "label_artery": mask,
                        }
                    ),
                    segment_arteries_from_ostia=Mock(
                        side_effect=lambda *args, **kwargs: {
                            "artery_mask": mask,
                            "raw_artery_mask": raw_mask,
                            "dice_artery": 0.75,
                            "artery_voxels_after_morphology": 2,
                        }
                    ),
                ),
            ):
                runs = []
                for mode, flags in (
                    ("disabled", []),
                    ("enabled", ["--save-artery-masks"]),
                    ("failed", ["--save-artery-masks"]),
                ):
                    argv = [
                        "segmentation_pipeline.py",
                        "--split",
                        "train",
                        "--resolution",
                        "mid",
                        "--num-batches",
                        "1",
                        "--no-gpu",
                        "--output-dir",
                        str(root / mode),
                        *flags,
                    ]
                    with patch("sys.argv", argv):
                        args = parse_pipeline_args(root, root / mode)
                    if mode == "failed":
                        with (
                            patch(
                                "utils.project.results.artery_masks.nib.save",
                                side_effect=OSError("Disco cheio"),
                            ),
                            self.assertLogs(
                                "utils.project.results.artery_masks", level="ERROR"
                            ),
                        ):
                            run_dir, _, _ = imagecas._run_from_args(args)
                    else:
                        run_dir, _, _ = imagecas._run_from_args(args)
                    runs.append(run_dir)

            disabled, enabled, failed = runs
            baseline, exported, failed_frame = [
                pd.read_csv(run_dir / "numeric/results_train.csv") for run_dir in runs
            ]
            self.assertFalse((disabled / "masks").exists())
            self.assertNotIn("artery_mask_status", baseline)
            self.assertEqual(exported.at[0, "artery_mask_status"], "saved")
            self.assertEqual(failed_frame.at[0, "artery_mask_status"], "error")
            self.assertIn("Disco cheio", failed_frame.at[0, "artery_mask_error"])
            self.assertTrue(pd.isna(failed_frame.at[0, "artery_mask_path"]))
            prediction = nib.load(enabled / exported.at[0, "artery_mask_path"])
            np.testing.assert_array_equal(prediction.dataobj, mask)
            self.assertEqual(exported.at[0, "artery_dice"], 0.75)
            scientific = [
                "IMG_ID",
                "status",
                "artery_dice",
                "artery_voxel_count_after_morphology",
            ]
            for frame in (exported, failed_frame):
                pd.testing.assert_frame_equal(frame[scientific], baseline[scientific])
            snapshot = json.loads(
                (enabled / "config/effective_pipeline_config.json").read_text()
            )
            self.assertTrue(snapshot["SAVE_ARTERY_MASKS"])
            self.assertFalse((failed / "masks/artery/train/17/metadata.json").exists())

    def test_resume_reads_export_setting_and_treats_old_snapshots_as_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config/effective_pipeline_config.json"
            config_path.parent.mkdir()
            argv = [
                "segmentation_pipeline.py",
                "--resume-batch",
                "2",
                "--resume-dir",
                str(root),
            ]
            with patch("sys.argv", argv):
                args = parse_pipeline_args(root, root)
            config_path.write_text("{}")
            self.assertFalse(
                imagecas.build_effective_config(args).get("SAVE_ARTERY_MASKS")
            )
            with patch("sys.argv", [*argv, "--save-artery-masks"]):
                overridden = parse_pipeline_args(root, root)
            with self.assertRaisesRegex(ValueError, "novo run"):
                imagecas.build_effective_config(overridden)
            config_path.write_text('{"SAVE_ARTERY_MASKS": true}')
            self.assertTrue(imagecas.build_effective_config(args)["SAVE_ARTERY_MASKS"])
