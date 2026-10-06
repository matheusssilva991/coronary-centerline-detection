"""Testes da exportação de uma aorta MM-WHS test para avaliação oficial."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from nibabel.loadsave import load as load_nifti
from nibabel.loadsave import save as save_nifti
from nibabel.nifti1 import Nifti1Image

from experiments.mmwhs_test_aorta import (
    evaluate_with_wine,
    load_run_config,
    predict_aorta,
    restore_native_mask,
    save_prediction,
    select_test_record,
    validate_saved_prediction,
    verify_run_result,
)
from utils.project.evaluation.mmwhs_official_aorta import (
    AortaPrediction,
    OFFICIAL_AORTA_METHOD,
    parse_dice_lo,
    parse_aorta_dice,
    parse_label_dice_output,
    aorta_dice_result_path,
)


class MmwhsTestAortaTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_loads_frozen_run_and_rejects_other_exam(self):
        run_dir = self.root / "run"
        config_dir = run_dir / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "effective_pipeline_config.json").write_text(
            json.dumps({"DOWNSCALE_FACTORS": [1, 1, 1], "USE_GPU": True})
        )
        (config_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "dataset": "mmwhs",
                    "resolution": "high",
                    "selected_exams": [{"subset": "test", "exam_id": "ct_test_2001"}],
                }
            )
        )

        config, resolution = load_run_config(run_dir, "ct_test_2001")

        self.assertEqual(resolution, "high")
        self.assertEqual(config["DOWNSCALE_FACTORS"], (1, 1, 1))
        self.assertFalse(config["USE_GPU"])
        with self.assertRaisesRegex(ValueError, "não pertence"):
            load_run_config(run_dir, "ct_test_2002")

    def test_selects_only_existing_ct_test_id(self):
        inventory = pd.DataFrame(
            {
                "subset": ["train", "test"],
                "exam_id": ["ct_train_1001", "ct_test_2001"],
            }
        )
        with patch(
            "experiments.mmwhs_test_aorta.discover_ccta_dataset",
            return_value=inventory,
        ):
            record = select_test_record(self.root, "ct_test_2001")
            self.assertEqual(record["exam_id"], "ct_test_2001")
            with self.assertRaisesRegex(ValueError, "Use um ID"):
                select_test_record(self.root, "ct_train_1001")
            with self.assertRaisesRegex(ValueError, "encontrados 0"):
                select_test_record(self.root, "ct_test_2002")

    @patch(
        "utils.project.evaluation.mmwhs_official_aorta.segment_aorta_with_diagnostics"
    )
    @patch(
        "utils.project.evaluation.mmwhs_official_aorta.locate_and_filter_aorta_circles"
    )
    @patch("utils.project.evaluation.mmwhs_official_aorta.preprocess_ccta_volume")
    @patch("utils.project.evaluation.mmwhs_official_aorta.load_ccta_volume")
    def test_reuses_batch_aorta_steps(self, load_volume, preprocess, locate, segment):
        load_volume.return_value = np.ones((4, 4, 2), dtype=np.float32)
        preprocess.return_value = {
            "lcc_image": np.ones((2, 2, 2), dtype=np.float32),
            "downscale_factors": (2, 2, 1),
            "scaled_spacing": (2.0, 2.0, 1.0),
        }
        locate.return_value = SimpleNamespace(
            original_circles=[{"z": 0}],
            filtered_circles=[{"z": 0}],
        )
        segment.return_value = SimpleNamespace(mask=np.ones((2, 2, 2), dtype=np.uint8))
        record = pd.Series(
            {
                "spacing_x_mm": 1.0,
                "spacing_y_mm": 1.0,
                "spacing_z_mm": 1.0,
            }
        )
        config = {"CIRCLE_DETECTION": {}, "LEVEL_SET": {}}

        prediction = predict_aorta(record, config)

        self.assertEqual(prediction.native_shape, (4, 4, 2))
        self.assertEqual(prediction.circle_count, 1)
        self.assertEqual(int(prediction.mask.sum()), 8)
        preprocess.assert_called_once()
        locate.assert_called_once()
        segment.assert_called_once()
        self.assertFalse(segment.call_args.kwargs["use_gpu"])

    def test_restores_native_shape_with_nearest_neighbor_and_flip(self):
        mask = np.zeros((2, 2, 2), dtype=np.uint8)
        mask[0, 0, :] = 1
        prediction = AortaPrediction(mask, (4, 4, 2), (0,), 1)

        restored = restore_native_mask(prediction)

        self.assertEqual(restored.shape, (4, 4, 2))
        self.assertSetEqual(set(np.unique(restored)), {0, 1})
        self.assertTrue(np.all(restored[2:, :2, :]))
        self.assertFalse(np.any(restored[:2, :, :]))

    def test_saves_label_820_in_original_nifti_geometry(self):
        reference_path = self.root / "ct_test_2001_image.nii.gz"
        output_path = self.root / "export" / "ct_test_2001_label.nii.gz"
        affine = np.diag([0.5, 0.5, 1.2, 1.0])
        reference = Nifti1Image(np.zeros((4, 4, 2), dtype=np.int16), affine)
        reference.set_qform(affine, code=1)
        reference.set_sform(affine, code=2)
        save_nifti(reference, str(reference_path))
        mask = np.zeros((4, 4, 2), dtype=np.uint8)
        mask[1:3, 1:3, :] = 1

        save_prediction(reference_path, mask, output_path)
        validate_saved_prediction(reference_path, output_path)
        saved = load_nifti(str(output_path))
        if not isinstance(saved, Nifti1Image):
            self.fail("O arquivo exportado deveria ser NIfTI-1.")

        self.assertEqual(saved.get_data_dtype(), np.dtype(np.int16))
        if saved.affine is None:
            self.fail("O NIfTI exportado deveria preservar o affine.")
        self.assertTrue(np.allclose(saved.affine, affine))
        self.assertSetEqual(set(np.unique(np.asarray(saved.dataobj))), {0, 820})
        self.assertEqual(saved.get_qform(coded=True)[1], 1)
        self.assertEqual(saved.get_sform(coded=True)[1], 2)
        with self.assertRaises(FileExistsError):
            save_prediction(reference_path, mask, output_path)

    def test_rejects_mask_that_does_not_reproduce_run(self):
        numeric = self.root / "numeric"
        numeric.mkdir()
        pd.DataFrame(
            [
                {
                    "subset": "test",
                    "exam_id": "ct_test_2001",
                    "aorta_mask_voxels": 3,
                    "aorta_circle_count": 1,
                }
            ]
        ).to_csv(numeric / "results_all.csv", index=False)
        prediction = AortaPrediction(
            np.ones((2, 2, 1), dtype=np.uint8), (2, 2, 1), (), 1
        )
        with self.assertRaisesRegex(ValueError, "não reproduz"):
            verify_run_result(self.root, "ct_test_2001", prediction)

    def test_parses_official_aorta_column_and_rejects_wrong_identity(self):
        path = self.root / "ct_test_2001_dice.xls"
        path.write_text(
            "0\t0\t0\t0\t0\t0.632263\t0\t0.076495\t\tct2001\t--decodeseg2\t\n",
            encoding="ascii",
        )
        self.assertAlmostEqual(parse_dice_lo(path, "ct_test_2001"), 0.632263)
        with self.assertRaisesRegex(ValueError, "Identidade"):
            parse_dice_lo(path, "ct_test_2002")
        path.write_text(
            "0\t0\t0\t0\t0\tnan\t0\t0.076495\t\tct2001\t--decodeseg2\t\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(ValueError, "fora de"):
            parse_dice_lo(path, "ct_test_2001")

    @patch(
        "utils.project.evaluation.mmwhs_official_aorta._wine_path",
        side_effect=lambda path: str(path),
    )
    @patch(
        "utils.project.evaluation.mmwhs_official_aorta.shutil.which",
        return_value="/usr/bin/wine",
    )
    @patch("utils.project.evaluation.mmwhs_official_aorta.subprocess.run")
    def test_wine_command_uses_encrypted_label_and_aorta_result(
        self, run_command, _which, _wine_path
    ):
        evaluator_dir = self.root / "evaluator"
        (evaluator_dir / "nii").mkdir(parents=True)
        (evaluator_dir / "zxhtransform.exe").touch()
        (evaluator_dir / "zxhCardWhsEvaluate.exe").touch()
        (evaluator_dir / "nii/ct_test_2001_label_encrypt_1mm.nii.gz").touch()
        prediction = self.root / "ct_test_2001_label.nii.gz"
        prediction.touch()

        def make_outputs(command, **_kwargs):
            if "zxhtransform.exe" in command[1]:
                (self.root / "ct_test_2001_label_1mm.nii.gz").touch()
            else:
                return SimpleNamespace(stdout="success: open image\n0.804026\t\n")

        run_command.side_effect = make_outputs
        result = evaluate_with_wine(
            prediction, "ct_test_2001", evaluator_dir, self.root
        )

        self.assertEqual(result.name, "ct_test_2001_aorta_dice.json")
        self.assertEqual(parse_aorta_dice(result, "ct_test_2001"), 0.804026)
        self.assertEqual(run_command.call_count, 2)
        transform_args = run_command.call_args_list[0].args[0]
        evaluate_args = run_command.call_args_list[1].args[0]
        self.assertIn("-nearest", transform_args)
        self.assertIn("--decodeseg2", evaluate_args)
        self.assertEqual(evaluate_args[2:5], ["-label", "820", "820"])
        self.assertNotIn("ALL", evaluate_args)
        self.assertIn(
            str(evaluator_dir / "nii/ct_test_2001_label_encrypt_1mm.nii.gz"),
            evaluate_args,
        )

    def test_scalar_output_rejects_invalid_or_multiple_scores(self):
        self.assertEqual(
            parse_label_dice_output("success: open image\n1.000000\t\n"), 1
        )
        for output in (
            "",
            "garbage",
            "nan",
            "inf",
            "-1.#IND00",
            "-0.1",
            "1.1",
            "0.2\n0.3",
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                parse_label_dice_output(output)

    def test_json_validates_identity_protocol_label_and_score(self):
        path = aorta_dice_result_path(self.root, "ct_test_2001")
        valid = {
            "exam_id": "ct_test_2001",
            "method": OFFICIAL_AORTA_METHOD,
            "label": 820,
            "dice": 0.804026,
        }
        path.write_text(json.dumps(valid))
        self.assertEqual(parse_aorta_dice(path, "ct_test_2001"), 0.804026)
        for key, value in (
            ("exam_id", "ct_test_2002"),
            ("method", "mmwhs_official_1mm_wine"),
            ("label", 850),
            ("dice", True),
            ("dice", None),
            ("dice", float("nan")),
            ("dice", 2),
        ):
            path.write_text(json.dumps({**valid, key: value}))
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                parse_aorta_dice(path, "ct_test_2001")

    @patch("utils.project.evaluation.mmwhs_official_aorta.preflight_evaluator")
    @patch("utils.project.evaluation.mmwhs_official_aorta._wine_path", side_effect=str)
    @patch("utils.project.evaluation.mmwhs_official_aorta.subprocess.run")
    def test_legacy_is_not_cached_new_result_is_atomic_and_reused(
        self, run_command, _path, _preflight
    ):
        exam_id = "ct_test_2001"
        legacy = self.root / f"{exam_id}_dice.xls"
        legacy.write_text("legacy result")
        (self.root / f"{exam_id}_label_1mm.nii.gz").touch()
        run_command.return_value = SimpleNamespace(stdout="invalid score")
        target = aorta_dice_result_path(self.root, exam_id)
        with self.assertRaises(ValueError):
            evaluate_with_wine(
                self.root / "native.nii.gz", exam_id, self.root, self.root
            )
        self.assertFalse(target.exists())
        self.assertEqual(legacy.read_text(), "legacy result")
        run_command.return_value = SimpleNamespace(stdout="0.804026\n")
        evaluate_with_wine(self.root / "native.nii.gz", exam_id, self.root, self.root)
        self.assertEqual(parse_aorta_dice(target, exam_id), 0.804026)
        self.assertEqual(list(self.root.glob("*.tmp")), [])
        run_command.reset_mock()
        evaluate_with_wine(self.root / "native.nii.gz", exam_id, self.root, self.root)
        run_command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
