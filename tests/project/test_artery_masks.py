import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from utils.project.results.artery_masks import ArteryMaskExport


class ArteryMasksTest(unittest.TestCase):
    def test_nifti_keeps_final_mask_in_the_pipeline_grid(self):
        original_shape = (9, 7, 5)
        original_affine = np.array(
            [[0, -1.2, 0, 12], [0.7, 0, 0, -8], [0, 0, 2.5, 4], [0, 0, 0, 1]]
        )
        cases = (
            ("mid", "opencv", (4, 3, 5), (9 / 4, 7 / 3, 1), (5 / 8, 2 / 3, 0)),
            ("mid", "scipy", (4, 4, 5), (8 / 3, 2, 1), (0, 0, 0)),
            ("high", "opencv", original_shape, (1, 1, 1), (0, 0, 0)),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "17.img.nii.gz"
            nib.save(
                nib.Nifti1Image(
                    np.zeros(original_shape, dtype=np.int16), original_affine
                ),
                source,
            )
            for resolution, backend, shape, scale, offset in cases:
                with self.subTest(resolution=resolution, backend=backend):
                    export = ArteryMaskExport(
                        root / f"{resolution}_{backend}",
                        "ImageCAS",
                        "train",
                        resolution,
                    )
                    mask = np.zeros(shape, dtype=bool)
                    mask[1:3, 1, 2] = True
                    before = mask.copy()
                    result = export.save(
                        "17", mask, source, {"DOWNSCALE_METHOD": backend}
                    )
                    self.assertEqual(result["artery_mask_status"], "saved")
                    path = export.run_dir / result["artery_mask_path"]
                    prediction = nib.load(path)
                    np.testing.assert_array_equal(mask, before)
                    np.testing.assert_array_equal(prediction.dataobj, before)
                    self.assertEqual(prediction.shape, shape)
                    self.assertEqual(prediction.get_data_dtype(), np.dtype(np.uint8))
                    test_point = np.array([2, 1, 3])
                    expected = nib.affines.apply_affine(
                        original_affine, test_point * scale + offset
                    )
                    np.testing.assert_allclose(
                        nib.affines.apply_affine(prediction.affine, test_point),
                        expected,
                        atol=1e-5,
                    )
                    manifest_path = export.run_dir / result["artery_mask_metadata_path"]
                    manifest = json.loads(manifest_path.read_text())
                    self.assertEqual(manifest["resolution"], resolution)
                    self.assertEqual(manifest["foreground_voxels"], 2)
                    self.assertEqual(
                        manifest["sha256"],
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                    self.assertEqual(set(path.parent.iterdir()), {path, manifest_path})

    def test_orcascore_inversion_is_recorded_in_affine_without_flipping_mask(self):
        original_shape = (9, 7, 3)
        mask = np.zeros((4, 3, 3), dtype=np.uint8)
        mask[1:3, 0, 1] = 1
        reference = sitk.GetImageFromArray(
            np.zeros(original_shape[::-1], dtype=np.int16)
        )
        reference.SetOrigin((10, -20, 30))
        reference.SetSpacing((0.7, 1.2, 2.5))
        reference.SetDirection((0, -1, 0, 1, 0, 0, 0, 0, 1))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "TRV1P1CTAI.mhd"
            sitk.WriteImage(reference, str(source), True)
            export = ArteryMaskExport(root, "OrCaScore", "test", "mid")
            result = export.save(
                "TRV1P1",
                mask,
                source,
                {"DOWNSCALE_METHOD": "opencv"},
                flipped_axes=(1,),
            )
            self.assertEqual(result["artery_mask_status"], "saved")
            prediction = nib.load(root / result["artery_mask_path"])
            np.testing.assert_array_equal(prediction.dataobj, mask)
            # O centro do voxel (2, 0, 1) na imagem reduzida é invertido no eixo 1.
            point_lps = reference.TransformContinuousIndexToPhysicalPoint(
                ((2.5 * 9 / 4) - 0.5, 6 - ((0.5 * 7 / 3) - 0.5), 1.0)
            )
            np.testing.assert_allclose(
                nib.affines.apply_affine(prediction.affine, (2, 0, 1)),
                np.array(point_lps) * [-1, -1, 1],
                atol=1e-5,
            )

    def test_empty_prediction_is_saved_and_write_failure_is_reported(self):
        mask = np.zeros((3, 3, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "image.nii.gz"
            nib.save(nib.Nifti1Image(mask, np.eye(4)), source)
            export = ArteryMaskExport(root, "MM-WHS", "train", "high")
            config = {"DOWNSCALE_METHOD": "opencv"}
            result = export.save("ct_train_1001", mask, source, config)
            self.assertEqual(result["artery_mask_status"], "saved")
            np.testing.assert_array_equal(
                nib.load(root / result["artery_mask_path"]).dataobj, mask
            )

            with (
                patch(
                    "utils.project.results.artery_masks.nib.save",
                    side_effect=OSError("Disco cheio"),
                ),
                self.assertLogs("utils.project.results.artery_masks", level="ERROR"),
            ):
                failed = export.save("ct_train_1002", mask, source, config)
            self.assertEqual(failed["artery_mask_status"], "error")
            self.assertIn("Disco cheio", failed["artery_mask_error"])
            self.assertIsNone(failed["artery_mask_path"])
            self.assertFalse(export.has_saved_prediction("ct_train_1002"))
