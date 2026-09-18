import tempfile
import unittest
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np

from utils.project.ccta_datasets import (
    discover_ccta_volumes,
    load_ccta_volume,
    load_mhd_volume,
    load_nifti_volume_xyz,
    select_representative_exams,
)


class CctaDatasetsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.orca = self.root / "orca"
        self.mmwhs = self.root / "mmwhs"
        self.imagecas = self.root / "imagecas"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_mhd(self, name: str, *, ccta: bool = True) -> tuple[Path, np.ndarray]:
        image_dir = self.orca / "Training_set" / "Images"
        image_dir.mkdir(parents=True, exist_ok=True)
        suffix = "CTAI" if ccta else "CTI"
        path = image_dir / f"{name}{suffix}.mhd"
        raw_path = path.with_suffix(".zraw")
        volume = np.arange(24, dtype="<i2").reshape(2, 3, 4)
        raw_path.write_bytes(zlib.compress(volume.tobytes()))
        path.write_text(
            "\n".join(
                [
                    "ObjectType = Image",
                    "NDims = 3",
                    "BinaryDataByteOrderMSB = False",
                    "CompressedData = True",
                    "ElementSpacing = 0.5 0.6 1.5",
                    "DimSize = 4 3 2",
                    "ElementType = MET_SHORT",
                    f"ElementDataFile = {raw_path.name}",
                ]
            ),
            encoding="utf-8",
        )
        return path, volume

    def _write_nifti(self, folder: str, name: str) -> Path:
        target = self.mmwhs / folder
        target.mkdir(parents=True, exist_ok=True)
        path = target / name
        image = nib.Nifti1Image(
            np.arange(60, dtype=np.int16).reshape(3, 4, 5),
            np.diag([0.7, 0.8, 1.2, 1]),
        )
        nib.save(image, path)
        return path

    def _write_imagecas(self, image_id: int) -> Path:
        self.imagecas.mkdir(parents=True, exist_ok=True)
        path = self.imagecas / f"{image_id}.img.nii.gz"
        image = nib.Nifti1Image(
            np.arange(60, dtype=np.int16).reshape(3, 4, 5),
            np.diag([-0.7, 0.8, 1.2, 1]),
        )
        nib.save(image, path)
        return path

    def test_loads_compressed_mhd_in_zyx_order(self):
        path, expected = self._write_mhd("CASE")

        loaded = load_mhd_volume(path)

        np.testing.assert_array_equal(loaded, expected)

    def test_inventory_keeps_only_ccta_images(self):
        self._write_mhd("CCTA")
        self._write_mhd("PLAIN", ccta=False)
        self._write_nifti("ct_train", "ct_train_1001_image.nii.gz")
        self._write_nifti("ct_train", "ct_train_1001_label.nii.gz")
        self._write_nifti("mr_train", "mr_train_1001_image.nii.gz")

        inventory = discover_ccta_volumes(self.orca, self.mmwhs)

        self.assertEqual(len(inventory), 2)
        self.assertSetEqual(set(inventory["dataset"]), {"OrCaScore", "MM-WHS"})
        self.assertFalse(
            inventory["path"].astype(str).str.contains("label|mr_train").any()
        )

    def test_loads_inventory_record_and_selects_representatives(self):
        _, mhd_zyx = self._write_mhd("CASE")
        self._write_nifti("ct_test", "ct_test_2001_image.nii.gz")
        inventory = discover_ccta_volumes(self.orca, self.mmwhs)

        volume = load_ccta_volume(inventory.loc[inventory.dataset.eq("MM-WHS")].iloc[0])
        mhd_volume = load_ccta_volume(
            inventory.loc[inventory.dataset.eq("OrCaScore")].iloc[0]
        )
        selected = select_representative_exams(inventory)

        self.assertEqual(volume.shape, (3, 4, 5))
        np.testing.assert_array_equal(mhd_volume, np.transpose(mhd_zyx, (2, 1, 0)))
        self.assertEqual(len(selected), 2)

    def test_native_nifti_loader_matches_preprocessing_layout(self):
        path = self._write_imagecas(90)
        raw = np.asanyarray(nib.load(path).dataobj)

        loaded = load_nifti_volume_xyz(path)

        np.testing.assert_array_equal(loaded, raw)
        np.testing.assert_array_equal(loaded[:, :, 2], raw[:, :, 2])

    def test_inventory_optionally_includes_imagecas(self):
        self._write_mhd("CASE")
        self._write_nifti("ct_test", "ct_test_2001_image.nii.gz")
        self._write_imagecas(1)

        inventory = discover_ccta_volumes(self.orca, self.mmwhs, self.imagecas)

        self.assertEqual(len(inventory), 3)
        self.assertSetEqual(
            set(inventory["dataset"]), {"OrCaScore", "MM-WHS", "ImageCAS"}
        )
        imagecas = inventory.loc[inventory["dataset"].eq("ImageCAS")].iloc[0]
        self.assertEqual(imagecas["subset"], "full")
        self.assertEqual(imagecas["reported_orientation"], "LAS")


if __name__ == "__main__":
    unittest.main()
