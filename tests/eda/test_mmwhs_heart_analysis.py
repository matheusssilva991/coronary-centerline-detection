import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from nibabel.funcs import as_closest_canonical
from nibabel.loadsave import save as save_nifti
from nibabel.nifti1 import Nifti1Image

from utils.project.analysis.mmwhs_heart_analysis import (
    aggregate_mmwhs_heart_hu,
    load_mmwhs_heart_volume,
    select_mmwhs_heart_exams,
    summarize_mmwhs_heart_hu,
)


class MmwhsHeartIntensityTest(unittest.TestCase):
    def setUp(self):
        self.image = np.full((5, 5, 5), 7.0, dtype=np.float32)
        self.label = np.zeros(self.image.shape, dtype=np.int16)
        self.label[1, 1, 1] = 500
        self.label[3, 1, 1] = 600
        self.image[1, 1, 1] = 10
        self.image[3, 1, 1] = 30
        self.image[0, 0, 0] = -1000

    def _analyze(self, image=None, label=None, **kwargs):
        return summarize_mmwhs_heart_hu(
            self.image if image is None else image,
            self.label if label is None else label,
            (1, 1, 1),
            exam_id=kwargs.pop("exam_id", "exam"),
            margin_mm=kwargs.pop("margin_mm", 0),
            **kwargs,
        )

    def test_regions_union_background_and_physical_margin(self):
        statistics, _ = self._analyze()
        rows = statistics.set_index("region")
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows.loc["label_500", "mean_hu"], 10)
        self.assertEqual(rows.loc["label_600", "mean_hu"], 30)
        self.assertEqual(rows.loc["heart", "mean_hu"], 20)
        self.assertEqual(rows.loc["heart", "voxel_count"], 2)
        self.assertAlmostEqual(rows.loc["heart", "volume_ml"], 0.002)
        self.assertEqual(rows.loc["background_full", "voxel_count"], 123)
        self.assertEqual(rows.loc["background_local", "voxel_count"], 1)
        self.assertEqual(rows.loc["background_local", "mean_hu"], 7)
        expanded, _ = self._analyze(margin_mm=1)
        self.assertEqual(
            expanded.set_index("region").loc["background_local", "voxel_count"], 43
        )

    def test_union_uses_voxels_instead_of_mean_of_label_means(self):
        self.label[2, 1, 1] = 500
        self.image[2, 1, 1] = 10
        statistics, _ = self._analyze()
        self.assertAlmostEqual(
            statistics.set_index("region").loc["heart", "mean_hu"], 50 / 3
        )

    def test_exact_statistics_nonfinite_values_and_outside_bins(self):
        values = np.array([-2000, 0, 4000, np.nan], dtype=np.float32).reshape(4, 1, 1)
        label = np.full(values.shape, 500, dtype=np.int16)
        before = values.copy()
        statistics, histograms = self._analyze(values, label)
        row = statistics.set_index("region").loc["label_500"]
        self.assertEqual(row["voxel_count"], 4)
        self.assertEqual(row["valid_voxel_count"], 3)
        self.assertEqual(row["nonfinite_voxel_count"], 1)
        self.assertEqual(row["below_range_count"], 1)
        self.assertEqual(row["above_range_count"], 1)
        self.assertEqual(row["min_hu"], -2000)
        self.assertEqual(row["max_hu"], 4000)
        self.assertEqual(row["median_hu"], 0)
        self.assertEqual(row["q1_hu"], -1000)
        self.assertEqual(row["q3_hu"], 2000)
        self.assertEqual(row["p5_hu"], -1800)
        self.assertAlmostEqual(row["p95_hu"], 3600)
        self.assertAlmostEqual(row["mean_hu"], 2000 / 3)
        self.assertAlmostEqual(row["std_hu"], np.std([-2000, 0, 4000]))
        density = histograms.loc[histograms["region"].eq("label_500"), "density"]
        self.assertAlmostEqual(float(density.sum()) * 10, 1 / 3)
        np.testing.assert_array_equal(values, before)

    def test_absent_labels_do_not_contribute_as_zero_hu(self):
        statistics, histograms = self._analyze()
        summary, densities = aggregate_mmwhs_heart_hu(statistics, histograms)
        missing = summary.set_index("region").loc["label_820"]
        self.assertEqual(missing["exam_count"], 0)
        self.assertTrue(pd.isna(missing["mean_of_means_hu"]))
        self.assertTrue(
            densities.loc[densities["region"].eq("label_820"), "density"].isna().all()
        )
        self.assertTrue(
            pd.isna(
                summary.set_index("region").loc["heart", "std_between_exam_means_hu"]
            )
        )

    def test_present_region_without_finite_hu_does_not_contribute(self):
        self.image[1, 1, 1] = np.nan
        statistics, histograms = self._analyze()
        row = statistics.set_index("region").loc["label_500"]
        self.assertEqual(row["voxel_count"], 1)
        self.assertEqual(row["valid_voxel_count"], 0)
        self.assertEqual(row["nonfinite_voxel_count"], 1)
        summary, _ = aggregate_mmwhs_heart_hu(statistics, histograms)
        self.assertEqual(summary.set_index("region").loc["label_500", "exam_count"], 0)

    def test_equal_exam_weight_despite_different_region_sizes(self):
        first_label = np.full((3, 3, 3), 500, dtype=np.int16)
        second_label = np.zeros(first_label.shape, dtype=np.int16)
        second_label[1, 1, 1] = 500
        edges = np.array([-1, 1, 101], dtype=float)
        first_stats, first_hist = self._analyze(
            np.zeros(first_label.shape), first_label, exam_id="a", bin_edges=edges
        )
        second_stats, second_hist = self._analyze(
            np.full(first_label.shape, 100.0),
            second_label,
            exam_id="b",
            bin_edges=edges,
        )
        summary, densities = aggregate_mmwhs_heart_hu(
            pd.concat([first_stats, second_stats]), pd.concat([first_hist, second_hist])
        )
        row = summary.set_index("region").loc["label_500"]
        self.assertEqual(row["exam_count"], 2)
        self.assertEqual(row["mean_of_means_hu"], 50)
        self.assertEqual(row["median_of_medians_hu"], 50)
        self.assertAlmostEqual(
            row["std_between_exam_means_hu"], np.std([0, 100], ddof=1)
        )
        np.testing.assert_allclose(
            densities.loc[densities["region"].eq("label_500"), "density"], [0.25, 0.005]
        )

    def test_rejects_empty_heart_unknown_labels_bad_bins_and_spacing(self):
        with self.assertRaisesRegex(ValueError, "vazia"):
            self._analyze(label=np.zeros(self.label.shape, dtype=np.int16))
        bad_label = self.label.copy()
        bad_label[0, 0, 0] = 123
        with self.assertRaisesRegex(ValueError, "desconhecidos"):
            self._analyze(label=bad_label)
        with self.assertRaisesRegex(ValueError, "bins"):
            self._analyze(bin_edges=np.array([1, 0]))
        with self.assertRaisesRegex(ValueError, "positivos"):
            summarize_mmwhs_heart_hu(self.image, self.label, (0, 1, 1), exam_id="bad")

    def test_loading_applies_hu_offset_and_canonical_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
            affine = np.diag([-0.5, 0.75, 1.25, 1])
            image = Nifti1Image(values, affine)
            image.header.set_slope_inter(1, -1024)
            label_values = np.full(values.shape, 500, dtype=np.int16)
            label_values[0, 0, 0] = 600
            label = Nifti1Image(label_values, affine)
            save_nifti(image, root / "image.nii.gz")
            # Imagem LAS e label RAS representam a mesma grade física.
            save_nifti(as_closest_canonical(label), root / "label.nii.gz")
            loaded, labels, spacing = load_mmwhs_heart_volume(
                root / "image.nii.gz", root / "label.nii.gz"
            )
            np.testing.assert_array_equal(loaded, np.flip(values - 1024, axis=0))
            self.assertEqual(loaded.dtype, np.float32)
            np.testing.assert_array_equal(labels, np.flip(label_values, axis=0))
            self.assertEqual(spacing, (0.5, 0.75, 1.25))

    def test_loading_rejects_different_affines_and_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_nifti(Nifti1Image(self.image, np.eye(4)), root / "image.nii.gz")
            shifted = np.eye(4)
            shifted[0, 3] = 5
            for data, affine in ((self.label, shifted), (self.label[:2], np.eye(4))):
                save_nifti(Nifti1Image(data, affine), root / "label.nii.gz")
                with self.assertRaisesRegex(ValueError, "geometria"):
                    load_mmwhs_heart_volume(
                        root / "image.nii.gz", root / "label.nii.gz"
                    )

    def test_selection_is_reproducible_and_explicit_ids_override_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "label.nii.gz"
            path.touch()
            inventory = pd.DataFrame(
                [
                    {
                        "dataset": "MM-WHS",
                        "subset": "train",
                        "exam_id": str(i),
                        "label_path": path,
                    }
                    for i in range(20)
                ]
                + [
                    {
                        "dataset": "MM-WHS",
                        "subset": "test",
                        "exam_id": "test",
                        "label_path": path,
                    }
                ]
            )
            a = select_mmwhs_heart_exams(inventory)
            b = select_mmwhs_heart_exams(inventory)
            pd.testing.assert_frame_equal(a, b)
            self.assertEqual(len(a), 10)
            self.assertEqual(len(select_mmwhs_heart_exams(inventory, n_exams=None)), 20)
            selected = select_mmwhs_heart_exams(
                inventory, n_exams=1, exam_ids=["3", "1"]
            )
            self.assertEqual(selected["exam_id"].tolist(), ["3", "1"])
            with self.assertRaisesRegex(ValueError, "sem par"):
                select_mmwhs_heart_exams(inventory, exam_ids=["test"])
            with self.assertRaisesRegex(ValueError, "distintos"):
                select_mmwhs_heart_exams(inventory, exam_ids=["1", "1"])


if __name__ == "__main__":
    unittest.main()
