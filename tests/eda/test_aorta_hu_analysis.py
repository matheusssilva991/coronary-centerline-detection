import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nibabel.loadsave import save as save_nifti
from nibabel.nifti1 import Nifti1Image

from utils.project.analysis.aorta_hu_analysis import (
    HU_REGION_NAMES,
    HuReferenceVolume,
    derive_hu_intervals,
    hu_interval_mask,
    load_imagecas_aorta_hu,
    load_mmwhs_hu_reference,
    select_hu_example_records,
    select_reviewed_imagecas_exams,
    summarize_hu_reference,
    summarize_hu_retention,
    verify_imagecas_aorta_prediction,
)
from utils.project.analysis.mmwhs_heart_analysis import aggregate_mmwhs_heart_hu
from utils.project.dataframe import numeric_series
from utils.segmentation.pipeline.detection import (
    AortaCircleTrackingResult,
    AortaSegmentationResult,
)
from utils.visualization.images.hu_threshold import (
    hu_axial_mip,
    reference_axial_slices,
)


from tests.notebook_helpers import load_presentation_helpers

_presentation = load_presentation_helpers("aorta_hu_threshold_comparison")
plot_hu_threshold_examples = _presentation["plot_hu_threshold_examples"]


class AortaHuAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.mask = np.ones((2, 2, 3), dtype=bool)
        self.tracking = AortaCircleTrackingResult(
            [{"slice_index": 0}, {"slice_index": 2}], [{"slice_index": 0}], {}
        )
        self.result = pd.Series(
            {
                "IMG_ID": 13,
                "aorta_mask_voxel_count": 12,
                "aorta_circle_count": 2,
                "aorta_circle_used_count": 1,
                "aorta_circle_first_slice": 0,
                "aorta_circle_last_slice": 2,
                "aorta_segmented_slice_count": 3,
                "image_slice_count": 3,
            }
        )

    def _statistics(self):
        rows = []
        for region in HU_REGION_NAMES:
            for exam_id, count, lower, upper in (
                ("a", 1, 10, 100),
                ("b", 100, 30, 300),
            ):
                rows.append(
                    {
                        "region": region,
                        "exam_id": exam_id,
                        "valid_voxel_count": count,
                        "lower_percentile_hu": lower,
                        "upper_percentile_hu": upper,
                    }
                )
        return pd.DataFrame(rows)

    def test_intervals_weight_exams_equally_and_support_absolute_overrides(self):
        stats = self._statistics()
        intervals = derive_hu_intervals(stats, overrides={"mmwhs_heart": (-100, 400)})
        self.assertEqual(intervals["mmwhs_heart"], (-100, 400))
        self.assertEqual(intervals["mmwhs_aorta"], (20, 200))
        self.assertEqual(intervals["imagecas_aorta"], (20, 200))

    def test_intervals_reject_duplicates_absence_and_invalid_limits(self):
        stats = self._statistics()
        with self.assertRaisesRegex(ValueError, "duplicadas"):
            derive_hu_intervals(pd.concat([stats, stats.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, "Nenhum exame"):
            derive_hu_intervals(stats.loc[stats["region"].ne("imagecas_aorta")])
        with self.assertRaisesRegex(ValueError, "desconhecidas"):
            derive_hu_intervals(stats, overrides={"unknown": (0, 1)})
        for interval in ((5, 1), (np.nan, 10), (-10, np.inf)):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                derive_hu_intervals(stats, overrides={"imagecas_aorta": interval})
        stats.loc[0, "lower_percentile_hu"] = np.nan
        with self.assertRaisesRegex(ValueError, "Percentis inválidos"):
            derive_hu_intervals(stats)

    def test_region_statistics_keep_hu_tails_nonfinite_and_custom_percentiles(self):
        image = np.array([-2000, 0, 100, 4000, np.nan], dtype=np.float32).reshape(
            5, 1, 1
        )
        before = image.copy()
        volume = HuReferenceVolume(
            image,
            {"imagecas_aorta": np.ones(image.shape, dtype=bool)},
            (1, 2, 3),
            "ImageCAS",
            "13",
        )
        stats, hist = summarize_hu_reference(volume, percentiles=(10, 90))
        row = stats.iloc[0]
        self.assertEqual(row["voxel_count"], 5)
        self.assertEqual(row["valid_voxel_count"], 4)
        self.assertEqual(row["nonfinite_voxel_count"], 1)
        self.assertEqual(row["volume_ml"], 0.03)
        self.assertEqual(row["outside_range_percent"], 50)
        self.assertEqual(row["lower_percentile_hu"], -1400)
        self.assertAlmostEqual(row["upper_percentile_hu"], 2830)
        self.assertAlmostEqual(float(numeric_series(hist, "density").sum()) * 10, 0.5)
        np.testing.assert_array_equal(image, before)
        for percentiles in ((95, 5), (-1, 90), (5, 101), (np.nan, 95)):
            with self.subTest(percentiles=percentiles), self.assertRaises(ValueError):
                summarize_hu_reference(volume, percentiles=percentiles)

    def test_empty_region_is_not_zero_and_does_not_contribute(self):
        volume = HuReferenceVolume(
            np.zeros((2, 2, 2), dtype=np.float32),
            {"imagecas_aorta": np.zeros((2, 2, 2), dtype=bool)},
            (1, 1, 1),
            "ImageCAS",
            "13",
        )
        stats, hist = summarize_hu_reference(volume)
        self.assertTrue(pd.isna(stats.iloc[0]["mean_hu"]))
        summary, _ = aggregate_mmwhs_heart_hu(stats, hist)
        self.assertEqual(summary.iloc[0]["exam_count"], 0)
        retention = summarize_hu_retention(volume, {"mmwhs_aorta": (-1, 1)})
        self.assertTrue(pd.isna(retention.iloc[0]["retained_region_percent"]))

    def test_threshold_is_inclusive_and_excludes_nonfinite_hu(self):
        values = np.array([-1, 0, 1, 2, np.nan, np.inf], dtype=float)
        np.testing.assert_array_equal(
            hu_interval_mask(values, (0, 1)), [False, True, True, False, False, False]
        )
        with self.assertRaises(ValueError):
            hu_interval_mask(values, (2, 1))

    def test_retention_uses_finite_denominators_and_no_imagecas_heart_reference(self):
        image = np.array([0, 10, 20, np.nan], dtype=np.float32).reshape(4, 1, 1)
        volume = HuReferenceVolume(
            image,
            {"imagecas_aorta": np.ones(image.shape, dtype=bool)},
            (1, 1, 1),
            "ImageCAS",
            "13",
        )
        retention = summarize_hu_retention(
            volume, {key: (0, 10) for key in HU_REGION_NAMES}
        )
        self.assertEqual(len(retention), 3)
        self.assertEqual(set(retention["region"]), {"imagecas_aorta"})
        self.assertEqual(retention.iloc[0]["finite_volume_voxels"], 3)
        self.assertAlmostEqual(retention.iloc[0]["retained_region_percent"], 200 / 3)
        self.assertAlmostEqual(retention.iloc[0]["selected_volume_percent"], 200 / 3)

    def test_reconstruction_checks_voxels_circles_bounds_and_slices(self):
        verify_imagecas_aorta_prediction(self.mask, self.tracking, self.result)
        for column in self.result.index:
            if column == "IMG_ID":
                continue
            changed = self.result.copy()
            changed[column] += 1
            with (
                self.subTest(column=column),
                self.assertRaisesRegex(ValueError, "divergente"),
            ):
                verify_imagecas_aorta_prediction(self.mask, self.tracking, changed)
        with self.assertRaisesRegex(ValueError, "ausente"):
            verify_imagecas_aorta_prediction(
                self.mask, self.tracking, self.result.drop("aorta_circle_count")
            )

    def test_imagecas_restoration_scales_hu_and_reorders_mask_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            affine = np.diag([-0.5, 0.75, 1.25, 1])
            values = np.arange(48, dtype=np.int16).reshape(4, 4, 3)
            nii = Nifti1Image(values, affine)
            nii.header.set_slope_inter(1, -1024)
            save_nifti(nii, root / "13.img.nii.gz")
            processed_mask = np.zeros(self.mask.shape, dtype=bool)
            processed_mask[0, 0, :] = True
            result = self.result.copy()
            result["aorta_mask_voxel_count"] = 3
            with (
                patch(
                    "utils.project.analysis.aorta_hu_analysis.load_config_json",
                    return_value={"CIRCLE_DETECTION": {}, "LEVEL_SET": {}},
                ),
                patch(
                    "utils.project.analysis.aorta_hu_analysis.preprocess_ccta_volume",
                    return_value={
                        "lcc_image": self.mask,
                        "downscale_factors": (2, 2, 1),
                        "scaled_spacing": (1, 1.5, 1.25),
                    },
                ) as preprocess,
                patch(
                    "utils.project.analysis.aorta_hu_analysis.locate_and_filter_aorta_circles",
                    return_value=self.tracking,
                ),
                patch(
                    "utils.project.analysis.aorta_hu_analysis.segment_aorta_with_diagnostics",
                    return_value=AortaSegmentationResult(processed_mask, {}),
                ) as segment,
            ):
                volume = load_imagecas_aorta_hu(root, root, result)
            np.testing.assert_array_equal(volume.image, np.flip(values - 1024, axis=0))
            self.assertEqual(volume.regions["imagecas_aorta"].shape, values.shape)
            expected_mask = np.zeros(values.shape, dtype=bool)
            expected_mask[2:, :2, :] = True
            np.testing.assert_array_equal(
                volume.regions["imagecas_aorta"], expected_mask
            )
            self.assertEqual(volume.spacing, (0.5, 0.75, 1.25))
            self.assertFalse(segment.call_args.kwargs["use_gpu"])
            np.testing.assert_array_equal(preprocess.call_args.args[0], values - 1024)

    def test_mmwhs_heart_union_and_aorta_share_native_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            labels = np.zeros(values.shape, dtype=np.int16)
            labels[0, 0, 0] = 820
            labels[1, 1, 1] = 500
            save_nifti(Nifti1Image(values, np.eye(4)), root / "ct.nii.gz")
            save_nifti(Nifti1Image(labels, np.eye(4)), root / "label.nii.gz")
            record = pd.Series(
                {
                    "path": root / "ct.nii.gz",
                    "label_path": root / "label.nii.gz",
                    "exam_id": "ct_train_1001",
                }
            )
            volume = load_mmwhs_hu_reference(record)
            self.assertEqual(np.count_nonzero(volume.regions["mmwhs_heart"]), 2)
            self.assertEqual(np.count_nonzero(volume.regions["mmwhs_aorta"]), 1)
            np.testing.assert_array_equal(volume.image, values)

    def test_selection_keeps_reviewed_run_and_good_ids_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run/config").mkdir(parents=True)
            (root / "run/config/effective_pipeline_config.json").touch()
            review = {"run_dir": "run", "aorta_good_ids": {13, 28, 44}}
            with (
                patch(
                    "utils.project.analysis.aorta_hu_analysis.load_aorta_visual_reviews",
                    return_value={},
                ),
                patch(
                    "utils.project.analysis.aorta_hu_analysis.get_aorta_visual_review",
                    return_value=review,
                ),
                patch(
                    "utils.project.analysis.aorta_hu_analysis.load_aorta_review_cohort",
                    return_value=pd.DataFrame({"IMG_ID": [13, 28, 44, 603]}),
                ),
            ):
                run, first = select_reviewed_imagecas_exams(root, n_exams=2)
                _, second = select_reviewed_imagecas_exams(root, n_exams=2)
                self.assertEqual(run, root / "run")
                self.assertTrue(set(first["IMG_ID"]).issubset({13, 28, 44}))
                pd.testing.assert_frame_equal(first, second)
                _, explicit = select_reviewed_imagecas_exams(root, exam_ids=[44, 13])
                self.assertEqual(list(explicit["IMG_ID"]), [44, 13])
                for ids in ([603], [13, 13], []):
                    with self.subTest(ids=ids), self.assertRaises(ValueError):
                        select_reviewed_imagecas_exams(root, exam_ids=ids)

    def test_example_selection_rejects_unknown_or_duplicate_ids(self):
        frame = pd.DataFrame({"IMG_ID": [13, 44, 28]})
        self.assertEqual(
            list(select_hu_example_records(frame, "IMG_ID")["IMG_ID"]), [13, 44]
        )
        self.assertEqual(
            list(
                select_hu_example_records(frame, "IMG_ID", requested=[28, 13])["IMG_ID"]
            ),
            [28, 13],
        )
        for ids in ([603], [13, 13], []):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                select_hu_example_records(frame, "IMG_ID", requested=ids)


class HuThresholdViewsTest(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_mip_thresholds_before_max_and_does_not_replace_negative_hu_with_zero(self):
        values = np.array([[[-20, -10, 100], [100, np.nan, np.inf]]], dtype=np.float32)
        mip = hu_axial_mip(values, (-20, -10))
        self.assertEqual(mip[0, 0], -10)
        self.assertTrue(np.isnan(mip[0, 1]))
        np.testing.assert_array_equal(hu_axial_mip(values), [[100, 100]])

    def test_axial_positions_follow_reference_extent(self):
        mask = np.zeros((2, 2, 20), dtype=bool)
        mask[0, 0, 4:17] = True
        self.assertEqual(reference_axial_slices(mask), (7, 10, 13))
        with self.assertRaisesRegex(ValueError, "vazia"):
            reference_axial_slices(np.zeros_like(mask))

    def test_plot_preserves_window_extent_and_contains_four_by_four_views(self):
        image = np.full((4, 5, 6), 200, dtype=np.float32)
        volume = HuReferenceVolume(
            image,
            {"imagecas_aorta": np.ones(image.shape, dtype=bool)},
            (0.5, 2, 3),
            "ImageCAS",
            "13",
        )
        figure = plot_hu_threshold_examples(
            volume, {key: (100, 300) for key in HU_REGION_NAMES}
        )
        self.assertEqual(len(figure.axes), 16)
        for ax in figure.axes:
            artist = ax.images[0]
            self.assertEqual(artist.get_clim(), (-200, 1000))
            self.assertEqual(tuple(artist.get_extent()), (-0.25, 1.75, -1, 9))
            self.assertEqual(artist.origin, "lower")
        with self.assertRaises(ValueError):
            plot_hu_threshold_examples(volume, {}, hu_window=(100, 0))
