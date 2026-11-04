"""Comparação descritiva sem remover silenciosamente falhas ou volumes ausentes."""

import unittest

import numpy as np
import pandas as pd

from utils.comparison_utils.artery_volume import (
    compare_artery_volume_cohorts,
    prepare_artery_volume_dice_data,
    prepare_predicted_artery_volumes,
    summarize_artery_volume_bands,
)


class VolumeCohortComparisonTest(unittest.TestCase):
    def setUp(self):
        self.results = pd.DataFrame(
            {
                "IMG_ID": [1, 2, 3],
                "dice_artery": [0.8, 0.2, np.nan],
                "status": ["both_correct", "one_correct", "error"],
            }
        )
        self.volumes = pd.DataFrame(
            {"IMG_ID": [1, 2], "volume_outlier_group": ["in_range", "high"]}
        )

    def test_retains_missing_volume_and_documented_failure_as_zero(self):
        summary = compare_artery_volume_cohorts(
            {("mid_res", "test"): self.results}, self.volumes
        )
        row = summary.loc[summary.split.eq("test")].iloc[0]
        self.assertEqual(row["missing_volume_count"], 1)
        self.assertEqual(row["excluded_exams"], 1)
        self.assertEqual(row["remaining_exams"], 2)
        self.assertAlmostEqual(row["dice_before"], 1 / 3)
        self.assertAlmostEqual(row["dice_after"], 0.4)
        self.assertAlmostEqual(row["delta_dice"], 0.4 - 1 / 3)
        self.assertEqual(row["ostia_success_count_before"], 1)
        self.assertEqual(row["ostia_success_count_after"], 1)
        self.assertAlmostEqual(row["ostia_success_percent_before"], 100 / 3)
        self.assertEqual(row["ostia_success_percent_after"], 50)
        self.assertTrue(np.isnan(self.results.loc[2, "dice_artery"]))

    def test_same_exclusions_in_both_resolutions_and_weighted_all(self):
        groups = {}
        for resolution in ("mid_res", "high_res"):
            groups[resolution, "train"] = self.results.iloc[:1]
            groups[resolution, "test"] = self.results.iloc[1:]
        result = compare_artery_volume_cohorts(groups, self.volumes)
        complete = result.loc[result.split.eq("all")]
        self.assertTrue(complete.total_exams.eq(3).all())
        self.assertTrue(complete.excluded_exams.eq(1).all())
        np.testing.assert_allclose(complete.dice_before, 1 / 3)

    def test_removing_high_dice_outlier_can_reduce_performance(self):
        volumes = self.volumes.copy()
        volumes["volume_outlier_group"] = ["low", "in_range"]
        summary = compare_artery_volume_cohorts(
            {("mid_res", "test"): self.results}, volumes
        )
        self.assertTrue(summary.delta_dice.lt(0).all())

    def test_unexplained_missing_dice_and_invalid_range_raise(self):
        for value in (np.nan, 1.5, -0.1):
            frame = self.results.copy()
            frame.loc[0, "dice_artery"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                compare_artery_volume_cohorts(
                    {("mid_res", "test"): frame}, self.volumes
                )

    def test_duplicate_result_volume_and_overlapping_split_ids_raise(self):
        for groups, volumes in (
            (
                {("mid_res", "test"): pd.concat([self.results, self.results])},
                self.volumes,
            ),
            (
                {("mid_res", "test"): self.results},
                pd.concat([self.volumes, self.volumes]),
            ),
            (
                {("mid_res", "train"): self.results, ("mid_res", "test"): self.results},
                self.volumes,
            ),
        ):
            with self.assertRaises(ValueError):
                compare_artery_volume_cohorts(groups, volumes)

    def test_no_outlier_preserves_dice_and_all_excluded_reports_nan(self):
        volumes = pd.DataFrame(
            {"IMG_ID": [1, 2, 3], "volume_outlier_group": "in_range"}
        )
        result = compare_artery_volume_cohorts(
            {("mid_res", "test"): self.results}, volumes
        )
        self.assertTrue(result.delta_dice.eq(0).all())
        volumes["volume_outlier_group"] = "high"
        result = compare_artery_volume_cohorts(
            {("mid_res", "test"): self.results}, volumes
        )
        self.assertTrue(result.remaining_exams.eq(0).all())
        self.assertTrue(result.dice_after.isna().all())


class VolumeDicePairingTest(unittest.TestCase):
    def setUp(self):
        self.mid = pd.DataFrame(
            {
                "IMG_ID": [1, 2, 3, 4],
                "dice_artery": [0.8, 0, np.nan, 0.5],
                "status": ["both_correct", "both_tolerable", "error", "one_correct"],
            }
        )
        self.high = pd.DataFrame(
            {
                "IMG_ID": [3, 2, 1, 5],
                "dice_artery": [0.3, 0.1, 0.7, 0.9],
                "status": [
                    "one_correct",
                    "both_correct",
                    "both_tolerable",
                    "both_correct",
                ],
            }
        )
        self.volumes = pd.DataFrame(
            {
                "IMG_ID": [1, 2, 3, 5],
                "volume_ml": [2.0, 0, 12.0, 8.0],
                "volume_outlier_group": ["in_range", "low", "high", "in_range"],
            }
        )
        self.groups = {("mid_res", "test"): self.mid, ("high_res", "test"): self.high}

    def test_pairs_by_id_and_preserves_zeros_failures_and_outliers(self):
        plotted, paired, coverage = prepare_artery_volume_dice_data(
            self.groups, self.volumes
        )
        paired = paired.set_index("IMG_ID")
        self.assertEqual(sorted(paired.index), [1, 2, 3])
        np.testing.assert_allclose(
            paired.loc[[1, 2, 3], "delta_dice"], [-0.1, 0.1, 0.3]
        )
        self.assertEqual(paired.loc[2, "dice_mid"], 0)
        self.assertEqual(paired.loc[3, "dice_mid"], 0)
        self.assertTrue(paired.loc[2, "ostia_success_both"])
        self.assertFalse(paired.loc[3, "ostia_success_both"])
        self.assertTrue(paired.loc[[2, 3], "is_volume_outlier"].all())
        mid_coverage = coverage.set_index("resolution").loc["mid_res"]
        self.assertEqual(mid_coverage["missing_volume_ids"], [4])
        self.assertEqual(mid_coverage["missing_pair_ids"], [4])
        self.assertEqual(len(plotted), 7)
        self.assertTrue(np.isnan(self.mid.loc[2, "dice_artery"]))

    def test_absent_resolution_and_no_volumes_return_empty_pairs(self):
        for groups, volumes in (
            ({("mid_res", "test"): self.mid}, self.volumes),
            (self.groups, self.volumes.iloc[:0]),
            ({}, self.volumes),
        ):
            with self.subTest(groups=list(groups), volumes=len(volumes)):
                _, paired, coverage = prepare_artery_volume_dice_data(groups, volumes)
                self.assertTrue(paired.empty)
                self.assertEqual(len(coverage), 2)

    def test_invalid_data_raise_without_silent_dropping(self):
        for column, value in (
            ("dice_artery", np.nan),
            ("dice_artery", 1.1),
            ("IMG_ID", 2),
        ):
            changed = self.mid.copy()
            changed.loc[0, column] = value
            with (
                self.subTest(column=column, value=value),
                self.assertRaises(ValueError),
            ):
                prepare_artery_volume_dice_data(
                    {("mid_res", "test"): changed}, self.volumes
                )
        with self.assertRaises(ValueError):
            prepare_artery_volume_dice_data(self.groups, pd.concat([self.volumes] * 2))
        for value in (np.nan, np.inf, -1):
            volumes = self.volumes.copy()
            volumes.loc[0, "volume_ml"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepare_artery_volume_dice_data(self.groups, volumes)
        with self.assertRaises(ValueError):
            prepare_artery_volume_dice_data(self.groups, self.volumes, split="invalid")

    def test_all_combines_splits_and_rejects_overlapping_ids(self):
        groups = {}
        for resolution in ("mid_res", "high_res"):
            for split, position in (("train", 0), ("val", 1), ("test", 2)):
                groups[resolution, split] = self.mid.iloc[[position]]
        data, pairs, coverage = prepare_artery_volume_dice_data(
            groups, self.volumes, split="all"
        )
        self.assertEqual(len(data), 6)
        self.assertEqual(len(pairs), 3)
        self.assertTrue(coverage.source_available.all())
        self.assertTrue(coverage.missing_splits.map(len).eq(0).all())
        groups["mid_res", "train"] = self.mid.iloc[[1]]
        with self.assertRaisesRegex(ValueError, "compartilhados"):
            prepare_artery_volume_dice_data(groups, self.volumes, split="all")
        _, _, partial = prepare_artery_volume_dice_data(
            self.groups, self.volumes, split="all"
        )
        self.assertFalse(partial.source_available.any())
        self.assertTrue(
            partial.missing_splits.map(lambda value: value == ["train", "val"]).all()
        )

    def test_all_predicted_volumes_preserve_each_split_configuration(self):
        volumes = self.volumes.assign(spacing_x_mm=1, spacing_y_mm=2, spacing_z_mm=3)
        groups = {
            ("mid_res", "train"): self.mid.iloc[[0]].assign(artery_voxel_count=10),
            ("mid_res", "val"): self.mid.iloc[[1]].assign(artery_voxel_count=10),
            ("mid_res", "test"): self.mid.iloc[[2]].assign(artery_voxel_count=10),
        }
        configs = {
            ("mid_res", subset): {"DOWNSCALE_FACTORS": factors}
            for subset, factors in (("train", [2, 2, 1]), ("val", [1, 1, 1]))
        }
        data, coverage = prepare_predicted_artery_volumes(
            groups, volumes, configs, split="all"
        )
        data = data.set_index("IMG_ID")
        self.assertAlmostEqual(data.loc[1, "predicted_volume_ml"], 0.24)
        self.assertAlmostEqual(data.loc[2, "predicted_volume_ml"], 0.06)
        self.assertTrue(pd.isna(data.loc[3, "predicted_volume_ml"]))
        self.assertEqual(
            coverage.set_index("resolution").loc["mid_res", "total_exams"], 3
        )
        self.assertEqual(
            coverage.set_index("resolution").loc["mid_res", "valid_predictions"], 2
        )

    def test_predicted_volume_uses_post_morphology_and_anisotropic_spacing(self):
        volumes = self.volumes.assign(
            spacing_x_mm=2.0, spacing_y_mm=3.0, spacing_z_mm=0.5
        )
        groups = {
            ("mid_res", "test"): self.mid.assign(
                artery_voxel_count_after_morphology=[10, 0, np.nan, 15],
                artery_voxel_count=999,
            ),
            ("high_res", "test"): self.high.assign(
                artery_voxel_count_after_morphology=[5, 0, 10, 20],
            ),
        }
        configs = {
            ("mid_res", "test"): {"DOWNSCALE_FACTORS": [2, 2, 1]},
            ("high_res", "test"): {"DOWNSCALE_FACTORS": [1, 1, 1]},
        }
        data, coverage = prepare_predicted_artery_volumes(groups, volumes, configs)
        data = data.set_index(["resolution", "IMG_ID"])
        self.assertAlmostEqual(data.loc[("mid_res", 1), "predicted_volume_ml"], 0.12)
        self.assertAlmostEqual(data.loc[("high_res", 1), "predicted_volume_ml"], 0.03)
        self.assertEqual(data.loc[("mid_res", 2), "predicted_volume_ml"], 0)
        self.assertTrue(pd.isna(data.loc[("mid_res", 3), "predicted_volume_ml"]))
        self.assertEqual(
            coverage.set_index("resolution").loc["mid_res", "excluded_ids"], [3]
        )

    def test_predicted_volume_reports_missing_or_invalid_geometry_and_counts(self):
        volumes = self.volumes.assign(spacing_x_mm=1, spacing_y_mm=1, spacing_z_mm=1)
        groups = {("mid_res", "test"): self.mid.assign(artery_voxel_count=[0, 4, 5, 6])}
        for factors in (None, [1, 1], [0, 1, 1], [np.inf, 1, 1], "invalid"):
            with self.subTest(factors=factors):
                data, _ = prepare_predicted_artery_volumes(
                    groups,
                    volumes,
                    {("mid_res", "test"): {"DOWNSCALE_FACTORS": factors}},
                )
                self.assertTrue(data.predicted_volume_ml.isna().all())
        configs = {("mid_res", "test"): {"DOWNSCALE_FACTORS": [1, 1, 1]}}
        for column, value in (("spacing_x_mm", 0), ("spacing_z_mm", np.nan)):
            changed = volumes.copy()
            changed.loc[0, column] = value
            data, _ = prepare_predicted_artery_volumes(groups, changed, configs)
            self.assertTrue(
                pd.isna(data.loc[data.IMG_ID.eq(1), "predicted_volume_ml"]).all()
            )
        for count in (-1, 1.5, np.inf, np.nan):
            changed = self.mid.assign(artery_voxel_count=[count, 0, 10, 5])
            data, _ = prepare_predicted_artery_volumes(
                {("mid_res", "test"): changed},
                volumes,
                configs,
            )
            self.assertTrue(
                pd.isna(data.loc[data.IMG_ID.eq(1), "predicted_volume_ml"]).all()
            )
        empty, report = prepare_predicted_artery_volumes({}, volumes, {})
        self.assertTrue(empty.empty)
        self.assertEqual(report.valid_predictions.sum(), 0)

    def test_band_boundaries_are_common_and_not_weighted_by_resolution_count(self):
        data, _, _ = prepare_artery_volume_dice_data(self.groups, self.volumes)
        summary, bounds = summarize_artery_volume_bands(data)
        self.assertEqual(len(bounds), 4)
        np.testing.assert_allclose(bounds.upper_ml, [1.5, 5.0, 9.0, 12.0])
        counts = summary.groupby("resolution").exam_count.sum().to_dict()
        self.assertEqual(counts, {"mid_res": 3, "high_res": 4})
        mid_zero = summary.loc[
            summary.resolution.eq("mid_res") & summary.lower_ml.eq(0)
        ].iloc[0]
        self.assertEqual(mid_zero.dice_mean, 0)
        self.assertEqual(mid_zero.ostia_success_percent, 100)
        self.assertTrue(pd.isna(mid_zero.dice_std))

    def test_bands_handle_constant_repeated_and_empty_volumes(self):
        data, _, _ = prepare_artery_volume_dice_data(self.groups, self.volumes)
        constant = data.assign(volume_ml=3.0)
        summary, bounds = summarize_artery_volume_bands(constant)
        self.assertEqual(len(bounds), 1)
        self.assertEqual(summary.exam_count.sum(), len(data))
        self.assertTrue(bounds.lower_ml.eq(bounds.upper_ml).all())
        repeated = data.copy()
        repeated["volume_ml"] = repeated.IMG_ID.map({1: 1, 2: 1, 3: 1, 5: 2})
        _, bounds = summarize_artery_volume_bands(repeated)
        self.assertEqual(len(bounds), 2)
        a, b = summarize_artery_volume_bands(data.iloc[:0])
        self.assertTrue(a.empty and b.empty)
        with self.assertRaises(ValueError):
            summarize_artery_volume_bands(pd.concat([data, data]))
