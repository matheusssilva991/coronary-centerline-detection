import unittest

import pandas as pd

from utils.project.results import (
    CANONICAL_COLUMN_NAMES,
    READABLE_COLUMN_NAMES,
    RESULT_COLUMNS,
    add_internal_result_aliases,
    build_result_row,
    make_readable_results_dataframe,
    make_result_dataframe,
    ostia_status_label_pt,
    result_status_label_pt,
    summarize_results_df,
)


class ResultAliasTests(unittest.TestCase):
    def test_adds_typed_internal_aliases_without_removing_readable_columns(self):
        readable_df = pd.DataFrame(
            {
                "artery_dice": [0.75, 0.25],
                "ostia_detected": ["yes", "no"],
                "both_ostia_correct": ["yes", "no"],
                "ostia_detection_status": ["both correct", "not found"],
            }
        )

        result = add_internal_result_aliases(readable_df)

        self.assertEqual(result["dice_artery"].tolist(), [0.75, 0.25])
        self.assertEqual(result["ostia_found"].tolist(), [True, False])
        self.assertEqual(result["both_correct"].tolist(), [True, False])
        self.assertEqual(
            result["ostia_status"].tolist(),
            ["both_correct", "not_found"],
        )
        self.assertEqual(
            result["ostia_detection_status"].tolist(),
            ["both_correct", "not_found"],
        )
        self.assertIn("artery_dice", result.columns)

    def test_persists_status_codes_and_translates_only_for_presentation(self):
        internal = make_result_dataframe(
            [
                {"IMG_ID": 1, "status": "ambos toleráveis"},
                {"IMG_ID": 2, "status": "no ostium correct"},
            ]
        )
        readable = make_readable_results_dataframe(internal)

        self.assertEqual(
            readable["status"].tolist(),
            ["both_tolerable", "none_correct"],
        )
        self.assertEqual(result_status_label_pt("both_tolerable"), "ambos toleráveis")
        self.assertEqual(ostia_status_label_pt("both_correct"), "ambos corretos")

    def test_preserves_image_and_aorta_volume_fields_in_readable_schema(self):
        internal = make_result_dataframe(
            [
                {
                    "IMG_ID": 1,
                    "effective_upper_threshold_hu": 845.5,
                    "image_voxels": 1_000,
                    "aorta_mask_voxels": 125,
                    "aorta_segmented_slice_count": 5,
                    "aorta_voxels_per_segmented_slice": 25.0,
                    "aorta_volume_fraction": 0.125,
                    "aorta_circle_radius_median_mm": 14.2,
                    "aorta_circle_radius_max_step_change_mm": 1.3,
                    "aorta_circle_upper_radius_bound_fraction": 0.25,
                    "aorta_level_set_initial_voxel_count": 40,
                    "aorta_level_set_raw_voxel_count": 180,
                    "aorta_level_set_initial_volume_fraction": 0.04,
                    "aorta_level_set_raw_volume_fraction": 0.18,
                    "aorta_level_set_iterations_used": 31,
                    "aorta_level_set_circle_fill_q25": 0.87,
                    "aorta_level_set_circle_area_ratio_p90": 1.4,
                    "aorta_slice_area_jump_p95": 0.2,
                    "aorta_segmentation_feedback": "adequate",
                }
            ]
        )

        readable = make_readable_results_dataframe(internal)

        self.assertEqual(readable.loc[0, "effective_upper_threshold_hu"], 845.5)
        self.assertEqual(readable.loc[0, "image_voxel_count"], 1_000)
        self.assertEqual(readable.loc[0, "aorta_mask_voxel_count"], 125)
        self.assertEqual(readable.loc[0, "aorta_segmented_slice_count"], 5)
        self.assertEqual(readable.loc[0, "aorta_voxels_per_segmented_slice"], 25.0)
        self.assertEqual(readable.loc[0, "aorta_volume_fraction"], 0.125)
        self.assertEqual(readable.loc[0, "aorta_circle_radius_median_mm"], 14.2)
        self.assertEqual(readable.loc[0, "aorta_circle_radius_max_step_change_mm"], 1.3)
        self.assertEqual(
            readable.loc[0, "aorta_circle_upper_radius_bound_fraction"], 0.25
        )
        self.assertEqual(readable.loc[0, "aorta_level_set_initial_voxel_count"], 40)
        self.assertEqual(readable.loc[0, "aorta_level_set_raw_voxel_count"], 180)
        self.assertEqual(
            readable.loc[0, "aorta_level_set_initial_volume_fraction"], 0.04
        )
        self.assertEqual(readable.loc[0, "aorta_level_set_raw_volume_fraction"], 0.18)
        self.assertEqual(readable.loc[0, "aorta_level_set_iterations_used"], 31)
        self.assertEqual(readable.loc[0, "aorta_level_set_circle_fill_q25"], 0.87)
        self.assertEqual(readable.loc[0, "aorta_level_set_circle_area_ratio_p90"], 1.4)
        self.assertEqual(readable.loc[0, "aorta_slice_area_jump_p95"], 0.2)
        self.assertEqual(
            readable.loc[0, "aorta_segmentation_feedback"],
            "adequate",
        )

    def test_omits_removed_aorta_controller_fields_from_new_results(self):
        internal = make_result_dataframe(
            [
                {
                    "IMG_ID": 1,
                    "aorta_level_set_checkpoint_count": 4,
                    "aorta_level_set_controller_state": "oversegmented",
                    "aorta_level_set_refinement_applied": True,
                }
            ]
        )

        self.assertNotIn("aorta_level_set_checkpoint_count", internal.columns)
        self.assertNotIn("aorta_level_set_controller_state", internal.columns)
        self.assertNotIn("aorta_level_set_refinement_applied", internal.columns)

    def test_summarizes_effective_upper_threshold_values(self):
        internal = make_result_dataframe(
            [
                {"IMG_ID": 1, "effective_upper_threshold_hu": 800.0},
                {"IMG_ID": 2, "effective_upper_threshold_hu": 900.0},
                {"IMG_ID": 3, "effective_upper_threshold_hu": None},
            ]
        )

        summary = summarize_results_df(internal)

        self.assertEqual(summary["effective_upper_threshold_hu_count"], 2)
        self.assertEqual(summary["effective_upper_threshold_hu_mean"], 850.0)
        self.assertEqual(summary["effective_upper_threshold_hu_min"], 800.0)
        self.assertEqual(summary["effective_upper_threshold_hu_max"], 900.0)

    def test_uses_legacy_max_threshold_as_effective_upper_threshold(self):
        internal = make_result_dataframe([{"IMG_ID": 1, "max_threshold": 875.0}])

        self.assertEqual(internal.loc[0, "effective_upper_threshold_hu"], 875.0)

    def test_readable_column_map_and_build_result_row_preserve_contract(self):
        self.assertEqual(
            CANONICAL_COLUMN_NAMES,
            {
                readable: internal
                for internal, readable in READABLE_COLUMN_NAMES.items()
            },
        )
        self.assertEqual(READABLE_COLUMN_NAMES["dice_artery"], "artery_dice")
        self.assertEqual(
            READABLE_COLUMN_NAMES["aorta_circle_filter_method"],
            "aorta_circle_filter_method",
        )
        self.assertEqual(READABLE_COLUMN_NAMES["min_threshold"], "min_threshold_hu")
        self.assertEqual(READABLE_COLUMN_NAMES["max_threshold"], "max_threshold_hu")

        empty_row = build_result_row({})
        for column in RESULT_COLUMNS:
            self.assertIn(column, empty_row)
        self.assertEqual(empty_row["artery_segmentation_method"], "region_growing")
        self.assertEqual(empty_row["aorta_circle_filter_method"], "none")
        self.assertFalse(empty_row["ostia_found"])
        self.assertEqual(empty_row["status"], "none_correct")

        legacy_row = build_result_row(
            {
                "IMG_ID": 42,
                "artery_dice": 0.81,
                "max_threshold_hu": 910.0,
                "ostia_detected": "yes",
                "ostia_detection_status": "both correct",
                "both_ostia_correct": "yes",
            }
        )
        self.assertEqual(legacy_row["IMG_ID"], 42)
        self.assertEqual(legacy_row["dice_artery"], 0.81)
        self.assertEqual(legacy_row["effective_upper_threshold_hu"], 910.0)
        self.assertTrue(legacy_row["ostia_found"])
        self.assertEqual(legacy_row["ostia_status"], "both_correct")
        self.assertEqual(legacy_row["status"], "both_correct")


if __name__ == "__main__":
    unittest.main()
