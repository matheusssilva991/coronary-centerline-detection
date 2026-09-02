import unittest

import pandas as pd

from utils.project.results import (
    add_config_columns,
    add_internal_result_aliases,
    make_readable_results_dataframe,
    make_result_dataframe,
)


class ResultAliasTests(unittest.TestCase):
    def test_adds_aorta_opening_radius_from_effective_config(self):
        result = add_config_columns(
            pd.DataFrame({"IMG_ID": [1]}),
            {"LEVEL_SET": {"leak_removal_radius": 2}},
        )

        self.assertEqual(result.loc[0, "aorta_opening_radius"], 2)

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
        self.assertIn("artery_dice", result.columns)

    def test_preserves_image_and_aorta_volume_fields_in_readable_schema(self):
        internal = make_result_dataframe(
            [
                {
                    "IMG_ID": 1,
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

        self.assertEqual(readable.loc[0, "image_voxel_count"], 1_000)
        self.assertEqual(readable.loc[0, "aorta_mask_voxel_count"], 125)
        self.assertEqual(readable.loc[0, "aorta_segmented_slice_count"], 5)
        self.assertEqual(readable.loc[0, "aorta_voxels_per_segmented_slice"], 25.0)
        self.assertEqual(readable.loc[0, "aorta_volume_fraction"], 0.125)
        self.assertEqual(readable.loc[0, "aorta_circle_radius_median_mm"], 14.2)
        self.assertEqual(
            readable.loc[0, "aorta_circle_radius_max_step_change_mm"], 1.3
        )
        self.assertEqual(
            readable.loc[0, "aorta_circle_upper_radius_bound_fraction"], 0.25
        )
        self.assertEqual(
            readable.loc[0, "aorta_level_set_initial_voxel_count"], 40
        )
        self.assertEqual(readable.loc[0, "aorta_level_set_raw_voxel_count"], 180)
        self.assertEqual(
            readable.loc[0, "aorta_level_set_initial_volume_fraction"], 0.04
        )
        self.assertEqual(
            readable.loc[0, "aorta_level_set_raw_volume_fraction"], 0.18
        )
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


if __name__ == "__main__":
    unittest.main()
