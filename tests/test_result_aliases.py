import unittest

import pandas as pd

from utils.project.results import (
    add_internal_result_aliases,
    make_readable_results_dataframe,
    make_result_dataframe,
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
                    "aorta_level_set_mode": "adaptive",
                    "aorta_level_set_iterations_used": 31,
                    "aorta_level_set_stop_reason": "nominal_corrected",
                    "aorta_level_set_checkpoint_count": 4,
                    "aorta_level_set_rolled_back": True,
                    "aorta_level_set_mask_change_fraction": 0.008,
                    "aorta_level_set_circle_fill_q25": 0.87,
                    "aorta_level_set_circle_area_ratio_p90": 1.4,
                    "aorta_level_set_leak_suspected": True,
                    "aorta_level_set_localization_suspected": False,
                    "aorta_level_set_leak_signal_count": 2,
                    "aorta_level_set_trigger_iteration": 26,
                    "aorta_level_set_trigger_volume_fraction": 0.019,
                    "aorta_level_set_trigger_relative_growth": 0.3,
                    "aorta_level_set_trigger_mask_change_fraction": 0.2,
                    "aorta_level_set_trigger_circle_fill_q25": 0.86,
                    "aorta_level_set_trigger_circle_area_ratio_p90": 2.2,
                    "aorta_level_set_correction_applied": True,
                    "aorta_level_set_correction_method": "contractive_level_set",
                    "aorta_level_set_refinement_applied": True,
                    "aorta_level_set_refinement_accepted": True,
                    "aorta_level_set_refinement_iterations": 3,
                    "aorta_level_set_refinement_balloon": -0.25,
                    "aorta_level_set_refinement_smoothing": 1,
                    "aorta_level_set_refinement_transition_mode": "gradual",
                    "aorta_level_set_refinement_anomaly_margin_slices": 10,
                    "aorta_level_set_refinement_volume_loss_fraction": 0.12,
                    "aorta_level_set_slice_area_jump_p95_before": 0.3,
                    "aorta_level_set_slice_area_jump_p95_after": 0.2,
                    "aorta_level_set_refinement_rejection_reason": "accepted",
                    "aorta_level_set_controller_state": "oversegmented",
                    "aorta_level_set_profile_used": "conservative",
                    "aorta_level_set_rollback_iteration": 26,
                    "aorta_level_set_circle_confidence_signal_count": 3,
                    "aorta_level_set_alternative_attempted": True,
                    "aorta_level_set_alternative_accepted": True,
                    "aorta_level_set_conservative_attempted": True,
                    "aorta_level_set_conservative_accepted": True,
                    "aorta_level_set_nominal_volume_fraction": 0.025,
                    "aorta_level_set_final_volume_fraction": 0.018,
                    "aorta_level_set_decision_reason": "accepted",
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
        self.assertEqual(readable.loc[0, "aorta_level_set_mode"], "adaptive")
        self.assertEqual(readable.loc[0, "aorta_level_set_iterations_used"], 31)
        self.assertEqual(
            readable.loc[0, "aorta_level_set_stop_reason"], "nominal_corrected"
        )
        self.assertEqual(readable.loc[0, "aorta_level_set_checkpoint_count"], 4)
        self.assertEqual(readable.loc[0, "aorta_level_set_rolled_back"], "yes")
        self.assertEqual(readable.loc[0, "aorta_level_set_mask_change_fraction"], 0.008)
        self.assertEqual(readable.loc[0, "aorta_level_set_circle_fill_q25"], 0.87)
        self.assertEqual(readable.loc[0, "aorta_level_set_circle_area_ratio_p90"], 1.4)
        self.assertEqual(readable.loc[0, "aorta_level_set_leak_suspected"], "yes")
        self.assertEqual(
            readable.loc[0, "aorta_level_set_localization_suspected"], "no"
        )
        self.assertEqual(readable.loc[0, "aorta_level_set_leak_signal_count"], 2)
        self.assertEqual(readable.loc[0, "aorta_level_set_trigger_iteration"], 26)
        self.assertEqual(
            readable.loc[0, "aorta_level_set_correction_applied"], "yes"
        )
        self.assertEqual(
            readable.loc[0, "aorta_level_set_correction_method"],
            "contractive_level_set",
        )
        self.assertEqual(
            readable.loc[0, "aorta_level_set_refinement_applied"], "yes"
        )
        self.assertEqual(
            readable.loc[0, "aorta_level_set_refinement_accepted"], "yes"
        )
        self.assertEqual(readable.loc[0, "aorta_level_set_refinement_iterations"], 3)
        self.assertEqual(
            readable.loc[0, "aorta_level_set_refinement_transition_mode"],
            "gradual",
        )
        self.assertEqual(
            readable.loc[0, "aorta_level_set_refinement_anomaly_margin_slices"],
            10,
        )
        self.assertEqual(
            readable.loc[0, "aorta_level_set_refinement_volume_loss_fraction"],
            0.12,
        )
        self.assertEqual(
            readable.loc[0, "aorta_level_set_slice_area_jump_p95_after"], 0.2
        )
        self.assertEqual(
            readable.loc[0, "aorta_level_set_controller_state"], "oversegmented"
        )
        self.assertEqual(
            readable.loc[0, "aorta_level_set_profile_used"], "conservative"
        )
        self.assertEqual(readable.loc[0, "aorta_level_set_rollback_iteration"], 26)
        self.assertEqual(
            readable.loc[0, "aorta_level_set_alternative_accepted"], "yes"
        )
        self.assertEqual(
            readable.loc[0, "aorta_level_set_final_volume_fraction"], 0.018
        )


if __name__ == "__main__":
    unittest.main()
