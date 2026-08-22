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
                }
            ]
        )

        readable = make_readable_results_dataframe(internal)

        self.assertEqual(readable.loc[0, "image_voxel_count"], 1_000)
        self.assertEqual(readable.loc[0, "aorta_mask_voxel_count"], 125)
        self.assertEqual(readable.loc[0, "aorta_segmented_slice_count"], 5)
        self.assertEqual(readable.loc[0, "aorta_voxels_per_segmented_slice"], 25.0)
        self.assertEqual(readable.loc[0, "aorta_volume_fraction"], 0.125)


if __name__ == "__main__":
    unittest.main()
