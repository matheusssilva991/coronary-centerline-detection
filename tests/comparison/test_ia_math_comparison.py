import unittest

import pandas as pd

from utils.comparison_utils.ostia_scenarios import (
    build_ostia_image_comparison_df,
)


class IaMathComparisonTests(unittest.TestCase):
    def test_image_comparison_keeps_each_ia_method(self) -> None:
        ia_results = pd.DataFrame(
            [
                {
                    "target_resolution": "mid_res",
                    "img_id": 10,
                    "dice": 0.70,
                    "method": "mid::direct_fcn_4",
                },
                {
                    "target_resolution": "mid_res",
                    "img_id": 10,
                    "dice": 0.75,
                    "method": "mid::direct_fcn_ag_4",
                },
            ]
        )
        math_results = pd.DataFrame(
            [
                {
                    "target_resolution": "mid_res",
                    "img_id": 10,
                    "dice": 0.60,
                    "method": "pipeline_matematico",
                }
            ]
        )

        comparison = build_ostia_image_comparison_df(ia_results, math_results)

        self.assertEqual(len(comparison), 2)
        self.assertEqual(
            set(comparison["ia_method"]),
            {"mid::direct_fcn_4", "mid::direct_fcn_ag_4"},
        )
        self.assertEqual(comparison["math_dice"].tolist(), [0.60, 0.60])


if __name__ == "__main__":
    unittest.main()
