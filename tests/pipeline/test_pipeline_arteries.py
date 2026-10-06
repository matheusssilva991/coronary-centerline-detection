import unittest
from unittest.mock import patch

import numpy as np

from utils.segmentation.pipeline.arteries import (
    get_artery_postprocessing_stages,
    postprocess_artery_mask,
    segment_artery_masks_from_vesselness,
)


class ArteryPostprocessingTests(unittest.TestCase):
    def test_stages_preserve_public_postprocessing_result(self):
        mask = np.zeros((9, 9, 9), dtype=np.uint8)
        mask[4, 4, 3:6] = 1
        config = {"POSTPROCESSING": {"closing_radius": 1, "dilation_radius": 1}}

        stages = get_artery_postprocessing_stages(mask, config)
        result = postprocess_artery_mask(mask, config)

        self.assertEqual(
            set(stages),
            {"raw_mask", "closed_mask", "final_mask"},
        )
        np.testing.assert_array_equal(stages["raw_mask"], mask)
        np.testing.assert_array_equal(stages["final_mask"], result)
        self.assertEqual(result.dtype, np.uint8)

    @patch("utils.segmentation.pipeline.arteries._segment_with_fuzzy_connectedness")
    @patch("utils.segmentation.pipeline.arteries.normal_region_growing_from_ostia")
    def test_unlabeled_segmentation_preserves_rg_stages_with_explicit_override(
        self,
        region_growing,
        fuzzy_connectedness,
    ):
        raw_mask = np.zeros((9, 9, 9), dtype=np.uint8)
        raw_mask[4, 4, 3:6] = 1
        region_growing.return_value = raw_mask
        config = {
            "ARTERY_SEGMENTATION": {"method": "fuzzy_connectedness"},
            "POSTPROCESSING": {"closing_radius": 1, "dilation_radius": 1},
        }

        result = segment_artery_masks_from_vesselness(
            np.zeros_like(raw_mask),
            np.zeros_like(raw_mask),
            (4, 4, 3),
            (4, 4, 5),
            config,
            method="region_growing",
        )
        expected = get_artery_postprocessing_stages(raw_mask, config)

        self.assertEqual(result.method, "region_growing")
        np.testing.assert_array_equal(result.raw_mask, expected["raw_mask"])
        np.testing.assert_array_equal(result.closed_mask, expected["closed_mask"])
        np.testing.assert_array_equal(result.final_mask, expected["final_mask"])
        region_growing.assert_called_once()
        fuzzy_connectedness.assert_not_called()

    @patch("utils.segmentation.fuzzy.connectedness.segment_artery_fuzzy_connectedness")
    def test_unlabeled_segmentation_honors_configured_fuzzy_connectedness(
        self,
        fuzzy_connectedness,
    ):
        raw_mask = np.zeros((5, 5, 5), dtype=np.uint8)
        raw_mask[2, 2, 2] = 1
        fuzzy_connectedness.return_value = {
            "raw_mask": raw_mask,
            "artery_mask": raw_mask,
            "details": {"processed_voxels": 12},
        }
        config = {
            "ARTERY_SEGMENTATION": {"method": "fuzzy_connectedness"},
            "FUZZY_CONNECTEDNESS": {
                "alpha": 0.5,
                "sigma_hu": 50.0,
                "neighborhood": 6,
            },
            "POSTPROCESSING": {"closing_radius": 1, "dilation_radius": 1},
            "MIN_THRESHOLD": -300,
        }

        result = segment_artery_masks_from_vesselness(
            np.zeros_like(raw_mask, dtype=np.float32),
            np.zeros_like(raw_mask, dtype=np.float32),
            (2, 2, 2),
            None,
            config,
        )

        self.assertEqual(result.method, "fuzzy_connectedness")
        self.assertEqual(result.details["processed_voxels"], 12)
        self.assertFalse(fuzzy_connectedness.call_args.kwargs["apply_postprocessing"])


if __name__ == "__main__":
    unittest.main()
