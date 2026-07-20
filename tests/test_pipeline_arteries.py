import unittest

import numpy as np

from utils.segmentation.pipeline_arteries import (
    get_artery_postprocessing_stages,
    postprocess_artery_mask,
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


if __name__ == "__main__":
    unittest.main()
