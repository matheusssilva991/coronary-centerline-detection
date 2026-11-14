import unittest
from unittest.mock import patch

import numpy as np

from utils.segmentation.pipeline.arteries import (
    evaluate_artery_postprocessing_stages,
    get_artery_postprocessing_stages,
    postprocess_artery_mask,
    segment_artery_masks_from_vesselness,
    segment_arteries_from_vesselness,
)
from utils.utils.metrics import dice_score


class ArteryPostprocessingTests(unittest.TestCase):
    def test_measures_distinct_stages_and_negative_dilation_gain_without_mutation(self):
        label = np.zeros((15, 15, 15), dtype=np.uint8)
        label[5:10, 5:10, 5:10] = 1
        raw = label.copy()
        raw[7, 7, 7] = 0
        stages = get_artery_postprocessing_stages(
            raw, {"POSTPROCESSING": {"closing_radius": 1, "dilation_radius": 1}}
        )
        original = {name: mask.copy() for name, mask in stages.items()}
        metrics = evaluate_artery_postprocessing_stages(
            stages["raw_mask"],
            stages["closed_mask"],
            stages["final_mask"],
            label_artery=label,
        )
        self.assertGreater(metrics["dice_artery_closing_delta"], 0)
        self.assertLess(metrics["dice_artery_dilation_delta"], 0)
        self.assertEqual(metrics["dice_artery_after_closing"], 1)
        self.assertAlmostEqual(
            metrics["dice_artery_morphology_delta"],
            metrics["dice_artery_closing_delta"]
            + metrics["dice_artery_dilation_delta"],
        )
        for name in stages:
            np.testing.assert_array_equal(stages[name], original[name])

    def test_empty_masks_keep_existing_dice_convention(self):
        empty = np.zeros((5, 5, 5), dtype=np.uint8)
        for reference in (empty, np.ones_like(empty)):
            with self.subTest(reference_voxels=reference.sum()):
                metrics = evaluate_artery_postprocessing_stages(
                    empty, empty, empty, label_artery=reference
                )
                self.assertEqual(metrics["artery_voxels_after_closing"], 0)
                self.assertEqual(
                    metrics["dice_artery_after_closing"], dice_score(empty, reference)
                )
                self.assertEqual(metrics["dice_artery_closing_delta"], 0)
                self.assertEqual(metrics["dice_artery_dilation_delta"], 0)

    def test_unlabeled_anisotropic_volumes_do_not_invent_dice(self):
        masks = [np.ones((count, 1, 1), dtype=np.uint8) for count in (2, 4, 8)]
        metrics = evaluate_artery_postprocessing_stages(*masks, spacing=(0.5, 2.0, 3.0))
        self.assertEqual(metrics["artery_volume_after_closing_ml"], 0.012)
        self.assertEqual(metrics["artery_volume_before_morphology_ml"], 0.006)
        self.assertEqual(metrics["artery_volume_after_morphology_ml"], 0.024)
        self.assertFalse(any(key.startswith("dice_") for key in metrics))
        for spacing in ((0, 1, 1), (1, 1), (1, np.nan, 1), (1, np.inf, 1)):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                evaluate_artery_postprocessing_stages(*masks, spacing=spacing)

    def test_rg_and_fc_evaluation_preserve_masks_and_apply_morphology_once(self):
        raw = np.zeros((15, 15, 15), dtype=np.uint8)
        raw[5:10, 5:10, 5:10] = 1
        label = raw.copy()
        raw[7, 7, 7] = 0
        for method in ("region_growing", "fuzzy_connectedness"):
            with self.subTest(method=method):
                config = {
                    "ARTERY_SEGMENTATION": {"method": method},
                    "POSTPROCESSING": {"closing_radius": 1, "dilation_radius": 1},
                }
                expected = get_artery_postprocessing_stages(raw, config)
                with (
                    patch(
                        "utils.segmentation.pipeline.arteries._segment_with_region_growing",
                        return_value=(raw, {}),
                    ),
                    patch(
                        "utils.segmentation.pipeline.arteries._segment_with_fuzzy_connectedness",
                        return_value=(raw, {}),
                    ),
                    patch(
                        "utils.segmentation.pipeline.arteries.get_artery_postprocessing_stages",
                        wraps=get_artery_postprocessing_stages,
                    ) as postprocess,
                ):
                    result = segment_arteries_from_vesselness(
                        raw, label, raw, (7, 7, 7), None, config
                    )
                postprocess.assert_called_once()
                np.testing.assert_array_equal(
                    result["artery_mask"], expected["final_mask"]
                )
                np.testing.assert_array_equal(result["raw_artery_mask"], raw)
                self.assertEqual(
                    result["dice_artery"], dice_score(expected["final_mask"], label)
                )
                self.assertEqual(
                    result["dice_artery_before_morphology"], dice_score(raw, label)
                )
                self.assertEqual(
                    result["artery_voxels_after_closing"],
                    int(expected["closed_mask"].sum()),
                )

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
