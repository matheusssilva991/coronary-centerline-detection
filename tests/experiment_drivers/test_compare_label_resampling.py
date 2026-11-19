"""Verifica o pareamento de dois Dice com uma única predição sintética."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from nibabel.loadsave import save
from nibabel.nifti1 import Nifti1Image
import numpy as np

from experiments.compare_label_resampling import compare_case
from utils.utils.metrics import dice_score


class CompareLabelResamplingTests(TestCase):
    def test_one_prediction_is_scored_on_two_different_grids(self):
        label = np.zeros((8, 4, 2), dtype=np.uint8)
        label[1] = 1
        prediction = np.zeros((4, 2, 2), dtype=np.uint8)
        prediction[0] = 1
        config = {"DOWNSCALE_FACTORS": [2, 2, 1], "DOWNSCALE_METHOD": "opencv"}

        def predict(case, row, config, experiment, *, include_artery_mask):
            self.assertTrue(include_artery_mask)
            row["artery_mask"] = prediction
            row["dice_artery"] = dice_score(prediction, case["down_label"])

        with TemporaryDirectory() as temp:
            root = Path(temp)
            for kind in ("img", "label"):
                save(Nifti1Image(label, np.eye(4)), root / f"1.{kind}.nii.gz")
            with (
                patch(
                    "experiments.compare_label_resampling.prepare_image_context",
                    side_effect=lambda case, *_: case,
                ),
                patch(
                    "experiments.compare_label_resampling.evaluate_prepared_image",
                    side_effect=predict,
                ) as evaluate,
            ):
                result = compare_case(1, root, config, {})
        evaluate.assert_called_once()
        self.assertEqual(result, {"dice_legacy": 0, "dice_corrected": 1, "delta": 1})
