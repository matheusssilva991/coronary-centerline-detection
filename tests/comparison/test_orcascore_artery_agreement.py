import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from utils.comparison_utils.orcascore_artery_agreement import (
    build_orcascore_pairs,
    common_orcascore_dice,
    compare_orcascore_pairs,
    select_orcascore_dice_case,
    summarize_orcascore_agreement,
    summarize_orcascore_dice_histogram,
)
from utils.project.analysis.artery_mask_agreement import (
    align_artery_prediction,
    artery_pair_metrics,
)


class OrcascoreArteryAgreementTest(unittest.TestCase):
    def test_example_selection_uses_valid_dice_and_prefers_a_case_near_the_target(self):
        results = pd.DataFrame(
            {
                "exam_id": ["TEV1P1", "TEV4P3", "TRV3P4", "TRV3P6", "TRV1P2"],
                "dice": [0.601, 0.599, 0.497, 0.765, None],
            }
        )
        self.assertEqual(select_orcascore_dice_case(results)["exam_id"], "TRV3P6")
        preferred = select_orcascore_dice_case(
            results, 0.60, preferred_exam_id="TEV1P1"
        )
        self.assertEqual(preferred["exam_id"], "TEV1P1")
        closest = select_orcascore_dice_case(results, 0.50, preferred_exam_id="TEV1P1")
        self.assertEqual(closest["exam_id"], "TRV3P4")
        self.assertIsNone(select_orcascore_dice_case(results.tail(1)))

    def test_histogram_counts_zero_boundary_one_and_omits_missing_values(self):
        results = pd.DataFrame(
            {
                "subset": ["train", "train", "test", "test"],
                "dice": [0.0, 0.3, 1.0, None],
            }
        )
        histogram = summarize_orcascore_dice_histogram(results)
        self.assertEqual(histogram["Geral"].sum(), 3)
        self.assertEqual(histogram.at[0, "Treino"], 1)
        self.assertEqual(histogram.at[3, "Treino"], 1)
        self.assertEqual(histogram.at[9, "Teste"], 1)

    def test_uses_physical_geometry_for_flipped_prediction_at_another_resolution(self):
        source = np.zeros((6, 4, 2), dtype=np.uint8)
        source[3, 1, 0] = 1
        source_affine = np.diag([-1, 1, 1, 1])
        source_affine[0, 3] = 5
        target = np.zeros((3, 4, 2), dtype=np.uint8)
        target[1, 1, 0] = 1
        aligned, resampled, source_voxels = align_artery_prediction(
            nib.Nifti1Image(source, source_affine),
            nib.Nifti1Image(target, np.diag([2, 1, 1, 1])),
        )
        np.testing.assert_array_equal(aligned, target)
        self.assertTrue(resampled)
        self.assertEqual(source_voxels, 1)
        metrics = artery_pair_metrics(target.astype(bool), aligned, source_voxels, 2.0)
        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["ai_volume_ml"], 0.002)

    def test_geo_mismatch_is_rejected_and_empty_predictions_are_distinguished(self):
        empty = np.zeros((3, 4, 2), dtype=bool)
        nonempty = empty.copy()
        nonempty[1, 1, 0] = True
        source_affine = np.eye(4)
        source_affine[0, 3] = 100
        with self.assertRaisesRegex(ValueError, "campo de visão"):
            align_artery_prediction(
                nib.Nifti1Image(nonempty.astype(np.uint8), source_affine),
                nib.Nifti1Image(empty.astype(np.uint8), np.eye(4)),
            )
        self.assertIsNone(artery_pair_metrics(empty, empty, 0, 1.0)["dice"])
        self.assertEqual(artery_pair_metrics(empty, nonempty, 1, 1.0)["dice"], 0.0)
        lost = artery_pair_metrics(nonempty, empty, 1, 1.0)
        self.assertEqual((lost["status"], lost["dice"]), ("ai_lost_in_resampling", 0.0))

    def test_keeps_missing_masks_in_coverage_and_compares_the_same_exam_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, ai_root = root / "run", root / "ai"
            (run_dir / "config").mkdir(parents=True)
            (run_dir / "numeric").mkdir()
            selected = [
                {"subset": "train", "exam_id": "TRV1P1"},
                {"subset": "test", "exam_id": "TEV1P1"},
                {"subset": "test", "exam_id": "TEV1P2"},
            ]
            (run_dir / "config/run_manifest.json").write_text(
                json.dumps({"dataset": "orcascore", "selected_exams": selected})
            )
            mask = np.zeros((3, 4, 2), dtype=np.uint8)
            mask[1, 1, 0] = 1
            prediction = nib.Nifti1Image(mask, np.eye(4))
            rows = []
            for case in selected[:2]:
                relative = Path(
                    "masks/artery",
                    case["subset"],
                    case["exam_id"],
                    "artery_processed.nii.gz",
                )
                path = run_dir / relative
                path.parent.mkdir(parents=True)
                nib.save(prediction, path)
                rows.append(
                    {
                        **case,
                        "status": "success",
                        "error": None,
                        "artery_mask_status": "saved",
                        "artery_mask_path": relative.as_posix(),
                    }
                )
            pd.DataFrame(rows).to_csv(run_dir / "numeric/results_all.csv", index=False)
            for variant, cases in (("a", selected), ("b", selected[1:])):
                for case in cases:
                    path = ai_root / variant / f"{case['exam_id']}CTAI/pre_label.nii.gz"
                    path.parent.mkdir(parents=True)
                    nib.save(prediction, path)
            results = compare_orcascore_pairs(
                build_orcascore_pairs(run_dir, ai_root, ["a", "b"])
            )
            coverage = results["status"].value_counts().to_dict()
            self.assertEqual(
                coverage, {"ok": 3, "pipeline_missing": 2, "ai_missing": 1}
            )
            common = common_orcascore_dice(results)
            self.assertEqual(set(common["exam_id"]), {"TEV1P1"})
            summary = summarize_orcascore_agreement(results, common)
            totals = summary.loc[summary["subset"].eq("Geral")].set_index("variant")
            self.assertEqual(totals["selected"].tolist(), [3, 3])
            self.assertEqual(totals["pipeline_masks"].tolist(), [2, 2])
            self.assertEqual(totals["dice_available"].tolist(), [2, 1])
            self.assertEqual(totals["dice_mean"].tolist(), [1.0, 1.0])
