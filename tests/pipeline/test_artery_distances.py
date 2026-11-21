"""Integração opcional de HD/AHD sem executar exames científicos."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

import segmentation_pipeline as pipeline
from utils.project.evaluation.artery_distances import (
    DISTANCE_PROTOCOL,
    DISTANCE_RESULT_COLUMNS,
    apply_distance_metric_options,
    evaluate_artery_distances,
    summarize_artery_distances,
)
from utils.project.results.io import merge_batch_results, save_results
from utils.project.results.metadata import build_metadata
from utils.project.results.schema import (
    add_internal_result_aliases,
    make_result_dataframe,
)
from utils.segmentation.pipeline.arteries import (
    segment_arteries_from_ostia,
    segment_arteries_from_vesselness,
)
from utils.segmentation.pipeline.cli import build_parser
from utils.segmentation.pipeline.orchestration import (
    _load_previous_batches,
    process_image,
)


class DistanceOptionsTests(TestCase):
    def test_imagecas_cli_has_tristate_flags(self):
        parser = build_parser(Path("/data"), Path("/out"))
        self.assertIsNone(parser.parse_args([]).distance_metrics)
        self.assertTrue(parser.parse_args(["--distance-metrics"]).distance_metrics)
        self.assertFalse(parser.parse_args(["--no-distance-metrics"]).distance_metrics)

    def test_options_preserve_config_unless_explicitly_overridden(self):
        for original in (False, True):
            for override in (None, False, True):
                config = {
                    "EVALUATION": {"distance_metrics": original},
                    "MIN_THRESHOLD": -300,
                }
                apply_distance_metric_options(
                    config, SimpleNamespace(distance_metrics=override)
                )
                self.assertEqual(
                    config["EVALUATION"]["distance_metrics"],
                    original if override is None else override,
                )
                self.assertEqual(
                    config["EVALUATION"]["distance_metrics_protocol"], DISTANCE_PROTOCOL
                )
                self.assertEqual(config["MIN_THRESHOLD"], -300)

    def test_resume_preserves_saved_option_and_cli_or_file_can_override(self):
        parser = build_parser(Path("/data"), Path("/out"))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            snapshot_path = root / "config/effective_pipeline_config.json"
            snapshot_path.write_text(
                json.dumps({"EVALUATION": {"distance_metrics": True}})
            )
            args = ["--resume-dir", str(root), "--resume-batch", "2"]
            restored = pipeline.build_effective_config(parser.parse_args(args))
            disabled = pipeline.build_effective_config(
                parser.parse_args([*args, "--no-distance-metrics"])
            )
            self.assertEqual(
                restored["EVALUATION"]["distance_metrics_protocol"],
                DISTANCE_PROTOCOL,
            )
            mask = np.ones((3, 3, 3))
            legacy = evaluate_artery_distances(mask, mask, (1, 2, 3), restored)
            self.assertNotIn("artery_hd_mm", legacy)
            self.assertEqual(legacy["artery_avghd_foreground_mm"], 0.0)
            self.assertTrue(restored["EVALUATION"]["distance_metrics"])
            self.assertFalse(disabled["EVALUATION"]["distance_metrics"])
            config_path = root / "override.json"
            config_path.write_text(
                json.dumps({"EVALUATION": {"distance_metrics": False}})
            )
            overridden = pipeline.build_effective_config(
                parser.parse_args([*args, "--config-file", str(config_path)])
            )
            self.assertFalse(overridden["EVALUATION"]["distance_metrics"])
            snapshot_path.write_text("{}")
            old = pipeline.build_effective_config(parser.parse_args(args))
            self.assertFalse(old["EVALUATION"]["distance_metrics"])


class ArteryDistanceIntegrationTests(TestCase):
    def test_distances_match_explicit_foreground_and_surface_point_sets(self):
        from scipy.spatial.distance import cdist

        spacing = np.array([0.5, 1.5, 3.0])
        config = {"EVALUATION": {"distance_metrics": True}}
        first = np.zeros((8, 8, 8), dtype=bool)
        first[1:6, 1:6, 1:6] = True
        reference = first.copy()
        reference[2:5, 2:5, 2:5] = False
        first[7, 7, 7] = True
        result = evaluate_artery_distances(first, reference, spacing, config)
        self.assertEqual(result["artery_distance_metrics_status"], "ok")

        def directed(a, b):
            distances = cdist(np.argwhere(a) * spacing, np.argwhere(b) * spacing)
            return distances.min(axis=0), distances.min(axis=1)

        def surface(mask):
            padded = np.pad(mask, 1)
            interior = mask.copy()
            for axis in range(3):
                for offset in (-1, 1):
                    interior &= np.roll(padded, offset, axis=axis)[1:-1, 1:-1, 1:-1]
            return mask & ~interior

        f, b = directed(first, reference)
        self.assertAlmostEqual(
            result["artery_hd_foreground_mm"], max(f.max(), b.max()), places=5
        )
        self.assertAlmostEqual(
            result["artery_avghd_foreground_mm"], (f.mean() + b.mean()) / 2, places=5
        )
        f, b = directed(surface(first), surface(reference))
        self.assertAlmostEqual(
            result["artery_hd95_surface_mm"],
            max(np.percentile(f, 95), np.percentile(b, 95)),
        )
        self.assertNotIn("artery_ahd_mm", result)
        self.assertNotIn("artery_hd_mm", result)
        self.assertLess(
            result["artery_hd95_surface_mm"], result["artery_hd_foreground_mm"]
        )

    def test_foreground_spacing_matches_each_numpy_axis(self):
        first = np.zeros((4, 5, 6), dtype=bool)
        first[1, 1, 1] = True
        spacing = (0.25, 1.5, 3.0)
        for axis, step in enumerate(spacing):
            second = np.roll(first, 1, axis)
            result = evaluate_artery_distances(
                first, second, spacing, {"EVALUATION": {"distance_metrics": True}}
            )
            for column in (
                "artery_hd_foreground_mm",
                "artery_avghd_foreground_mm",
                "artery_hd95_surface_mm",
            ):
                self.assertAlmostEqual(result[column], step)

    def test_disabled_evaluation_does_not_compute_or_require_spacing(self):
        with patch(
            "utils.project.evaluation.artery_distances.hausdorff_distances"
        ) as calculate:
            result = evaluate_artery_distances(None, None, None, {})
        calculate.assert_not_called()
        self.assertEqual(result["artery_distance_metrics_status"], "disabled")
        self.assertIsNone(result["artery_hd_foreground_mm"])

    def test_evaluation_errors_are_separate_from_scientific_results(self):
        config = {"EVALUATION": {"distance_metrics": True}}
        for spacing in (None, (1, 0, 1)):
            result = evaluate_artery_distances(
                np.ones((3, 3, 3)), np.ones((3, 3, 3)), spacing, config
            )
            self.assertEqual(result["artery_distance_metrics_status"], "error")
            self.assertIn("ValueError", result["artery_distance_metrics_error"])
            self.assertIsNone(result["artery_hd_foreground_mm"])
            self.assertNotIn("error", result)
        with patch(
            "utils.project.evaluation.artery_distances.hausdorff_distances",
            side_effect=RuntimeError("synthetic"),
        ):
            result = evaluate_artery_distances(
                np.ones((2, 2, 2)), np.ones((2, 2, 2)), (1, 1, 1), config
            )
        self.assertIn("synthetic", result["artery_distance_metrics_error"])

    def test_rg_fc_masks_dice_and_counts_stay_identical_with_distances_enabled(self):
        raw = np.zeros((11, 11, 11), dtype=np.uint8)
        raw[4:7, 4:7, 4:7] = 1
        label = raw.copy()
        label[5, 5, 5] = 0
        config: dict[str, Any] = {
            "POSTPROCESSING": {"closing_radius": 1, "dilation_radius": 1}
        }
        for method in ("region_growing", "fuzzy_connectedness"):
            with (
                self.subTest(method=method),
                patch(
                    "utils.segmentation.pipeline.arteries._segment_with_region_growing",
                    return_value=(raw, {}),
                ),
                patch(
                    "utils.segmentation.pipeline.arteries._segment_with_fuzzy_connectedness",
                    return_value=(raw, {}),
                ),
            ):
                config["ARTERY_SEGMENTATION"] = {"method": method}
                disabled = segment_arteries_from_vesselness(
                    raw, label, raw, (5, 5, 5), None, config
                )
                enabled_config = {**config, "EVALUATION": {"distance_metrics": True}}
                enabled = segment_arteries_from_vesselness(
                    raw,
                    label,
                    raw,
                    (5, 5, 5),
                    None,
                    enabled_config,
                    spacing=(2, 1, 0.5),
                )
                for key in disabled:
                    if key in DISTANCE_RESULT_COLUMNS:
                        continue
                    if isinstance(disabled[key], np.ndarray):
                        np.testing.assert_array_equal(disabled[key], enabled[key])
                    else:
                        self.assertEqual(disabled[key], enabled[key])
                self.assertEqual(enabled["artery_distance_metrics_status"], "ok")
                self.assertGreater(enabled["artery_hd_foreground_mm"], 0)
                invalid = segment_arteries_from_vesselness(
                    raw, label, raw, (5, 5, 5), None, enabled_config
                )
                self.assertEqual(invalid["artery_distance_metrics_status"], "error")
                self.assertEqual(invalid["dice_artery"], enabled["dice_artery"])

    def test_ostia_wrapper_propagates_effective_spacing(self):
        mask = np.ones((3, 3, 3))
        with (
            patch(
                "utils.segmentation.pipeline.arteries.compute_vesselness",
                return_value=mask,
            ),
            patch(
                "utils.segmentation.pipeline.arteries.segment_arteries_from_vesselness",
                return_value={},
            ) as segment,
        ):
            segment_arteries_from_ostia(
                mask,
                mask,
                None,
                None,
                {"VESSELNESS_ARTERY": {}},
                spacing=(0.7, 1.2, 2.5),
            )
        self.assertEqual(segment.call_args.kwargs["spacing"], (0.7, 1.2, 2.5))

    def test_failure_before_segmentation_stays_not_evaluated(self):
        with patch(
            "utils.segmentation.pipeline.orchestration.load_and_preprocess_image",
            side_effect=OSError("synthetic"),
        ):
            result = process_image(
                1, {"EVALUATION": {"distance_metrics": True}}, "/unused"
            )
        self.assertEqual(result["artery_distance_metrics_status"], "not_evaluated")
        self.assertIsNone(result["artery_hd_foreground_mm"])
        self.assertIn("synthetic", result["error"])


class DistancePersistenceTests(TestCase):
    def test_csv_merge_resume_and_aliases_preserve_old_missing_and_valid_zero(self):
        measured = {
            "IMG_ID": 2,
            "artery_distance_protocol_version": 3,
            "artery_hd_foreground_mm": 1.5,
            "artery_avghd_foreground_mm": 0.25,
            "artery_hd95_surface_mm": 2.0,
            "artery_distance_metrics_status": "ok",
            "artery_distance_metrics_error": None,
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "results_train_lote_1.csv"
            pd.DataFrame(
                [
                    {
                        "IMG_ID": 1,
                        "artery_dice": 0.5,
                        "artery_hd_mm": 0.0,
                        "artery_ahd_mm": 0.0,
                    }
                ]
            ).to_csv(old, index=False)
            original = old.read_bytes()
            save_results([measured], "train_lote_2", root)
            previous, batches = _load_previous_batches(root, "train", 2)
            self.assertEqual(batches, [1, 2])
            self.assertNotIn("artery_hd_mm", previous[1])
            self.assertEqual(previous[0]["artery_hd_mm"], 0)
            merged_path = merge_batch_results("train", root)
            assert merged_path is not None
            frame = pd.read_csv(merged_path)
            self.assertTrue(pd.isna(frame.loc[1, "artery_hd_mm"]))
            self.assertTrue(pd.isna(frame.loc[0, "artery_distance_metrics_status"]))
            self.assertEqual(frame.loc[0, "artery_hd_mm"], 0)
            normalized = add_internal_result_aliases(frame)
            recreated = make_result_dataframe(normalized.to_dict("records"))
            self.assertEqual(recreated.loc[0, "artery_hd_mm"], 0)
            self.assertTrue(pd.isna(recreated.loc[1, "artery_hd_mm"]))
            for column in (
                "artery_hd_foreground_mm",
                "artery_avghd_foreground_mm",
                "artery_hd95_surface_mm",
            ):
                self.assertEqual(recreated.loc[1, column], measured[column])
                self.assertTrue(pd.isna(recreated.loc[0, column]))
            self.assertEqual(old.read_bytes(), original)

    def test_summaries_count_only_finite_ok_values_and_keep_status_denominators(self):
        frame = pd.DataFrame(
            {
                "IMG_ID": [1, 2, 3, 4, 5, 6],
                "artery_dice": [0.5] * 6,
                "artery_hd_mm": [0, 4, None, None, 999, np.inf],
                "artery_ahd_mm": [0, 2, None, None, 999, np.inf],
                "artery_distance_metrics_status": [
                    "ok",
                    "ok",
                    "prediction_empty",
                    None,
                    "error",
                    "ok",
                ],
            }
        )
        summary = summarize_artery_distances(frame)
        self.assertEqual(summary["artery_hd_mm"]["count"], 2)
        self.assertEqual(summary["artery_hd_mm"]["mean"], 2)
        self.assertEqual(summary["artery_hd_mm"]["median"], 2)
        self.assertAlmostEqual(summary["artery_hd_mm"]["std"], np.sqrt(8))
        self.assertEqual(summary["status_counts"]["ok"], 3)
        self.assertEqual(summary["missing_status_count"], 1)
        metadata = build_metadata(
            "train", {"EVALUATION": {"distance_metrics": True}}, results=frame
        )
        self.assertEqual(metadata["results"]["artery_distances"], summary)
        protocol = metadata["configuration"]["artery_distance_metrics"]
        self.assertEqual(protocol["unit"], "mm")
        self.assertEqual(protocol["stage"], "after_morphology")
        self.assertEqual(
            protocol["foreground_filter"], "SimpleITK.HausdorffDistanceImageFilter"
        )
        legacy = summarize_artery_distances(pd.DataFrame({"IMG_ID": [1, 2]}))
        self.assertEqual(legacy["missing_status_count"], 2)
        self.assertEqual(legacy["status_counts"], {})
        self.assertEqual(
            legacy["artery_hd_foreground_mm"],
            {"count": 0, "mean": None, "std": None, "median": None},
        )
