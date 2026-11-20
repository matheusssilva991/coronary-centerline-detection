"""Verifica a geometria da redução de rótulos no pipeline e no experimento."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

import segmentation_pipeline
from utils.processing.preprocessing import downscale_label
from utils.project.config import (
    IMAGE_GRID_LABEL_PROTOCOL,
    LEGACY_LABEL_PROTOCOL,
    load_config_json,
)
from utils.segmentation.pipeline.cli import build_parser
from utils.segmentation.pipeline.preprocessing import preprocess_ccta_volume


class LabelResamplingTests(TestCase):
    def test_mid_samples_expected_centers_and_preserves_intensities(self):
        # As classes identificam a coordenada original de cada plano.
        for size, chosen in [(8, [1, 3, 5, 7]), (7, [1, 3, 5])]:
            with self.subTest(size=size):
                label = np.broadcast_to(
                    np.arange(size, dtype=np.uint8)[:, None, None], (size, 6, 3)
                ).copy()
                image = label.astype(np.float32)
                config = {
                    "DOWNSCALE_FACTORS": [2, 2, 1],
                    "DOWNSCALE_METHOD": "opencv",
                    "OPENCV_INTERPOLATION": "linear",
                    "THRESHOLDING": {"method": "none"},
                }
                pipeline = preprocess_ccta_volume(
                    image, (1, 1, 1), config, label=label, include_intermediates=True
                )
                expected = label[chosen, 1::2, :]
                np.testing.assert_array_equal(pipeline["label"], expected)
                centers = (np.arange(len(chosen)) + 0.5) * size / len(chosen) - 0.5
                np.testing.assert_allclose(
                    pipeline["down_image"][:, 0, 0], centers, atol=1e-6
                )

    def test_snapshots_restore_the_protocol_and_new_runs_record_it(self):
        parser = build_parser(Path("/dataset"), Path("/output"))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            path = root / "config/effective_pipeline_config.json"
            for snapshot, expected in [
                ({}, LEGACY_LABEL_PROTOCOL),
                (
                    {"LABEL_RESAMPLING": IMAGE_GRID_LABEL_PROTOCOL},
                    IMAGE_GRID_LABEL_PROTOCOL,
                ),
            ]:
                path.write_text(json.dumps(snapshot))
                args = parser.parse_args(
                    ["--resume-dir", directory, "--resume-batch", "2"]
                )
                self.assertEqual(
                    segmentation_pipeline.build_effective_config(args)[
                        "LABEL_RESAMPLING"
                    ],
                    expected,
                )
                self.assertEqual(
                    load_config_json(path, {})["LABEL_RESAMPLING"], expected
                )
        fresh = segmentation_pipeline.build_effective_config(parser.parse_args([]))
        self.assertEqual(fresh["LABEL_RESAMPLING"], IMAGE_GRID_LABEL_PROTOCOL)

    def test_high_identity_and_alternative_sampling_conventions(self):
        label = np.arange(8, dtype=np.uint8).reshape(8, 1, 1)
        config = {"DOWNSCALE_FACTORS": [2, 1, 1], "DOWNSCALE_METHOD": "opencv"}
        for changes, indices in [
            ({"DOWNSCALE_FACTORS": [1, 1, 1]}, list(range(8))),
            ({"DOWNSCALE_METHOD": "scipy"}, [0, 2, 5, 7]),
            ({"LABEL_RESAMPLING": LEGACY_LABEL_PROTOCOL}, [0, 2, 5, 7]),
            ({"OPENCV_INTERPOLATION": "nearest"}, [0, 2, 4, 6]),
        ]:
            with self.subTest(changes=changes):
                np.testing.assert_array_equal(
                    downscale_label(label, config | changes), label[indices]
                )
