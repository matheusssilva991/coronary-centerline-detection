"""Testa o carregamento integral de snapshots para comparações controladas."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import segmentation_pipeline as pipeline
from utils.project.config import serialize_config_for_json
from utils.segmentation.pipeline_cli import build_parser, parse_pipeline_args


class ConfigReplaceTests(TestCase):
    def test_replace_requires_config_file(self):
        with patch("sys.argv", ["pipeline", "--config-replace"]):
            with self.assertRaises(SystemExit) as error:
                parse_pipeline_args(Path("/dataset"), Path("/output"))
        self.assertEqual(error.exception.code, 2)

    def test_replace_changes_only_downscale_method(self):
        snapshot = json.loads(Path("config/pipeline_config.json").read_text())
        snapshot.pop("VISUAL_OUTPUT_DIR", None)
        snapshot["CIRCLE_DETECTION"].pop("trajectory_filter", None)
        current = {**snapshot, "NEW_DEFAULT": "must not leak"}

        with TemporaryDirectory() as temporary_dir:
            config_file = Path(temporary_dir) / "baseline.json"
            config_file.write_text(json.dumps(snapshot), encoding="utf-8")
            parser = build_parser(Path("/dataset"), Path("/output"))
            base_args = [
                "--split",
                "train",
                "--resolution",
                "mid",
                "--config-file",
                str(config_file),
                "--downscale-method",
                "scipy",
            ]
            with patch.object(pipeline, "CONFIG_MID_RES", current):
                merged = pipeline.build_effective_config(parser.parse_args(base_args))
                replaced = pipeline.build_effective_config(
                    parser.parse_args([*base_args, "--config-replace"])
                )

        self.assertEqual(merged["NEW_DEFAULT"], "must not leak")
        self.assertNotIn("NEW_DEFAULT", replaced)
        expected = dict(snapshot)
        expected["DOWNSCALE_METHOD"] = "scipy"
        self.assertEqual(serialize_config_for_json(replaced), expected)
