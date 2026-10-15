"""Confere que as EDAs usam apenas referências canônicas disponíveis."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from utils.project.runtime.notebook_env import (
    get_default_split_paths,
    get_refined_pipeline_split_paths,
)


class CanonicalSplitPathsTests(TestCase):
    def test_returns_only_canonical_results_and_preserves_public_alias(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            numeric = (
                root / "output/segmentation/canonical/mid_res/train/current/numeric"
            )
            numeric.mkdir(parents=True)
            (numeric / "results_train.csv").write_text("IMG_ID,artery_dice\n1,0.5\n")
            # Este run não pode substituir uma referência canônica ausente.
            unrelated = root / "output/segmentation/runs/mid_res/2099-01-01/numeric"
            unrelated.mkdir(parents=True)
            (unrelated / "results_val.csv").write_text("IMG_ID,artery_dice\n2,0.9\n")
            result = get_default_split_paths(root)
            self.assertEqual(result, {"mid_res": {"train": numeric}, "high_res": {}})
            self.assertEqual(get_refined_pipeline_split_paths(root), result)

    def test_follows_high_reference_link_without_copying_results(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs/high/reference"
            (run / "numeric").mkdir(parents=True)
            (run / "numeric/results_test.csv").write_text("IMG_ID,artery_dice\n1,0.5\n")
            parent = root / "output/segmentation/canonical/high_res/test"
            parent.mkdir(parents=True)
            (parent / "current").symlink_to(run, target_is_directory=True)
            result = get_default_split_paths(root)
            self.assertEqual(result["high_res"]["test"].resolve(), run / "numeric")

    def test_missing_canonical_never_uses_unrelated_run(self):
        with TemporaryDirectory() as directory:
            self.assertEqual(
                get_default_split_paths(Path(directory)),
                {"mid_res": {}, "high_res": {}},
            )
