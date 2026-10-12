"""Testes dos caminhos de resultados selecionados para as EDAs."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from utils.project.runtime.notebook_env import get_refined_pipeline_split_paths


class RefinedPipelineSplitPathsTests(TestCase):
    """Valida a resolução dos runs refinados sem depender do workspace real."""

    def test_returns_only_complete_split_results(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            train_numeric = repo_root / "runs/mid/train/run/numeric"
            train_numeric.mkdir(parents=True)
            (train_numeric / "results_train.csv").write_text(
                "IMG_ID,artery_dice\n1,0.5\n",
                encoding="utf-8",
            )

            selected_runs = {
                "mid_res": {
                    "train": "runs/mid/train/run",
                    "val": "runs/mid/val/missing",
                },
                "high_res": {},
            }
            with patch(
                "utils.project.runtime.notebook_env._REFINED_PIPELINE_RUNS",
                selected_runs,
            ):
                result = get_refined_pipeline_split_paths(repo_root)

        self.assertEqual(result["mid_res"], {"train": train_numeric})
        self.assertEqual(result["high_res"], {})
