"""Exporta e avalia a aorta prevista de um CT test do MM-WHS."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.project.ccta_datasets import discover_ccta_dataset  # noqa: E402
from utils.project.config import load_config_json  # noqa: E402
from utils.project.dataframe import require_series_column  # noqa: E402
from utils.project.mmwhs_official_aorta import (  # noqa: E402
    EVALUATOR_FOLDER,
    evaluate_with_wine,
    parse_dice_lo,
    predict_aorta,
    restore_native_mask,
    save_prediction,
    validate_saved_prediction,
    verify_run_result,
)

EXAM_ID_PATTERN = re.compile(r"ct_test_20\d{2}\Z")


def build_parser() -> argparse.ArgumentParser:
    """Monta a CLI para um único exame CT de teste."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=os.environ.get("MMWHS_BASE_PATH"),
        help="Raiz do MM-WHS; aceita MMWHS_BASE_PATH.",
    )
    parser.add_argument("--exam-id", required=True, help="Ex.: ct_test_2001.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--evaluator-dir",
        type=Path,
        help="Pasta do avaliador criptografado; por padrão, dentro do dataset.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--evaluate",
        action="store_true",
        help="Exporta a máscara e chama o avaliador oficial pelo Wine.",
    )
    mode.add_argument(
        "--evaluate-existing",
        action="store_true",
        help="Usa a máscara já exportada para tentar novamente somente o Wine.",
    )
    return parser


def load_run_config(run_dir: Path, exam_id: str) -> tuple[dict[str, Any], str]:
    """Carrega o snapshot efetivo e valida a identidade do run."""
    config_path = run_dir / "config/effective_pipeline_config.json"
    manifest_path = run_dir / "config/run_manifest.json"
    if not config_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "O run precisa conter config/effective_pipeline_config.json "
            "e config/run_manifest.json."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolution = manifest.get("resolution")
    if manifest.get("dataset") != "mmwhs" or resolution not in {"mid", "high"}:
        raise ValueError("O run deve ser MM-WHS em resolução mid ou high.")
    selected = manifest.get("selected_exams")
    if not isinstance(selected, list) or not any(
        isinstance(record, dict)
        and record.get("subset") == "test"
        and record.get("exam_id") == exam_id
        for record in selected
    ):
        raise ValueError(f"{exam_id} não pertence à seleção de teste desse run.")
    config = load_config_json(str(config_path), {})
    expected_factors = (1, 1, 1) if resolution == "high" else (2, 2, 1)
    if tuple(config["DOWNSCALE_FACTORS"]) != expected_factors:
        raise ValueError(
            f"Downscale do snapshot diverge da resolução {resolution}: "
            f"{config['DOWNSCALE_FACTORS']}."
        )
    config["USE_GPU"] = False
    return config, resolution


def select_test_record(dataset_root: Path, exam_id: str) -> pd.Series:
    """Seleciona exatamente um CT test do inventário MM-WHS."""
    if not EXAM_ID_PATTERN.fullmatch(exam_id):
        raise ValueError("Use um ID como ct_test_2001.")
    inventory = discover_ccta_dataset("mmwhs", dataset_root)
    matches = inventory.loc[
        require_series_column(inventory, "subset").eq("test")
        & require_series_column(inventory, "exam_id").eq(exam_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Esperado um único exame de teste {exam_id}; encontrados {len(matches)}."
        )
    return matches.iloc[0]


def main(argv: list[str] | None = None) -> int:
    """Executa exportação e, opcionalmente, avaliação de um exame."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dataset_root is None:
        parser.error("Informe --dataset-root ou MMWHS_BASE_PATH.")
    dataset_root = Path(args.dataset_root).expanduser()
    run_dir = args.run_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Banco MM-WHS não encontrado: {dataset_root}")
    config, resolution = load_run_config(run_dir, args.exam_id)
    record = select_test_record(dataset_root, args.exam_id)
    image_path = Path(str(record["path"]))
    prediction_path = output_dir / f"{args.exam_id}_label.nii.gz"
    evaluator_dir = args.evaluator_dir or dataset_root / EVALUATOR_FOLDER

    if args.evaluate_existing:
        if not prediction_path.is_file():
            raise FileNotFoundError(
                f"Predição exportada não encontrada: {prediction_path}"
            )
        validate_saved_prediction(image_path, prediction_path)
    else:
        if prediction_path.exists():
            raise FileExistsError(f"Predição já existe: {prediction_path}")
        prediction = predict_aorta(record, config)
        verify_run_result(run_dir, args.exam_id, prediction)
        native_mask = restore_native_mask(prediction)
        save_prediction(image_path, native_mask, prediction_path)
        validate_saved_prediction(image_path, prediction_path)
        print(
            f"Aorta {resolution} exportada: {prediction_path} "
            f"({int(native_mask.sum())} voxels na geometria original)"
        )

    if args.evaluate or args.evaluate_existing:
        dice_path = evaluate_with_wine(
            prediction_path, args.exam_id, evaluator_dir, output_dir
        )
        print(f"Avaliação oficial salva em: {dice_path}")
        print(
            f"DiceLO (aorta, label 820): {parse_dice_lo(dice_path, args.exam_id):.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
