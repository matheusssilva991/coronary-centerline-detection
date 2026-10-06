"""Integra a avaliação oficial da aorta aos resultados dos runs externos."""

import logging
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd

from utils.project.results.external import has_valid_aorta_dice
from utils.project.evaluation.mmwhs_official_aorta import (
    OFFICIAL_AORTA_METHOD,
    has_aorta_mask,
    validate_saved_prediction,
    predict_aorta,
    verify_run_result,
    restore_native_mask,
    save_prediction,
    evaluate_with_wine,
    parse_aorta_dice,
    aorta_dice_result_path,
)

LOGGER = logging.getLogger(__name__)


def evaluate_test_aorta_result(
    row: Mapping[str, Any],
    record: pd.Series,
    config: dict[str, Any],
    run_dir: Path,
    evaluator_dir: Path,
    *,
    rebuild_mask: bool,
    align_volume: bool = True,
) -> dict[str, Any]:
    """Avalia a aorta sem alterar o resultado científico do exame."""
    updated = dict(row)
    exam_id = str(record["exam_id"])
    updated["aorta_evaluation_method"] = OFFICIAL_AORTA_METHOD
    updated["aorta_ground_truth_available"] = True
    updated["aorta_evaluation_error"] = None
    if not has_aorta_mask(row):
        updated["aorta_evaluation_status"] = "unavailable"
        updated["aorta_ground_truth_evaluated"] = False
        updated["aorta_dice"] = None
        return updated

    output_dir = run_dir / "evaluation" / "aorta" / "test" / exam_id
    prediction_path = output_dir / f"{exam_id}_label.nii.gz"
    try:
        reference_path = Path(str(record["path"]))
        if prediction_path.exists():
            validate_saved_prediction(reference_path, prediction_path)
        elif rebuild_mask:
            prediction = predict_aorta(
                record, {**config, "USE_GPU": False}, align_volume=align_volume
            )
            verify_run_result(run_dir, exam_id, prediction)
            native_mask = restore_native_mask(prediction)
            save_prediction(reference_path, native_mask, prediction_path)
            del native_mask, prediction
        else:
            raise FileNotFoundError(f"Predição da aorta ausente: {prediction_path}")
        dice_path = evaluate_with_wine(
            prediction_path, exam_id, evaluator_dir, output_dir
        )
        updated["aorta_dice"] = parse_aorta_dice(dice_path, exam_id)
        updated["aorta_ground_truth_evaluated"] = True
        updated["aorta_evaluation_status"] = "success"
    except Exception as error:
        LOGGER.exception("Falha na avaliação oficial da aorta de %s", exam_id)
        updated["aorta_dice"] = None
        updated["aorta_ground_truth_evaluated"] = False
        updated["aorta_evaluation_status"] = "error"
        updated["aorta_evaluation_error"] = str(error)
    return updated


def official_evaluation_complete(row: Mapping[str, Any], run_dir: Path) -> bool:
    """Confere que o Dice persistido corresponde ao arquivo oficial do exame."""
    if row.get("aorta_evaluation_method") != OFFICIAL_AORTA_METHOD:
        return False
    if row.get("aorta_evaluation_status") != "success" or not has_valid_aorta_dice(row):
        return False
    exam_id = str(row["exam_id"])
    dice_path = aorta_dice_result_path(
        run_dir / "evaluation" / "aorta" / "test" / exam_id, exam_id
    )
    if not dice_path.is_file():
        return False
    saved_dice = parse_aorta_dice(dice_path, exam_id)
    if not np.isclose(saved_dice, float(row["aorta_dice"]), rtol=0, atol=1e-6):
        raise ValueError(f"Dice do CSV diverge do arquivo oficial: {exam_id}")
    return True
