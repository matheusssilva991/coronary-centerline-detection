"""Carrega métricas quantitativas dos runs CCTA externos."""

from pathlib import Path

import numpy as np
import pandas as pd

from utils.project.dataframe import numeric_series, require_series_column
from utils.project.mmwhs_official_aorta import OFFICIAL_AORTA_METHOD


def load_mmwhs_train_aorta_dice(results_path: str | Path) -> pd.Series:
    """Carrega o Dice da aorta dos exames de treino MM-WHS avaliados."""
    path = Path(results_path)
    if not path.is_file():
        raise FileNotFoundError(f"Resultados MM-WHS ausentes: {path}")

    frame = pd.read_csv(path)
    required = {
        "dataset",
        "subset",
        "exam_id",
        "aorta_ground_truth_evaluated",
        "aorta_dice",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: colunas ausentes: {missing}")
    train = frame.loc[
        frame["subset"].eq("train") & frame["dataset"].eq("MM-WHS")
    ].copy()
    if (
        train.empty
        or train["exam_id"].isna().any()
        or train["exam_id"].duplicated().any()
    ):
        raise ValueError(f"{path}: IDs de treino ausentes ou duplicados")
    if not train["aorta_ground_truth_evaluated"].eq(True).all():
        raise ValueError(f"{path}: há exames de treino sem avaliação da aorta")
    dice = numeric_series(train, "aorta_dice")
    if (
        dice.isna().any()
        or not np.isfinite(dice.to_numpy()).all()
        or not dice.between(0, 1).all()
    ):
        raise ValueError(f"{path}: Dice da aorta ausente ou fora de [0, 1]")
    return train.assign(aorta_dice=dice).set_index("exam_id")["aorta_dice"].sort_index()


def load_mmwhs_test_aorta_dice(results_path: str | Path) -> pd.Series:
    """Carrega o Dice oficial do MM-WHS test, mantendo falhas como ausentes."""
    path = Path(results_path)
    if not path.is_file():
        raise FileNotFoundError(f"Resultados MM-WHS ausentes: {path}")
    frame = pd.read_csv(path)
    required = {
        "dataset",
        "subset",
        "exam_id",
        "aorta_dice",
        "aorta_evaluation_method",
        "aorta_evaluation_status",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: colunas ausentes: {missing}")
    test = frame.loc[frame["subset"].eq("test") & frame["dataset"].eq("MM-WHS")].copy()
    if test.empty or test["exam_id"].isna().any() or test["exam_id"].duplicated().any():
        raise ValueError(f"{path}: IDs de teste ausentes ou duplicados")
    if (
        not require_series_column(test, "aorta_evaluation_method")
        .eq(OFFICIAL_AORTA_METHOD)
        .all()
    ):
        raise ValueError(f"{path}: avaliação oficial ausente em exames de teste")
    status = require_series_column(test, "aorta_evaluation_status")
    if not status.isin(["success", "unavailable"]).all():
        raise ValueError(f"{path}: há avaliações oficiais pendentes ou com erro")
    dice = numeric_series(test, "aorta_dice")
    valid = dice.notna() & np.isfinite(dice) & dice.between(0, 1)
    if not valid.eq(status.eq("success")).all():
        raise ValueError(f"{path}: Dice oficial incompatível com o estado da avaliação")
    return test.assign(aorta_dice=dice).set_index("exam_id")["aorta_dice"].sort_index()


__all__ = ["load_mmwhs_train_aorta_dice", "load_mmwhs_test_aorta_dice"]
