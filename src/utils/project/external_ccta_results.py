"""Carrega métricas quantitativas dos runs CCTA externos."""

from pathlib import Path

import numpy as np
import pandas as pd


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
    dice = pd.to_numeric(train["aorta_dice"], errors="coerce")
    if (
        dice.isna().any()
        or not np.isfinite(dice.to_numpy()).all()
        or not dice.between(0, 1).all()
    ):
        raise ValueError(f"{path}: Dice da aorta ausente ou fora de [0, 1]")
    return train.assign(aorta_dice=dice).set_index("exam_id")["aorta_dice"].sort_index()


__all__ = ["load_mmwhs_train_aorta_dice"]
