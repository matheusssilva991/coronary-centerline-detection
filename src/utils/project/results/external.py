"""Persiste resultados externos, resume execuções e carrega métricas para EDA."""

from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import pandas as pd

from utils.project.dataframe import numeric_series, require_series_column
from utils.project.evaluation.mmwhs_official_aorta import (
    LEGACY_WHS_METHOD,
    OFFICIAL_AORTA_METHOD,
)
from utils.project.results.io import save_dataframe_atomic
from utils.project.results.timing import duration_breakdown

AORTA_EVALUATION_LABELS = {
    OFFICIAL_AORTA_METHOD: "Aorta isolada · label 820 · 1 mm",
    LEGACY_WHS_METHOD: "Legado WHS · coração completo · não corrigido",
}


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
    """Carrega um protocolo homogêneo e o identifica nos atributos da série."""
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
    methods = require_series_column(test, "aorta_evaluation_method")
    if not methods.isin(tuple(AORTA_EVALUATION_LABELS)).all():
        raise ValueError(f"{path}: protocolo da avaliação ausente ou desconhecido")
    if methods.nunique() != 1:
        raise ValueError(
            f"{path}: protocolos misturados; conclua a reavaliação da aorta"
        )
    method = str(methods.iloc[0])
    status = require_series_column(test, "aorta_evaluation_status")
    if not status.isin(["success", "unavailable"]).all():
        raise ValueError(f"{path}: há avaliações oficiais pendentes ou com erro")
    dice = numeric_series(test, "aorta_dice")
    valid = dice.notna() & np.isfinite(dice) & dice.between(0, 1)
    if not valid.eq(status.eq("success")).all():
        raise ValueError(f"{path}: Dice oficial incompatível com o estado da avaliação")
    result = (
        test.assign(aorta_dice=dice).set_index("exam_id")["aorta_dice"].sort_index()
    )
    result.attrs["aorta_evaluation_method"] = method
    result.attrs["aorta_evaluation_label"] = AORTA_EVALUATION_LABELS[method]
    return result


__all__ = [
    "load_mmwhs_train_aorta_dice",
    "load_mmwhs_test_aorta_dice",
    "external_result_key",
    "upsert_external_result",
    "save_numeric_results",
    "load_existing_results",
    "build_external_metadata",
    "has_valid_aorta_dice",
    "is_terminal_external_result",
]


def external_result_key(row: Mapping[str, Any]) -> tuple[str, str]:
    """Retorna a chave única de conjunto e exame de um resultado."""
    return str(row["subset"]), str(row["exam_id"])


def _result_sort_key(row: Mapping[str, Any]) -> tuple[int, str]:
    """Retorna a chave determinística de ordenação de um resultado."""
    subset_order = {"train": 0, "test": 1}
    return subset_order.get(str(row["subset"]), 2), str(row["exam_id"])


def upsert_external_result(
    rows: list[dict[str, Any]],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Insere ou substitui um resultado preservando a ordem determinística."""
    key = external_result_key(result)
    updated = [row for row in rows if external_result_key(row) != key]
    updated.append(result)
    return sorted(updated, key=_result_sort_key)


def save_numeric_results(rows: list[dict[str, Any]], numeric_dir: Path) -> None:
    """Salva resultados consolidados e por subset de forma atômica."""
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return
    subset_values = require_series_column(dataframe, "subset")
    subset_order = subset_values.map(
        lambda value: {"train": 0, "test": 1}.get(str(value))
    ).fillna(2)
    dataframe = (
        dataframe.assign(_subset_order=subset_order)
        .sort_values(["_subset_order", "exam_id"], kind="stable")
        .drop(columns="_subset_order")
    )
    save_dataframe_atomic(dataframe, numeric_dir / "results_all.csv")
    for subset, subset_frame in dataframe.groupby("subset", sort=False):
        save_dataframe_atomic(
            subset_frame.reset_index(drop=True),
            numeric_dir / f"results_{subset}.csv",
        )


def load_existing_results(numeric_dir: Path) -> list[dict[str, Any]]:
    """Carrega resultados já persistidos para permitir retomadas."""
    path = numeric_dir / "results_all.csv"
    if not path.is_file():
        return []
    records = pd.read_csv(path).where(pd.notna, None).to_dict(orient="records")
    return [{str(key): value for key, value in record.items()} for record in records]


def build_external_metadata(
    *,
    dataset: str,
    resolution: str,
    subset: str,
    rows: list[dict[str, Any]],
    started_at: datetime,
    state: str,
) -> dict[str, Any]:
    """Monta o metadata compacto da execução externa."""
    status_counts = pd.Series(
        [row.get("status", "unknown") for row in rows]
    ).value_counts()
    total_seconds = sum(float(row.get("execution_time_seconds") or 0.0) for row in rows)
    aorta_dice_values = [
        float(value)
        for row in rows
        if str(row.get("subset")) == "train"
        if isinstance(
            value := row.get("aorta_dice"),
            (int, float, np.integer, np.floating),
        )
        and np.isfinite(float(value))
    ]
    official_test_rows = [
        row
        for row in rows
        if str(row.get("subset")) == "test"
        and row.get("aorta_evaluation_method") == OFFICIAL_AORTA_METHOD
    ]
    official_test_dice = [
        float(row["aorta_dice"])
        for row in official_test_rows
        if has_valid_aorta_dice(row) and row.get("aorta_evaluation_status") == "success"
    ]
    return {
        "schema_version": 3,
        "dataset": dataset,
        "resolution": resolution,
        "selected_subset": subset,
        "state": state,
        "started_at": started_at.isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "processed_exam_count": len(rows),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "execution_time": duration_breakdown(total_seconds),
        "ground_truth_metrics": {
            "aorta": {
                "available_exam_count": sum(
                    str(row.get("subset")) == "train"
                    and row.get("aorta_ground_truth_available") is True
                    for row in rows
                ),
                "evaluated_exam_count": len(aorta_dice_values),
                "dice_mean": (
                    float(np.mean(aorta_dice_values)) if aorta_dice_values else None
                ),
                "test_official": {
                    "method": OFFICIAL_AORTA_METHOD,
                    "legacy_exam_count": sum(
                        row.get("subset") == "test"
                        and row.get("aorta_evaluation_method") == LEGACY_WHS_METHOD
                        for row in rows
                    ),
                    "available_exam_count": len(official_test_rows),
                    "evaluated_exam_count": len(official_test_dice),
                    "unavailable_exam_count": sum(
                        row.get("aorta_evaluation_status") == "unavailable"
                        for row in official_test_rows
                    ),
                    "error_exam_count": sum(
                        row.get("aorta_evaluation_status") == "error"
                        for row in official_test_rows
                    ),
                    "dice_mean": (
                        float(np.mean(official_test_dice))
                        if official_test_dice
                        else None
                    ),
                },
            },
            "ostia_accuracy": None,
            "coronary_reason": (
                "Os bancos externos não possuem referência coronariana compatível."
            ),
        },
    }


def has_valid_aorta_dice(row: Mapping[str, Any]) -> bool:
    """Verifica se um resultado possui Dice da aorta finito e normalizado."""
    value = row.get("aorta_dice")
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and np.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def is_terminal_external_result(
    row: Mapping[str, Any],
    aorta_ground_truth_keys: set[tuple[str, str]],
) -> bool:
    """Valida conclusão do exame, exigindo Dice quando há referência."""
    if str(row.get("status")) not in {"success", "ostia_not_found"}:
        return False
    key = external_result_key(row)
    return key not in aorta_ground_truth_keys or has_valid_aorta_dice(row)
