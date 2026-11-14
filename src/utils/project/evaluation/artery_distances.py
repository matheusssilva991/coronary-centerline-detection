"""Opções, resultados e agregação das distâncias arteriais opcionais."""

from __future__ import annotations

import argparse
from typing import Any, Sequence

import numpy as np
import pandas as pd

from utils.project.dataframe import numeric_series, require_series_column
from utils.utils.metrics import surface_distance_metrics


DISTANCE_RESULT_COLUMNS = (
    "artery_hd_mm",
    "artery_ahd_mm",
    "artery_distance_metrics_status",
    "artery_distance_metrics_error",
)
DISTANCE_PROTOCOL = {
    "unit": "mm",
    "stage": "after_morphology",
    "point_set": "surface_voxel_centers",
    "surface_connectivity": 6,
    "exterior_is_background": True,
    "hd_definition": "max_of_bidirectional_nearest_distances",
    "ahd_definition": "arithmetic_mean_of_directed_means",
}


def add_distance_metric_arguments(parser: argparse.ArgumentParser) -> None:
    """Adiciona flags sem sobrescrever a opção da configuração por omissão."""
    parser.add_argument(
        "--distance-metrics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Calcula HD e AHD em mm na máscara arterial final (custo adicional).",
    )


def distance_metric_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Descreve a convenção, sem presumir equivalência com métricas de IA."""
    return {
        "enabled": bool(
            (config.get("EVALUATION") or {}).get("distance_metrics", False)
        ),
        **DISTANCE_PROTOCOL,
    }


def apply_distance_metric_options(config: dict[str, Any], args: Any) -> None:
    """Aplica somente uma flag explícita e registra o protocolo no snapshot."""
    override = getattr(args, "distance_metrics", None)
    if "EVALUATION" not in config and override is None:
        return
    evaluation = dict(config.get("EVALUATION") or {})
    if override is not None:
        evaluation["distance_metrics"] = bool(override)
    evaluation.setdefault("distance_metrics", False)
    evaluation["distance_metrics_protocol"] = dict(DISTANCE_PROTOCOL)
    config["EVALUATION"] = evaluation


def pending_artery_distance_metrics(config: dict[str, Any]) -> dict[str, Any]:
    """Distingue avaliação desativada de segmentação ainda não executada."""
    enabled = bool((config.get("EVALUATION") or {}).get("distance_metrics", False))
    return {
        "artery_hd_mm": None,
        "artery_ahd_mm": None,
        "artery_distance_metrics_status": "not_evaluated" if enabled else "disabled",
        "artery_distance_metrics_error": None,
    }


def evaluate_artery_distances(
    prediction: Any,
    reference: Any,
    spacing: Sequence[float] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Isola falhas da avaliação, sem descartar máscaras ou métricas existentes."""
    result = pending_artery_distance_metrics(config)
    if result["artery_distance_metrics_status"] == "disabled":
        return result
    try:
        if spacing is None:
            raise ValueError(
                "Espaçamento efetivo ausente para calcular distâncias em mm."
            )
        measured = surface_distance_metrics(prediction, reference, spacing)
        result.update(
            artery_hd_mm=measured["hd_mm"],
            artery_ahd_mm=measured["ahd_mm"],
            artery_distance_metrics_status=measured["status"],
        )
    except Exception as error:
        result["artery_distance_metrics_status"] = "error"
        result["artery_distance_metrics_error"] = f"{type(error).__name__}: {error}"
    return result


def summarize_artery_distances(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Resume só medidas finitas avaliadas, sem preencher falhas ou runs antigos."""
    status_column = "artery_distance_metrics_status"
    statuses = (
        require_series_column(dataframe, status_column)
        if status_column in dataframe.columns
        else pd.Series(index=dataframe.index, dtype=object)
    )
    known_statuses = statuses.dropna()
    summary: dict[str, Any] = {
        "status_counts": {
            str(key): int(value) for key, value in known_statuses.value_counts().items()
        },
        "missing_status_count": int(statuses.isna().sum()),
    }
    for column in ("artery_hd_mm", "artery_ahd_mm"):
        values = (
            numeric_series(dataframe, column).where(statuses.eq("ok"))
            if column in dataframe.columns
            else pd.Series(dtype=float)
        )
        valid = values.where(np.isfinite(values) & values.ge(0)).dropna()
        summary[column] = {
            "count": int(len(valid)),
            "mean": float(valid.mean()) if len(valid) else None,
            "std": float(np.asarray(valid.std(ddof=1)).item())
            if len(valid) > 1
            else None,
            "median": float(valid.median()) if len(valid) else None,
        }
    return summary
