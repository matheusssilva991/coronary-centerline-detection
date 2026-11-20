"""Opções, resultados e agregação das distâncias arteriais opcionais."""

from __future__ import annotations

import argparse
from typing import Any, Sequence

import numpy as np
import pandas as pd

from utils.project.dataframe import numeric_series, require_series_column
from utils.utils.metrics import foreground_hausdorff_distances, surface_hd95

ADDITIONAL_DISTANCE_METRICS = {
    "artery_hd_foreground_mm": "HD da máscara inteira",
    "artery_avghd_foreground_mm": "AvgHD da máscara inteira",
    "artery_hd95_surface_mm": "HD95 de superfície",
}
LEGACY_DISTANCE_COLUMNS = ("artery_hd_mm", "artery_ahd_mm")
DISTANCE_METRIC_COLUMNS = tuple(ADDITIONAL_DISTANCE_METRICS)
DISTANCE_RESULT_COLUMNS = (
    *DISTANCE_METRIC_COLUMNS,
    "artery_distance_protocol_version",
    "artery_distance_metrics_status",
    "artery_distance_metrics_error",
)
DISTANCE_PROTOCOL = {
    "unit": "mm",
    "stage": "after_morphology",
    "version": 3,
    "surface_connectivity": 6,
    "exterior_is_background": True,
    "foreground_filter": "SimpleITK.HausdorffDistanceImageFilter",
    "foreground_point_set": "nonzero_voxel_centers",
    "hd95_surface_definition": "max_of_directional_95th_percentiles_linear",
}


def add_distance_metric_arguments(parser: argparse.ArgumentParser) -> None:
    """Adiciona a flag ``--distance-metrics`` sem sobrescrever o padrão atual.

    Args:
        parser (argparse.ArgumentParser): Instância de ``ArgumentParser`` onde a
            flag é registrada.
    """
    parser.add_argument(
        "--distance-metrics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Calcula distâncias de Hausdorff de volume e superfície em mm na máscara arterial final (custo adicional).",
    )


def distance_metric_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Descreve a convenção das distâncias arteriais sem equivalência com IA.

    Args:
        config (dict[str, Any]): Configuração contendo a seção opcional ``EVALUATION``.

    Returns:
        dict[str, Any]: Dicionário com ``enabled`` e os campos do protocolo
            ``DISTANCE_PROTOCOL`` aplicado.
    """
    return {
        "enabled": bool(
            (config.get("EVALUATION") or {}).get("distance_metrics", False)
        ),
        **(config.get("EVALUATION") or {}).get(
            "distance_metrics_protocol", DISTANCE_PROTOCOL
        ),
    }


def apply_distance_metric_options(config: dict[str, Any], args: Any) -> None:
    """Aplica ``--distance-metrics`` quando explícito e grava o protocolo em ``config``.

    Args:
        config (dict[str, Any]): Configuração atualizada in-place na chave
            ``EVALUATION``.
        args (Any): Namespace CLI contendo o atributo opcional ``distance_metrics``.
    """
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
    """Inicializa colunas de distância distinguindo desativação de não avaliação.

    Args:
        config (dict[str, Any]): Configuração efetiva usada para verificar se
            ``EVALUATION.distance_metrics`` está ativo.

    Returns:
        dict[str, Any]: Dicionário com colunas ``artery_hd_*`` zeradas como ``None``,
            versão e status inicial.
    """
    enabled = bool((config.get("EVALUATION") or {}).get("distance_metrics", False))
    return {
        **dict.fromkeys(DISTANCE_METRIC_COLUMNS),
        "artery_distance_protocol_version": DISTANCE_PROTOCOL["version"]
        if enabled
        else None,
        "artery_distance_metrics_status": "not_evaluated" if enabled else "disabled",
        "artery_distance_metrics_error": None,
    }


def evaluate_artery_distances(
    prediction: Any,
    reference: Any,
    spacing: Sequence[float] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Calcula distâncias físicas arteriais isolando falhas sem descartar máscaras.

    Args:
        prediction (Any): Máscara binária 3D predita em ordem de eixos ``(y, x, z)``
            após morfologia.
        reference (Any): Máscara binária 3D de referência no mesmo shape.
        spacing (Sequence[float] | None): Espaçamento físico em ``mm`` na ordem
            ``(dy, dx, dz)``, ou ``None`` (falha registrada como erro).
        config (dict[str, Any]): Configuração efetiva que define
            ``EVALUATION.distance_metrics``.

    Returns:
        dict[str, Any]: Dicionário com ``artery_hd_foreground_mm``,
            ``artery_avghd_foreground_mm``, ``artery_hd95_surface_mm``,
            ``artery_distance_protocol_version``, ``artery_distance_metrics_status``
            e ``artery_distance_metrics_error``.
    """
    result = pending_artery_distance_metrics(config)
    if result["artery_distance_metrics_status"] == "disabled":
        return result
    try:
        # Etapa 1: valida dimensões das máscaras e espaçamento físico positivo em mm.
        if spacing is None:
            raise ValueError(
                "Espaçamento efetivo ausente para calcular distâncias em mm."
            )
        pred = np.asarray(prediction) > 0
        target = np.asarray(reference) > 0
        voxel_spacing = np.asarray(spacing, dtype=float)
        if (
            pred.ndim != 3
            or pred.shape != target.shape
            or any(size == 0 for size in pred.shape)
        ):
            raise ValueError(
                "As máscaras devem ser volumes 3D com o mesmo shape não vazio."
            )
        if (
            voxel_spacing.shape != (3,)
            or not np.isfinite(voxel_spacing).all()
            or np.any(voxel_spacing <= 0)
        ):
            raise ValueError(
                "O espaçamento deve conter três valores positivos e finitos em mm."
            )

        # Etapa 2: registra casos com máscara vazia sem erro de distância indefinida.
        pred_exists, target_exists = bool(pred.any()), bool(target.any())
        if not pred_exists or not target_exists:
            status = (
                "both_empty"
                if not pred_exists and not target_exists
                else "prediction_empty"
                if not pred_exists
                else "reference_empty"
            )
            result["artery_distance_metrics_status"] = status
            return result

        # Etapa 3: calcula HD/AvgHD na máscara e HD95 simétrico em superfície (mm).
        hd, average = foreground_hausdorff_distances(pred, target, spacing)
        result.update(
            artery_hd_foreground_mm=hd,
            artery_avghd_foreground_mm=average,
            artery_hd95_surface_mm=surface_hd95(pred, target, spacing),
            artery_distance_metrics_status="ok",
        )
    except Exception as error:
        # Isola falhas de validação ou cálculo preservando demais métricas do exame.
        result["artery_distance_metrics_status"] = "error"
        result["artery_distance_metrics_error"] = f"{type(error).__name__}: {error}"
    return result


def summarize_artery_distances(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Resume medidas finitas avaliadas em mm, sem preencher falhas ou runs antigos.

    Args:
        dataframe (pd.DataFrame): DataFrame com resultados por imagem contendo
            ``artery_distance_metrics_status`` e colunas de métricas.

    Returns:
        dict[str, Any]: Dicionário com ``status_counts``, ``missing_status_count``
            e estatísticas (``count``, ``mean``, ``std``, ``median``) por coluna.
    """
    # Contabiliza os status de avaliação e registros ainda sem status na coorte.
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

    # Agrega apenas valores finitos e não negativos das linhas com status "ok".
    for column in (*DISTANCE_METRIC_COLUMNS, *LEGACY_DISTANCE_COLUMNS):
        if column in LEGACY_DISTANCE_COLUMNS and column not in dataframe:
            continue
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
