"""Registro e consolidação dos tempos de execução por lote."""

from __future__ import annotations

import math
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd


type PathInput = str | PathLike[str]


BATCH_TIMING_COLUMNS: list[str] = [
    "split_name",
    "batch_number",
    "total_batches",
    "num_images",
    "first_img_id",
    "last_img_id",
    "result_file",
    "started_at",
    "finished_at",
    "duration_seconds",
    "duration_minutes",
    "duration_hours",
]


def duration_breakdown(duration_seconds: Any) -> dict[str, Any]:
    """Retorna duração em segundos, minutos e horas."""
    if duration_seconds is None:
        return {
            "seconds": None,
            "minutes": None,
            "hours": None,
        }

    try:
        seconds = float(duration_seconds)
    except (TypeError, ValueError):
        return {"seconds": None, "minutes": None, "hours": None}

    if math.isnan(seconds):
        return {"seconds": None, "minutes": None, "hours": None}
    return {
        "seconds": seconds,
        "minutes": seconds / 60,
        "hours": seconds / 3600,
    }


def batch_timing_manifest_path(output_dir: PathInput, split_name: str) -> Path:
    """Retorna o caminho do CSV com tempos por lote."""
    return Path(output_dir) / f"ostios_{split_name}_batch_timings.csv"


def load_batch_timing_records(
    output_dir: PathInput,
    split_name: str,
) -> list[dict[str, Any]]:
    """Carrega tempos por lote já salvos."""
    manifest_path = batch_timing_manifest_path(output_dir, split_name)
    if not manifest_path.exists():
        return []

    df = pd.read_csv(manifest_path)
    if df.empty:
        return []
    return [
        {str(key): value for key, value in record.items()}
        for record in df.to_dict("records")
    ]


def save_batch_timing_record(
    output_dir: PathInput,
    split_name: str,
    record: dict[str, Any],
) -> str:
    """Insere ou atualiza o tempo de um lote no manifest incremental."""
    manifest_path = batch_timing_manifest_path(output_dir, split_name)
    records = load_batch_timing_records(output_dir, split_name)
    batch_number = int(record["batch_number"])

    # Em retomadas, substitui o tempo antigo do lote em vez de duplicá-lo.
    records = [
        item for item in records if int(item.get("batch_number", -1)) != batch_number
    ]
    records.append(record)
    records = sorted(records, key=lambda item: int(item["batch_number"]))

    # Mantém a ordem das linhas e das colunas estável entre execuções.
    df = pd.DataFrame(records)
    df = df.reindex(columns=BATCH_TIMING_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    return str(manifest_path)


def summarize_batch_timing_records(
    records: list[dict[str, Any]],
    expected_batches: list[int] | None = None,
) -> dict[str, Any]:
    """Soma tempos conhecidos dos lotes e marca lotes sem tempo salvo."""
    if not records:
        return {
            "total_known_batches": 0,
            "total_known_duration_seconds": None,
            "total_known_duration_minutes": None,
            "total_known_duration_hours": None,
            "missing_timing_batches": expected_batches or [],
        }

    # Descarta durações ausentes ou inválidas antes de calcular o total.
    durations_by_batch: dict[int, float] = {}
    for record in records:
        batch_number = int(record["batch_number"])
        duration_value = record.get("duration_seconds")
        if duration_value is None:
            continue
        try:
            duration = float(duration_value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(duration):
            durations_by_batch[batch_number] = duration

    known_batches = sorted(durations_by_batch)
    total_seconds = sum(durations_by_batch.values()) if durations_by_batch else None

    # Identifica lotes antigos que ainda não possuíam registro de tempo.
    missing_timing_batches: list[int] = []
    if expected_batches is not None:
        missing_timing_batches = [
            batch_number
            for batch_number in expected_batches
            if batch_number not in durations_by_batch
        ]

    total = duration_breakdown(total_seconds)
    return {
        "total_known_batches": len(known_batches),
        "known_batches": known_batches,
        "total_known_duration_seconds": total["seconds"],
        "total_known_duration_minutes": total["minutes"],
        "total_known_duration_hours": total["hours"],
        "missing_timing_batches": missing_timing_batches,
    }
