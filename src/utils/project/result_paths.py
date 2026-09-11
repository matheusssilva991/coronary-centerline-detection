"""Contrato central de nomes dos artefatos numéricos de um run."""

from __future__ import annotations

import re
from pathlib import Path


SPLIT_NAMES = ("train", "val", "test", "full")


def results_filename(split: str) -> str:
    return f"results_{split}.csv"


def batch_results_filename(split: str, batch_number: int) -> str:
    return f"results_{split}_lote_{batch_number}.csv"


def summary_filename(split: str) -> str:
    return f"summary_{split}.csv"


def metadata_filename(split: str) -> str:
    return f"metadata_{split}.json"


def batch_timings_filename(split: str) -> str:
    return f"batch_timings_{split}.csv"


def integrity_filename(split: str) -> str:
    return f"integrity_{split}.json"


def result_candidates(directory: Path, split: str) -> tuple[Path, ...]:
    """Retorna resultados em ordem de preferência, incluindo layouts legados."""
    return (
        directory / results_filename(split),
        directory / f"ostios_{split}_results.csv",
        directory / f"ostios_{split}_summary.csv",
    )


def summary_candidates(directory: Path, split: str) -> tuple[Path, ...]:
    return (
        directory / summary_filename(split),
        directory / f"ostios_{split}_summary.csv",
    )


def metadata_candidates(directory: Path, split: str) -> tuple[Path, ...]:
    return (
        directory / metadata_filename(split),
        directory / f"ostios_{split}_metadata.json",
    )


def batch_timings_candidates(directory: Path, split: str) -> tuple[Path, ...]:
    return (
        directory / batch_timings_filename(split),
        directory / f"ostios_{split}_batch_timings.csv",
    )


def batch_result_number(path: Path, split: str) -> int | None:
    """Extrai lote tanto do nome atual quanto do nome histórico."""
    escaped = re.escape(split)
    patterns = (
        rf"^results_{escaped}_lote_(\d+)\.csv$",
        rf"^ostios_{escaped}_lote_(\d+)_summary\.csv$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, path.name)
        if match:
            return int(match.group(1))
    return None


__all__ = [
    "SPLIT_NAMES",
    "batch_result_number",
    "batch_results_filename",
    "batch_timings_candidates",
    "batch_timings_filename",
    "integrity_filename",
    "metadata_candidates",
    "metadata_filename",
    "result_candidates",
    "results_filename",
    "summary_candidates",
    "summary_filename",
]
