from pathlib import Path

import pandas as pd

from ..project.results import add_internal_result_aliases
from ..project.result_paths import (
    batch_timings_candidates,
    metadata_candidates,
    result_candidates,
    summary_candidates,
)
from ..utils import load_json_file


def load_split_metadata(split_paths_by_resolution, resolution, subset_name):
    """Load metadata JSON for a given resolution/split, or None when unavailable."""
    # Busca a pasta da resolucao solicitada.
    split_paths = split_paths_by_resolution.get(resolution, {})
    if subset_name not in split_paths:
        # Subset ausente: nada para carregar.
        return None

    numeric_dir = Path(split_paths[subset_name])
    metadata_path = next(
        (
            path
            for path in metadata_candidates(numeric_dir, subset_name)
            if path.is_file()
        ),
        None,
    )
    return load_json_file(str(metadata_path)) if metadata_path else None


def _split_numeric_dir(split_paths_by_resolution, resolution, subset_name):
    """Resolve o diretório numérico de um split configurado."""
    # Busca a pasta da resolucao solicitada.
    split_paths = split_paths_by_resolution.get(resolution, {})
    if subset_name not in split_paths:
        return None
    return Path(split_paths[subset_name])


def load_split_results(split_paths_by_resolution, resolution, subset_name):
    """Carrega resultados por imagem, com fallback para runs legados."""
    numeric_dir = _split_numeric_dir(split_paths_by_resolution, resolution, subset_name)
    if numeric_dir is None:
        return None

    inspected: list[Path] = []
    for results_path in result_candidates(numeric_dir, subset_name):
        if not results_path.is_file():
            continue
        inspected.append(results_path)
        result = pd.read_csv(results_path)
        if "IMG_ID" in result.columns:
            return add_internal_result_aliases(result)
    raise FileNotFoundError(
        f"Resultados por imagem não encontrados para {resolution}/{subset_name}: "
        + ", ".join(
            str(path)
            for path in inspected or result_candidates(numeric_dir, subset_name)
        )
    )


def load_split_batch_timings(split_paths_by_resolution, resolution, subset_name):
    """Carrega o manifest de tempos do split, com fallback para o nome legado."""
    numeric_dir = _split_numeric_dir(split_paths_by_resolution, resolution, subset_name)
    if numeric_dir is None:
        return None
    timing_path = next(
        (
            path
            for path in batch_timings_candidates(numeric_dir, subset_name)
            if path.is_file()
        ),
        None,
    )
    return pd.read_csv(timing_path) if timing_path is not None else None


def load_split_summary(split_paths_by_resolution, resolution, subset_name):
    """Carrega um resumo legado; novos runs calculam agregados dos resultados."""
    numeric_dir = _split_numeric_dir(split_paths_by_resolution, resolution, subset_name)
    if numeric_dir is None:
        return None

    legacy_per_image = None
    for summary_path in summary_candidates(numeric_dir, subset_name):
        if not summary_path.is_file():
            continue
        summary = pd.read_csv(summary_path)
        if "IMG_ID" not in summary.columns:
            return summary
        legacy_per_image = summary_path
    if legacy_per_image is not None:
        raise ValueError(
            f"{legacy_per_image} usa o formato legado por imagem; "
            "use load_split_results()."
        )
    raise FileNotFoundError(
        f"Resumo agregado não encontrado para {resolution}/{subset_name}."
    )
