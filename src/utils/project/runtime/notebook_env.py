"""Configura o ambiente compartilhado dos notebooks exploratórios."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from utils.project.runtime.paths import find_repository_root

from utils.project.config import (
    load_config_json,
    scale_config_to_resolution,
)
from utils.project.results.paths import results_filename


def configure_notebook_environment(chdir_to_src: bool = True) -> Path:
    """Adiciona `src` ao caminho e opcionalmente altera o diretório atual.

    Retorna o caminho da raiz do repositório.
    """
    repo_root = find_repository_root(Path(__file__))
    src_dir = repo_root / "src"

    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)

    if chdir_to_src:
        os.chdir(src_dir)
    return repo_root


def resolve_existing_path(
    env_var: str,
    candidates: list[Path],
    description: str,
) -> Path:
    """Resolve um caminho por variável de ambiente ou candidatos locais."""
    env_value = os.environ.get(env_var)
    if env_value:
        path = Path(env_value).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(
            f"{description} definido em {env_var} não existe: {path}"
        )

    resolved = next((path for path in candidates if path.exists()), None)
    if resolved is not None:
        return resolved

    candidate_text = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        f"Nenhum caminho encontrado para {description}. "
        f"Exporte {env_var} ou ajuste os candidatos:\n{candidate_text}"
    )


def resolve_imagecas_base_path() -> Path:
    """Resolve o diretório ImageCAS com arquivos ``*.img.nii.gz``."""
    return resolve_existing_path(
        "IMAGECAS_BASE_PATH",
        [
            Path("/run/media/matheus/HD/DatasetsCCTA/ImageCAS/1-1000"),
            Path("/data04/home/mpmaia/ImageCAS/database/1-1000"),
            Path("/home/matheus/DatasetsCCTA/ImageCAS/1-1000"),
        ],
        "ImageCAS",
    )


def load_notebook_pipeline_config(
    config_file: str | Path,
    resolution: str = "mid",
) -> dict:
    """Carrega e escala a configuração usada em notebooks interativos.

    Preserva a ordem aplicada historicamente no ``main.ipynb``: carrega o
    JSON, força fatores unitários em alta resolução e, por fim, escala os
    parâmetros espaciais.
    """
    if resolution not in {"mid", "high"}:
        raise ValueError("resolution deve ser 'mid' ou 'high'.")

    config_path = Path(config_file)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuração não encontrada: {config_path}")

    config = load_config_json(str(config_path), {})
    if resolution == "high":
        config["DOWNSCALE_FACTORS"] = [1, 1, 1]

    return scale_config_to_resolution(config)


def _numeric_result_dir(path: Path) -> Path:
    """Resolve o subdiretório numérico ou preserva caminhos legados."""
    numeric_dir = path / "numeric"
    return numeric_dir if numeric_dir.exists() else path


def _latest_split_result_dir(parent: Path, split: str) -> Path | None:
    """Localiza o resultado consolidado mais recente de um split."""
    results_name = results_filename(split)
    candidates = []

    # Aceita tanto ``<split>/numeric`` quanto ``<split>/<timestamp>/numeric``.
    for run_dir in (parent, *sorted(parent.glob("*"))):
        if not run_dir.is_dir():
            continue
        numeric_dir = _numeric_result_dir(run_dir)
        legacy_summary = numeric_dir / f"ostios_{split}_summary.csv"
        has_results = (numeric_dir / results_name).is_file()
        if not has_results:
            has_results = (numeric_dir / f"ostios_{split}_results.csv").is_file()
        has_legacy_results = False
        if legacy_summary.is_file() and not has_results:
            try:
                has_legacy_results = (
                    "IMG_ID" in pd.read_csv(legacy_summary, nrows=0).columns
                )
            except (OSError, pd.errors.ParserError):
                has_legacy_results = False
        if has_results or has_legacy_results:
            candidates.append(numeric_dir)

    return max(candidates, key=lambda path: str(path)) if candidates else None


def _resolve_split_result_dir(
    repo_root: Path,
    resolution: str,
    split: str,
) -> Path | None:
    """Resolve um split priorizando runs canônicos e padrão."""
    canonical_parent = repo_root / "output/segmentation/canonical" / resolution / split
    canonical_result = _latest_split_result_dir(canonical_parent, split)
    if canonical_result is not None:
        return canonical_result

    # Inspeciona apenas runs datados diretos, sem entrar em experimentos.
    runs_parent = repo_root / "output/segmentation/runs" / resolution
    return _latest_split_result_dir(runs_parent, split)


def get_default_split_paths(repo_root: Path) -> dict[str, dict[str, Path]]:
    """Retorna pastas de resultados consolidados usadas pelas EDAs.

    Pastas canônicas podem conter um nível de data entre o split e ``numeric``.
    Combinações ausentes de resolução e split são omitidas.
    """
    result: dict[str, dict[str, Path]] = {"mid_res": {}, "high_res": {}}
    for resolution in result:
        for split in ("train", "val", "test"):
            resolved = _resolve_split_result_dir(
                repo_root,
                resolution,
                split,
            )
            if resolved is not None:
                result[resolution][split] = resolved

    return result
