"""Notebook bootstrap helpers for exploratory notebooks."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .config import (
    apply_aorta_ostia_method,
    load_config_json,
    scale_config_to_resolution,
)


def configure_notebook_environment(chdir_to_src: bool = True) -> Path:
    """Add src to sys.path and optionally switch cwd to the src directory.

    Returns:
        Path to the repository root.
    """
    repo_root = Path(__file__).resolve().parents[3]
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
    """Resolve a dataset/output path from an env var or known local candidates."""
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
            Path("/media/matheus/HD/DatasetsCCTA/ImageCAS/1-1000"),
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
    JSON, força fatores unitários em alta resolução, aplica um perfil de
    aorta/óstios não padrão e, por fim, escala os parâmetros espaciais.
    """
    if resolution not in {"mid", "high"}:
        raise ValueError("resolution deve ser 'mid' ou 'high'.")

    config_path = Path(config_file)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuração não encontrada: {config_path}")

    config = load_config_json(str(config_path), {})
    if resolution == "high":
        config["DOWNSCALE_FACTORS"] = [1, 1, 1]

    selected_method = config.get("AORTA_OSTIA_METHOD", {}).get(
        "method", "standard"
    )
    if selected_method != "standard":
        config = apply_aorta_ostia_method(config, method=selected_method)

    return scale_config_to_resolution(config)


def _numeric_result_dir(path: Path) -> Path:
    """Return the numeric subdir for new runs, otherwise keep legacy paths."""
    numeric_dir = path / "numeric"
    return numeric_dir if numeric_dir.exists() else path


def _latest_split_result_dir(parent: Path, split: str) -> Path | None:
    """Find the newest consolidated result below a split/run directory."""
    summary_name = f"ostios_{split}_summary.csv"
    candidates = []

    # Supports both ``<split>/numeric`` and ``<split>/<timestamp>/numeric``.
    for run_dir in (parent, *sorted(parent.glob("*"))):
        if not run_dir.is_dir():
            continue
        numeric_dir = _numeric_result_dir(run_dir)
        if (numeric_dir / summary_name).is_file():
            candidates.append(numeric_dir)

    return max(candidates, key=lambda path: str(path)) if candidates else None


def _resolve_split_result_dir(
    repo_root: Path,
    resolution: str,
    split: str,
) -> Path | None:
    """Resolve one split, preferring canonical and standard runs."""
    canonical_parent = repo_root / "output/segmentation/canonical" / resolution / split
    canonical_result = _latest_split_result_dir(canonical_parent, split)
    if canonical_result is not None:
        return canonical_result

    # Only inspect direct timestamped runs, avoiding experiment subdirectories.
    runs_parent = repo_root / "output/segmentation/runs" / resolution
    return _latest_split_result_dir(runs_parent, split)


def get_default_split_paths(repo_root: Path) -> dict[str, dict[str, Path]]:
    """Return available consolidated result folders used by EDA notebooks.

    Canonical folders may contain a timestamp level between the split and its
    ``numeric`` directory. Missing resolution/split combinations are omitted.
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


def get_bad_cases_export_dir(repo_root: Path) -> Path:
    """Return the shared bad-cases export directory."""
    return repo_root / "output/segmentation/analysis/bad_cases"


def get_cases_analysis_output_dir(repo_root: Path) -> Path:
    """Return the HTML output directory for cases-analysis notebooks."""
    return repo_root / "output/segmentation/analysis/cases_analysis/visual"
