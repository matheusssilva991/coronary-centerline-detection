"""Notebook bootstrap helpers for exploratory notebooks."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_notebook_environment(chdir_to_src: bool = True) -> Path:
    """Add src to sys.path and optionally switch cwd to the src directory.

    Returns:
        Path to the repository root.
    """
    repo_root = Path(__file__).resolve().parents[2]
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
        raise FileNotFoundError(f"{description} definido em {env_var} não existe: {path}")

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
        ],
        "ImageCAS",
    )


def resolve_processed_imagecas_path() -> Path:
    """Resolve o diretório de caches/resultados processados do ImageCAS."""
    return resolve_existing_path(
        "IMAGECAS_PROCESSED_PATH",
        [
            Path("/media/matheus/HD/DatasetsCCTA/Processed_ImageCAS"),
            Path("/data04/home/mpmaia/ImageCAS/Processed_ImageCAS"),
        ],
        "Processed ImageCAS",
    )


def get_default_split_paths(repo_root: Path) -> dict[str, dict[str, Path]]:
    """Return the current canonical final-result folders used by EDA notebooks."""
    base_dir = repo_root / "output/segmentation/8.final_results"
    return {
        "mid_res": {
            "train": base_dir / "mid_res/2026-04-30_14-33-37",
            "val": base_dir / "mid_res/2026-04-30_13-24-40",
            "test": base_dir / "mid_res/2026-05-02_10-48-13",
        },
        "high_res": {
            "train": base_dir / "high_res/2026-04-21_08-42-13",
            "val": base_dir / "high_res/2026-04-21_08-42-13",
            "test": base_dir / "high_res/2026-04-28_14-28-44",
        },
    }


def get_bad_cases_export_dir(repo_root: Path) -> Path:
    """Return the shared bad-cases export directory."""
    return repo_root / "output/segmentation/8.final_results/bad_cases_exports"


def get_cases_analysis_output_dir(repo_root: Path) -> Path:
    """Return the HTML output directory for cases-analysis notebooks."""
    return repo_root / "output/cases_analysis_3d"


def get_cases_analysis_cache_dir(repo_root: Path) -> Path:
    """Return the cache directory for cases-analysis notebooks."""
    return repo_root / "output/cases_analysis_cache"
