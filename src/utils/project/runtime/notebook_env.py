"""Configura o ambiente compartilhado dos notebooks exploratórios."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from utils.project.config import (
    load_config_json,
    scale_config_to_resolution,
)
from utils.project.results.io import results_filename
from utils.project.runtime.paths import find_repository_root


def configure_notebook_environment(chdir_to_src: bool = True) -> Path:
    """Adiciona `src` ao caminho e opcionalmente altera o diretório atual.

    Args:
        chdir_to_src (bool): Se ``True``, altera o diretório de trabalho para ``src/``.

    Returns:
        Path: Caminho da raiz do repositório encontrada.
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
    """Resolve um caminho por variável de ambiente ou candidatos locais.

    Args:
        env_var (str): Nome da variável de ambiente verificada com prioridade.
        candidates (list[Path]): Caminhos locais alternativos testados em ordem.
        description (str): Descrição legível do recurso exibida em mensagens de erro.

    Returns:
        Path: Primeiro caminho existente encontrado.
    """
    # Prioriza o caminho configurado explicitamente na variável de ambiente.
    env_value = os.environ.get(env_var)
    if env_value:
        path = Path(env_value).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(
            f"{description} definido em {env_var} não existe: {path}"
        )

    # Recorre aos caminhos candidatos locais quando a variável não está definida.
    resolved = next((path for path in candidates if path.exists()), None)
    if resolved is not None:
        return resolved

    candidate_text = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        f"Nenhum caminho encontrado para {description}. "
        f"Exporte {env_var} ou ajuste os candidatos:\n{candidate_text}"
    )


def resolve_imagecas_base_path() -> Path:
    """Resolve o diretório ImageCAS com arquivos ``*.img.nii.gz``.

    Returns:
        Path: Diretório raiz existente do conjunto ImageCAS.
    """
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

    Args:
        config_file (str | Path): Caminho do arquivo JSON de configuração base.
        resolution (str): Perfil de resolução desejado (``"mid"`` ou ``"high"``).

    Returns:
        dict[str, Any]: Configuração carregada e escalada para a resolução solicitada.
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
    """Resolve o subdiretório numérico ou preserva caminhos legados.

    Args:
        path (Path): Diretório raiz de um run ou split.

    Returns:
        Path: Subdiretório ``numeric/`` quando existente, ou o próprio ``path`` em
            layouts legados.
    """
    numeric_dir = path / "numeric"
    return numeric_dir if numeric_dir.exists() else path


def _latest_split_result_dir(parent: Path, split: str) -> Path | None:
    """Localiza o resultado consolidado mais recente de um split.

    Args:
        parent (Path): Diretório pai de um split contendo runs diretos ou com
            timestamp.
        split (str): Nome do split buscado (``train``, ``val`` ou ``test``).

    Returns:
        Path | None: Diretório numérico válido mais recente encontrado, ou ``None``
            se ausente.
    """
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
        # Em summaries legados, confirma IMG_ID para distinguir tabelas agregadas.
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
    """Resolve somente a referência canônica, sem recorrer a outro run.

    Args:
        repo_root (Path): Caminho raiz do repositório.
        resolution (str): Nome da pasta de resolução (``mid_res`` ou ``high_res``).
        split (str): Nome do split buscado.

    Returns:
        Path | None: Diretório numérico canônico encontrado, ou ``None`` se ausente.
    """
    canonical_parent = repo_root / "output/segmentation/canonical" / resolution / split
    return _latest_split_result_dir(canonical_parent, split)


def get_default_split_paths(repo_root: Path) -> dict[str, dict[str, Path]]:
    """Retorna pastas de resultados consolidados usadas pelas EDAs.

    Pastas canônicas podem conter um nível de data entre o split e ``numeric``.
    Combinações ausentes de resolução e split são omitidas, sem substituição
    por runs fora de canonical. A escolha oficial pertence aos arquivos/links
    de canonical, não a uma lista duplicada em código.

    Args:
        repo_root (Path): Caminho raiz do repositório.

    Returns:
        dict[str, dict[str, Path]]: Mapeamento por resolução (``mid_res``,
            ``high_res``) e split para os diretórios disponíveis.
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
