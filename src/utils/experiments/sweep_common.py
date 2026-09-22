"""Reúne utilitários compartilhados pelos scripts de varredura.

Os arquivos em ``src/experiments`` são executados diretamente em uma estação
ou servidor. Este módulo uniformiza caminhos da CLI, variantes, amostragem de
splits e serialização CSV.
"""

from __future__ import annotations

import copy
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Protege máquinas com uma GPU quando o ambiente não seleciona um dispositivo.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from utils.project.dataset import get_data_splits  # noqa: E402
from utils.project.results import make_json_safe  # noqa: E402


def resolve_cli_path(path: Path | None) -> Path | None:
    """Resolve caminhos relativos da CLI a partir da raiz do repositório."""
    if path is None:
        return None
    return path if path.is_absolute() else REPO_ROOT / path


def load_json_arg(value: str | None) -> Any:
    """Carrega JSON de texto ou de arquivo relativo ao repositório."""
    if value is None:
        return None

    path = resolve_cli_path(Path(value))
    try:
        path_exists = path is not None and path.exists()
    except OSError:
        path_exists = False
    if path_exists:
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def load_json_file(path: Path) -> Any:
    """Carrega JSON de um caminho absoluto ou relativo ao repositório."""
    resolved_path = resolve_cli_path(path)
    if resolved_path is None:
        raise ValueError("JSON file path cannot be None")
    return json.loads(resolved_path.read_text(encoding="utf-8"))


def sanitize_name(name: str) -> str:
    """Normaliza um nome para uso em pastas e campos CSV."""
    safe = "".join(
        char if char.isalnum() or char in {"_", "-", "."} else "_" for char in str(name)
    )
    return safe.strip("_") or "variant"


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Define ``A.B.C`` dentro de um dicionário aninhado."""
    keys = dotted_key.split(".")
    target = config
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = copy.deepcopy(value)


def get_nested(data: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Obtém ``A.B.C`` de um dicionário aninhado."""
    target: Any = data
    for key in dotted_key.split("."):
        if not isinstance(target, dict) or key not in target:
            return default
        target = target[key]
    return target


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Combina dicionários recursivamente sem alterar as entradas."""
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_overrides(
    config: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Aplica sobrescritas pontuadas ou aninhadas em uma cópia da configuração."""
    updated = copy.deepcopy(config)
    for key, value in overrides.items():
        if "." in key:
            set_nested(updated, key, value)
        elif isinstance(value, dict) and isinstance(updated.get(key), dict):
            updated[key] = deep_update(updated[key], value)
        else:
            updated[key] = copy.deepcopy(value)
    return updated


def make_grid_variants(grid: dict[str, Any]) -> list[dict[str, Any]]:
    """Monta variantes cartesianas a partir de uma grade de chaves pontuadas."""
    keys = list(grid)
    values = [value if isinstance(value, list) else [value] for value in grid.values()]
    variants = []
    for index, combination in enumerate(itertools.product(*values), start=1):
        overrides = dict(zip(keys, combination))
        name_parts = [
            f"{key.split('.')[-1]}={value}" for key, value in overrides.items()
        ]
        variant_name = sanitize_name(f"grid_{index:03d}_{'_'.join(name_parts)}")
        variants.append({"name": variant_name, "overrides": overrides})
    return variants


def select_ids(
    split: str,
    sample_size: int,
    start_index: int,
    ids_arg: str | None,
    base_path: Path,
    split_config_path: str | Path | None = None,
) -> list[int]:
    """Seleciona IDs por split fixo ou lista explícita separada por vírgulas."""
    if ids_arg:
        return [int(item.strip()) for item in ids_arg.split(",") if item.strip()]
    if start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if sample_size <= 0:
        raise ValueError("--sample-size must be > 0")

    train_ids, val_ids, test_ids, _ = get_data_splits(
        str(base_path),
        split_config_path=split_config_path,
    )
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}[split]
    return split_ids[start_index : start_index + sample_size]


def csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Serializa listas, dicionários e valores NumPy antes do CSV."""
    out = df.copy()
    for column in out.columns:
        if out[column].dtype != "object":
            continue
        out[column] = out[column].map(
            lambda value: (
                json.dumps(make_json_safe(value), ensure_ascii=False)
                if isinstance(value, (dict, list, tuple)) or hasattr(value, "tolist")
                else value
            )
        )
    return out


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Salva JSON usando a serialização segura do projeto."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(make_json_safe(data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
