"""Shared helpers for experiment sweep scripts.

The files in ``src/experiments`` are meant to be executed directly on a
workstation or server. This module keeps CLI path handling, variant expansion,
split sampling and CSV serialization consistent across those scripts.
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

# Keep single-GPU machines safe when the environment does not select a device.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from utils.project.dataset import get_data_splits  # noqa: E402
from utils.project.results import make_json_safe  # noqa: E402


def resolve_cli_path(path: Path | None) -> Path | None:
    """Resolve relative CLI paths from the repository root."""
    if path is None:
        return None
    return path if path.is_absolute() else REPO_ROOT / path


def load_json_arg(value: str | None) -> Any:
    """Load a JSON argument from inline text or from a repository-relative file."""
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
    """Load JSON from an absolute or repository-relative path."""
    resolved_path = resolve_cli_path(path)
    if resolved_path is None:
        raise ValueError("JSON file path cannot be None")
    return json.loads(resolved_path.read_text(encoding="utf-8"))


def sanitize_name(name: str) -> str:
    """Make a name safe for folders and CSV fields."""
    safe = "".join(
        char if char.isalnum() or char in {"_", "-", "."} else "_"
        for char in str(name)
    )
    return safe.strip("_") or "variant"


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``A.B.C`` inside a nested dictionary."""
    keys = dotted_key.split(".")
    target = config
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = copy.deepcopy(value)


def get_nested(data: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Get ``A.B.C`` from a nested dictionary."""
    target: Any = data
    for key in dotted_key.split("."):
        if not isinstance(target, dict) or key not in target:
            return default
        target = target[key]
    return target


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating the inputs."""
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply dotted-key or nested-dict overrides to a copied config."""
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
    """Build cartesian-product variants from a dotted-key grid."""
    keys = list(grid)
    values = [value if isinstance(value, list) else [value] for value in grid.values()]
    variants = []
    for index, combination in enumerate(itertools.product(*values), start=1):
        overrides = dict(zip(keys, combination))
        name_parts = [f"{key.split('.')[-1]}={value}" for key, value in overrides.items()]
        variant_name = sanitize_name(f"grid_{index:03d}_{'_'.join(name_parts)}")
        variants.append({"name": variant_name, "overrides": overrides})
    return variants


def select_ids(
    split: str,
    sample_size: int,
    start_index: int,
    ids_arg: str | None,
    base_path: Path,
) -> list[int]:
    """Select image IDs from a fixed split or an explicit comma-separated list."""
    if ids_arg:
        return [int(item.strip()) for item in ids_arg.split(",") if item.strip()]
    if start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if sample_size <= 0:
        raise ValueError("--sample-size must be > 0")

    train_ids, val_ids, test_ids, _ = get_data_splits(str(base_path))
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}[split]
    return split_ids[start_index : start_index + sample_size]


def csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Serialize list/dict/numpy-like values before CSV writing."""
    out = df.copy()
    for column in out.columns:
        if out[column].dtype != "object":
            continue
        out[column] = out[column].map(
            lambda value: json.dumps(make_json_safe(value), ensure_ascii=False)
            if isinstance(value, (dict, list, tuple)) or hasattr(value, "tolist")
            else value
        )
    return out


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON with project-safe serialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(make_json_safe(data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
