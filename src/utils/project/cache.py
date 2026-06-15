"""Helpers pequenos para cache em disco."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json_cache(path: str | Path, enabled: bool = True) -> Any | None:
    """Carrega um cache JSON quando habilitado e existente."""
    path = Path(path)
    if not enabled or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def save_json_cache(data: Any, path: str | Path, enabled: bool = True) -> None:
    """Salva um cache JSON quando habilitado."""
    if not enabled:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=4)


def load_npy_cache(path: str | Path, enabled: bool = True) -> Any | None:
    """Carrega um cache NumPy quando habilitado e existente."""
    path = Path(path)
    if not enabled or not path.exists():
        return None
    return np.load(path)


def save_npy_cache(array: Any, path: str | Path, enabled: bool = True) -> None:
    """Salva um cache NumPy quando habilitado."""
    if not enabled:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
