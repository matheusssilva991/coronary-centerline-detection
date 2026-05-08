"""Notebook bootstrap helpers for the EDA notebooks."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_notebook_environment():
    """Add src to sys.path and switch cwd to the src directory."""
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"

    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)

    os.chdir(src_dir)
    return repo_root
