from pathlib import Path

import pandas as pd

from ..utils import load_json_file


def load_split_metadata(split_paths_by_resolution, resolution, subset_name):
    """Load metadata JSON for a given resolution/split, or None when unavailable."""
    split_paths = split_paths_by_resolution.get(resolution, {})
    if subset_name not in split_paths:
        return None

    metadata_path = (
        Path(split_paths[subset_name]) / f"ostios_{subset_name}_metadata.json"
    )
    return load_json_file(str(metadata_path))


def load_split_summary(split_paths_by_resolution, resolution, subset_name):
    """Load summary CSV for a given resolution/split, or None when unavailable."""
    split_paths = split_paths_by_resolution.get(resolution, {})
    if subset_name not in split_paths:
        return None

    summary_path = Path(split_paths[subset_name]) / f"ostios_{subset_name}_summary.csv"
    return pd.read_csv(summary_path)
