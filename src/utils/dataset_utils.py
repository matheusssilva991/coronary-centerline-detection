"""Utilities para listagem e divisão de datasets."""

import os
from glob import glob
from typing import List, Tuple

from sklearn.model_selection import train_test_split


def get_data_splits(
    base_path: str,
    test_size: float = 0.7,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Divide o dataset em treino, validação e teste.

    Returns:
        (train_ids, val_ids, test_ids, all_ids)
    """
    img_files = sorted(glob(os.path.join(base_path, "*.img.nii.gz")))
    all_ids = [int(os.path.basename(f).split(".")[0]) for f in img_files]

    train_val_ids, test_ids = train_test_split(
        all_ids, test_size=test_size, random_state=random_state
    )
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_size, random_state=random_state
    )

    return train_ids, val_ids, test_ids, all_ids
