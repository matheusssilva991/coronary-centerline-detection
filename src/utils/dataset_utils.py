"""Utilities para listagem e divisão de datasets."""

import os
from glob import glob

from sklearn.model_selection import train_test_split


def get_data_splits(base_path, test_size=0.7, val_size=0.1, random_state=42):
    """Divide o dataset em treino, validação e teste."""
    img_files = sorted(glob(os.path.join(base_path, "*.img.nii.gz")))
    all_ids = [int(os.path.basename(f).split(".")[0]) for f in img_files]

    train_val_ids, test_ids = train_test_split(
        all_ids, test_size=test_size, random_state=random_state
    )
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_size, random_state=random_state
    )

    return train_ids, val_ids, test_ids, all_ids