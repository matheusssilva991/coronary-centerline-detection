"""Utilities para listagem e divisão de datasets."""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

from sklearn.model_selection import train_test_split

DEFAULT_TEST_SIZE = 0.7
DEFAULT_VAL_SIZE = 0.1
DEFAULT_RANDOM_STATE = 42
DEFAULT_SPLIT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "imagecas_splits.json"
)


def _load_fixed_splits(
    split_config_path: Path,
    available_ids: List[int],
) -> Tuple[List[int], List[int], List[int]]:
    """Load frozen train/val/test IDs and validate them against the dataset."""
    data = json.loads(split_config_path.read_text(encoding="utf-8"))
    splits = data.get("splits", {})
    required_keys = ("train", "val", "test")
    missing_keys = [key for key in required_keys if key not in splits]
    if missing_keys:
        raise ValueError(
            f"Arquivo de splits {split_config_path} sem chaves: {missing_keys}"
        )

    # Converte os IDs uma única vez antes das validações de integridade.
    train_ids = [int(img_id) for img_id in splits["train"]]
    val_ids = [int(img_id) for img_id in splits["val"]]
    test_ids = [int(img_id) for img_id in splits["test"]]

    split_ids = train_ids + val_ids + test_ids
    duplicate_count = len(split_ids) - len(set(split_ids))
    if duplicate_count:
        raise ValueError(
            f"Arquivo de splits {split_config_path} contém IDs duplicados."
        )

    # Exige correspondência exata para evitar mudanças silenciosas na coorte.
    available_set = set(available_ids)
    split_set = set(split_ids)
    missing_from_dataset = sorted(split_set - available_set)
    extra_in_dataset = sorted(available_set - split_set)
    if missing_from_dataset or extra_in_dataset:
        raise ValueError(
            "IDs do dataset não batem com o arquivo de splits fixo "
            f"{split_config_path}. "
            f"Faltando no dataset: {missing_from_dataset[:10]} "
            f"(total={len(missing_from_dataset)}); "
            f"extras no dataset: {extra_in_dataset[:10]} "
            f"(total={len(extra_in_dataset)})."
        )

    return train_ids, val_ids, test_ids


def _should_use_default_fixed_splits(
    test_size: float,
    val_size: float,
    random_state: int,
) -> bool:
    """Use frozen ImageCAS splits only for the canonical split parameters."""
    return (
        test_size == DEFAULT_TEST_SIZE
        and val_size == DEFAULT_VAL_SIZE
        and random_state == DEFAULT_RANDOM_STATE
        and DEFAULT_SPLIT_CONFIG_PATH.exists()
    )


def get_data_splits(
    base_path: str,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    split_config_path: Optional[Union[str, Path]] = None,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Divide o dataset em treino, validação e teste.

    Com os parâmetros canônicos, usa ``config/imagecas_splits.json`` para manter
    os mesmos IDs entre execuções e refactors. Ao informar ``split_config_path``,
    o arquivo indicado é usado independentemente dos demais parâmetros.

    Returns:
        (train_ids, val_ids, test_ids, all_ids)
    """
    dataset_path = Path(base_path)
    # A ordenação numérica evita que o ID 10 apareça antes do ID 2.
    img_files = sorted(
        dataset_path.glob("*.img.nii.gz"),
        key=lambda path: int(path.name.split(".")[0]),
    )
    if not img_files:
        raise FileNotFoundError(
            f"Nenhuma imagem *.img.nii.gz encontrada em {dataset_path}"
        )

    all_ids = [int(path.name.split(".")[0]) for path in img_files]

    # Um arquivo explícito tem precedência sobre o split canônico do projeto.
    fixed_split_path = Path(split_config_path) if split_config_path else None
    if fixed_split_path is None and _should_use_default_fixed_splits(
        test_size, val_size, random_state
    ):
        fixed_split_path = DEFAULT_SPLIT_CONFIG_PATH

    if fixed_split_path is not None:
        train_ids, val_ids, test_ids = _load_fixed_splits(fixed_split_path, all_ids)
        return train_ids, val_ids, test_ids, all_ids

    # Parâmetros não canônicos usam divisão reproduzível em duas etapas.
    train_val_ids, test_ids = train_test_split(
        all_ids, test_size=test_size, random_state=random_state
    )
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_size, random_state=random_state
    )

    return train_ids, val_ids, test_ids, all_ids
