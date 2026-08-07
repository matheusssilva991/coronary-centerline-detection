"""Persistência e consolidação dos arquivos de resultado."""

from __future__ import annotations

import re
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd

from .results_schema import (
    add_config_columns,
    make_readable_results_dataframe,
    make_result_dataframe,
)


type PathInput = str | PathLike[str]


def create_timestamped_output_dir(
    base_output_dir: PathInput,
    experiment_name: str = "segmentation",
) -> str:
    """Cria diretório de saída com timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(base_output_dir) / experiment_name / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def save_results(
    results: list[dict[str, Any]],
    split_name: str,
    output_dir: PathInput,
    config: dict[str, Any] | None = None,
) -> str:
    """Salva resultados em CSV."""
    # Padroniza o schema interno antes de expor nomes legíveis no CSV.
    df = make_result_dataframe(results)
    if config is not None:
        df = add_config_columns(df, config)
    df = make_readable_results_dataframe(df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ostios_{split_name}_summary.csv"
    df.to_csv(output_path, index=False)
    return str(output_path)


def batch_result_number(path: PathInput, split_name: str) -> int | None:
    """Extrai o número do lote de um arquivo de resultado."""
    filename = Path(path).name
    match = re.match(
        rf"^ostios_{re.escape(split_name)}_lote_(\d+)_summary\.csv$",
        filename,
    )
    return int(match.group(1)) if match else None


def list_batch_result_files(split_name: str, output_dir: PathInput) -> list[Path]:
    """Lista os CSVs de lote atuais em ordem numérica."""
    output_dir = Path(output_dir)
    # Ignora arquivos parecidos que não seguem o padrão oficial de lote.
    candidates = [
        path
        for path in output_dir.glob(f"ostios_{split_name}_lote_*_summary.csv")
        if batch_result_number(path, split_name) is not None
    ]

    def batch_sort_key(path: Path) -> int:
        batch_number = batch_result_number(path, split_name)
        return batch_number if batch_number is not None else -1

    return sorted(candidates, key=batch_sort_key)


def get_batch_result_file(
    output_dir: PathInput,
    split_name: str,
    batch_number: int,
) -> Path | None:
    """Retorna o CSV de um lote quando ele existe."""
    batch_file = (
        Path(output_dir) / f"ostios_{split_name}_lote_{batch_number}_summary.csv"
    )
    return batch_file if batch_file.exists() else None


def merge_batch_results(split_name: str, output_dir: PathInput) -> str | None:
    """Mescla todos os CSVs de lotes em um único arquivo final."""
    output_dir = Path(output_dir)
    batch_files = list_batch_result_files(split_name, output_dir)

    if not batch_files:
        print(f"⚠️  Nenhum arquivo de lote encontrado em {output_dir}")
        return None

    print(f"\n🔄 Mesclando {len(batch_files)} arquivo(s) de lote...")
    dfs: list[pd.DataFrame] = []

    # Carrega cada lote e uniformiza possíveis aliases antes da consolidação.
    for batch_file in batch_files:
        df = pd.read_csv(batch_file)
        df = make_readable_results_dataframe(df)
        dfs.append(df)
        print(f"   ✓ {batch_file.name} ({len(df)} registros)")

    # Recria o índice para produzir um CSV final contínuo entre os lotes.
    merged_df = pd.concat(dfs, ignore_index=True)
    final_path = output_dir / f"ostios_{split_name}_summary.csv"
    merged_df.to_csv(final_path, index=False)

    print(
        f"✅ Arquivo final mesclado: {final_path} ({len(merged_df)} registros totais)\n"
    )
    return str(final_path)
