"""Persistência e consolidação dos arquivos de resultado."""

from __future__ import annotations

from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd

from .result_paths import (
    batch_result_number as parse_batch_result_number,
    batch_results_filename,
    results_filename,
)
from .results_schema import (
    make_result_dataframe,
    select_per_image_result_columns,
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
    """Salva somente resultados e diagnósticos individuais em CSV."""
    # Padroniza o schema interno antes de expor nomes legíveis no CSV.
    df = make_result_dataframe(results)
    df = select_per_image_result_columns(df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / results_filename(split_name)
    df.to_csv(output_path, index=False)
    return str(output_path)


def batch_result_number(path: PathInput, split_name: str) -> int | None:
    """Extrai o número do lote de um arquivo de resultado."""
    filename = Path(path).name
    return parse_batch_result_number(Path(filename), split_name)


def list_batch_result_files(split_name: str, output_dir: PathInput) -> list[Path]:
    """Lista os CSVs de lote atuais em ordem numérica."""
    output_dir = Path(output_dir)
    # Ignora arquivos parecidos que não seguem o padrão oficial de lote.
    candidates = [
        path
        for path in output_dir.glob("*.csv")
        if batch_result_number(path, split_name) is not None
    ]

    by_number: dict[int, list[Path]] = {}
    for path in candidates:
        number = batch_result_number(path, split_name)
        if number is not None:
            by_number.setdefault(number, []).append(path)
    duplicates = {
        number: paths for number, paths in by_number.items() if len(paths) > 1
    }
    if duplicates:
        detail = "; ".join(
            f"lote {number}: {', '.join(path.name for path in paths)}"
            for number, paths in sorted(duplicates.items())
        )
        raise ValueError(f"Lotes duplicados entre formatos: {detail}")

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
    output_dir = Path(output_dir)
    candidates = (
        output_dir / batch_results_filename(split_name, batch_number),
        output_dir / f"ostios_{split_name}_lote_{batch_number}_summary.csv",
    )
    existing = [path for path in candidates if path.exists()]
    if len(existing) > 1:
        raise ValueError(
            f"Lote {batch_number} duplicado entre formatos: "
            + ", ".join(path.name for path in existing)
        )
    return existing[0] if existing else None


def merge_batch_results(split_name: str, output_dir: PathInput) -> str | None:
    """Mescla todos os CSVs de lotes no consolidado por imagem."""
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
        # Lotes legados podem conter configurações repetidas em cada linha.
        # A projeção também limpa esses campos durante um ``--merge-only``.
        df = select_per_image_result_columns(df)
        dfs.append(df)
        print(f"   ✓ {batch_file.name} ({len(df)} registros)")

    # Recria o índice para produzir um CSV final contínuo entre os lotes.
    merged_df = pd.concat(dfs, ignore_index=True)
    final_path = output_dir / results_filename(split_name)
    temporary_path = final_path.with_suffix(".csv.tmp")
    merged_df.to_csv(temporary_path, index=False)
    temporary_path.replace(final_path)

    print(
        f"✅ Arquivo final mesclado: {final_path} ({len(merged_df)} registros totais)\n"
    )
    return str(final_path)
