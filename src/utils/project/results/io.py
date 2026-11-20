"""Persistência e consolidação dos arquivos de resultado."""

from __future__ import annotations

import math
import re
import tempfile
from collections.abc import Iterable
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd

from utils.project.dataframe import require_series_column
from utils.project.results.schema import (
    make_result_dataframe,
    select_per_image_result_columns,
)

type PathInput = str | PathLike[str]

SPLIT_NAMES = ("train", "val", "test", "full")

BATCH_TIMING_COLUMNS: list[str] = [
    "split_name",
    "batch_number",
    "total_batches",
    "num_images",
    "first_img_id",
    "last_img_id",
    "result_file",
    "started_at",
    "finished_at",
    "duration_seconds",
    "duration_minutes",
    "duration_hours",
]


def results_filename(split: str) -> str:
    """Retorna o nome do resultado consolidado de um split.

    Args:
        split (str): Nome do split avaliado (ex.: ``train``, ``val`` ou ``test``).

    Returns:
        str: Nome do arquivo CSV consolidado do split.
    """
    return f"results_{split}.csv"


def batch_results_filename(split: str, batch_number: int) -> str:
    """Retorna o nome do resultado de um lote.

    Args:
        split (str): Nome do split avaliado.
        batch_number (int): Índice numérico do lote processado.

    Returns:
        str: Nome do arquivo CSV correspondente ao lote.
    """
    return f"results_{split}_lote_{batch_number}.csv"


def metadata_filename(split: str) -> str:
    """Retorna o nome do metadata de um split.

    Args:
        split (str): Nome do split avaliado.

    Returns:
        str: Nome do arquivo JSON de metadados do split.
    """
    return f"metadata_{split}.json"


def batch_timings_filename(split: str) -> str:
    """Retorna o nome do manifest de tempos dos lotes.

    Args:
        split (str): Nome do split avaliado.

    Returns:
        str: Nome do arquivo CSV com tempos por lote.
    """
    return f"batch_timings_{split}.csv"


def integrity_filename(split: str) -> str:
    """Retorna o nome do marcador de integridade de um split.

    Args:
        split (str): Nome do split avaliado.

    Returns:
        str: Nome do arquivo JSON com o marcador de integridade.
    """
    return f"integrity_{split}.json"


def result_candidates(directory: Path, split: str) -> tuple[Path, ...]:
    """Retorna resultados em ordem de preferência, incluindo layouts legados.

    Args:
        directory (Path): Diretório onde os arquivos de resultado são buscados.
        split (str): Nome do split avaliado.

    Returns:
        tuple[Path, ...]: Caminhos candidatos ordenados do padrão atual para os legados.
    """
    return (
        directory / results_filename(split),
        directory / f"ostios_{split}_results.csv",
        directory / f"ostios_{split}_summary.csv",
    )


def metadata_candidates(directory: Path, split: str) -> tuple[Path, ...]:
    """Retorna candidatos atuais e legados de metadata.

    Args:
        directory (Path): Diretório onde os arquivos de metadados são buscados.
        split (str): Nome do split avaliado.

    Returns:
        tuple[Path, ...]: Caminhos candidatos de metadados em ordem de prioridade.
    """
    return (
        directory / metadata_filename(split),
        directory / f"ostios_{split}_metadata.json",
    )


def batch_timings_candidates(directory: Path, split: str) -> tuple[Path, ...]:
    """Retorna candidatos atuais e legados de tempos por lote.

    Args:
        directory (Path): Diretório onde o manifest de tempos é buscado.
        split (str): Nome do split avaliado.

    Returns:
        tuple[Path, ...]: Caminhos candidatos do manifest de tempos por lote.
    """
    return (
        directory / batch_timings_filename(split),
        directory / f"ostios_{split}_batch_timings.csv",
    )


def save_dataframe_atomic(dataframe: pd.DataFrame, path: PathInput) -> None:
    """Salva CSV sem índice por substituição atômica, limpando temporários.

    Args:
        dataframe (pd.DataFrame): DataFrame a persistir em formato CSV.
        path (PathInput): Caminho de destino do arquivo CSV.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    # Grava primeiro em arquivo temporário no mesmo diretório para substituição atômica.
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            dataframe.to_csv(stream, index=False)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def create_timestamped_output_dir(
    base_output_dir: PathInput,
    experiment_name: str = "segmentation",
) -> str:
    """Cria diretório de saída com timestamp.

    Args:
        base_output_dir (PathInput): Diretório base onde a execução será armazenada.
        experiment_name (str): Subdiretório do experimento em ``base_output_dir``.

    Returns:
        str: Caminho completo do diretório criado com timestamp.
    """
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
    """Salva somente resultados e diagnósticos individuais em CSV.

    Args:
        results (list[dict[str, Any]]): Lista de resultados individuais por imagem.
        split_name (str): Nome do split avaliado.
        output_dir (PathInput): Diretório onde o CSV consolidado será gravado.
        config (dict[str, Any] | None): Argumento opcional legado mantido por
            compatibilidade (não utilizado).

    Returns:
        str: Caminho do arquivo CSV consolidado salvo em disco.
    """
    # Padroniza o schema interno antes de expor nomes legíveis no CSV.
    df = make_result_dataframe(results)
    df = select_per_image_result_columns(df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / results_filename(split_name)
    save_dataframe_atomic(df, output_path)
    return str(output_path)


def batch_result_number(path: PathInput, split_name: str) -> int | None:
    """Extrai o número do lote tanto do nome atual quanto do nome histórico.

    Args:
        path (PathInput): Caminho do arquivo CSV de lote.
        split_name (str): Nome do split ao qual o lote pertence.

    Returns:
        int | None: Número do lote extraído do nome, ou ``None`` se inválido.
    """
    escaped = re.escape(split_name)
    filename = Path(path).name
    patterns = (
        rf"^results_{escaped}_lote_(\d+)\.csv$",
        rf"^ostios_{escaped}_lote_(\d+)_summary\.csv$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, filename)
        if match:
            return int(match.group(1))
    return None


def list_batch_result_files(split_name: str, output_dir: PathInput) -> list[Path]:
    """Lista os CSVs de lote atuais em ordem numérica.

    Args:
        split_name (str): Nome do split avaliado.
        output_dir (PathInput): Diretório onde os arquivos de lote estão armazenados.

    Returns:
        list[Path]: Lista de caminhos dos CSVs de lote ordenada numericamente.
    """
    output_dir = Path(output_dir)
    # Ignora arquivos parecidos que não seguem o padrão oficial de lote.
    candidates = [
        path
        for path in output_dir.glob("*.csv")
        if batch_result_number(path, split_name) is not None
    ]

    # Garante que o mesmo número de lote não exista simultaneamente em dois formatos.
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
        """Retorna o índice numérico do lote para ordenação determinística."""
        batch_number = batch_result_number(path, split_name)
        return batch_number if batch_number is not None else -1

    return sorted(candidates, key=batch_sort_key)


def get_batch_result_file(
    output_dir: PathInput,
    split_name: str,
    batch_number: int,
) -> Path | None:
    """Retorna o CSV de um lote quando ele existe.

    Args:
        output_dir (PathInput): Diretório que contém os arquivos de lote.
        split_name (str): Nome do split avaliado.
        batch_number (int): Número do lote buscado.

    Returns:
        Path | None: Caminho do CSV encontrado ou ``None`` quando o lote não existe.
    """
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
    """Mescla todos os CSVs de lotes no consolidado por imagem.

    Args:
        split_name (str): Nome do split a consolidar.
        output_dir (PathInput): Diretório contendo os CSVs individuais de lote.

    Returns:
        str | None: Caminho do CSV consolidado final, ou ``None`` se sem lotes.
    """
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
    save_dataframe_atomic(merged_df, final_path)

    print(
        f"✅ Arquivo final mesclado: {final_path} ({len(merged_df)} registros totais)\n"
    )
    return str(final_path)


def duration_breakdown(duration_seconds: Any) -> dict[str, Any]:
    """Retorna duração em segundos, minutos e horas.

    Args:
        duration_seconds (Any): Duração bruta em segundos (numérico ou ``None``).

    Returns:
        dict[str, Any]: Dicionário com ``seconds``, ``minutes`` e ``hours``
            (ou ``None`` se inválido).
    """
    if duration_seconds is None:
        return {
            "seconds": None,
            "minutes": None,
            "hours": None,
        }

    # Converte para float rejeitando entradas não numéricas ou NaN antes da divisão.
    try:
        seconds = float(duration_seconds)
    except (TypeError, ValueError):
        return {"seconds": None, "minutes": None, "hours": None}

    if math.isnan(seconds):
        return {"seconds": None, "minutes": None, "hours": None}
    return {
        "seconds": seconds,
        "minutes": seconds / 60,
        "hours": seconds / 3600,
    }


def batch_timing_manifest_path(output_dir: PathInput, split_name: str) -> Path:
    """Retorna o caminho do CSV com tempos por lote.

    Args:
        output_dir (PathInput): Diretório de saída numérica da execução.
        split_name (str): Nome do split avaliado.

    Returns:
        Path: Caminho do arquivo CSV de manifest de tempos dos lotes.
    """
    return Path(output_dir) / batch_timings_filename(split_name)


def load_batch_timing_records(
    output_dir: PathInput,
    split_name: str,
) -> list[dict[str, Any]]:
    """Carrega tempos por lote já salvos.

    Args:
        output_dir (PathInput): Diretório onde o manifest de tempos pode existir.
        split_name (str): Nome do split avaliado.

    Returns:
        list[dict[str, Any]]: Lista de registros de tempo carregados, ou lista vazia.
    """
    # Seleciona o primeiro manifest disponível entre os formatos atual e legado.
    manifest_path = next(
        (
            path
            for path in batch_timings_candidates(Path(output_dir), split_name)
            if path.is_file()
        ),
        None,
    )
    if manifest_path is None:
        return []

    # Lê o CSV de tempos encontrado e normaliza as chaves para string.
    df = pd.read_csv(manifest_path)
    if df.empty:
        return []
    return [
        {str(key): value for key, value in record.items()}
        for record in df.to_dict("records")
    ]


def save_batch_timing_record(
    output_dir: PathInput,
    split_name: str,
    record: dict[str, Any],
) -> str:
    """Insere ou atualiza o tempo de um lote no manifest incremental.

    Args:
        output_dir (PathInput): Diretório onde o manifest de tempos é armazenado.
        split_name (str): Nome do split avaliado.
        record (dict[str, Any]): Registro com metadados e duração do lote concluído.

    Returns:
        str: Caminho do arquivo CSV de manifest atualizado em disco.
    """
    manifest_path = batch_timing_manifest_path(output_dir, split_name)
    records = load_batch_timing_records(output_dir, split_name)
    batch_number = int(record["batch_number"])

    # Em retomadas, substitui o tempo antigo do lote em vez de duplicá-lo.
    records = [
        item for item in records if int(item.get("batch_number", -1)) != batch_number
    ]
    records.append(record)
    records = sorted(records, key=lambda item: int(item["batch_number"]))

    # Mantém a ordem das linhas e das colunas estável entre execuções.
    df = pd.DataFrame(records)
    df = df.reindex(columns=BATCH_TIMING_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataframe_atomic(df, manifest_path)
    return str(manifest_path)


def summarize_batch_timing_records(
    records: list[dict[str, Any]],
    expected_batches: list[int] | None = None,
) -> dict[str, Any]:
    """Soma tempos conhecidos dos lotes e marca lotes sem tempo salvo.

    Args:
        records (list[dict[str, Any]]): Registros de duração carregados do manifest.
        expected_batches (list[int] | None): Lista opcional com os índices de
            lotes esperados.

    Returns:
        dict[str, Any]: Resumo com lotes conhecidos, duração total e faltantes.
    """
    if not records:
        return {
            "total_known_batches": 0,
            "total_known_duration_seconds": None,
            "total_known_duration_minutes": None,
            "total_known_duration_hours": None,
            "missing_timing_batches": expected_batches or [],
        }

    # Descarta durações ausentes ou inválidas antes de calcular o total.
    durations_by_batch: dict[int, float] = {}
    for record in records:
        batch_number = int(record["batch_number"])
        duration_value = record.get("duration_seconds")
        if duration_value is None:
            continue
        try:
            duration = float(duration_value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(duration):
            durations_by_batch[batch_number] = duration

    known_batches = sorted(durations_by_batch)
    total_seconds = sum(durations_by_batch.values()) if durations_by_batch else None

    # Identifica lotes antigos que ainda não possuíam registro de tempo.
    missing_timing_batches: list[int] = []
    if expected_batches is not None:
        missing_timing_batches = [
            batch_number
            for batch_number in expected_batches
            if batch_number not in durations_by_batch
        ]

    total = duration_breakdown(total_seconds)
    return {
        "total_known_batches": len(known_batches),
        "known_batches": known_batches,
        "total_known_duration_seconds": total["seconds"],
        "total_known_duration_minutes": total["minutes"],
        "total_known_duration_hours": total["hours"],
        "missing_timing_batches": missing_timing_batches,
    }


class ResultIntegrityError(ValueError):
    """Indica que o consolidado não representa exatamente a coorte esperada."""

    def __init__(self, report: dict[str, Any]):
        """Inicializa a exceção formatando os problemas encontrados na coorte.

        Args:
            report (dict[str, Any]): Relatório com listas de IDs duplicados,
                ausentes e inesperados.
        """
        self.report = report
        problems = []
        if report["duplicate_image_ids"]:
            problems.append(f"IDs duplicados: {report['duplicate_image_ids']}")
        if report["missing_image_ids"]:
            problems.append(f"IDs ausentes: {report['missing_image_ids']}")
        if report["unexpected_image_ids"]:
            problems.append(f"IDs inesperados: {report['unexpected_image_ids']}")
        super().__init__("; ".join(problems) or "Resultados inconsistentes.")


def _normalized_image_ids(values: Iterable[Any]) -> list[int]:
    """Converte valores não nulos de identificadores em lista de inteiros.

    Args:
        values (Iterable[Any]): Sequência bruta de identificadores de imagem.

    Returns:
        list[int]: Identificadores válidos convertidos para ``int``.
    """
    return [int(value) for value in values if pd.notna(value)]


def validate_result_integrity(
    dataframe: pd.DataFrame,
    expected_image_ids: Iterable[Any],
) -> dict[str, Any]:
    """Valida unicidade e igualdade exata entre IDs persistidos e esperados.

    Args:
        dataframe (pd.DataFrame): DataFrame consolidado contendo a coluna ``IMG_ID``.
        expected_image_ids (Iterable[Any]): Identificadores esperados na coorte.

    Returns:
        dict[str, Any]: Relatório de integridade com status ``complete`` e contagens.
    """
    # Garante presença da coluna de identificação antes de comparar os conjuntos.
    expected_ids = _normalized_image_ids(expected_image_ids)
    if "IMG_ID" not in dataframe.columns:
        raise ResultIntegrityError(
            {
                "status": "incomplete",
                "expected_image_count": len(expected_ids),
                "persisted_image_count": len(dataframe),
                "missing_image_ids": expected_ids,
                "unexpected_image_ids": [],
                "duplicate_image_ids": [],
                "reason": "missing_img_id_column",
            }
        )

    # Compara os IDs persistidos com os esperados para detectar duplicatas ou faltas.
    persisted_ids = _normalized_image_ids(
        require_series_column(dataframe, "IMG_ID").tolist()
    )
    persisted_series = pd.Series(persisted_ids, dtype=int)
    duplicate_ids = sorted(persisted_series.loc[persisted_series.duplicated()].unique())
    expected_set = set(expected_ids)
    persisted_set = set(persisted_ids)
    report = {
        "status": "complete",
        "expected_image_count": len(expected_ids),
        "persisted_image_count": len(persisted_ids),
        "missing_image_ids": sorted(expected_set - persisted_set),
        "unexpected_image_ids": sorted(persisted_set - expected_set),
        "duplicate_image_ids": duplicate_ids,
        "reason": None,
    }
    # Dispara erro detalhado caso haja divergência de cardinalidade ou unicidade.
    if (
        duplicate_ids
        or report["missing_image_ids"]
        or report["unexpected_image_ids"]
        or len(expected_ids) != len(persisted_ids)
    ):
        report["status"] = "incomplete"
        report["reason"] = "image_id_mismatch"
        raise ResultIntegrityError(report)
    return report


__all__ = [
    "BATCH_TIMING_COLUMNS",
    "ResultIntegrityError",
    "SPLIT_NAMES",
    "batch_result_number",
    "batch_results_filename",
    "batch_timing_manifest_path",
    "batch_timings_candidates",
    "batch_timings_filename",
    "create_timestamped_output_dir",
    "duration_breakdown",
    "get_batch_result_file",
    "integrity_filename",
    "list_batch_result_files",
    "load_batch_timing_records",
    "merge_batch_results",
    "metadata_candidates",
    "metadata_filename",
    "result_candidates",
    "results_filename",
    "save_batch_timing_record",
    "save_dataframe_atomic",
    "save_results",
    "summarize_batch_timing_records",
    "validate_result_integrity",
]
