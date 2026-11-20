"""Carregamento de resultados por split/variante, tempos de lote e resumo por resolução."""

from pathlib import Path

import numpy as np
import pandas as pd

from utils.project.results.io import (
    batch_timings_candidates,
    result_candidates,
)

from ..project.dataframe import numeric_series, require_series_column
from ..project.results import add_internal_result_aliases
from ..project.results.schema import normalize_result_status


def load_experiment_variant_results(
    run_dir: str | Path, *, variant: str, split: str
) -> pd.DataFrame:
    """Carrega uma variante de sweep no schema de resultados por exame.

    Preserva o Dice registrado e atribui zero apenas a falhas documentadas sem
    tentativa de segmentação. Ausências sem diagnóstico são rejeitadas.

    Args:
        run_dir (str | Path): Diretório raiz do run contendo
            ``results/image_results.csv``.
        variant (str): Nome da variante do sweep a filtrar.
        split (str): Nome do subconjunto do dataset (ex.: ``"val"`` ou ``"test"``).

    Returns:
        pd.DataFrame: Tabela validada por exame com aliases internos e status
        normalizados dos óstios.
    """
    # Lê o CSV consolidado e valida colunas obrigatórias, variante, split e IDs.
    path = Path(run_dir) / "results" / "image_results.csv"
    frame = pd.read_csv(path)
    required = {
        "variant",
        "split",
        "IMG_ID",
        "dice_artery",
        "ostia_status",
        "error",
        "segmentation_attempted",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: colunas ausentes: {sorted(missing)}")
    frame = frame.loc[frame["variant"].eq(variant) & frame["split"].eq(split)].copy()
    if frame.empty:
        raise ValueError(f"{path}: variante/conjunto ausente: {variant}/{split}")
    if frame["IMG_ID"].isna().any() or frame["IMG_ID"].duplicated().any():
        raise ValueError(f"{path}: IDs ausentes ou repetidos em {variant}/{split}")

    # Falhas registradas permanecem na coorte, com Dice geral zero.
    dice = pd.to_numeric(frame["dice_artery"], errors="raise")
    failed = frame["error"].notna() & frame["segmentation_attempted"].isin(
        [False, "False", "false", 0, "0"]
    )
    if (dice.isna() & ~failed).any():
        raise ValueError(f"{path}: Dice ausente sem falha registrada")
    if not dice.dropna().between(0, 1).all():
        raise ValueError(f"{path}: Dice fora de [0, 1]")
    # Normaliza colunas de Dice, status dos óstios e métricas opcionais de morfologia.
    frame["recorded_dice"] = dice
    frame["evaluable_dice"] = dice.mask(failed)
    frame["dice_artery"] = dice.fillna(0.0)
    frame["artery_dice"] = frame["dice_artery"]
    frame["status"] = frame["ostia_status"].map(normalize_result_status)
    # Distância zero significa interseção com o label na avaliação dos óstios.
    if {"left_dist_mm", "right_dist_mm"}.issubset(frame.columns):
        left = pd.to_numeric(frame["left_dist_mm"], errors="raise")
        right = pd.to_numeric(frame["right_dist_mm"], errors="raise")
        wrong = frame["status"].eq("found_but_wrong") & left.notna() & right.notna()
        frame.loc[wrong, "status"] = "none_correct"
        frame.loc[wrong & (left.eq(0) | right.eq(0)), "status"] = "one_correct"
    frame["ostia_detection_status"] = frame["status"]
    frame["ostia_detected"] = frame.get("ostia_found", False)
    frame["pipeline_error"] = frame["error"]
    frame["source_run"] = str(Path(run_dir))
    for stage in ("before", "after"):
        source = f"dice_artery_{stage}_morphology"
        if source in frame:
            frame[f"artery_dice_{stage}_morphology"] = frame[source]
    return add_internal_result_aliases(frame)


def _split_numeric_dir(split_paths_by_resolution, resolution, subset_name):
    """Resolve o diretório numérico de um split configurado.

    Args:
        split_paths_by_resolution (Mapping[str, Mapping[str, str | Path]]): Mapa
            aninhado de resolução e split para diretórios numéricos.
        resolution (str): Chave da resolução desejada (ex.: ``"mid_res"``).
        subset_name (str): Nome do split (ex.: ``"train"``, ``"val"`` ou ``"test"``).

    Returns:
        Path | None: Caminho do diretório numérico ou ``None`` se não configurado.
    """
    # Busca a pasta da resolucao solicitada.
    split_paths = split_paths_by_resolution.get(resolution, {})
    if subset_name not in split_paths:
        return None
    return Path(split_paths[subset_name])


def load_split_results(split_paths_by_resolution, resolution, subset_name):
    """Carrega resultados por imagem, com fallback para runs legados.

    Args:
        split_paths_by_resolution (Mapping[str, Mapping[str, str | Path]]): Mapa
            aninhado de resolução e split para diretórios numéricos.
        resolution (str): Chave da resolução desejada.
        subset_name (str): Nome do split desejado.

    Returns:
        pd.DataFrame | None: Tabela com aliases internos ou ``None`` quando o
        par resolução/split não existe no mapeamento.
    """
    # Resolve o diretório configurado e percorre candidatos atuais e legados.
    numeric_dir = _split_numeric_dir(split_paths_by_resolution, resolution, subset_name)
    if numeric_dir is None:
        return None

    inspected: list[Path] = []
    for results_path in result_candidates(numeric_dir, subset_name):
        if not results_path.is_file():
            continue
        inspected.append(results_path)
        result = pd.read_csv(results_path)
        if "IMG_ID" in result.columns:
            return add_internal_result_aliases(result)
    # Dispara erro explícito caso nenhum candidato válido contenha IMG_ID.
    raise FileNotFoundError(
        f"Resultados por imagem não encontrados para {resolution}/{subset_name}: "
        + ", ".join(
            str(path)
            for path in inspected or result_candidates(numeric_dir, subset_name)
        )
    )


def load_split_batch_timings(split_paths_by_resolution, resolution, subset_name):
    """Carrega o manifest de tempos do split, com fallback para o nome legado.

    Args:
        split_paths_by_resolution (Mapping[str, Mapping[str, str | Path]]): Mapa
            aninhado de resolução e split para diretórios numéricos.
        resolution (str): Chave da resolução desejada.
        subset_name (str): Nome do split desejado.

    Returns:
        pd.DataFrame | None: Tabela de tempos do lote ou ``None`` se ausente.
    """
    numeric_dir = _split_numeric_dir(split_paths_by_resolution, resolution, subset_name)
    if numeric_dir is None:
        return None
    timing_path = next(
        (
            path
            for path in batch_timings_candidates(numeric_dir, subset_name)
            if path.is_file()
        ),
        None,
    )
    return pd.read_csv(timing_path) if timing_path is not None else None


def build_split_resolution_summary(
    split_paths_by_resolution,
    valid_splits=("train", "val", "test"),
):
    """Monta a tabela consumida pelos gráficos de split e resolução.

    Mantém pares ausentes com ``is_available=False`` para que os notebooks
    relatem coleções incompletas sem lógica especial de carregamento.

    Args:
        split_paths_by_resolution (Mapping[str, Mapping[str, str | Path]]): Mapa
            aninhado de resolução e split para diretórios numéricos.
        valid_splits (Sequence[str]): Ordem dos splits incluídos no resumo
            (padrão ``("train", "val", "test")``).

    Returns:
        pd.DataFrame: Tabela ordenada por split e resolução com médias de Dice,
        sucesso dos óstios, tempo em minutos e aliases em português.
    """
    from utils.project.results.io import summarize_batch_timing_records
    from utils.project.results.schema import summarize_results_df

    from .bad_cases import filter_correct_ostia_cases

    # Percorre cada par resolução/split registrando marcador vazio se indisponível.
    rows = []
    for resolution in split_paths_by_resolution:
        for subset_name in valid_splits:
            results_df = load_split_results(
                split_paths_by_resolution,
                resolution,
                subset_name,
            )

            if results_df is None:
                rows.append(
                    {
                        "subset": subset_name,
                        "resolution": resolution,
                        "num_images": np.nan,
                        "mean_dice": np.nan,
                        "mean_dice_correct": np.nan,
                        "mean_dice_all": np.nan,
                        "execution_time_min": np.nan,
                        "total_success_percent": np.nan,
                        "total_ostia_success": np.nan,
                        "is_available": False,
                    }
                )
                continue

            # Resume Dice para todos os casos e para óstios aceitos.
            dice_all = numeric_series(results_df, "dice_artery").dropna()
            correct_cases = filter_correct_ostia_cases(results_df)
            dice_correct = numeric_series(correct_cases, "dice_artery").dropna()

            # Consolida tempos de execução do lote e taxa de sucesso total dos óstios.
            result_summary = summarize_results_df(results_df)
            timings = load_split_batch_timings(
                split_paths_by_resolution,
                resolution,
                subset_name,
            )
            timing_records = [] if timings is None else timings.to_dict("records")
            execution_value = summarize_batch_timing_records(timing_records).get(
                "total_known_duration_seconds"
            )
            execution_time_seconds = (
                float(execution_value) if execution_value is not None else None
            )
            num_images = len(results_df)
            success_value = result_summary["total_success_percent"]
            total_success_percent = (
                float(success_value) if success_value is not None else None
            )
            if total_success_percent is not None:
                total_ostia_success = (num_images * 2) * (total_success_percent / 100)
            else:
                total_ostia_success = np.nan

            rows.append(
                {
                    "subset": subset_name,
                    "resolution": resolution,
                    "num_images": num_images,
                    "mean_dice": (
                        dice_correct.mean() if not dice_correct.empty else np.nan
                    ),
                    "mean_dice_correct": (
                        dice_correct.mean() if not dice_correct.empty else np.nan
                    ),
                    "mean_dice_all": (
                        dice_all.mean() if not dice_all.empty else np.nan
                    ),
                    "execution_time_min": (
                        execution_time_seconds / 60
                        if execution_time_seconds is not None
                        else np.nan
                    ),
                    "total_success_percent": total_success_percent,
                    "total_ostia_success": total_ostia_success,
                    "is_available": True,
                }
            )

    # Ordena os resultados segundo a sequência canônica de splits e resolução.
    summary = pd.DataFrame(rows)
    summary["subset"] = pd.Categorical(
        summary["subset"],
        categories=list(valid_splits),
        ordered=True,
    )
    summary = summary.sort_values(["subset", "resolution"]).reset_index(drop=True)

    # Mantém os aliases usados pelos helpers de visualização existentes.
    aliases = {
        "resolucao": "resolution",
        "num_imagens": "num_images",
        "dice_medio": "mean_dice",
        "dice_medio_correto": "mean_dice_correct",
        "dice_medio_todos": "mean_dice_all",
        "tempo_execucao_min": "execution_time_min",
        "sucesso_total_percent": "total_success_percent",
        "ostios_sucesso_total": "total_ostia_success",
        "disponivel": "is_available",
    }
    for alias, source in aliases.items():
        summary[alias] = require_series_column(summary, source)
    return summary
