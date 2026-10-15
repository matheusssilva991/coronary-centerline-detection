from pathlib import Path

import pandas as pd

from ..project.results import add_internal_result_aliases
from ..project.results.schema import normalize_result_status
from utils.project.results.paths import (
    batch_timings_candidates,
    metadata_candidates,
    result_candidates,
    summary_candidates,
)
from ..utils import load_json_file


def load_experiment_variant_results(
    run_dir: str | Path, *, variant: str, split: str
) -> pd.DataFrame:
    """Carrega uma variante de sweep no schema de resultados por exame.

    Preserva o Dice registrado e atribui zero apenas a falhas documentadas sem
    tentativa de segmentação. Ausências sem diagnóstico são rejeitadas.
    """
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


def load_split_metadata(split_paths_by_resolution, resolution, subset_name):
    """Carrega o metadata de uma resolução e split, se disponível."""
    # Busca a pasta da resolucao solicitada.
    split_paths = split_paths_by_resolution.get(resolution, {})
    if subset_name not in split_paths:
        # Subset ausente: nada para carregar.
        return None

    numeric_dir = Path(split_paths[subset_name])
    metadata_path = next(
        (
            path
            for path in metadata_candidates(numeric_dir, subset_name)
            if path.is_file()
        ),
        None,
    )
    return load_json_file(str(metadata_path)) if metadata_path else None


def _split_numeric_dir(split_paths_by_resolution, resolution, subset_name):
    """Resolve o diretório numérico de um split configurado."""
    # Busca a pasta da resolucao solicitada.
    split_paths = split_paths_by_resolution.get(resolution, {})
    if subset_name not in split_paths:
        return None
    return Path(split_paths[subset_name])


def load_split_results(split_paths_by_resolution, resolution, subset_name):
    """Carrega resultados por imagem, com fallback para runs legados."""
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
    raise FileNotFoundError(
        f"Resultados por imagem não encontrados para {resolution}/{subset_name}: "
        + ", ".join(
            str(path)
            for path in inspected or result_candidates(numeric_dir, subset_name)
        )
    )


def load_split_batch_timings(split_paths_by_resolution, resolution, subset_name):
    """Carrega o manifest de tempos do split, com fallback para o nome legado."""
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


def load_split_summary(split_paths_by_resolution, resolution, subset_name):
    """Carrega um resumo legado; novos runs calculam agregados dos resultados."""
    numeric_dir = _split_numeric_dir(split_paths_by_resolution, resolution, subset_name)
    if numeric_dir is None:
        return None

    legacy_per_image = None
    for summary_path in summary_candidates(numeric_dir, subset_name):
        if not summary_path.is_file():
            continue
        summary = pd.read_csv(summary_path)
        if "IMG_ID" not in summary.columns:
            return summary
        legacy_per_image = summary_path
    if legacy_per_image is not None:
        raise ValueError(
            f"{legacy_per_image} usa o formato legado por imagem; "
            "use load_split_results()."
        )
    raise FileNotFoundError(
        f"Resumo agregado não encontrado para {resolution}/{subset_name}."
    )
