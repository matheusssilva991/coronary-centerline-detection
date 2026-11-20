"""Carrega e valida revisões manuais usadas nas EDAs da aorta."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Collection

import numpy as np
import pandas as pd

from utils.project.results.io import results_filename
from utils.project.results.schema import normalize_ostia_status

from ..comparison_utils.io import load_split_results
from ..project.dataframe import (
    numeric_series,
    require_series_column,
    to_numeric_series,
)

AORTA_REVIEW_ID_FIELDS = (
    "aorta_good_ids",
    "aorta_bad_ids",
)
OSTIA_REVIEW_ID_FIELDS = (
    "ostia_good_ids",
    "ostia_bad_ids",
)


def load_aorta_visual_reviews(path: str | Path) -> dict[str, Any]:
    """Carrega o catálogo visual e valida cada variante e split.

    Args:
        path (str | Path): Caminho do arquivo JSON com o catálogo de revisões.

    Returns:
        dict[str, Any]: Estrutura JSON validada contendo grupos por variante e
        split.
    """
    review_path = Path(path)
    data = json.loads(review_path.read_text(encoding="utf-8"))
    variants = data.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise ValueError("The visual-review catalog must contain variants.")

    for variant, split_reviews in variants.items():
        if not isinstance(split_reviews, dict):
            raise ValueError(f"Invalid review groups for variant {variant!r}.")
        for split, review in split_reviews.items():
            _validate_review(review, variant, split)
    return data


def get_aorta_visual_review(
    catalog: dict[str, Any],
    variant: str,
    split: str,
) -> dict[str, Any]:
    """Retorna uma revisão com IDs em conjuntos e chaves de notas inteiras.

    Args:
        catalog (dict[str, Any]): Catálogo carregado por
            ``load_aorta_visual_reviews``.
        variant (str): Nome da variante desejada.
        split (str): Nome do split desejado.

    Returns:
        dict[str, Any]: Dicionário da revisão com conjuntos de IDs inteiros e
        notas indexadas por ``IMG_ID`` inteiro.
    """
    try:
        raw_review = catalog["variants"][variant][split]
    except KeyError as exc:
        raise KeyError(
            f"Review not found for variant={variant!r}, split={split!r}."
        ) from exc

    # Converte listas de IDs e chaves das notas para inteiros visando buscas O(1).
    review = dict(raw_review)
    for field in (*AORTA_REVIEW_ID_FIELDS, *OSTIA_REVIEW_ID_FIELDS):
        if field in raw_review:
            review[field] = {int(img_id) for img_id in raw_review[field]}
    review["notes"] = {
        int(img_id): note for img_id, note in raw_review.get("notes", {}).items()
    }
    return review


def resolve_aorta_review_results_path(
    repo_root: str | Path,
    review: dict[str, Any],
    split: str,
) -> Path:
    """Resolve o CSV por imagem associado a uma entrada do catálogo.

    Args:
        repo_root (str | Path): Raiz do repositório usada com ``review["run_dir"]``.
        review (dict[str, Any]): Dicionário de revisão contendo ``run_dir``.
        split (str): Nome do split associado ao arquivo de resultados.

    Returns:
        Path: Caminho do CSV atual por imagem ou do arquivo legado equivalente.
    """
    numeric_dir = Path(repo_root) / review["run_dir"] / "numeric"
    current = numeric_dir / results_filename(split)
    legacy = numeric_dir / f"ostios_{split}_results.csv"
    return current if current.is_file() or not legacy.is_file() else legacy


# Compatibilidade com notebooks externos anteriores à separação results/summary.
resolve_aorta_review_summary_path = resolve_aorta_review_results_path


def add_aorta_extent_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas axiais comparáveis entre círculos e máscara da aorta.

    Valores positivos em ``segmented_minus_circle_slices`` indicam uma máscara
    final mais extensa que a trajetória; valores negativos indicam retração.

    Args:
        dataframe (pd.DataFrame): Tabela de resultados contendo contagens de
            fatias da imagem, dos círculos e da aorta segmentada.

    Returns:
        pd.DataFrame: Cópia enriquecida com frações de cobertura axial, delta de
        fatias e posições normalizadas da trajetória quando disponíveis.
    """
    # Valida colunas obrigatórias e evita divisões por zero nas razões axiais.
    df = dataframe.copy()
    required = {
        "image_slice_count",
        "aorta_circle_count",
        "aorta_segmented_slice_count",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing aorta extent columns: {sorted(missing)}")

    image_slices = numeric_series(df, "image_slice_count")
    circle_slices = numeric_series(df, "aorta_circle_count")
    segmented_slices = numeric_series(df, "aorta_segmented_slice_count")
    valid_image_slices = image_slices.where(image_slices.gt(0))
    valid_circle_slices = circle_slices.where(circle_slices.gt(0))

    # Calcula frações axiais e diferenças entre a máscara segmentada e os círculos.
    df["circle_slice_fraction"] = circle_slices / valid_image_slices
    df["segmented_slice_fraction"] = segmented_slices / valid_image_slices
    df["segmented_minus_circle_slices"] = segmented_slices - circle_slices
    df["segmented_vs_circle_change_fraction"] = (
        segmented_slices - circle_slices
    ) / valid_circle_slices

    # Adiciona percentual volumétrico e posições normalizadas quando disponíveis.
    if "aorta_volume_fraction" in df.columns:
        df["aorta_volume_percentage"] = (
            numeric_series(df, "aorta_volume_fraction") * 100.0
        )
    if {
        "aorta_circle_first_slice",
        "aorta_circle_last_slice",
    }.issubset(df.columns):
        first = numeric_series(df, "aorta_circle_first_slice")
        last = numeric_series(df, "aorta_circle_last_slice")
        df["circle_first_position"] = first / valid_image_slices
        df["circle_last_position"] = last / valid_image_slices
        df["circle_center_position"] = (first + last) / (2.0 * valid_image_slices)
    return df


def load_aorta_review_cohort(
    repo_root: str | Path,
    review: dict[str, Any],
    split: str,
    *,
    cohort_name: str | None = None,
    required_columns: Collection[str] = (),
    use_reviewed_ostia_labels: bool = False,
) -> pd.DataFrame:
    """Carrega um run revisado e adiciona rótulos visuais, de óstios e axiais.

    Args:
        repo_root (str | Path): Raiz do repositório para resolver o diretório do run.
        review (dict[str, Any]): Entrada de revisão obtida via
            ``get_aorta_visual_review``.
        split (str): Nome do split carregado.
        cohort_name (str | None): Nome opcional para a coluna ``coorte``; se
            ``None``, utiliza o valor de ``split``.
        required_columns (Collection[str]): Colunas extras exigidas no CSV.
        use_reviewed_ostia_labels (bool): Se ``True``, utiliza ``ostia_bad_ids``
            da revisão manual em vez do status registrado no CSV.

    Returns:
        pd.DataFrame: Tabela ordenada por ``IMG_ID`` com rótulos visuais da
        aorta, desfecho dos óstios e métricas de extensão axial.
    """
    # Carrega o CSV do run revisado e verifica presença das colunas exigidas.
    results_path = resolve_aorta_review_results_path(repo_root, review, split)
    numeric_dir = results_path.parent
    dataframe = load_split_results({"mid_res": {split: numeric_dir}}, "mid_res", split)
    if dataframe is None:
        raise RuntimeError(f"Could not load the {split!r} results.")

    missing = set(required_columns).difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing result columns for {split!r}: {sorted(missing)}")

    # Garante correspondência exata entre os IDs do CSV e os grupos revisados.
    df = dataframe.copy()
    image_ids = to_numeric_series(require_series_column(df, "IMG_ID"), errors="raise")
    df["IMG_ID"] = image_ids.astype(int)
    good_ids = {int(img_id) for img_id in review["aorta_good_ids"]}
    bad_ids = {int(img_id) for img_id in review["aorta_bad_ids"]}
    expected_ids = good_ids | bad_ids
    image_ids = require_series_column(df, "IMG_ID")
    observed_ids = set(image_ids)
    if expected_ids != observed_ids:
        raise ValueError(
            f"Incompatible IDs for {split!r}. "
            f"Missing={sorted(expected_ids - observed_ids)}; "
            f"unclassified={sorted(observed_ids - expected_ids)}"
        )

    # Anexa qualidade visual, notas manuais, desfecho dos óstios e métricas axiais.
    df["visual_aorta_quality"] = np.where(
        image_ids.isin(tuple(good_ids)), "boa", "ruim"
    )
    notes = review.get("notes", {})
    df["visual_review_note"] = image_ids.map(lambda value: notes.get(value)).fillna("")
    ostia_status = require_series_column(df, "ostia_detection_status")
    normalized_status = ostia_status.map(normalize_ostia_status)
    csv_success = normalized_status.isin(("both_correct", "both_tolerable"))
    if use_reviewed_ostia_labels:
        bad_ostia_ids = {int(img_id) for img_id in review.get("ostia_bad_ids", ())}
        df["ostia_success"] = ~image_ids.isin(tuple(bad_ostia_ids))
    else:
        df["ostia_success"] = csv_success
    df["ostia_outcome"] = np.where(df["ostia_success"], "sucesso", "falha")
    df["coorte"] = cohort_name or split
    extent_columns = {
        "image_slice_count",
        "aorta_circle_count",
        "aorta_segmented_slice_count",
    }
    if extent_columns.issubset(df.columns):
        df = add_aorta_extent_metrics(df)
    return df.sort_values("IMG_ID").reset_index(drop=True)


def _validate_review(review: Any, variant: str, split: str) -> None:
    """Rejeita classificações manuais incompletas ou contraditórias.

    Args:
        review (Any): Entrada de revisão lida do catálogo JSON.
        variant (str): Nome da variante usado em mensagens de erro.
        split (str): Nome do split usado em mensagens de erro.
    """
    # Garante presença de run_dir e dos campos obrigatórios de revisão da aorta.
    if not isinstance(review, dict) or not review.get("run_dir"):
        raise ValueError(f"Missing run_dir for variant={variant!r}, split={split!r}.")
    missing = [field for field in AORTA_REVIEW_ID_FIELDS if field not in review]
    if missing:
        raise ValueError(
            f"Missing review fields for variant={variant!r}, split={split!r}: {missing}"
        )

    # Converte IDs e exige par completo caso a revisão de óstios esteja presente.
    groups = {
        field: {int(img_id) for img_id in review[field]}
        for field in (*AORTA_REVIEW_ID_FIELDS, *OSTIA_REVIEW_ID_FIELDS)
        if field in review
    }
    ostia_fields_present = [field in review for field in OSTIA_REVIEW_ID_FIELDS]
    if any(ostia_fields_present) and not all(ostia_fields_present):
        raise ValueError(
            f"Incomplete ostia labels for variant={variant!r}, split={split!r}."
        )

    # Verifica ausência de sobreposições e cobertura idêntica entre aorta e óstios.
    subjects = ["aorta"]
    if all(ostia_fields_present):
        subjects.append("ostia")
    for subject in subjects:
        good = groups[f"{subject}_good_ids"]
        bad = groups[f"{subject}_bad_ids"]
        overlap = good & bad
        if overlap:
            raise ValueError(
                f"Contradictory {subject} labels for variant={variant!r}, "
                f"split={split!r}: {sorted(overlap)}"
            )
        if good | bad != groups["aorta_good_ids"] | groups["aorta_bad_ids"]:
            raise ValueError(
                f"The {subject} labels do not cover the same cohort for "
                f"variant={variant!r}, split={split!r}."
            )


__all__ = [
    "add_aorta_extent_metrics",
    "get_aorta_visual_review",
    "load_aorta_review_cohort",
    "load_aorta_visual_reviews",
    "resolve_aorta_review_results_path",
    "resolve_aorta_review_summary_path",
]
