"""Associa os alertas automáticos da aorta à avaliação visual externa."""

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from utils.project.dataframe import require_series_column

AORTA_FEEDBACK_LABELS = {
    "adequate": "Adequada",
    "suspected_undersegmentation": "Suspeita de subsegmentação",
    "suspected_oversegmentation": "Suspeita de sobresegmentação",
    "insufficient_data": "Dados insuficientes",
}
AORTA_FEEDBACK_ORDER = (*AORTA_FEEDBACK_LABELS.values(), "Sem feedback")
RESULT_COLUMNS = ("dataset", "exam_id", "aorta_segmentation_feedback")


def attach_aorta_feedback(
    assessments: pd.DataFrame,
    result_paths: Mapping[str, str | Path],
) -> pd.DataFrame:
    """Pareia alertas e exames visuais, preservando exames sem feedback."""
    required_visual = {"dataset", "image_id"}
    missing = sorted(required_visual.difference(assessments.columns))
    if missing:
        raise ValueError(f"Colunas ausentes na avaliação visual: {missing}")
    visual_identity = assessments.loc[:, ["dataset", "image_id"]]
    if visual_identity.isna().any().any():
        raise ValueError("A avaliação visual contém banco ou ID ausente.")
    if visual_identity.duplicated().any():
        raise ValueError("A avaliação visual contém IDs duplicados.")
    if set(require_series_column(assessments, "dataset")) != set(result_paths):
        raise ValueError("Os bancos da avaliação visual e dos resultados diferem.")

    frames: list[pd.DataFrame] = []
    for dataset, raw_path in result_paths.items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"CSV de resultados não encontrado: {path}")
        results = pd.read_csv(path, dtype={"dataset": "string", "exam_id": "string"})
        missing = sorted(set(RESULT_COLUMNS).difference(results.columns))
        if missing:
            raise ValueError(f"{path}: colunas obrigatórias ausentes: {missing}")
        results = results.loc[:, RESULT_COLUMNS].copy()
        dataset_values = require_series_column(results, "dataset")
        ids = require_series_column(results, "exam_id")
        if (
            dataset_values.isna().any()
            or not dataset_values.eq(dataset).all()
            or ids.isna().any()
            or ids.duplicated().any()
        ):
            raise ValueError(f"{path}: banco ou IDs inválidos nos resultados.")

        feedback = require_series_column(results, "aorta_segmentation_feedback")
        unknown = sorted(set(feedback.dropna()).difference(AORTA_FEEDBACK_LABELS))
        if unknown:
            raise ValueError(f"{path}: feedback desconhecido: {unknown}")
        visual_ids = require_series_column(
            assessments.loc[require_series_column(assessments, "dataset").eq(dataset)],
            "image_id",
        )
        extra = ids.loc[~ids.isin(visual_ids)]
        if not extra.empty:
            raise ValueError(
                f"{path}: resultados sem avaliação visual: {extra.tolist()}"
            )
        frames.append(results.rename(columns={"exam_id": "image_id"}))

    numeric = pd.concat(frames, ignore_index=True)
    paired = assessments.merge(
        numeric.loc[:, ["dataset", "image_id", "aorta_segmentation_feedback"]],
        on=["dataset", "image_id"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    paired["feedback"] = (
        require_series_column(paired, "aorta_segmentation_feedback")
        .map(AORTA_FEEDBACK_LABELS)
        .fillna("Sem feedback")
    )
    return paired


__all__ = [
    "AORTA_FEEDBACK_LABELS",
    "AORTA_FEEDBACK_ORDER",
    "attach_aorta_feedback",
]
