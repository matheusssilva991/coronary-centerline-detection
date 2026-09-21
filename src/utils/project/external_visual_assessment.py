"""Loading and summaries for external-CCTA visual assessments."""

from collections.abc import Mapping, Sequence
from pathlib import Path
import warnings

import pandas as pd

ASSESSMENT_SHEET_NAME = "Avaliação visual"
ASSESSMENT_COLUMNS = (
    "image_id",
    "aorta_result",
    "ostia_result",
    "artery_result",
    "visual_assessment_notes",
)
REQUIRED_VALUE_COLUMNS = ASSESSMENT_COLUMNS[:-1]

AORTA_ARTERY_STATUS_ORDER = (
    "Adequada",
    "Parcial",
    "Inadequada",
    "Não detectada",
    "Não avaliável",
)
OSTIA_STATUS_ORDER = (
    "Ambos adequados",
    "Um adequado",
    "Inadequados",
    "Não detectados",
    "Não avaliável",
)
STATUS_ORDER_BY_COLUMN = {
    "aorta_result": AORTA_ARTERY_STATUS_ORDER,
    "ostia_result": OSTIA_STATUS_ORDER,
    "artery_result": AORTA_ARTERY_STATUS_ORDER,
}


def _clean_text_column(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _validate_assessment_frame(
    frame: pd.DataFrame,
    *,
    dataset: str,
    source_path: Path,
) -> pd.DataFrame:
    missing_columns = sorted(set(ASSESSMENT_COLUMNS).difference(frame.columns))
    if missing_columns:
        raise ValueError(
            f"{dataset}: colunas obrigatórias ausentes em {source_path}: "
            f"{missing_columns}"
        )

    result = frame.loc[:, ASSESSMENT_COLUMNS].copy()
    result = result.dropna(subset=list(ASSESSMENT_COLUMNS), how="all")
    for column in ASSESSMENT_COLUMNS:
        result[column] = _clean_text_column(result[column])

    for column in REQUIRED_VALUE_COLUMNS:
        missing_values = result[column].isna() | result[column].eq("")
        if missing_values.any():
            raise ValueError(
                f"{dataset}: {int(missing_values.sum())} valor(es) ausente(s) "
                f"na coluna {column!r}."
            )

    duplicated_ids = result.loc[
        result["image_id"].duplicated(keep=False), "image_id"
    ].unique()
    if len(duplicated_ids):
        raise ValueError(
            f"{dataset}: IDs duplicados na avaliação visual: "
            f"{sorted(duplicated_ids.tolist())}"
        )

    for column, allowed_statuses in STATUS_ORDER_BY_COLUMN.items():
        unexpected = sorted(set(result[column]).difference(allowed_statuses))
        if unexpected:
            raise ValueError(
                f"{dataset}: status desconhecido(s) em {column!r}: {unexpected}"
            )

    result["visual_assessment_notes"] = result["visual_assessment_notes"].fillna("")
    result.insert(0, "dataset", dataset)
    return result


def load_external_visual_assessments(
    assessment_paths: Mapping[str, str | Path],
) -> pd.DataFrame:
    """Load and validate one visual-assessment workbook per dataset."""
    if not assessment_paths:
        raise ValueError("Nenhum arquivo de avaliação visual foi informado.")

    frames: list[pd.DataFrame] = []
    for dataset, raw_path in assessment_paths.items():
        source_path = Path(raw_path)
        if not source_path.is_file():
            raise FileNotFoundError(
                f"{dataset}: arquivo de avaliação visual não encontrado: {source_path}"
            )
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        "(Conditional Formatting|Data Validation) extension "
                        "is not supported and will be removed"
                    ),
                    category=UserWarning,
                    module="openpyxl.worksheet._reader",
                )
                frame = pd.read_excel(
                    source_path,
                    sheet_name=ASSESSMENT_SHEET_NAME,
                    dtype="string",
                )
        except ValueError as error:
            raise ValueError(
                f"{dataset}: não foi possível ler a aba "
                f"{ASSESSMENT_SHEET_NAME!r} de {source_path}."
            ) from error
        frames.append(
            _validate_assessment_frame(
                frame,
                dataset=dataset,
                source_path=source_path,
            )
        )

    return pd.concat(frames, ignore_index=True)


def summarize_visual_status(
    assessments: pd.DataFrame,
    column: str,
    status_order: Sequence[str],
    dataset_order: Sequence[str],
) -> pd.DataFrame:
    """Return counts and percentages by dataset plus a weighted aggregate."""
    rows: list[dict[str, object]] = []
    cohorts = [
        *(
            (dataset, assessments.loc[assessments["dataset"].eq(dataset)])
            for dataset in dataset_order
        ),
        ("Geral", assessments),
    ]
    for dataset, cohort in cohorts:
        if cohort.empty:
            raise ValueError(f"Nenhuma avaliação encontrada para {dataset!r}.")
        counts = cohort[column].value_counts().reindex(status_order, fill_value=0)
        for status, count in counts.items():
            rows.append(
                {
                    "dataset": dataset,
                    "status": status,
                    "count": int(count),
                    "percent": 100.0 * int(count) / len(cohort),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "AORTA_ARTERY_STATUS_ORDER",
    "ASSESSMENT_COLUMNS",
    "ASSESSMENT_SHEET_NAME",
    "OSTIA_STATUS_ORDER",
    "load_external_visual_assessments",
    "summarize_visual_status",
]
