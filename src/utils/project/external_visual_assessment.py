"""Carrega e resume avaliações visuais de CCTA externas."""

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
VISUAL_SUBSETS = ("train", "test")
FULLY_ADEQUATE_STATUS = {
    "aorta": ("aorta_result", "Adequada"),
    "ostia": ("ostia_result", "Ambos adequados"),
    "artery": ("artery_result", "Adequada"),
}


def _clean_text_column(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _validate_assessment_frame(
    frame: pd.DataFrame,
    *,
    dataset: str,
    source_path: Path,
) -> pd.DataFrame:
    """Valida colunas, IDs e status de uma avaliação visual."""
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
    """Carrega e valida uma planilha de avaliação visual por banco."""
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


def load_assessment_subset_lookup(
    result_paths: Mapping[str, Sequence[str | Path]],
) -> pd.DataFrame:
    """Carrega o split de cada exame a partir dos resultados de vários runs."""
    if not result_paths:
        raise ValueError("Nenhum CSV de resultados foi informado.")

    frames: list[pd.DataFrame] = []
    required = {"dataset", "exam_id", "subset"}
    for dataset, paths in result_paths.items():
        if not paths:
            raise ValueError(f"{dataset}: nenhum CSV de resultados foi informado.")
        for raw_path in paths:
            path = Path(raw_path)
            if not path.is_file():
                raise FileNotFoundError(f"CSV de resultados não encontrado: {path}")
            frame = pd.read_csv(
                path,
                dtype={"dataset": "string", "exam_id": "string", "subset": "string"},
            )
            missing = sorted(required.difference(frame.columns))
            if missing:
                raise ValueError(f"{path}: colunas obrigatórias ausentes: {missing}")
            identity = frame.loc[:, ["dataset", "exam_id", "subset"]].copy()
            if identity.empty or identity.isna().any().any():
                raise ValueError(f"{path}: identidade de exames vazia ou incompleta.")
            if not identity["dataset"].eq(dataset).all():
                raise ValueError(f"{path}: banco diferente de {dataset!r}.")
            if not identity["subset"].isin(VISUAL_SUBSETS).all():
                raise ValueError(f"{path}: split diferente de train/test.")
            if identity["exam_id"].duplicated().any():
                raise ValueError(f"{path}: IDs de exame duplicados.")
            frames.append(identity.rename(columns={"exam_id": "image_id"}))

    lookup = pd.concat(frames, ignore_index=True)
    conflicts = lookup.groupby(["dataset", "image_id"])["subset"].nunique()
    if conflicts.gt(1).any():
        raise ValueError(
            "Splits contraditórios entre runs: "
            f"{conflicts.index[conflicts.gt(1)].tolist()}"
        )
    return lookup.drop_duplicates(["dataset", "image_id"]).reset_index(drop=True)


def attach_assessment_subsets(
    assessments: pd.DataFrame,
    subset_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Associa splits às avaliações visuais e rejeita exames sem correspondência."""
    if "subset" in assessments.columns:
        raise ValueError("As avaliações visuais já contêm a coluna subset.")
    required_lookup = {"dataset", "image_id", "subset"}
    missing = sorted(required_lookup.difference(subset_lookup.columns))
    if missing:
        raise ValueError(f"Colunas ausentes no mapa de splits: {missing}")
    if subset_lookup.duplicated(["dataset", "image_id"]).any():
        raise ValueError("O mapa de splits contém IDs duplicados.")

    result = assessments.merge(
        subset_lookup.loc[:, ["dataset", "image_id", "subset"]],
        on=["dataset", "image_id"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    unknown = result.loc[result["subset"].isna(), ["dataset", "image_id"]]
    if not unknown.empty:
        raise ValueError(
            "Avaliações visuais sem split nos CSVs: "
            f"{list(unknown.itertuples(index=False, name=None))}"
        )
    return result


def summarize_visual_overview(
    assessments: pd.DataFrame,
    dataset_order: Sequence[str],
) -> pd.DataFrame:
    """Resume acertos visuais por banco e split, com totais ponderados."""
    required = {"dataset", "subset", *STATUS_ORDER_BY_COLUMN}
    missing = sorted(required.difference(assessments.columns))
    if missing:
        raise ValueError(f"Colunas ausentes na avaliação visual: {missing}")
    if not assessments["subset"].isin(VISUAL_SUBSETS).all():
        raise ValueError("A avaliação visual contém split diferente de train/test.")

    cohorts = [
        (
            dataset,
            subset,
            assessments.loc[
                assessments["dataset"].eq(dataset) & assessments["subset"].eq(subset)
            ],
        )
        for dataset in dataset_order
        for subset in VISUAL_SUBSETS
    ]
    cohorts.extend(
        (
            "Geral",
            subset,
            assessments.loc[assessments["subset"].eq(subset)],
        )
        for subset in VISUAL_SUBSETS
    )
    cohorts.append(("Geral", "total", assessments))

    rows: list[dict[str, object]] = []
    for dataset, subset, cohort in cohorts:
        if cohort.empty:
            raise ValueError(f"Nenhuma avaliação encontrada para {dataset}/{subset}.")
        row: dict[str, object] = {
            "dataset": dataset,
            "subset": subset,
            "exam_count": len(cohort),
        }
        for stage, (column, adequate_status) in FULLY_ADEQUATE_STATUS.items():
            count = int(cohort[column].eq(adequate_status).sum())
            row[f"{stage}_adequate_count"] = count
            row[f"{stage}_adequate_percent"] = 100.0 * count / len(cohort)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_visual_status(
    assessments: pd.DataFrame,
    column: str,
    status_order: Sequence[str],
    dataset_order: Sequence[str],
) -> pd.DataFrame:
    """Retorna contagens e percentuais por banco e no agregado ponderado."""
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
    "attach_assessment_subsets",
    "load_assessment_subset_lookup",
    "load_external_visual_assessments",
    "summarize_visual_overview",
    "summarize_visual_status",
]
