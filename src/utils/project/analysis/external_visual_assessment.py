"""Carrega e resume avaliações visuais de CCTA externas."""

import warnings
from collections.abc import Mapping, Sequence
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
    """Converte uma coluna para string do pandas removendo espaços nas pontas.

    Args:
        series (pd.Series): Série tabular contendo rótulos textuais ou valores nulos.

    Returns:
        pd.Series: Série textual normalizada com dtype ``string``.
    """
    return series.astype("string").str.strip()


def _validate_assessment_frame(
    frame: pd.DataFrame,
    *,
    dataset: str,
    source_path: Path,
) -> pd.DataFrame:
    """Valida colunas, IDs e status de uma avaliação visual.

    Args:
        frame (pd.DataFrame): DataFrame bruto lido da aba de avaliação visual.
        dataset (str): Nome do banco associado à planilha.
        source_path (Path): Caminho do arquivo Excel utilizado em mensagens de erro.

    Returns:
        pd.DataFrame: DataFrame limpo e validado com a coluna ``dataset`` inserida.
    """
    # Garante presença das colunas obrigatórias, descarta linhas vazias e limpa textos.
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

    # Verifica preenchimento das colunas exigidas e unicidade dos IDs de exame.
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

    # Valida categorias permitidas para cada etapa e preenche notas opcionais vazias.
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
    """Carrega e valida uma planilha de avaliação visual por banco.

    Args:
        assessment_paths (Mapping[str, str | Path]): Mapeamento de nome do banco para
            o caminho da planilha Excel de avaliação.

    Returns:
        pd.DataFrame: Avaliações visuais validadas e concatenadas entre os bancos.
    """
    if not assessment_paths:
        raise ValueError("Nenhum arquivo de avaliação visual foi informado.")

    frames: list[pd.DataFrame] = []
    for dataset, raw_path in assessment_paths.items():
        # Verifica existência da planilha antes de tentar abrir a aba esperada.
        source_path = Path(raw_path)
        if not source_path.is_file():
            raise FileNotFoundError(
                f"{dataset}: arquivo de avaliação visual não encontrado: {source_path}"
            )

        # Lê a aba de avaliação visual suprimindo avisos conhecidos do openpyxl.
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

        # Valida esquema e status da planilha antes de incorporar ao resultado.
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
    """Carrega o split de cada exame a partir dos resultados de vários runs.

    Args:
        result_paths (Mapping[str, Sequence[str | Path]]): Mapeamento de cada banco
            para sequências de CSVs de resultados numéricos.

    Returns:
        pd.DataFrame: Tabela única com ``dataset``, ``image_id`` e ``subset`` sem
            conflitos entre runs.
    """
    if not result_paths:
        raise ValueError("Nenhum CSV de resultados foi informado.")

    # Lê e valida colunas de identidade, banco e split em cada CSV informado.
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

    # Garante ausência de splits conflitantes para o mesmo exame entre runs.
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
    """Associa splits às avaliações visuais e rejeita exames sem correspondência.

    Args:
        assessments (pd.DataFrame): Avaliações visuais ainda sem coluna ``subset``.
        subset_lookup (pd.DataFrame): Tabela de correspondência gerada por
            ``load_assessment_subset_lookup``.

    Returns:
        pd.DataFrame: Avaliações visuais enriquecidas com a coluna ``subset``.
    """
    # Valida pré-condições das tabelas antes de executar a junção.
    if "subset" in assessments.columns:
        raise ValueError("As avaliações visuais já contêm a coluna subset.")
    required_lookup = {"dataset", "image_id", "subset"}
    missing = sorted(required_lookup.difference(subset_lookup.columns))
    if missing:
        raise ValueError(f"Colunas ausentes no mapa de splits: {missing}")
    if subset_lookup.duplicated(["dataset", "image_id"]).any():
        raise ValueError("O mapa de splits contém IDs duplicados.")

    # Associa splits por (dataset, image_id) e rejeita avaliações sem correspondência.
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


def attach_aorta_feedback(
    assessments: pd.DataFrame,
    result_paths: Mapping[str, str | Path],
) -> pd.DataFrame:
    """Pareia alertas e exames visuais, preservando exames sem feedback.

    Args:
        assessments (pd.DataFrame): Avaliações visuais contendo ``dataset`` e
            ``image_id``.
        result_paths (Mapping[str, str | Path]): Mapeamento de cada banco para o
            caminho do CSV de resultados correspondente.

    Returns:
        pd.DataFrame: Avaliações pareadas com ``aorta_segmentation_feedback`` e
            rótulo descritivo em ``feedback``.
    """
    # Valida chaves de identidade nas avaliações e correspondência exata de bancos.
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

    # Lê e valida feedbacks de cada CSV, garantindo cobertura na avaliação visual.
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

    # Realiza merge um-para-um e rotula exames sem alerta como 'Sem feedback'.
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


def summarize_visual_overview(
    assessments: pd.DataFrame,
    dataset_order: Sequence[str],
) -> pd.DataFrame:
    """Resume acertos visuais por banco e split, com totais ponderados.

    Args:
        assessments (pd.DataFrame): Avaliações visuais com colunas ``dataset``,
            ``subset`` e desfechos por etapa.
        dataset_order (Sequence[str]): Ordem explícita de exibição dos bancos.

    Returns:
        pd.DataFrame: Contagens e percentuais de adequação total por coorte e no
            agregado ``Geral``.
    """
    # Valida colunas de desfecho e garante splits restritos a treino e teste.
    required = {"dataset", "subset", *STATUS_ORDER_BY_COLUMN}
    missing = sorted(required.difference(assessments.columns))
    if missing:
        raise ValueError(f"Colunas ausentes na avaliação visual: {missing}")
    if not assessments["subset"].isin(VISUAL_SUBSETS).all():
        raise ValueError("A avaliação visual contém split diferente de train/test.")

    # Constrói coortes por banco/split seguidas dos totais ponderados em Geral.
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

    # Contabiliza casos plenamente adequados e respectivos percentuais por etapa.
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
    """Retorna contagens e percentuais por banco e no agregado ponderado.

    Args:
        assessments (pd.DataFrame): Tabela consolidada de avaliações visuais.
        column (str): Nome da coluna de resultado a tabular.
        status_order (Sequence[str]): Ordem fixa dos status na saída.
        dataset_order (Sequence[str]): Ordem explícita dos bancos na tabela.

    Returns:
        pd.DataFrame: Distribuição absoluta e percentual dos status por banco e em
            ``Geral``.
    """
    # Monta coortes individuais na ordem pedida e acrescenta o agregado Geral.
    rows: list[dict[str, object]] = []
    cohorts = [
        *(
            (dataset, assessments.loc[assessments["dataset"].eq(dataset)])
            for dataset in dataset_order
        ),
        ("Geral", assessments),
    ]

    # Reindexa contagens pela ordem fixa dos status e calcula percentuais por coorte.
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
    "AORTA_FEEDBACK_LABELS",
    "AORTA_FEEDBACK_ORDER",
    "ASSESSMENT_COLUMNS",
    "ASSESSMENT_SHEET_NAME",
    "OSTIA_STATUS_ORDER",
    "attach_aorta_feedback",
    "attach_assessment_subsets",
    "load_assessment_subset_lookup",
    "load_external_visual_assessments",
    "summarize_visual_overview",
    "summarize_visual_status",
]
