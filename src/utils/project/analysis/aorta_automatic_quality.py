"""Carrega e compara diagnósticos persistidos da aorta entre bancos e coortes."""

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from utils.project.analysis.external_aorta_assessment import (
    AORTA_FEEDBACK_LABELS,
    AORTA_FEEDBACK_ORDER,
)
from utils.project.runtime.notebook_env import get_default_split_paths

GROUP_COLUMNS = ["reference", "dataset", "resolution", "subset"]
RECORD_COLUMNS = [
    *GROUP_COLUMNS,
    "image_id",
    "aorta_segmentation_feedback",
    "feedback",
    "diagnostic_available",
    "source_path",
    "source_run",
]


@dataclass(frozen=True)
class AortaFeedbackSource:
    """Identifica uma coorte sem substituir fontes ausentes por outro run."""

    dataset: str
    resolution: str
    subset: str
    reference: str
    path: Path | None
    expected_count: int | None = None
    applicable: bool = True


def default_aorta_feedback_sources(
    repo_root: Path, external_root: Path | None = None
) -> list[AortaFeedbackSource]:
    """Seleciona canônicos ImageCAS e os CSVs externos locais já produzidos.

    Uma raiz externa explícita mantém a seleção dos runs externos corrigidos,
    sem recorrer aos locais se o arquivo solicitado estiver ausente.
    """
    canonical = get_default_split_paths(repo_root)
    splits = json.loads((repo_root / "config/imagecas_splits.json").read_text())[
        "splits"
    ]
    sources = []
    for resolution in ("mid", "high"):
        for subset in ("train", "val", "test"):
            directory = canonical[f"{resolution}_res"].get(subset)
            sources.append(
                AortaFeedbackSource(
                    "ImageCAS",
                    resolution,
                    subset,
                    "Atual",
                    directory / f"results_{subset}.csv" if directory else None,
                    len(splits[subset]),
                )
            )
    external_runs = {
        ("MM-WHS", "mid"): "mmwhs/mid_res/2026-10-06_07-53-03",
        ("MM-WHS", "high"): "mmwhs/high_res/2026-10-06_08-33-19",
        ("OrCaScore", "mid"): "orcascore/mid_res/2026-09-19_10-23-50",
        ("OrCaScore", "high"): "orcascore/high_res/2026-09-22_19-12-41",
    }
    if external_root is None:
        external_root = repo_root / "output/segmentation/runs"
        external_runs = {
            ("MM-WHS", "mid"): "mid_res/external_ccta/mmwhs/2026-09-22_18-51-28",
            ("MM-WHS", "high"): "high_res/external_ccta/mmwhs/2026-09-22_17-47-45",
            ("OrCaScore", "mid"): "mid_res/external_ccta/orcascore/2026-09-19_10-23-50",
            (
                "OrCaScore",
                "high",
            ): "high_res/external_ccta/orcascore/2026-09-22_19-12-41",
        }
    for (dataset, resolution), run in external_runs.items():
        for subset in ("train", "val", "test"):
            expected = (20 if dataset == "MM-WHS" else 32) if subset == "train" else 40
            sources.append(
                AortaFeedbackSource(
                    dataset,
                    resolution,
                    subset,
                    "Atual",
                    external_root / run / "numeric/results_all.csv"
                    if subset != "val"
                    else None,
                    expected if subset != "val" else None,
                    applicable=subset != "val",
                )
            )
    return sources


def load_aorta_feedback_sources(
    sources: Sequence[AortaFeedbackSource],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normaliza CSVs ImageCAS/externos e informa a cobertura de cada fonte.

    Coluna ausente difere de feedback vazio por exame. Somente o segundo recebe
    o rótulo 'Sem feedback'; nenhum diagnóstico é recalculado nesta análise.
    """
    frames = []
    coverage = []
    identities = set()
    for source in sources:
        group = dict(
            zip(
                GROUP_COLUMNS,
                (source.reference, source.dataset, source.resolution, source.subset),
                strict=True,
            )
        )
        key = tuple(group.values())
        if key in identities:
            raise ValueError(f"Coorte duplicada: {key}")
        identities.add(key)
        source_run = ""
        if source.path is not None:
            run_dir = source.path.parent.parent
            manifest = run_dir / "source_manifest.json"
            source_run = str(run_dir)
            if manifest.is_file():
                source_run = str(
                    json.loads(manifest.read_text()).get("source_run", run_dir)
                )
        row = {
            **group,
            "source_path": str(source.path) if source.path is not None else "",
            "source_run": source_run,
            "expected_count": source.expected_count,
            "exam_count": pd.NA,
            "missing_count": pd.NA,
            "with_feedback": pd.NA,
            "status": "Não se aplica" if not source.applicable else "Arquivo ausente",
        }
        if source.applicable and source.path is not None and source.path.is_file():
            raw = pd.read_csv(source.path, dtype="string")
            id_column = "IMG_ID" if source.dataset == "ImageCAS" else "exam_id"
            if id_column not in raw:
                raise ValueError(f"{source.path}: coluna de ID ausente: {id_column}")
            if (
                "dataset" in raw
                and not raw["dataset"].eq(source.dataset).fillna(False).all()
            ):
                raise ValueError(f"{source.path}: banco incompatível")
            split_column = next((c for c in ("subset", "split") if c in raw), None)
            if split_column is None and source.dataset != "ImageCAS":
                raise ValueError(f"{source.path}: coluna de subconjunto ausente")
            if split_column is not None:
                if not raw[split_column].isin(["train", "val", "test"]).all():
                    raise ValueError(f"{source.path}: subconjunto inválido")
                raw = raw.loc[raw[split_column].eq(source.subset)].copy()
            ids = raw[id_column].str.strip()
            if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
                raise ValueError(f"{source.path}: IDs ausentes ou duplicados")
            available = "aorta_segmentation_feedback" in raw
            feedback = (
                raw["aorta_segmentation_feedback"].str.strip().replace("", pd.NA)
                if available
                else pd.Series(pd.NA, index=raw.index, dtype="string")
            )
            unknown = set(feedback.dropna()) - set(AORTA_FEEDBACK_LABELS)
            if unknown:
                raise ValueError(
                    f"{source.path}: feedback desconhecido: {sorted(unknown)}"
                )
            frame = pd.DataFrame(
                {"image_id": ids, "aorta_segmentation_feedback": feedback}
            )
            for column, value in group.items():
                frame[column] = value
            frame["feedback"] = (
                feedback.map(AORTA_FEEDBACK_LABELS).fillna("Sem feedback")
                if available
                else "Diagnóstico não persistido"
            )
            frame["diagnostic_available"] = available
            frame["source_path"] = str(source.path)
            frame["source_run"] = source_run
            frames.append(frame.loc[:, RECORD_COLUMNS])
            row.update(
                exam_count=len(raw),
                with_feedback=int(feedback.notna().sum()) if available else pd.NA,
                status=(
                    "Coorte vazia"
                    if raw.empty
                    else "Disponível"
                    if available
                    else "Diagnóstico não persistido"
                ),
            )
            if source.expected_count is not None:
                if len(raw) > source.expected_count:
                    raise ValueError(f"{source.path}: coorte excede o tamanho esperado")
                row["missing_count"] = source.expected_count - len(raw)
        coverage.append(row)
    records = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=RECORD_COLUMNS)
    )
    return records, pd.DataFrame(
        coverage,
        columns=[
            *GROUP_COLUMNS,
            "source_path",
            "source_run",
            "expected_count",
            "exam_count",
            "missing_count",
            "with_feedback",
            "status",
        ],
    )


def summarize_aorta_feedback(records: pd.DataFrame) -> pd.DataFrame:
    """Conta categorias por coorte, incluindo feedback vazio no denominador."""
    rows = []
    available = records.loc[records["diagnostic_available"].eq(True)]
    for key, frame in available.groupby(GROUP_COLUMNS, sort=False):
        counts = frame["feedback"].value_counts()
        for status in AORTA_FEEDBACK_ORDER:
            count = int(counts.get(status, 0))
            rows.append(
                {
                    **dict(zip(GROUP_COLUMNS, key, strict=True)),
                    "status": status,
                    "count": count,
                    "total": len(frame),
                    "percent": 100 * count / len(frame),
                }
            )
    return pd.DataFrame(
        rows, columns=[*GROUP_COLUMNS, "status", "count", "total", "percent"]
    )


def summarize_aorta_feedback_comparison(
    records: pd.DataFrame, coverage: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Acrescenta o Geral pela soma dos exames, sem misturar splits/resoluções.

    Fontes sem coluna de diagnóstico não entram no agregado. O Geral fica
    parcial quando algum banco aplicável não contribui; registros com feedback
    vazio continuam no denominador como 'Sem feedback'. Exames ausentes do CSV
    são descritos separadamente pela contagem esperada.
    """
    if coverage.duplicated(GROUP_COLUMNS).any():
        raise ValueError("Coortes duplicadas na cobertura.")
    if coverage["dataset"].eq("Geral").any() or records["dataset"].eq("Geral").any():
        raise ValueError("A entrada já contém um agregado Geral.")
    summary = summarize_aorta_feedback(records)
    totals = summary.drop_duplicates(GROUP_COLUMNS)
    known = set(map(tuple, coverage[GROUP_COLUMNS].to_numpy()))
    if not set(map(tuple, totals[GROUP_COLUMNS].to_numpy())).issubset(known):
        raise ValueError("Há diagnósticos sem identificação na cobertura.")
    aggregate_rows = []
    aggregate_coverage = []
    keys = ["reference", "resolution", "subset"]
    for key, cohort in coverage.groupby(keys, sort=False):
        group = dict(zip(keys, key, strict=True))
        applicable = cohort.loc[cohort["status"].ne("Não se aplica")]
        selection = summary
        for column, value in group.items():
            selection = selection.loc[selection[column].eq(value)]
        selection = selection.loc[selection["dataset"].isin(applicable["dataset"])]
        included = selection["dataset"].drop_duplicates().tolist()
        contributing = selection.drop_duplicates("dataset")
        total = int(contributing["total"].sum())
        available = total > 0
        complete = available and set(included) == set(applicable["dataset"])
        expected = applicable["expected_count"]
        aggregate_coverage.append(
            {
                **group,
                "dataset": "Geral",
                "source_path": "",
                "source_run": "",
                "exam_count": total if available else pd.NA,
                "expected_count": int(expected.sum())
                if expected.notna().all()
                else pd.NA,
                "missing_count": pd.NA,
                "with_feedback": pd.NA,
                "status": "Disponível"
                if complete
                else "Parcial"
                if available
                else "Indisponível",
                "included_datasets": ", ".join(included),
                "applicable_datasets": ", ".join(applicable["dataset"]),
            }
        )
        if available:
            counts = selection.groupby("status")["count"].sum()
            for status in AORTA_FEEDBACK_ORDER:
                count = int(counts.get(status, 0))
                aggregate_rows.append(
                    {
                        **group,
                        "dataset": "Geral",
                        "status": status,
                        "count": count,
                        "total": total,
                        "percent": 100 * count / total,
                    }
                )
    comparison = pd.concat(
        [summary, pd.DataFrame(aggregate_rows, columns=summary.columns)],
        ignore_index=True,
    )
    compared_coverage = pd.concat(
        [coverage, pd.DataFrame(aggregate_coverage)], ignore_index=True
    )
    return comparison, compared_coverage


def compare_aorta_resolutions(
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pareia IDs por coorte e expõe exames exclusivos de cada resolução."""
    paired_frames = []
    coverage = []
    for key, frame in records.groupby(["reference", "dataset", "subset"], sort=False):
        group = dict(zip(["reference", "dataset", "subset"], key, strict=True))
        sides = []
        for resolution in ("mid", "high"):
            side = frame.loc[frame["resolution"].eq(resolution)]
            if side["image_id"].duplicated().any():
                raise ValueError(f"IDs duplicados no pareamento: {key}, {resolution}")
            sides.append(side.set_index("image_id"))
        mid, high = sides
        common = mid.index.intersection(high.index)
        coverage.append(
            {
                **group,
                "common_count": len(common),
                "mid_only_ids": mid.index.difference(high.index).tolist(),
                "high_only_ids": high.index.difference(mid.index).tolist(),
                "paired_diagnostics": int(
                    (
                        mid.reindex(common)["diagnostic_available"].eq(True)
                        & high.reindex(common)["diagnostic_available"].eq(True)
                    ).sum()
                ),
            }
        )
        paired = (
            pd.DataFrame(
                {
                    "feedback_mid": mid.reindex(common)["feedback"],
                    "feedback_high": high.reindex(common)["feedback"],
                    "diagnostic_mid_available": mid.reindex(common)[
                        "diagnostic_available"
                    ],
                    "diagnostic_high_available": high.reindex(common)[
                        "diagnostic_available"
                    ],
                }
            )
            .rename_axis("image_id")
            .reset_index()
        )
        for column, value in group.items():
            paired[column] = value
        paired_frames.append(paired)
    return (
        pd.concat(paired_frames, ignore_index=True)
        if paired_frames
        else pd.DataFrame(),
        pd.DataFrame(coverage),
    )
