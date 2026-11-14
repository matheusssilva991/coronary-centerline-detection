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
    repo_root: Path, external_root: Path
) -> list[AortaFeedbackSource]:
    """Seleciona referências atuais e uma coorte histórica explicitamente separada."""
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
    historical = {
        "train": "train/ostia_localization/lower100_pad2/2026-09-06_08-08-25",
        "val": "val/ostia_validation270/lower100_pad2/2026-09-07_07-15-15",
        "test": "test/ostia_comparison/lower100_pad2/2026-09-07_10-37-38",
    }
    base = repo_root / "output/segmentation/runs/mid_res/aorta_segmentation_experiments"
    for subset, run in historical.items():
        sources.append(
            AortaFeedbackSource(
                "ImageCAS",
                "mid",
                subset,
                "Histórico",
                base / run / f"numeric/results_{subset}.csv",
                len(splits[subset]),
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
