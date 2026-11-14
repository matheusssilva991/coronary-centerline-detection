"""Reexecuta uma referência com seu snapshot efetivo e promove métricas completas."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import math
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.comparison_utils.io import load_experiment_variant_results
from utils.experiments.fuzzy_pipeline_comparison import run_image
from utils.project.evaluation.artery_distances import apply_distance_metric_options
from utils.project.analysis.external_aorta_assessment import AORTA_FEEDBACK_LABELS
from utils.project.results.columns import READABLE_COLUMN_NAMES
from utils.project.results.io import save_dataframe_atomic
from utils.project.results.schema import (
    make_readable_results_dataframe,
    make_result_dataframe,
)
from utils.project.runtime.paths import find_repository_root
from utils.segmentation.pipeline.orchestration import process_image
from utils.utils.json_io import save_json_atomic


def fingerprint(path: Path) -> str:
    """Identifica exatamente os arquivos usados como referência."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_plan(repo_root: Path, resolution: str, split: str) -> dict:
    """Lê IDs/config efetiva sem reescalonar nem consultar o config corrente."""
    source = (
        repo_root / f"output/segmentation/canonical/{resolution}_res/{split}/current"
    )
    csv = source / f"numeric/results_{split}.csv"
    config_path = source / "config/effective_pipeline_config.json"
    frame = pd.read_csv(csv)
    ids = frame["IMG_ID"].astype(int).tolist()
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("A referência contém IDs vazios ou duplicados.")
    splits = json.loads((repo_root / "config/imagecas_splits.json").read_text())[
        "splits"
    ]
    if set(ids) != set(splits[split]):
        raise ValueError("Os IDs canônicos diferem do split completo registrado.")
    settings = source / "config/experiment_settings.json"
    config = json.loads(config_path.read_text())
    apply_distance_metric_options(config, argparse.Namespace(distance_metrics=True))
    manifest = source / "source_manifest.json"
    return {
        "resolution": resolution,
        "split": split,
        "ids": ids,
        "source_csv": csv.relative_to(repo_root).as_posix(),
        "source_csv_sha256": fingerprint(csv),
        "source_config_sha256": fingerprint(config_path),
        "implementation_sha256": {
            name: fingerprint(repo_root / name)
            for name in (
                "src/experiments/rebuild_canonical_metrics.py",
                "src/utils/experiments/fuzzy_pipeline_comparison.py",
                "src/utils/segmentation/pipeline/detection.py",
                "src/utils/segmentation/pipeline/arteries.py",
                "src/utils/segmentation/pipeline/orchestration.py",
                "src/utils/segmentation/aorta/segmentation.py",
                "src/utils/utils/metrics.py",
            )
            if (repo_root / name).is_file()
        },
        "source_manifest": json.loads(manifest.read_text())
        if manifest.is_file()
        else None,
        "effective_config": config,
        "experiment": json.loads(settings.read_text()) if settings.is_file() else {},
        "execution_flow": "shared_experiment" if resolution == "mid" else "pipeline",
    }


def validate_complete(candidate: Path, plan: dict) -> pd.DataFrame:
    """Exige coorte completa e campos novos, preservando métricas indisponíveis."""
    frame = pd.read_csv(candidate / f"numeric/results_{plan['split']}.csv")
    if frame["IMG_ID"].duplicated().any() or set(frame["IMG_ID"]) != set(plan["ids"]):
        raise ValueError("Candidato incompleto ou com IDs repetidos.")
    required = {
        "aorta_segmentation_feedback",
        "aorta_level_set_circle_fill_q25",
        "aorta_level_set_circle_area_ratio_p90",
        "artery_hd_mm",
        "artery_ahd_mm",
        "artery_distance_metrics_status",
        "artery_dice_after_closing",
    }
    if not required.issubset(frame):
        raise ValueError(f"Métricas ausentes: {sorted(required - set(frame))}")
    if not frame["aorta_segmentation_feedback"].isin(AORTA_FEEDBACK_LABELS).all():
        raise ValueError("Há exames sem classificação registrada.")
    if (
        not frame["artery_distance_metrics_status"]
        .isin(
            [
                "ok",
                "not_evaluated",
                "error",
                "prediction_empty",
                "reference_empty",
                "both_empty",
            ]
        )
        .all()
    ):
        raise ValueError("Há distâncias desativadas ou status inválido.")
    evaluated = frame["artery_distance_metrics_status"].eq("ok")
    for metric in ("artery_hd_mm", "artery_ahd_mm"):
        values = pd.to_numeric(frame.loc[evaluated, metric], errors="coerce")
        if not values.map(lambda value: math.isfinite(value) and value >= 0).all():
            raise ValueError(f"{metric}: status ok com medida inválida.")
    return frame


def rebuild(
    candidate: Path, plan: dict, base_path: Path, *, resume: bool = False
) -> None:
    """Executa uma coorte com checkpoint por exame em um diretório novo."""
    missing = [
        str(base_path / f"{img_id}.{kind}.nii.gz")
        for img_id in plan["ids"]
        for kind in ("img", "label")
        if not (base_path / f"{img_id}.{kind}.nii.gz").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Volumes ausentes ({len(missing)}): {missing[:5]}")
    plan_file = candidate / "rebuild_plan.json"
    if candidate.exists():
        if not resume or not plan_file.is_file():
            raise ValueError(
                "Destino já existe; use --resume para retomar a mesma execução."
            )
        if json.loads(plan_file.read_text()) != plan:
            raise ValueError("O snapshot/IDs de origem mudou; não é possível retomar.")
    else:
        candidate.mkdir(parents=True)
        save_json_atomic(plan, plan_file)
    raw_path = candidate / "results/image_results.csv"
    rows = pd.read_csv(raw_path).to_dict("records") if raw_path.is_file() else []
    completed = {int(row["IMG_ID"]) for row in rows}
    if len(completed) != len(rows) or not completed.issubset(plan["ids"]):
        raise ValueError("Checkpoint contém IDs inválidos.")
    for img_id in plan["ids"]:
        if img_id in completed:
            continue
        config = copy.deepcopy(plan["effective_config"])
        if plan["execution_flow"] == "shared_experiment":
            row = run_image(
                img_id, "baseline", plan["split"], base_path, config, plan["experiment"]
            )
        else:
            row = process_image(img_id, config, str(base_path))
        rows.append(row)
        save_dataframe_atomic(pd.DataFrame(rows), raw_path)
        print(
            f"{plan['resolution']}/{plan['split']}: {len(rows)}/{len(plan['ids'])}; IMG_ID={img_id}; erro={row.get('error')}",
            flush=True,
        )
    if plan["execution_flow"] == "shared_experiment":
        frame = load_experiment_variant_results(
            candidate, variant="baseline", split=plan["split"]
        )
        redundant = [
            k
            for k, v in READABLE_COLUMN_NAMES.items()
            if k != v and k in frame and v in frame
        ]
        frame = make_readable_results_dataframe(frame.drop(columns=redundant))
    else:
        frame = make_readable_results_dataframe(make_result_dataframe(rows))
    save_dataframe_atomic(frame, candidate / f"numeric/results_{plan['split']}.csv")
    save_json_atomic(
        plan["effective_config"], candidate / "config/effective_pipeline_config.json"
    )
    save_json_atomic(plan["experiment"], candidate / "config/experiment_settings.json")
    validate_complete(candidate, plan)
    save_json_atomic(
        {
            "artifact_kind": "reexecuted_complete_metrics",
            "resolution": f"{plan['resolution']}_res",
            "split": plan["split"],
            "exam_count": len(rows),
            "exam_ids": plan["ids"],
            "results_sha256": fingerprint(
                candidate / f"numeric/results_{plan['split']}.csv"
            ),
            "previous_reference": {
                k: v
                for k, v in plan.items()
                if k not in {"effective_config", "experiment", "ids"}
            },
            "timing_note": "Reexecução por exame; sem tempos de batch sintetizados.",
        },
        candidate / "source_manifest.json",
    )


def promote(repo_root: Path, candidate: Path, plan: dict) -> None:
    """Promove uma coorte concluída preservando o destino canônico anterior."""
    frame = validate_complete(candidate, plan)
    manifest_path = candidate / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["results_sha256"] != fingerprint(
        candidate / f"numeric/results_{plan['split']}.csv"
    ):
        raise ValueError("O CSV candidato mudou após a conclusão.")
    current = (
        repo_root
        / f"output/segmentation/canonical/{plan['resolution']}_res/{plan['split']}/current"
    )
    if current.resolve() == candidate.resolve():
        print("Esta execução já é a referência atual.")
        return
    if (
        fingerprint(current / f"numeric/results_{plan['split']}.csv")
        != plan["source_csv_sha256"]
    ):
        raise ValueError("A referência mudou desde o início da reexecução.")
    if (
        fingerprint(current / "config/effective_pipeline_config.json")
        != plan["source_config_sha256"]
    ):
        raise ValueError("A configuração canônica mudou desde o início da reexecução.")
    archive = current.with_name(f"before_{candidate.parents[1].name}")
    if archive.exists() or archive.is_symlink():
        raise ValueError(f"Arquivo histórico já existe: {archive}")
    registry_path = repo_root / "output/segmentation/canonical/references_manifest.json"
    registry = json.loads(registry_path.read_text())
    new_reference = {
        **manifest,
        "source_run": candidate.relative_to(repo_root).as_posix(),
        "source_csv_sha256": fingerprint(
            candidate / f"numeric/results_{plan['split']}.csv"
        ),
        "source_config_sha256": fingerprint(
            candidate / "config/effective_pipeline_config.json"
        ),
    }
    previous = [
        entry
        for entry in registry["current_references"]
        if entry["resolution"] == f"{plan['resolution']}_res"
        and entry["split"] == plan["split"]
    ]
    registry.setdefault("metric_refresh_history", []).append(
        {
            "previous": previous,
            "archive": archive.relative_to(repo_root).as_posix(),
            "new": new_reference,
        }
    )
    registry["current_references"] = [
        entry for entry in registry["current_references"] if entry not in previous
    ] + [new_reference]
    current.rename(archive)
    try:
        current.symlink_to(
            os.path.relpath(candidate, current.parent), target_is_directory=True
        )
        save_json_atomic(registry, registry_path)
    except Exception:
        if current.is_symlink():
            current.unlink()
        archive.rename(current)
        raise
    print(f"Canonical atualizado: {current}; {len(frame)} exames; anterior: {archive}")


def main() -> None:
    """Executa somente a resolução/conjunto solicitados, sem variantes extras."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", choices=["mid", "high"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--base-path", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--promote-only", action="store_true")
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        parser.error("--run-name deve ser um nome simples de diretório.")
    root = find_repository_root(Path(__file__))
    candidate = (
        root
        / f"output/segmentation/runs/canonical_metrics/{args.run_name}/{args.resolution}_res/{args.split}"
    )
    if args.promote_only:
        if args.dry_run or args.resume:
            parser.error("--promote-only não aceita --dry-run/--resume")
        plan = json.loads((candidate / "rebuild_plan.json").read_text())
        promote(root, candidate, plan)
        return
    plan = build_plan(root, args.resolution, args.split)
    print(
        f"{args.resolution}/{args.split}: {len(plan['ids'])} exames; destino: {candidate}"
    )
    print(
        "Configuração efetiva preservada; HD/AHD habilitados; sem novo escalonamento."
    )
    if args.dry_run:
        return
    base_path = args.base_path or (
        Path(os.environ["IMAGECAS_BASE_PATH"])
        if os.environ.get("IMAGECAS_BASE_PATH")
        else None
    )
    if base_path is None or not base_path.is_dir():
        parser.error(
            "Informe --base-path ou IMAGECAS_BASE_PATH com os volumes ImageCAS."
        )
    rebuild(candidate, plan, base_path, resume=args.resume)
    print(
        "Execução concluída. Revise os resultados e use --promote-only para atualizar canonical."
    )


if __name__ == "__main__":
    main()
