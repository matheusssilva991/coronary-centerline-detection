"""Batch execution of the segmentation pipeline on OrCaScore and MM-WHS CCTA."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from utils.processing.gpu_utils import use_gpu
from utils.project.config import load_config_json, save_config_json
from utils.project.ccta_datasets import (
    align_ccta_volume_to_imagecas_view,
    discover_ccta_dataset,
    load_ccta_volume,
)
from utils.project.notebook_env import load_notebook_pipeline_config
from utils.project.results import make_json_safe
from utils.segmentation.artery_segmentation import normal_region_growing_from_ostia
from utils.segmentation.aorta_segmentation import (
    classify_aorta_segmentation_feedback,
)
from utils.segmentation.pipeline_arteries import get_artery_postprocessing_stages
from utils.segmentation.pipeline_detection import (
    detect_ostia,
    filter_located_aorta_circles,
    locate_aorta_circles,
    segment_aorta_with_diagnostics,
)
from utils.segmentation.pipeline_orchestration import (
    summarize_aorta_circles,
    summarize_aorta_volume,
)
from utils.segmentation.pipeline_preprocessing import (
    compute_vesselness,
    preprocess_ccta_volume,
)
from utils.visualization.pipeline_artifacts import (
    save_detected_circles_figure,
    save_stage_views,
)
from utils.visualization.volume import visualize_aorta_ostia_artery


LOGGER = logging.getLogger("external_ccta_batch_pipeline")
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "pipeline_config.json"
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "CCTA_RESULTS_ROOT",
        "/media/matheus/HD/Results_dataset_ccta",
    )
)
DATASET_SPECS = {
    "orcascore": {
        "display_name": "OrCaScore",
        "environment": "ORCASCORE_BASE_PATH",
        "default_path": Path("/media/matheus/HD/DatasetsCCTA/Orca_Score_Calcium"),
    },
    "mmwhs": {
        "display_name": "MM-WHS",
        "environment": "MMWHS_BASE_PATH",
        "default_path": Path("/media/matheus/HD/DatasetsCCTA/MM-WHS-2017-Dataset"),
    },
}


@dataclass(frozen=True)
class RunPaths:
    """Structured directories for one external-dataset run."""

    run_dir: Path
    numeric_dir: Path
    config_dir: Path
    visual_dir: Path
    logs_dir: Path


def normalize_dataset_name(value: str) -> str:
    """Normalize CLI aliases to the two supported dataset keys."""
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "orca": "orcascore",
        "orca-score": "orcascore",
        "orcascore": "orcascore",
        "mm-whs": "mmwhs",
        "mmwhs": "mmwhs",
        "whs": "mmwhs",
        "owhs": "mmwhs",
    }
    if normalized not in aliases:
        raise argparse.ArgumentTypeError("dataset inválido; use orcascore ou mmwhs.")
    return aliases[normalized]


def build_parser() -> argparse.ArgumentParser:
    """Build the external CCTA batch CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Executa o pipeline completo em todas as CCTA do OrCaScore ou MM-WHS "
            "e salva resultados numéricos e visuais por exame."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=normalize_dataset_name,
        choices=tuple(DATASET_SPECS),
        help="Banco CCTA: orcascore ou mmwhs (aliases: orca, whs, owhs).",
    )
    parser.add_argument(
        "--resolution",
        required=True,
        choices=("mid", "high"),
        help="Resolução efetiva do pipeline.",
    )
    parser.add_argument(
        "--subset",
        default="all",
        choices=("all", "train", "test"),
        help="Por padrão processa train e test do banco selecionado.",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        help="Sobrescreve o caminho do banco selecionado.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Raiz dos resultados (padrão: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Configuração base do pipeline.",
    )
    parser.add_argument(
        "--exam-ids",
        nargs="+",
        help="Processa somente os IDs informados, útil para validação controlada.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limita a quantidade de exames após os demais filtros.",
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        help="Retoma uma execução e ignora exames já concluídos com sucesso.",
    )
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Habilita/desabilita GPU nas etapas compatíveis.",
    )
    parser.add_argument(
        "--visuals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Salva os artefatos visuais por exame (habilitado por padrão).",
    )
    parser.add_argument(
        "--align-orcascore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Aplica a correção visual do OrCaScore usada no notebook.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Interrompe no primeiro exame com erro.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def resolve_dataset_path(dataset: str, explicit_path: Path | None) -> Path:
    """Resolve and validate the selected dataset root."""
    if explicit_path is not None:
        path = explicit_path.expanduser()
    else:
        spec = DATASET_SPECS[dataset]
        environment_value = os.environ.get(str(spec["environment"]))
        path = (
            Path(environment_value).expanduser()
            if environment_value
            else Path(spec["default_path"])
        )
    if not path.is_dir():
        raise FileNotFoundError(f"Banco não encontrado: {path}")
    return path


def create_run_paths(
    output_root: Path,
    dataset: str,
    resolution: str,
    *,
    resume_dir: Path | None = None,
) -> RunPaths:
    """Create a new timestamped run or resolve an existing run."""
    if resume_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = output_root / dataset / f"{resolution}_res" / timestamp
    else:
        run_dir = resume_dir.expanduser()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run para retomada não encontrado: {run_dir}")

    paths = RunPaths(
        run_dir=run_dir,
        numeric_dir=run_dir / "numeric",
        config_dir=run_dir / "config",
        visual_dir=run_dir / "visual",
        logs_dir=run_dir / "logs",
    )
    for directory in (
        paths.run_dir,
        paths.numeric_dir,
        paths.config_dir,
        paths.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def select_inventory(
    inventory: pd.DataFrame,
    *,
    subset: str,
    exam_ids: Sequence[str] | None,
    limit: int | None,
) -> pd.DataFrame:
    """Filter the discovered inventory while preserving a deterministic order."""
    selected = inventory.copy()
    if subset != "all":
        selected = selected.loc[selected["subset"].eq(subset)]
    if exam_ids:
        requested = {str(exam_id) for exam_id in exam_ids}
        available = set(selected["exam_id"].astype(str))
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"IDs não encontrados no recorte selecionado: {missing}")
        selected = selected.loc[selected["exam_id"].astype(str).isin(requested)]
    subset_order = selected["subset"].map({"train": 0, "test": 1}).fillna(2)
    selected = (
        selected.assign(_subset_order=subset_order)
        .sort_values(["_subset_order", "exam_id"], kind="stable")
        .drop(columns="_subset_order")
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit deve ser positivo.")
        selected = selected.head(limit)
    if selected.empty:
        raise ValueError("Nenhum exame corresponde aos filtros informados.")
    return selected.reset_index(drop=True)


def _save_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(make_json_safe(dict(payload)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_dataframe_atomic(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_csv(temporary, index=False)
    temporary.replace(path)


def _coordinates_to_fields(prefix: str, coordinates: Any) -> dict[str, int | None]:
    values = (
        tuple(int(value) for value in coordinates) if coordinates is not None else ()
    )
    return {
        f"{prefix}_y": values[0] if len(values) > 0 else None,
        f"{prefix}_x": values[1] if len(values) > 1 else None,
        f"{prefix}_z": values[2] if len(values) > 2 else None,
    }


def _save_combined_visual(
    output_path: Path,
    *,
    exam_label: str,
    aorta_mask: Any,
    ostia_left: Any,
    ostia_right: Any,
    artery_mask: Any,
    spacing: Sequence[float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visualize_aorta_ostia_artery(
        aorta_mask,
        ostia_left,
        ostia_right,
        artery_mask=artery_mask,
        spacing=spacing,
        use_physical_coords=True,
        save_html_path=str(output_path),
        display_plot=False,
        plot_name=f"{exam_label}: aorta, óstios e artérias",
    )


def _save_stage(
    exam_dir: Path | None,
    stage_name: str,
    volume: Any,
    *,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    if exam_dir is None:
        return
    save_stage_views(
        np.asarray(volume),
        exam_dir / "stages" / stage_name,
        title=title,
        vmin=vmin,
        vmax=vmax,
    )


def process_external_exam(
    record: Mapping[str, Any] | pd.Series,
    config: dict[str, Any],
    resolution: str,
    *,
    visual_root: Path | None,
    align_orcascore: bool = True,
) -> dict[str, Any]:
    """Run all notebook stages for one unlabeled external CCTA exam."""
    started = time.perf_counter()
    dataset = str(record["dataset"])
    subset = str(record["subset"])
    exam_id = str(record["exam_id"])
    exam_dir = visual_root / subset / exam_id if visual_root is not None else None
    result: dict[str, Any] = {
        "dataset": dataset,
        "subset": subset,
        "exam_id": exam_id,
        "resolution": resolution,
        "status": "error",
        "error": None,
        "reported_orientation": str(record["reported_orientation"]),
        "quality_validated": False,
        "dice": None,
        "ostia_accuracy": None,
    }

    try:
        native_image = load_ccta_volume(record).astype(np.float32, copy=False)
        spacing = (
            float(record["spacing_x_mm"]),
            float(record["spacing_y_mm"]),
            float(record["spacing_z_mm"]),
        )
        image = native_image
        flipped_axes: tuple[int, ...] = ()
        if align_orcascore:
            image, flipped_axes = align_ccta_volume_to_imagecas_view(image, dataset)
        result.update(
            {
                "visual_flip_axes": ",".join(map(str, flipped_axes)),
                "original_size_x": int(image.shape[0]),
                "original_size_y": int(image.shape[1]),
                "original_slice_count": int(image.shape[2]),
                "spacing_x_mm": spacing[0],
                "spacing_y_mm": spacing[1],
                "spacing_z_mm": spacing[2],
            }
        )
        _save_stage(
            exam_dir,
            "00_input",
            image,
            title="CCTA de entrada",
            vmin=-200.0,
            vmax=1000.0,
        )

        image_data = preprocess_ccta_volume(
            image,
            spacing,
            config,
            include_intermediates=True,
        )
        threshold_mask = image_data["threshold_mask"]
        lcc_image = image_data["lcc_image"]
        downscale_factors = image_data["downscale_factors"]
        scaled_spacing = tuple(float(value) for value in image_data["scaled_spacing"])
        visual_spacing = (scaled_spacing[1], scaled_spacing[0], scaled_spacing[2])
        preprocessing_details = dict(image_data["preprocessing_details"])
        result.update(preprocessing_details)
        result.update(
            {
                "processed_size_x": int(lcc_image.shape[0]),
                "processed_size_y": int(lcc_image.shape[1]),
                "processed_slice_count": int(lcc_image.shape[2]),
                "processed_voxel_count": int(lcc_image.size),
                "scaled_spacing_x_mm": scaled_spacing[0],
                "scaled_spacing_y_mm": scaled_spacing[1],
                "scaled_spacing_z_mm": scaled_spacing[2],
            }
        )
        _save_stage(
            exam_dir,
            "01_threshold",
            threshold_mask,
            title="Máscara após threshold",
            vmin=0.0,
            vmax=1.0,
        )
        _save_stage(
            exam_dir,
            "02_lcc",
            lcc_image,
            title="Imagem após LCC",
            vmin=-200.0,
            vmax=1000.0,
        )

        raw_circles = locate_aorta_circles(
            lcc_image,
            downscale_factors,
            scaled_spacing,
            config["CIRCLE_DETECTION"],
        )
        if not raw_circles:
            raise RuntimeError("Nenhum círculo da aorta foi detectado.")
        detected_circles, filter_details = filter_located_aorta_circles(
            raw_circles,
            scaled_spacing,
            lcc_image.shape[2],
            config["CIRCLE_DETECTION"],
        )
        if not detected_circles:
            raise RuntimeError("O filtro removeu todos os círculos da aorta.")
        result["aorta_circle_count_before_filter"] = len(raw_circles)
        result.update(filter_details)
        result.update(
            summarize_aorta_circles(
                detected_circles,
                lcc_image.shape[2],
                scaled_spacing,
                config["CIRCLE_DETECTION"],
            )
        )
        if exam_dir is not None:
            save_detected_circles_figure(
                lcc_image,
                detected_circles,
                exam_dir / "aorta_circles.png",
                vmin=-200.0,
                vmax=1000.0,
            )

        aorta_segmentation = segment_aorta_with_diagnostics(
            lcc_image,
            detected_circles,
            config["LEVEL_SET"],
            use_gpu=config.get("USE_GPU", False),
        )
        aorta_mask = aorta_segmentation.mask.astype(np.uint8)
        if not np.any(aorta_mask):
            raise RuntimeError("A segmentação da aorta produziu uma máscara vazia.")
        result.update(aorta_segmentation.diagnostics)
        result.update(summarize_aorta_volume(aorta_mask, lcc_image.size))
        result["aorta_segmentation_feedback"] = classify_aorta_segmentation_feedback(
            result.get("aorta_level_set_circle_fill_q25"),
            result.get("aorta_level_set_circle_area_ratio_p90"),
            result.get("aorta_volume_fraction"),
            config.get("LEVEL_SET", {}).get("quality_feedback"),
        )
        voxel_volume_mm3 = float(np.prod(scaled_spacing))
        result["aorta_volume_ml"] = float(aorta_mask.sum() * voxel_volume_mm3 / 1000.0)
        _save_stage(
            exam_dir,
            "03_aorta",
            aorta_mask,
            title="Máscara final da aorta",
            vmin=0.0,
            vmax=1.0,
        )

        vesselness_ostia = compute_vesselness(
            lcc_image,
            vesselness_config=config["VESSELNESS_AORTA"],
            use_gpu=config.get("USE_GPU", False),
        )
        _save_stage(
            exam_dir,
            "04_vesselness_ostia",
            vesselness_ostia,
            title="Vesselness para detecção dos óstios",
        )
        vesselness_artery = compute_vesselness(
            lcc_image,
            vesselness_config=config["VESSELNESS_ARTERY"],
            use_gpu=config.get("USE_GPU", False),
        )
        _save_stage(
            exam_dir,
            "05_vesselness_artery",
            vesselness_artery,
            title="Vesselness para segmentação arterial",
        )

        try:
            ostia_left, ostia_right = detect_ostia(
                aorta_mask,
                vesselness_ostia,
                scaled_spacing,
                config,
            )
        except ValueError as error:
            result["status"] = "ostia_not_found"
            result["error"] = str(error)
            if exam_dir is not None:
                _save_combined_visual(
                    exam_dir / "aorta_ostia_artery.html",
                    exam_label=f"{dataset} {exam_id}",
                    aorta_mask=aorta_mask,
                    ostia_left=None,
                    ostia_right=None,
                    artery_mask=None,
                    spacing=visual_spacing,
                )
            return result

        result.update(_coordinates_to_fields("ostia_left", ostia_left))
        result.update(_coordinates_to_fields("ostia_right", ostia_right))
        raw_artery_mask = normal_region_growing_from_ostia(
            vesselness_artery,
            ostia_left,
            ostia_right,
            config,
        ).astype(np.uint8)
        _save_stage(
            exam_dir,
            "06_artery_raw",
            raw_artery_mask,
            title="Artérias antes da morfologia",
            vmin=0.0,
            vmax=1.0,
        )
        postprocessing = get_artery_postprocessing_stages(raw_artery_mask, config)
        closed_mask = postprocessing["closed_mask"]
        artery_mask = postprocessing["final_mask"]
        _save_stage(
            exam_dir,
            "07_artery_closed",
            closed_mask,
            title="Artérias após fechamento",
            vmin=0.0,
            vmax=1.0,
        )
        _save_stage(
            exam_dir,
            "08_artery_final",
            artery_mask,
            title="Artérias após dilatação",
            vmin=0.0,
            vmax=1.0,
        )
        result.update(
            {
                "artery_voxels_before_morphology": int(raw_artery_mask.sum()),
                "artery_voxels_after_closing": int(closed_mask.sum()),
                "artery_voxels_after_morphology": int(artery_mask.sum()),
                "artery_volume_before_morphology_ml": float(
                    raw_artery_mask.sum() * voxel_volume_mm3 / 1000.0
                ),
                "artery_volume_after_morphology_ml": float(
                    artery_mask.sum() * voxel_volume_mm3 / 1000.0
                ),
            }
        )
        if exam_dir is not None:
            _save_combined_visual(
                exam_dir / "aorta_ostia_artery.html",
                exam_label=f"{dataset} {exam_id}",
                aorta_mask=aorta_mask,
                ostia_left=ostia_left,
                ostia_right=ostia_right,
                artery_mask=artery_mask,
                spacing=visual_spacing,
            )
        result["status"] = "success"
    except Exception as error:
        LOGGER.exception("Falha no exame %s/%s", subset, exam_id)
        result["status"] = "error"
        result["error"] = str(error)
        if exam_dir is not None:
            exam_dir.mkdir(parents=True, exist_ok=True)
            (exam_dir / "error.txt").write_text(str(error), encoding="utf-8")
    finally:
        result["execution_time_seconds"] = float(time.perf_counter() - started)
        if exam_dir is not None:
            _save_json_atomic(result, exam_dir / "result.json")
    return result


def _result_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["subset"]), str(row["exam_id"])


def _result_sort_key(row: Mapping[str, Any]) -> tuple[int, str]:
    subset_order = {"train": 0, "test": 1}
    return subset_order.get(str(row["subset"]), 2), str(row["exam_id"])


def _upsert_result(
    rows: list[dict[str, Any]],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    key = _result_key(result)
    updated = [row for row in rows if _result_key(row) != key]
    updated.append(result)
    return sorted(updated, key=_result_sort_key)


def save_numeric_results(rows: list[dict[str, Any]], numeric_dir: Path) -> None:
    """Persist the consolidated and per-subset external results atomically."""
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return
    subset_order = dataframe["subset"].map({"train": 0, "test": 1}).fillna(2)
    dataframe = (
        dataframe.assign(_subset_order=subset_order)
        .sort_values(["_subset_order", "exam_id"], kind="stable")
        .drop(columns="_subset_order")
    )
    _save_dataframe_atomic(dataframe, numeric_dir / "results_all.csv")
    for subset, subset_frame in dataframe.groupby("subset", sort=False):
        _save_dataframe_atomic(
            subset_frame.reset_index(drop=True),
            numeric_dir / f"results_{subset}.csv",
        )


def load_existing_results(numeric_dir: Path) -> list[dict[str, Any]]:
    path = numeric_dir / "results_all.csv"
    if not path.is_file():
        return []
    return pd.read_csv(path).where(pd.notna, None).to_dict(orient="records")


def _metadata_payload(
    *,
    dataset: str,
    resolution: str,
    subset: str,
    rows: list[dict[str, Any]],
    started_at: datetime,
    state: str,
) -> dict[str, Any]:
    status_counts = pd.Series(
        [row.get("status", "unknown") for row in rows]
    ).value_counts()
    total_seconds = sum(float(row.get("execution_time_seconds") or 0.0) for row in rows)
    return {
        "schema_version": 1,
        "dataset": dataset,
        "resolution": resolution,
        "selected_subset": subset,
        "state": state,
        "started_at": started_at.isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "processed_exam_count": len(rows),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "execution_time": {
            "seconds": total_seconds,
            "minutes": total_seconds / 60.0,
            "hours": total_seconds / 3600.0,
        },
        "ground_truth_metrics": {
            "dice": None,
            "ostia_accuracy": None,
            "reason": "Os bancos externos não possuem referência coronariana compatível.",
        },
    }


def _configure_logging(logs_dir: Path, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d] %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        root.addHandler(stream)
    file_handler = logging.FileHandler(logs_dir / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def run(args: argparse.Namespace) -> RunPaths:
    """Execute the selected external cohort and return its run paths."""
    base_path = resolve_dataset_path(args.dataset, args.base_path)
    inventory = discover_ccta_dataset(args.dataset, base_path)
    selected = select_inventory(
        inventory,
        subset=args.subset,
        exam_ids=args.exam_ids,
        limit=args.limit,
    )
    paths = create_run_paths(
        args.output_root,
        args.dataset,
        args.resolution,
        resume_dir=args.resume_dir,
    )
    _configure_logging(paths.logs_dir, args.verbose)
    selected_exam_records = [
        {"subset": str(record["subset"]), "exam_id": str(record["exam_id"])}
        for _, record in selected.iterrows()
    ]
    selected_keys = {
        (record["subset"], record["exam_id"]) for record in selected_exam_records
    }

    config_path = paths.config_dir / "effective_pipeline_config.json"
    manifest_path = paths.config_dir / "run_manifest.json"
    if args.resume_dir is None:
        config = load_notebook_pipeline_config(args.config_file, args.resolution)
    else:
        if not config_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                "Run retomado não possui effective_pipeline_config.json e "
                "run_manifest.json."
            )
        manifest = _load_json(manifest_path)
        expected_identity = (args.dataset, args.resolution, args.subset)
        saved_identity = (
            manifest.get("dataset"),
            manifest.get("resolution"),
            manifest.get("selected_subset"),
        )
        if saved_identity != expected_identity:
            raise ValueError(
                "Identidade do run retomado diverge da CLI: "
                f"salva={saved_identity}, solicitada={expected_identity}."
            )
        if manifest.get("selected_exams") != selected_exam_records:
            raise ValueError(
                "A seleção de exames diverge do run retomado. Repita os mesmos "
                "--exam-ids e --limit usados na execução original."
            )
        saved_options = (
            bool(manifest.get("visuals_enabled")),
            bool(manifest.get("orcascore_visual_alignment")),
        )
        requested_options = (bool(args.visuals), bool(args.align_orcascore))
        if saved_options != requested_options:
            raise ValueError(
                "As opções de visualização/alinhamento divergem do run retomado: "
                f"salvas={saved_options}, solicitadas={requested_options}."
            )
        config = load_config_json(str(config_path), {})

    gpu_available = use_gpu()
    requested_gpu = config.get("USE_GPU", False) if args.gpu is None else args.gpu
    config["USE_GPU"] = bool(requested_gpu and gpu_available)
    if requested_gpu and not gpu_available:
        LOGGER.warning("GPU solicitada, mas indisponível; execução seguirá em CPU.")

    if args.resume_dir is None:
        save_config_json(config, str(config_path))
        try:
            config_source = str(args.config_file.resolve().relative_to(REPO_ROOT))
        except ValueError:
            config_source = args.config_file.name
        _save_json_atomic(
            {
                "schema_version": 1,
                "dataset": args.dataset,
                "resolution": args.resolution,
                "selected_subset": args.subset,
                "visuals_enabled": bool(args.visuals),
                "orcascore_visual_alignment": bool(args.align_orcascore),
                "config_source": config_source,
                "selected_exams": selected_exam_records,
            },
            manifest_path,
        )

    rows = load_existing_results(paths.numeric_dir)
    completed = {
        _result_key(row)
        for row in rows
        if str(row.get("status")) in {"success", "ostia_not_found"}
    }
    pending_records = [
        record
        for _, record in selected.iterrows()
        if (str(record["subset"]), str(record["exam_id"])) not in completed
    ]
    started_at = datetime.now(timezone.utc)
    LOGGER.info(
        "Run %s | dataset=%s resolution=%s selecionados=%d pendentes=%d",
        paths.run_dir,
        args.dataset,
        args.resolution,
        len(selected),
        len(pending_records),
    )

    metadata_path = paths.run_dir / "metadata.json"
    _save_json_atomic(
        _metadata_payload(
            dataset=args.dataset,
            resolution=args.resolution,
            subset=args.subset,
            rows=rows,
            started_at=started_at,
            state="running",
        ),
        metadata_path,
    )
    for position, record in enumerate(pending_records, start=1):
        LOGGER.info(
            "[%d/%d] Processando %s/%s",
            position,
            len(pending_records),
            record["subset"],
            record["exam_id"],
        )
        result = process_external_exam(
            record,
            config,
            args.resolution,
            visual_root=paths.visual_dir if args.visuals else None,
            align_orcascore=args.align_orcascore,
        )
        rows = _upsert_result(rows, result)
        save_numeric_results(rows, paths.numeric_dir)
        _save_json_atomic(
            _metadata_payload(
                dataset=args.dataset,
                resolution=args.resolution,
                subset=args.subset,
                rows=rows,
                started_at=started_at,
                state="running",
            ),
            metadata_path,
        )
        if result["status"] != "success" and args.fail_fast:
            raise RuntimeError(
                f"Falha no exame {record['exam_id']}: {result.get('error')}"
            )

    terminal_keys = {
        _result_key(row)
        for row in rows
        if row.get("status") in {"success", "ostia_not_found"}
    }
    attempted_keys = {_result_key(row) for row in rows}
    all_attempted = selected_keys.issubset(attempted_keys)
    all_terminal = selected_keys.issubset(terminal_keys)
    final_state = (
        "complete"
        if all_terminal
        else "complete_with_errors"
        if all_attempted
        else "incomplete"
    )
    _save_json_atomic(
        _metadata_payload(
            dataset=args.dataset,
            resolution=args.resolution,
            subset=args.subset,
            rows=rows,
            started_at=started_at,
            state=final_state,
        ),
        metadata_path,
    )
    LOGGER.info("Execução finalizada em %s (state=%s)", paths.run_dir, final_state)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        paths = run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        LOGGER.error("%s", error)
        return 1
    print(f"Resultados salvos em: {paths.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
