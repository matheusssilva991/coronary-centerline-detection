"""Executa o pipeline em lote nas CCTA do OrCaScore e MM-WHS."""

from __future__ import annotations

import argparse
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
from utils.project.dataframe import require_series_column
from utils.project.datasets.ccta import (
    align_ccta_volume_to_imagecas_view,
    discover_ccta_dataset,
    load_ccta_aorta_ground_truth,
    load_ccta_volume,
)
from utils.project.evaluation.mmwhs_official_aorta import (
    EVALUATOR_FOLDER,
    OFFICIAL_AORTA_METHOD,
    AortaPrediction,
    has_aorta_mask,
    preflight_evaluator,
    restore_native_mask,
    save_prediction,
    validate_saved_prediction,
)
from utils.project.runtime.notebook_env import load_notebook_pipeline_config
from utils.project.runtime.run_notification import notify_run_completion
from utils.utils.json_io import load_json_file, save_json_atomic
from utils.project.runtime.run_logging import add_run_file_handler
from utils.project.results.external import (
    external_result_key,
    upsert_external_result,
    save_numeric_results,
    load_existing_results,
    build_external_metadata,
    is_terminal_external_result,
)
from utils.project.evaluation.mmwhs_aorta_evaluation import (
    evaluate_test_aorta_result,
    official_evaluation_complete,
)
from utils.segmentation.fuzzy.threshold import normalize_threshold_mode
from utils.segmentation.aorta.segmentation import (
    classify_aorta_segmentation_feedback,
)
from utils.segmentation.pipeline.arteries import (
    segment_artery_masks_from_vesselness,
)
from utils.segmentation.pipeline.detection import (
    detect_ostia,
    locate_and_filter_aorta_circles,
    segment_aorta_with_diagnostics,
)
from utils.segmentation.aorta.diagnostics import (
    summarize_aorta_circles,
    summarize_aorta_volume,
)
from utils.segmentation.pipeline.preprocessing import (
    compute_vesselness,
    preprocess_ccta_volume,
)
from utils.segmentation.pipeline.visuals import save_segmentation_visual_to_path
from utils.utils.metrics import dice_score
from utils.visualization.pipeline.pipeline_artifacts import (
    save_detected_circles_figure,
    save_stage_views,
)
from utils.visualization.images.volume import visualize_binary_masks_comparison


LOGGER = logging.getLogger("external_ccta_batch_pipeline")
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "pipeline_config.json"
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "CCTA_RESULTS_ROOT",
        "/run/media/matheus/HD/Results_dataset_ccta",
    )
)
DATASET_SPECS = {
    "orcascore": {
        "display_name": "OrCaScore",
        "environment": "ORCASCORE_BASE_PATH",
        "default_path": Path("/run/media/matheus/HD/DatasetsCCTA/Orca_Score_Calcium"),
    },
    "mmwhs": {
        "display_name": "MM-WHS",
        "environment": "MMWHS_BASE_PATH",
        "default_path": Path("/run/media/matheus/HD/DatasetsCCTA/MM-WHS-2017-Dataset"),
    },
}


@dataclass(frozen=True)
class RunPaths:
    """Representa os diretórios de uma execução em banco externo."""

    run_dir: Path
    numeric_dir: Path
    config_dir: Path
    visual_dir: Path
    logs_dir: Path


def normalize_dataset_name(value: str) -> str:
    """Normaliza aliases da CLI para os dois bancos suportados."""
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
    """Cria o parser da execução em lote de CCTA externas."""
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
        "--no-hu-threshold",
        action="store_true",
        help="Preserva todos os voxels finitos após o downscale, sem corte de HU ou LCC.",
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
    parser.add_argument(
        "--test-aorta-dice",
        dest="evaluate_mmwhs_test_aorta",
        action="store_true",
        help="Calcula Dice isolado do label 820 do MM-WHS test em 1 mm via Wine.",
    )
    parser.add_argument(
        "--evaluate-mmwhs-test-aorta",
        dest="evaluate_mmwhs_test_aorta",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--aorta-eval-only",
        action="store_true",
        help="Atualiza apenas o Dice da aorta em um --resume-dir, sem refazer o pipeline.",
    )
    parser.add_argument(
        "--evaluator-dir",
        type=Path,
        help="Pasta do avaliador oficial; padrão: dentro do MM-WHS.",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Avisa no desktop e toca um som ao terminar o run.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def resolve_dataset_path(dataset: str, explicit_path: Path | None) -> Path:
    """Resolve e valida a raiz do banco selecionado."""
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
    """Cria uma execução datada ou recupera uma execução existente."""
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
    """Filtra o inventário mantendo uma ordem determinística."""
    selected = inventory.copy()
    if subset != "all":
        selected = selected.loc[selected["subset"].eq(subset)]
    if exam_ids:
        requested = {str(exam_id) for exam_id in exam_ids}
        exam_ids_series = require_series_column(selected, "exam_id").astype(str)
        available = set(exam_ids_series)
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"IDs não encontrados no recorte selecionado: {missing}")
        selected = selected.loc[exam_ids_series.isin(tuple(requested))]
    subset_values = require_series_column(selected, "subset")
    subset_order = subset_values.map(
        lambda value: {"train": 0, "test": 1}.get(str(value))
    ).fillna(2)
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


def _coordinates_to_fields(prefix: str, coordinates: Any) -> dict[str, int | None]:
    """Converte uma coordenada opcional em campos escalares nomeados."""
    values = (
        tuple(int(value) for value in coordinates) if coordinates is not None else ()
    )
    return {
        f"{prefix}_y": values[0] if len(values) > 0 else None,
        f"{prefix}_x": values[1] if len(values) > 1 else None,
        f"{prefix}_z": values[2] if len(values) > 2 else None,
    }


def _save_aorta_ground_truth_visual(
    output_path: Path,
    *,
    exam_label: str,
    ground_truth: Any,
    prediction: Any,
    spacing: Sequence[float],
) -> None:
    """Salva a comparação 3D entre a aorta de referência e a predita."""
    visualize_binary_masks_comparison(
        ground_truth,
        prediction,
        spacing=spacing,
        save_html_path=output_path,
        display_plot=False,
        plot_name=f"{exam_label}: aorta de referência vs predita",
        reference_name="Aorta ground truth",
        predicted_name="Aorta predita",
        reference_color=0x39B54A,
        predicted_color=0xFF5555,
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
    """Salva as vistas representativas de uma etapa intermediária."""
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
    aorta_export_dir: Path | None = None,
) -> dict[str, Any]:
    """Executa todas as etapas para um exame CCTA externo."""
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
        "aorta_ground_truth_available": False,
        "aorta_ground_truth_evaluated": False,
        "aorta_ground_truth_voxels": None,
        "aorta_dice": None,
        "aorta_evaluation_method": None,
        "aorta_evaluation_status": None,
        "aorta_evaluation_error": None,
    }

    try:
        native_image = load_ccta_volume(record).astype(np.float32, copy=False)
        native_aorta_ground_truth = load_ccta_aorta_ground_truth(record)
        if (
            native_aorta_ground_truth is not None
            and native_aorta_ground_truth.shape != native_image.shape
        ):
            raise ValueError(
                "Imagem e ground truth da aorta possuem shapes diferentes: "
                f"{native_image.shape} != {native_aorta_ground_truth.shape}."
            )
        result["aorta_ground_truth_available"] = native_aorta_ground_truth is not None
        spacing_values: list[float] = []
        for column in ("spacing_x_mm", "spacing_y_mm", "spacing_z_mm"):
            value = record.get(column)
            if value is None:
                raise TypeError(f"Espaçamento ausente no inventário: {column}.")
            spacing_values.append(float(value))
        spacing = tuple(spacing_values)
        image = native_image
        flipped_axes: tuple[int, ...] = ()
        if align_orcascore:
            image, flipped_axes = align_ccta_volume_to_imagecas_view(image, dataset)
            if native_aorta_ground_truth is not None:
                aorta_ground_truth, ground_truth_flipped_axes = (
                    align_ccta_volume_to_imagecas_view(
                        native_aorta_ground_truth,
                        dataset,
                    )
                )
                if ground_truth_flipped_axes != flipped_axes:
                    raise RuntimeError(
                        "Imagem e ground truth receberam alinhamentos diferentes."
                    )
            else:
                aorta_ground_truth = None
        else:
            aorta_ground_truth = native_aorta_ground_truth
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
            label=aorta_ground_truth,
            include_intermediates=True,
        )
        threshold_mask = image_data["threshold_mask"]
        lcc_image = image_data["lcc_image"]
        downscale_factors = image_data["downscale_factors"]
        scaled_spacing = tuple(float(value) for value in image_data["scaled_spacing"])
        visual_spacing = (scaled_spacing[1], scaled_spacing[0], scaled_spacing[2])
        preprocessing_details = dict(image_data["preprocessing_details"])
        processed_aorta_ground_truth = image_data["label"]
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
            title=(
                "Voxels finitos (sem limiar HU)"
                if preprocessing_details.get("threshold_mode") == "none"
                else "Máscara após threshold"
            ),
            vmin=0.0,
            vmax=1.0,
        )
        _save_stage(
            exam_dir,
            "02_lcc",
            lcc_image,
            title=(
                "Imagem sem corte de HU"
                if preprocessing_details.get("threshold_mode") == "none"
                else "Imagem após LCC"
            ),
            vmin=-200.0,
            vmax=1000.0,
        )
        del (
            image_data,
            threshold_mask,
            image,
            native_image,
            native_aorta_ground_truth,
            aorta_ground_truth,
        )

        circle_tracking = locate_and_filter_aorta_circles(
            lcc_image,
            downscale_factors,
            scaled_spacing,
            config["CIRCLE_DETECTION"],
        )
        raw_circles = circle_tracking.original_circles
        if not raw_circles:
            raise RuntimeError("Nenhum círculo da aorta foi detectado.")
        detected_circles = circle_tracking.filtered_circles
        if not detected_circles:
            raise RuntimeError("O filtro removeu todos os círculos da aorta.")
        result["aorta_circle_count_before_filter"] = len(raw_circles)
        result.update(circle_tracking.filter_diagnostics)
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
        if aorta_export_dir is not None and dataset == "MM-WHS" and subset == "test":
            result["aorta_evaluation_method"] = OFFICIAL_AORTA_METHOD
            prediction_path = aorta_export_dir / f"{exam_id}_label.nii.gz"
            try:
                if prediction_path.exists():
                    validate_saved_prediction(
                        Path(str(record["path"])), prediction_path
                    )
                else:
                    prediction = AortaPrediction(
                        mask=np.asarray(aorta_mask, dtype=np.uint8),
                        native_shape=(
                            int(result["original_size_x"]),
                            int(result["original_size_y"]),
                            int(result["original_slice_count"]),
                        ),
                        flipped_axes=flipped_axes,
                        circle_count=len(detected_circles),
                    )
                    native_mask = restore_native_mask(prediction)
                    save_prediction(
                        Path(str(record["path"])), native_mask, prediction_path
                    )
                    del native_mask, prediction
                result["aorta_evaluation_status"] = "pending"
            except Exception as error:
                LOGGER.exception("Falha ao exportar a aorta de %s", exam_id)
                result["aorta_evaluation_status"] = "error"
                result["aorta_evaluation_error"] = str(error)
        if processed_aorta_ground_truth is not None:
            aorta_ground_truth_mask = np.asarray(processed_aorta_ground_truth) > 0
            if aorta_ground_truth_mask.shape != aorta_mask.shape:
                raise ValueError(
                    "Aorta predita e ground truth processado possuem shapes diferentes: "
                    f"{aorta_mask.shape} != {aorta_ground_truth_mask.shape}."
                )
            if not np.any(aorta_ground_truth_mask):
                raise ValueError(
                    "O ground truth da aorta ficou vazio após o downscale."
                )
            result.update(
                {
                    "aorta_ground_truth_evaluated": True,
                    "aorta_ground_truth_voxels": int(aorta_ground_truth_mask.sum()),
                    "aorta_dice": float(
                        dice_score(aorta_mask, aorta_ground_truth_mask)
                    ),
                }
            )
            if exam_dir is not None:
                _save_aorta_ground_truth_visual(
                    exam_dir / "aorta_ground_truth_comparison.html",
                    exam_label=f"{dataset} {exam_id}",
                    ground_truth=aorta_ground_truth_mask,
                    prediction=aorta_mask,
                    spacing=visual_spacing,
                )
            del aorta_ground_truth_mask
        processed_aorta_ground_truth = None
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
            vesselness_ostia = None
            if exam_dir is not None:
                save_segmentation_visual_to_path(
                    exam_dir / "aorta_ostia_artery.html",
                    plot_name=f"{dataset} {exam_id}: aorta, óstios e artérias",
                    aorta_mask=aorta_mask,
                    ostia_left=None,
                    ostia_right=None,
                    artery_mask=None,
                    spacing=visual_spacing,
                )
            return result

        result.update(_coordinates_to_fields("ostia_left", ostia_left))
        result.update(_coordinates_to_fields("ostia_right", ostia_right))
        vesselness_ostia = None

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
        artery_segmentation = segment_artery_masks_from_vesselness(
            lcc_image,
            vesselness_artery,
            ostia_left,
            ostia_right,
            config,
            method="region_growing",
        )
        raw_artery_mask = artery_segmentation.raw_mask
        _save_stage(
            exam_dir,
            "06_artery_raw",
            raw_artery_mask,
            title="Artérias antes da morfologia",
            vmin=0.0,
            vmax=1.0,
        )
        closed_mask = artery_segmentation.closed_mask
        artery_mask = artery_segmentation.final_mask
        del vesselness_artery
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
            save_segmentation_visual_to_path(
                exam_dir / "aorta_ostia_artery.html",
                plot_name=f"{dataset} {exam_id}: aorta, óstios e artérias",
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
            save_json_atomic(result, exam_dir / "result.json")
    return result


def _configure_logging(logs_dir: Path, verbose: bool) -> None:
    """Configura logs em arquivo e no terminal para a execução."""
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
    add_run_file_handler(logs_dir / "pipeline.log", formatter=formatter)


def run(args: argparse.Namespace) -> RunPaths:
    """Executa a coorte externa selecionada e retorna seus diretórios."""
    base_path = resolve_dataset_path(args.dataset, args.base_path)
    inventory = discover_ccta_dataset(args.dataset, base_path)
    selected = select_inventory(
        inventory,
        subset=args.subset,
        exam_ids=args.exam_ids,
        limit=args.limit,
    )
    evaluate_official = bool(getattr(args, "evaluate_mmwhs_test_aorta", False))
    evaluation_only = bool(getattr(args, "aorta_eval_only", False))
    if evaluation_only and (not evaluate_official or args.resume_dir is None):
        raise ValueError("--aorta-eval-only exige --test-aorta-dice e --resume-dir.")
    if evaluate_official and args.dataset != "mmwhs":
        raise ValueError("A avaliação oficial da aorta é exclusiva do MM-WHS.")
    test_records = selected.loc[require_series_column(selected, "subset").eq("test")]
    if evaluate_official and test_records.empty:
        raise ValueError("A avaliação oficial exige exames MM-WHS test na seleção.")
    evaluator_dir = getattr(args, "evaluator_dir", None) or base_path / EVALUATOR_FOLDER
    if evaluate_official:
        preflight_evaluator(
            evaluator_dir,
            [str(value) for value in require_series_column(test_records, "exam_id")],
        )
    paths = create_run_paths(
        args.output_root,
        args.dataset,
        args.resolution,
        resume_dir=args.resume_dir,
    )
    args.notification_run_dir = paths.run_dir
    _configure_logging(paths.logs_dir, args.verbose)
    selected_exam_records = [
        {"subset": str(record["subset"]), "exam_id": str(record["exam_id"])}
        for _, record in selected.iterrows()
    ]
    selected_keys = {
        (record["subset"], record["exam_id"]) for record in selected_exam_records
    }
    aorta_ground_truth_keys = {
        (str(record["subset"]), str(record["exam_id"]))
        for _, record in selected.iterrows()
        if isinstance(record.get("label_path"), (str, Path))
    }

    config_path = paths.config_dir / "effective_pipeline_config.json"
    manifest_path = paths.config_dir / "run_manifest.json"
    if args.resume_dir is None:
        config = load_notebook_pipeline_config(args.config_file, args.resolution)
        if args.no_hu_threshold:
            thresholding = dict(config.get("THRESHOLDING", {}))
            thresholding["method"] = "none"
            config["THRESHOLDING"] = thresholding
    else:
        if not config_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                "Run retomado não possui effective_pipeline_config.json e "
                "run_manifest.json."
            )
        manifest = load_json_file(manifest_path)
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
        if (
            args.no_hu_threshold
            and normalize_threshold_mode(config.get("THRESHOLDING", {}).get("method"))
            != "none"
        ):
            raise ValueError(
                "--no-hu-threshold não corresponde à configuração salva do run."
            )

    if evaluation_only:
        config["USE_GPU"] = False
    else:
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
        save_json_atomic(
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
    if evaluation_only:
        existing_by_key = {external_result_key(row): row for row in rows}
        if len(existing_by_key) != len(rows):
            raise ValueError("O CSV do run contém IDs duplicados.")
        missing = [
            str(record["exam_id"])
            for _, record in test_records.iterrows()
            if (str(record["subset"]), str(record["exam_id"])) not in existing_by_key
        ]
        if missing:
            raise ValueError(f"Resultados de teste ausentes no run: {missing}")
        pending_records = [
            record
            for _, record in test_records.iterrows()
            if not (
                existing_by_key[(str(record["subset"]), str(record["exam_id"]))].get(
                    "aorta_evaluation_status"
                )
                == "unavailable"
                and existing_by_key[
                    (str(record["subset"]), str(record["exam_id"]))
                ].get("aorta_evaluation_method")
                == OFFICIAL_AORTA_METHOD
                and not has_aorta_mask(
                    existing_by_key[(str(record["subset"]), str(record["exam_id"]))]
                )
            )
            and not official_evaluation_complete(
                existing_by_key[(str(record["subset"]), str(record["exam_id"]))],
                paths.run_dir,
            )
        ]
    else:
        completed = {
            external_result_key(row)
            for row in rows
            if is_terminal_external_result(row, aorta_ground_truth_keys)
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
    if args.resume_dir is not None and metadata_path.is_file():
        previous_started_at = load_json_file(metadata_path).get("started_at")
        if isinstance(previous_started_at, str):
            started_at = datetime.fromisoformat(previous_started_at)
    save_json_atomic(
        build_external_metadata(
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
        if evaluation_only:
            result = evaluate_test_aorta_result(
                existing_by_key[(str(record["subset"]), str(record["exam_id"]))],
                record,
                config,
                paths.run_dir,
                evaluator_dir,
                rebuild_mask=True,
                align_volume=args.align_orcascore,
            )
        else:
            export_dir = (
                paths.run_dir / "evaluation" / "aorta" / "test" / str(record["exam_id"])
                if evaluate_official and str(record["subset"]) == "test"
                else None
            )
            result = process_external_exam(
                record,
                config,
                args.resolution,
                visual_root=paths.visual_dir if args.visuals else None,
                align_orcascore=args.align_orcascore,
                aorta_export_dir=export_dir,
            )
            if export_dir is not None:
                result = evaluate_test_aorta_result(
                    result,
                    record,
                    config,
                    paths.run_dir,
                    evaluator_dir,
                    rebuild_mask=False,
                    align_volume=args.align_orcascore,
                )
        rows = upsert_external_result(rows, result)
        save_numeric_results(rows, paths.numeric_dir)
        save_json_atomic(
            build_external_metadata(
                dataset=args.dataset,
                resolution=args.resolution,
                subset=args.subset,
                rows=rows,
                started_at=started_at,
                state="running",
            ),
            metadata_path,
        )
        if result["status"] != "success" and args.fail_fast and not evaluation_only:
            raise RuntimeError(
                f"Falha no exame {record['exam_id']}: {result.get('error')}"
            )

    terminal_keys = {
        external_result_key(row)
        for row in rows
        if is_terminal_external_result(row, aorta_ground_truth_keys)
    }
    attempted_keys = {external_result_key(row) for row in rows}
    all_attempted = selected_keys.issubset(attempted_keys)
    all_terminal = selected_keys.issubset(terminal_keys)
    final_state = (
        "complete"
        if all_terminal
        else "complete_with_errors"
        if all_attempted
        else "incomplete"
    )
    save_json_atomic(
        build_external_metadata(
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


def _notify_external_result(args: argparse.Namespace, paths: RunPaths) -> None:
    """Lê o metadata final e diferencia falhas científicas das avaliações."""
    try:
        metadata = load_json_file(paths.run_dir / "metadata.json")
        state = str(metadata.get("state") or "incomplete")
        if state not in {"complete", "complete_with_errors", "incomplete"}:
            state = "incomplete"
        status_counts = metadata.get("status_counts") or {}
        pipeline_errors = int(status_counts.get("error") or 0)
        official = (
            (metadata.get("ground_truth_metrics") or {}).get("aorta") or {}
        ).get("test_official") or {}
        evaluation_errors = int(official.get("error_exam_count") or 0)
        details = [
            f"{int(metadata.get('processed_exam_count') or 0)} exame(s) processado(s)."
        ]
        if pipeline_errors:
            details.append(f"Pipeline: {pipeline_errors} erro(s).")
        if evaluation_errors:
            details.append(f"Avaliação oficial: {evaluation_errors} falha(s).")
        if state == "complete" and evaluation_errors:
            state = "complete_with_warnings"
    except (OSError, ValueError, TypeError, AttributeError) as error:
        LOGGER.warning("Não foi possível ler o metadata final para o aviso: %s", error)
        state = "incomplete"
        details = ["Metadata final indisponível; confira o run."]

    notify_run_completion(
        pipeline=args.dataset,
        split=args.subset,
        resolution=args.resolution,
        run_dir=paths.run_dir,
        status=state,
        details=" ".join(details),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Executa o batch externo e avisa sobre o resultado quando solicitado."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        paths = run(args)
    except BaseException as error:
        if args.notify:
            run_dir = getattr(args, "notification_run_dir", args.resume_dir)
            notify_run_completion(
                pipeline=args.dataset,
                split=args.subset,
                resolution=args.resolution,
                run_dir=Path(run_dir) if run_dir is not None else None,
                status="interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                details="Consulte o terminal e os logs para detalhes.",
            )
        if isinstance(error, (FileNotFoundError, ValueError, RuntimeError)):
            LOGGER.error("%s", error)
            return 1
        raise
    if args.notify:
        _notify_external_result(args, paths)
    print(f"Resultados salvos em: {paths.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
