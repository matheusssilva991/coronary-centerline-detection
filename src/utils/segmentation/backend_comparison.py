"""Comparação etapa a etapa entre execuções CPU e GPU do pipeline.

Este módulo roda a mesma imagem nos dois backends, coleta métricas por etapa
e salva diferenças numéricas/binárias para identificar onde CPU e GPU divergem.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd
from skimage.morphology import ball

from ..processing.binary_operations import binary_closing, binary_dilation
from ..processing.gpu_utils import GPU_AVAILABLE, cp
from ..project.config import load_config_json, scale_config_to_resolution
from ..utils.metrics import dice_score
from .artery_segmentation import normal_region_growing_from_ostia
from .ostia_detection import (
    check_ostium_intersection,
    find_aorta_surface,
    find_ostia,
)
from .pipeline_detection import (
    get_or_detect_aorta_circles,
    get_or_segment_aorta,
)
from .pipeline_preprocessing import get_or_compute_vesselness, load_and_preprocess_image


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "pipeline_config.json"
DEFAULT_SPLIT_CONFIG_PATH = REPO_ROOT / "config" / "imagecas_splits.json"
DEFAULT_BASE_PATH = Path("/media/matheus/HD/DatasetsCCTA/ImageCAS/1-1000")
DEFAULT_BASE_PATH_FALLBACK = Path("/data04/home/mpmaia/ImageCAS/database/1-1000")
DEFAULT_BASE_SAVE_PATH = Path("/media/matheus/HD/DatasetsCCTA/Processed_ImageCAS")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "segmentation" / "backend_comparison"
T = TypeVar("T")


def _sync_gpu_if_needed(use_gpu: bool) -> None:
    """Sincroniza a GPU antes/depois de medir tempo de uma etapa."""
    if use_gpu and GPU_AVAILABLE and cp is not None:
        cp.cuda.Stream.null.synchronize()


def _time_call(
    timings: Dict[str, float],
    stage: str,
    use_gpu: bool,
    func: Callable[[], T],
) -> T:
    """Executa uma etapa cronometrada, sincronizando GPU quando necessário."""
    _sync_gpu_if_needed(use_gpu)
    start = time.perf_counter()
    result = func()
    _sync_gpu_if_needed(use_gpu)
    timings[stage] = time.perf_counter() - start
    return result


def has_imagecas_images(path: Path) -> bool:
    """Retorna True se o diretório contém arquivos ImageCAS."""
    return path.exists() and any(path.glob("*.img.nii.gz"))


def resolve_base_path(base_path: Path) -> Path:
    """Resolve o caminho do dataset usando fallback para o caminho padrão."""
    if has_imagecas_images(base_path):
        return base_path
    if base_path == DEFAULT_BASE_PATH and has_imagecas_images(DEFAULT_BASE_PATH_FALLBACK):
        return DEFAULT_BASE_PATH_FALLBACK
    return base_path


def load_comparison_config(config_path: Path, resolution: str) -> Dict[str, Any]:
    """Carrega e escala a configuração do pipeline para a resolução escolhida."""
    config = load_config_json(str(config_path), {})
    if resolution == "high":
        config = copy.deepcopy(config)
        config["DOWNSCALE_FACTORS"] = (1, 1, 1)
    config = scale_config_to_resolution(config)
    config["LOAD_CACHE"] = False
    config["SAVE_CACHE"] = False
    return config


def load_split_img_ids(split_config_path: Path, split_name: str) -> List[int]:
    """Carrega IDs de um split do arquivo JSON de splits."""
    data = json.loads(split_config_path.read_text(encoding="utf-8"))
    splits = data.get("splits", {})
    if split_name not in splits:
        available = ", ".join(sorted(splits)) or "nenhum"
        raise ValueError(
            f"Split '{split_name}' não encontrado em {split_config_path}. "
            f"Disponíveis: {available}."
        )
    return [int(img_id) for img_id in splits[split_name]]


def _coord_tuple(value: Optional[Sequence[int]]) -> Optional[Tuple[int, int, int]]:
    """Converte coordenadas para tupla inteira ou mantém None."""
    if value is None:
        return None
    return tuple(map(int, value))


def _coord_distance(
    cpu_coord: Optional[Sequence[int]],
    gpu_coord: Optional[Sequence[int]],
    spacing_yxz: Sequence[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Calcula distância CPU/GPU de uma coordenada em voxels e milímetros."""
    if cpu_coord is None or gpu_coord is None:
        return None, None

    cpu_arr = np.asarray(cpu_coord, dtype=float)
    gpu_arr = np.asarray(gpu_coord, dtype=float)
    diff = cpu_arr - gpu_arr
    dist_voxels = float(np.linalg.norm(diff))
    dist_mm = float(np.linalg.norm(diff * np.asarray(spacing_yxz, dtype=float)))
    return dist_voxels, dist_mm


def _binary_metrics(
    img_id: int,
    stage: str,
    cpu_array: Any,
    gpu_array: Any,
) -> Dict[str, Any]:
    """Calcula métricas de diferença para máscaras binárias."""
    cpu_mask = np.asarray(cpu_array) > 0
    gpu_mask = np.asarray(gpu_array) > 0
    cpu_voxels = int(cpu_mask.sum())
    gpu_voxels = int(gpu_mask.sum())
    intersection = int(np.logical_and(cpu_mask, gpu_mask).sum())
    union = int(np.logical_or(cpu_mask, gpu_mask).sum())
    xor_voxels = int(np.logical_xor(cpu_mask, gpu_mask).sum())

    if cpu_voxels + gpu_voxels == 0:
        dice = 1.0
    else:
        dice = float((2.0 * intersection) / (cpu_voxels + gpu_voxels))

    return {
        "IMG_ID": img_id,
        "stage": stage,
        "kind": "binary",
        "shape_equal": cpu_mask.shape == gpu_mask.shape,
        "cpu_voxels": cpu_voxels,
        "gpu_voxels": gpu_voxels,
        "voxel_delta": gpu_voxels - cpu_voxels,
        "abs_voxel_delta": abs(gpu_voxels - cpu_voxels),
        "dice_cpu_gpu": dice,
        "iou_cpu_gpu": float(intersection / union) if union else 1.0,
        "xor_voxels": xor_voxels,
        "xor_fraction": float(xor_voxels / cpu_mask.size) if cpu_mask.size else 0.0,
    }


def _float_metrics(
    img_id: int,
    stage: str,
    cpu_array: Any,
    gpu_array: Any,
    tolerance: float,
) -> Dict[str, Any]:
    """Calcula métricas de erro absoluto para mapas contínuos."""
    cpu = np.asarray(cpu_array, dtype=np.float64)
    gpu = np.asarray(gpu_array, dtype=np.float64)
    if cpu.shape != gpu.shape:
        return {
            "IMG_ID": img_id,
            "stage": stage,
            "kind": "float",
            "shape_equal": False,
            "cpu_shape": tuple(cpu.shape),
            "gpu_shape": tuple(gpu.shape),
        }

    diff = gpu - cpu
    abs_diff = np.abs(diff)
    return {
        "IMG_ID": img_id,
        "stage": stage,
        "kind": "float",
        "shape_equal": True,
        "cpu_min": float(np.min(cpu)),
        "gpu_min": float(np.min(gpu)),
        "cpu_max": float(np.max(cpu)),
        "gpu_max": float(np.max(gpu)),
        "cpu_mean": float(np.mean(cpu)),
        "gpu_mean": float(np.mean(gpu)),
        "mean_abs_error": float(np.mean(abs_diff)),
        "max_abs_error": float(np.max(abs_diff)),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "p95_abs_error": float(np.percentile(abs_diff, 95)),
        "p99_abs_error": float(np.percentile(abs_diff, 99)),
        "voxels_abs_gt_tolerance": int(np.sum(abs_diff > tolerance)),
        "fraction_abs_gt_tolerance": float(np.mean(abs_diff > tolerance)),
    }


def _circle_metrics(
    img_id: int,
    cpu_circles: List[Dict[str, Any]],
    gpu_circles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compara círculos detectados por fatia entre CPU e GPU."""
    cpu_by_slice = {int(item["slice_index"]): item for item in cpu_circles}
    gpu_by_slice = {int(item["slice_index"]): item for item in gpu_circles}
    common_slices = sorted(set(cpu_by_slice) & set(gpu_by_slice))

    center_distances = []
    radius_diffs = []
    for slice_idx in common_slices:
        # Compara apenas fatias detectadas nos dois backends.
        cpu_circle = cpu_by_slice[slice_idx]
        gpu_circle = gpu_by_slice[slice_idx]
        center_distances.append(
            float(
                np.linalg.norm(
                    [
                        gpu_circle["center_x"] - cpu_circle["center_x"],
                        gpu_circle["center_y"] - cpu_circle["center_y"],
                    ]
                )
            )
        )
        radius_diffs.append(float(abs(gpu_circle["radius"] - cpu_circle["radius"])))

    return {
        "IMG_ID": img_id,
        "stage": "aorta_circles",
        "kind": "circles",
        "cpu_count": len(cpu_circles),
        "gpu_count": len(gpu_circles),
        "common_slices": len(common_slices),
        "cpu_only_slices": len(set(cpu_by_slice) - set(gpu_by_slice)),
        "gpu_only_slices": len(set(gpu_by_slice) - set(cpu_by_slice)),
        "mean_center_distance_px": (
            float(np.mean(center_distances)) if center_distances else None
        ),
        "max_center_distance_px": (
            float(np.max(center_distances)) if center_distances else None
        ),
        "mean_radius_abs_diff_px": (
            float(np.mean(radius_diffs)) if radius_diffs else None
        ),
        "max_radius_abs_diff_px": (
            float(np.max(radius_diffs)) if radius_diffs else None
        ),
    }


def _evaluate_ostia(
    aorta_mask: Any,
    vesselness_ostios: Any,
    label: Any,
    scaled_spacing: Sequence[float],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Seleciona os óstios e calcula status para uma execução específica."""
    dx, dy, dz = scaled_spacing
    spacing_yxz = (dy, dx, dz)
    ostia_config = config["OSTIA_DETECTION"]

    # Seleciona os óstios na superfície da aorta usando o mapa de vesselness.
    ostia_left, ostia_right = find_ostia(
        aorta_mask,
        vesselness_ostios,
        spacing=spacing_yxz,
        top_n=ostia_config["top_n"],
        max_z_diff_mm=ostia_config["max_z_diff_mm"],
        lower_fraction=ostia_config["lower_fraction"],
        min_center_distance_factor=ostia_config["min_center_distance_factor"],
        min_lateral_factor=ostia_config["min_lateral_factor"],
        erosion_radius=ostia_config["erosion_radius"],
        verbose=False,
    )

    # Avalia se os óstios encontrados estão no label arterial ou próximos dele.
    label_artery = (label == 1).astype(np.uint8)
    left_info = check_ostium_intersection(
        ostia_left,
        label_artery,
        spacing=spacing_yxz,
        ostium_name="Óstio esquerdo",
    )
    right_info = check_ostium_intersection(
        ostia_right,
        label_artery,
        spacing=spacing_yxz,
        ostium_name="Óstio direito",
    )

    tolerable = config["OSTIA_VALIDATION"]["distance_threshold_mm"]
    both_correct = left_info["intersects"] and right_info["intersects"]
    both_tolerable = (
        left_info["intersects"] or left_info["physical_dist"] <= tolerable
    ) and (right_info["intersects"] or right_info["physical_dist"] <= tolerable)

    if both_correct:
        status = "both_correct"
    elif both_tolerable:
        status = "both_tolerable"
    else:
        status = "found_but_wrong"

    return {
        "ostia_left": ostia_left,
        "ostia_right": ostia_right,
        "label_artery": label_artery,
        "left_info": left_info,
        "right_info": right_info,
        "both_correct": both_correct,
        "both_tolerable": both_tolerable and not both_correct,
        "status": status,
    }


def _artery_steps(
    img_id: int,
    lcc_image: Any,
    label_artery: Any,
    ostia_left: Optional[Sequence[int]],
    ostia_right: Optional[Sequence[int]],
    config: Dict[str, Any],
    base_save_path: Path,
    use_gpu: bool,
    scaled_spacing: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Executa as etapas arteriais para um backend e retorna máscaras/tempos."""
    config = copy.deepcopy(config)
    config["USE_GPU"] = use_gpu
    timings: Dict[str, float] = {}
    vesselness_spacing = (
        (scaled_spacing[1], scaled_spacing[0], scaled_spacing[2])
        if scaled_spacing is not None
        else None
    )

    # Calcula o mapa de vasos arterial no backend em teste.
    vesselness_artery = _time_call(
        timings,
        "vesselness_artery",
        use_gpu,
        lambda: get_or_compute_vesselness(
            img_id,
            lcc_image,
            cache_dir=str(
                base_save_path
                / f"debug_vesselness_artery_cache_{'gpu' if use_gpu else 'cpu'}"
            ),
            vesselness_config=config["VESSELNESS_ARTERY"],
            load_cache=False,
            save_cache=False,
            use_gpu=use_gpu,
            spacing=vesselness_spacing,
        ),
    )

    def run_region_growing() -> Any:
        """Executa o mesmo region growing usado pelo pipeline principal."""
        return normal_region_growing_from_ostia(
            vesselness_artery,
            ostia_left,
            ostia_right,
            config,
        )

    raw_mask = _time_call(
        timings,
        "artery_region_growing",
        use_gpu,
        run_region_growing,
    )

    # Pós-processa a máscara bruta antes da comparação final.
    raw_mask = (raw_mask > 0).astype(np.uint8)
    post_config = config["POSTPROCESSING"]

    def run_postprocessing() -> Tuple[Any, Any]:
        """Aplica fechamento e dilatação finais, sempre na CPU."""
        closed = binary_closing(
            raw_mask > 0,
            structure=ball(post_config["closing_radius"]),
            gpu=False,
        )
        final = binary_dilation(
            closed,
            structure=ball(post_config["dilation_radius"]),
            gpu=False,
        )
        return closed, final

    closed_mask, final_mask = _time_call(
        timings,
        "artery_postprocessing",
        use_gpu,
        run_postprocessing,
    )
    timings["artery_total"] = sum(timings.values())

    return {
        "vesselness_artery": vesselness_artery,
        "artery_raw_mask": raw_mask,
        "artery_closed_mask": closed_mask,
        "artery_final_mask": final_mask,
        "artery_voxels": int(np.sum(final_mask)),
        "dice_artery": float(dice_score(final_mask, label_artery)),
        "__timings__": timings,
    }


def _run_steps_for_backend(
    img_id: int,
    image_data: Dict[str, Any],
    config: Dict[str, Any],
    base_save_path: Path,
    use_gpu: bool,
) -> Dict[str, Any]:
    """Roda todas as etapas do pipeline para um backend específico."""
    lcc_image = image_data["lcc_image"]
    label = image_data["label"]
    scaled_spacing = image_data["scaled_spacing"]
    vesselness_spacing = (scaled_spacing[1], scaled_spacing[0], scaled_spacing[2])
    downscale_factors = image_data["downscale_factors"]

    backend_config = copy.deepcopy(config)
    backend_config["USE_GPU"] = use_gpu
    timings: Dict[str, float] = {}
    total_start = time.perf_counter()

    # Calcula o mapa de vasos usado na seleção dos óstios.
    vesselness_ostios = _time_call(
        timings,
        "vesselness_ostios",
        use_gpu,
        lambda: get_or_compute_vesselness(
            img_id,
            lcc_image,
            cache_dir=str(
                base_save_path
                / f"debug_vesselness_ostios_cache_{'gpu' if use_gpu else 'cpu'}"
            ),
            vesselness_config=backend_config["VESSELNESS_AORTA"],
            load_cache=False,
            save_cache=False,
            use_gpu=use_gpu,
            spacing=vesselness_spacing,
        ),
    )
    # Detecta os círculos usados para localizar/segmentar a aorta.
    detected_circles = _time_call(
        timings,
        "aorta_circles",
        use_gpu,
        lambda: get_or_detect_aorta_circles(
            img_id,
            lcc_image,
            downscale_factors,
            scaled_spacing,
            backend_config["CIRCLE_DETECTION"],
            str(base_save_path),
            load_cache=False,
            save_cache=False,
        ),
    )
    # Segmenta a aorta a partir dos círculos detectados.
    aorta_mask = _time_call(
        timings,
        "aorta_mask",
        use_gpu,
        lambda: get_or_segment_aorta(
            img_id,
            lcc_image,
            detected_circles,
            backend_config["LEVEL_SET"],
            str(base_save_path),
            load_cache=False,
            save_cache=False,
            use_gpu=use_gpu,
        ),
    )
    # Extrai a superfície da aorta para comparação direta entre backends.
    aorta_surface = _time_call(
        timings,
        "aorta_surface",
        use_gpu,
        lambda: find_aorta_surface(
            aorta_mask,
            erosion_radius=backend_config["OSTIA_DETECTION"]["erosion_radius"],
        ),
    )
    # Seleciona os óstios e mede o status em relação ao label.
    ostia_eval = _time_call(
        timings,
        "ostia_detection",
        use_gpu,
        lambda: _evaluate_ostia(
            aorta_mask,
            vesselness_ostios,
            label,
            scaled_spacing,
            backend_config,
        ),
    )
    # Segmenta as artérias a partir dos óstios selecionados.
    artery = _artery_steps(
        img_id,
        lcc_image,
        ostia_eval["label_artery"],
        ostia_eval["ostia_left"],
        ostia_eval["ostia_right"],
        backend_config,
        base_save_path,
        use_gpu=use_gpu,
        scaled_spacing=scaled_spacing,
    )
    artery_timings = artery.pop("__timings__")
    timings.update(artery_timings)
    _sync_gpu_if_needed(use_gpu)
    timings["backend_total"] = time.perf_counter() - total_start

    return {
        "vesselness_ostios": vesselness_ostios,
        "aorta_circles": detected_circles,
        "aorta_mask": aorta_mask,
        "aorta_surface": aorta_surface,
        "ostia_eval": ostia_eval,
        **artery,
        "__timings__": timings,
    }


def _ostia_row(
    img_id: int,
    cpu_eval: Dict[str, Any],
    gpu_eval: Dict[str, Any],
    spacing_yxz: Sequence[float],
) -> Dict[str, Any]:
    """Monta a linha comparativa dos óstios CPU/GPU."""
    left_dist_voxels, left_dist_mm = _coord_distance(
        cpu_eval["ostia_left"], gpu_eval["ostia_left"], spacing_yxz
    )
    right_dist_voxels, right_dist_mm = _coord_distance(
        cpu_eval["ostia_right"], gpu_eval["ostia_right"], spacing_yxz
    )

    return {
        "IMG_ID": img_id,
        "cpu_status": cpu_eval["status"],
        "gpu_status": gpu_eval["status"],
        "status_equal": cpu_eval["status"] == gpu_eval["status"],
        "cpu_left_ostium": _coord_tuple(cpu_eval["ostia_left"]),
        "gpu_left_ostium": _coord_tuple(gpu_eval["ostia_left"]),
        "left_distance_voxels_cpu_gpu": left_dist_voxels,
        "left_distance_mm_cpu_gpu": left_dist_mm,
        "cpu_right_ostium": _coord_tuple(cpu_eval["ostia_right"]),
        "gpu_right_ostium": _coord_tuple(gpu_eval["ostia_right"]),
        "right_distance_voxels_cpu_gpu": right_dist_voxels,
        "right_distance_mm_cpu_gpu": right_dist_mm,
        "cpu_left_dist_to_label_mm": cpu_eval["left_info"]["physical_dist"],
        "gpu_left_dist_to_label_mm": gpu_eval["left_info"]["physical_dist"],
        "cpu_right_dist_to_label_mm": cpu_eval["right_info"]["physical_dist"],
        "gpu_right_dist_to_label_mm": gpu_eval["right_info"]["physical_dist"],
        "cpu_both_correct": cpu_eval["both_correct"],
        "gpu_both_correct": gpu_eval["both_correct"],
        "cpu_both_tolerable": cpu_eval["both_tolerable"],
        "gpu_both_tolerable": gpu_eval["both_tolerable"],
    }


def _timing_row(
    img_id: int,
    preprocess_seconds: float,
    cpu_timings: Dict[str, float],
    gpu_timings: Dict[str, float],
) -> Dict[str, Any]:
    """Monta a linha com tempos CPU/GPU por etapa."""
    row: Dict[str, Any] = {
        "IMG_ID": img_id,
        "preprocess_seconds": preprocess_seconds,
    }
    for stage in sorted(set(cpu_timings) | set(gpu_timings)):
        cpu_seconds = cpu_timings.get(stage)
        gpu_seconds = gpu_timings.get(stage)
        row[f"cpu_{stage}_seconds"] = cpu_seconds
        row[f"gpu_{stage}_seconds"] = gpu_seconds
        if cpu_seconds is not None and gpu_seconds is not None:
            # Speedup > 1 indica CPU mais lenta que GPU naquela etapa.
            row[f"{stage}_delta_gpu_minus_cpu_seconds"] = gpu_seconds - cpu_seconds
            row[f"{stage}_speedup_cpu_over_gpu"] = (
                cpu_seconds / gpu_seconds if gpu_seconds > 0 else None
            )
    return row


def compare_image_cpu_gpu(
    img_id: int,
    config: Dict[str, Any],
    base_path: Path,
    base_save_path: Path,
    float_tolerance: float = 1e-6,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Compara uma imagem nas execuções CPU e GPU."""
    # O pré-processamento é compartilhado para isolar diferenças do backend.
    preprocess_start = time.perf_counter()
    image_data = load_and_preprocess_image(img_id, str(base_path), config)
    preprocess_seconds = time.perf_counter() - preprocess_start
    cpu = _run_steps_for_backend(
        img_id, image_data, config, base_save_path, use_gpu=False
    )
    gpu = _run_steps_for_backend(
        img_id, image_data, config, base_save_path, use_gpu=True
    )

    # Compara mapas contínuos, máscaras binárias, círculos e segmentação final.
    rows = [
        _float_metrics(
            img_id,
            "vesselness_ostios",
            cpu["vesselness_ostios"],
            gpu["vesselness_ostios"],
            float_tolerance,
        ),
        _circle_metrics(img_id, cpu["aorta_circles"], gpu["aorta_circles"]),
        _binary_metrics(img_id, "aorta_mask", cpu["aorta_mask"], gpu["aorta_mask"]),
        _binary_metrics(
            img_id, "aorta_surface", cpu["aorta_surface"], gpu["aorta_surface"]
        ),
        _float_metrics(
            img_id,
            "vesselness_artery",
            cpu["vesselness_artery"],
            gpu["vesselness_artery"],
            float_tolerance,
        ),
        _binary_metrics(
            img_id, "artery_raw_mask", cpu["artery_raw_mask"], gpu["artery_raw_mask"]
        ),
        _binary_metrics(
            img_id,
            "artery_closed_mask",
            cpu["artery_closed_mask"],
            gpu["artery_closed_mask"],
        ),
        _binary_metrics(
            img_id,
            "artery_final_mask",
            cpu["artery_final_mask"],
            gpu["artery_final_mask"],
        ),
    ]

    dx, dy, dz = image_data["scaled_spacing"]
    ostia_row = _ostia_row(
        img_id, cpu["ostia_eval"], gpu["ostia_eval"], spacing_yxz=(dy, dx, dz)
    )
    ostia_row.update(
        {
            "cpu_artery_dice": cpu["dice_artery"],
            "gpu_artery_dice": gpu["dice_artery"],
            "artery_dice_delta_gpu_minus_cpu": (
                gpu["dice_artery"] - cpu["dice_artery"]
            ),
            "cpu_artery_voxels": cpu["artery_voxels"],
            "gpu_artery_voxels": gpu["artery_voxels"],
            "artery_voxel_delta_gpu_minus_cpu": (
                gpu["artery_voxels"] - cpu["artery_voxels"]
            ),
        }
    )
    timing_row = _timing_row(
        img_id,
        preprocess_seconds,
        cpu["__timings__"],
        gpu["__timings__"],
    )

    return rows, ostia_row, timing_row


def compare_images_cpu_gpu(
    img_ids: Iterable[int],
    config: Dict[str, Any],
    base_path: Path,
    base_save_path: Path,
    output_dir: Path,
    float_tolerance: float = 1e-6,
) -> Dict[str, Path]:
    """Executa a comparação e salva CSVs com os resultados."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_rows = []
    ostia_rows = []
    timing_rows = []

    for img_id in img_ids:
        print(f"Comparando IMG_ID {img_id}...")
        rows, ostia_row, timing_row = compare_image_cpu_gpu(
            int(img_id),
            config,
            base_path,
            base_save_path,
            float_tolerance=float_tolerance,
        )
        stage_rows.extend(rows)
        ostia_rows.append(ostia_row)
        timing_rows.append(timing_row)

    stage_path = output_dir / "stage_comparison.csv"
    ostia_path = output_dir / "ostia_comparison.csv"
    timing_path = output_dir / "timing_comparison.csv"
    pd.DataFrame(stage_rows).to_csv(stage_path, index=False)
    pd.DataFrame(ostia_rows).to_csv(ostia_path, index=False)
    pd.DataFrame(timing_rows).to_csv(timing_path, index=False)

    config_path = output_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as file_handle:
        json.dump(
            {
                "gpu_available": GPU_AVAILABLE,
                "base_path": str(base_path),
                "base_save_path": str(base_save_path),
                "float_tolerance": float_tolerance,
                "config": _json_safe(config),
            },
            file_handle,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "stage_comparison": stage_path,
        "ostia_comparison": ostia_path,
        "timing_comparison": timing_path,
        "run_config": config_path,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if hasattr(value, "as_posix"):
        return value.as_posix()
    return value


def build_parser() -> argparse.ArgumentParser:
    """Cria parser da comparação CPU vs GPU."""
    parser = argparse.ArgumentParser(
        description="Compara etapas do pipeline executadas em CPU e GPU."
    )
    parser.add_argument(
        "--img-id",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Um ou mais IDs ImageCAS para comparar. "
            "Se omitido, usa os IDs do split selecionado."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Split usado quando --img-id não é informado. Padrão: train.",
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        default=DEFAULT_SPLIT_CONFIG_PATH,
        help=f"Arquivo JSON com splits. Padrão: {DEFAULT_SPLIT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--resolution",
        choices=["mid", "high"],
        default="mid",
        help="Resolução usada no pipeline.",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Config do pipeline. Padrão: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=DEFAULT_BASE_PATH,
        help=f"Caminho do dataset ImageCAS. Padrão: {DEFAULT_BASE_PATH}",
    )
    parser.add_argument(
        "--base-save-path",
        type=Path,
        default=DEFAULT_BASE_SAVE_PATH,
        help=f"Caminho base de artefatos/cache. Padrão: {DEFAULT_BASE_SAVE_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Diretório de saída. Padrão: output/segmentation/backend_comparison/<timestamp>",
    )
    parser.add_argument(
        "--float-tolerance",
        type=float,
        default=1e-6,
        help="Tolerância para contar diferenças em arrays float.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entrada CLI da comparação CPU vs GPU."""
    parser = build_parser()
    args = parser.parse_args(argv)

    base_path = resolve_base_path(args.base_path)
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = DEFAULT_OUTPUT_ROOT / f"{args.resolution}_res" / timestamp

    config = load_comparison_config(args.config_file, args.resolution)
    img_ids = args.img_id
    if img_ids is None:
        img_ids = load_split_img_ids(args.split_config, args.split)
        print(
            f"Usando {len(img_ids)} imagens do split '{args.split}' "
            f"em {args.split_config}."
        )

    paths = compare_images_cpu_gpu(
        img_ids,
        config,
        base_path,
        args.base_save_path,
        output_dir,
        float_tolerance=args.float_tolerance,
    )

    print("\nComparação salva em:")
    for label, path in paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
