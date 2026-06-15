# Generated script version of src/fuzzy.ipynb.
# Keep experiment logic aligned with the notebook when changing sweep variants.


# %% [notebook cell 2]
# ruff: noqa: E402
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ORIGINAL_CWD = Path.cwd().resolve()
NOTEBOOK_CWD = ORIGINAL_CWD
for candidate in (NOTEBOOK_CWD, NOTEBOOK_CWD.parent, NOTEBOOK_CWD.parent.parent):
    src_dir = candidate / "src"
    if src_dir.exists():
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        break

from utils.project.notebook_env import configure_notebook_environment  # noqa: E402

REPO_ROOT = configure_notebook_environment()

# %% [notebook cell 3]
import copy
import heapq

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, uniform_filter
from skimage.morphology import ball

from utils.project.config import load_config_json, scale_config_to_resolution
from utils.project.dataset import get_data_splits
from utils.project.notebook_env import resolve_imagecas_base_path
from utils.processing import (
    binary_closing,
    binary_dilation,
    downscale_image_ndi,
    threshold_image_with_offset,
)
from utils.segmentation import (
    build_lcc_image_from_mask,
    detect_and_evaluate_ostia,
    get_or_compute_vesselness,
    get_or_detect_aorta_circles,
    get_or_segment_aorta,
    region_growing_segmentation,
)
from utils.utils import dice_score
from utils.utils.nifti_io import load_raw_img_and_label


def _parse_json_arg(value: str | None) -> dict:
    """Carrega overrides JSON a partir de string ou caminho de arquivo."""
    if not value:
        return {}
    path = resolve_cli_path(Path(value))
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def resolve_cli_path(path: Path) -> Path:
    """Resolve paths relativos a partir do diretório original do comando."""
    return path if path.is_absolute() else ORIGINAL_CWD / path


def parse_args() -> argparse.Namespace:
    """Argumentos para rodar o sweep fora do Jupyter."""
    parser = argparse.ArgumentParser(
        description="Executa o sweep do fuzzy.ipynb e salva tabelas em CSV."
    )
    parser.add_argument(
        "--phase",
        choices=("ostia", "segmentation", "all"),
        default="ostia",
        help="Fase do sweep a executar.",
    )
    parser.add_argument("--train-size", type=int, default=30)
    parser.add_argument("--val-size", type=int, default=10)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=REPO_ROOT / "config/pipeline_config.json",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=None,
        help="Diretório ImageCAS. Se omitido, usa notebook_env.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output/segmentation/analysis/fuzzy_sweep",
        help="Diretório raiz para salvar os CSVs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Nome da subpasta da execução. Default: timestamp.",
    )
    parser.add_argument(
        "--fixed-detection-overrides",
        default=None,
        help="JSON string ou arquivo JSON com overrides fixos de detecção.",
    )
    parser.add_argument("--fixed-max-threshold-percentile", type=float, default=None)
    parser.add_argument("--load-cache", action="store_true")
    parser.add_argument("--save-cache", action="store_true")
    parser.add_argument(
        "--no-partial-csv",
        action="store_true",
        help="Desativa salvamento parcial após cada imagem.",
    )
    return parser.parse_args()


ARGS = parse_args()
RUN_STARTED_AT = datetime.now()
RUN_NAME = ARGS.run_name or RUN_STARTED_AT.strftime("%Y-%m-%d_%H-%M-%S")
RUN_OUTPUT_DIR = (resolve_cli_path(ARGS.output_dir) / RUN_NAME).resolve()
RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _csv_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Converte listas/dicts/arrays em strings estáveis antes do CSV."""
    if df.empty:
        return df
    safe_df = df.copy()
    for column in safe_df.columns:
        if safe_df[column].dtype != "object":
            continue
        safe_df[column] = safe_df[column].map(
            lambda value: json.dumps(value.tolist(), ensure_ascii=False)
            if hasattr(value, "tolist")
            else json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list, tuple))
            else value
        )
    return safe_df


def save_csv(df: pd.DataFrame, name: str, output_dir: Path = RUN_OUTPUT_DIR) -> None:
    """Salva DataFrame em CSV quando houver colunas definidas."""
    if df is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    _csv_safe_df(df).to_csv(output_dir / f"{name}.csv", index=False)


def save_json(data: dict, name: str, output_dir: Path = RUN_OUTPUT_DIR) -> None:
    """Salva metadados da execução em JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# %% [notebook cell 5]

CONFIG_PATH = resolve_cli_path(ARGS.config_path)
BASE_PATH = resolve_cli_path(ARGS.base_path) if ARGS.base_path else resolve_imagecas_base_path()

TRAIN_SAMPLE_SIZE = ARGS.train_size
VAL_SAMPLE_SIZE = ARGS.val_size
DOWNSCALE_FACTORS = (2, 2, 1)

# Fases recomendadas:
# - "ostia": escolhe parametros de threshold/aorta/ostios sem rodar segmentacao arterial.
# - "segmentation": fixa a melhor deteccao e varia vesselness arterial/region growing/pos-processamento.
# - "all": roda todas as variantes cadastradas.
OPTIMIZATION_PHASE = ARGS.phase
RUN_SEGMENTATION = OPTIMIZATION_PHASE in {"segmentation", "all"}

# Use estes campos depois da fase "ostia" para fixar a melhor configuracao
# antes de otimizar a segmentacao arterial.
FIXED_DETECTION_OVERRIDES = _parse_json_arg(ARGS.fixed_detection_overrides)
FIXED_MAX_THRESHOLD_PERCENTILE = ARGS.fixed_max_threshold_percentile

MIN_HU = -300

CONTEXTUAL_FUZZY_OBJECT_PERCENTILE = 99.8
CONTEXTUAL_FUZZY_DENSE_PERCENTILE = 99.95
CONTEXTUAL_FUZZY_SOFT_MARGIN_HU = 160
CONTEXTUAL_FUZZY_SMOOTH_RADIUS = 1
CONTEXTUAL_FUZZY_SMOOTH_MODE = "mean"

DETECTION_CONFIG_KEYS = (
    "CIRCLE_DETECTION",
    "LEVEL_SET",
    "VESSELNESS_AORTA",
    "OSTIA_DETECTION",
)
BASELINE_APPROACH = "normal_standard_baseline"


def deep_update_config(base: dict, overrides: dict | None) -> dict:
    """Atualiza dicionarios aninhados sem alterar a configuracao base."""
    result = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


CONFIG = load_config_json(str(CONFIG_PATH), {})
CONFIG["DOWNSCALE_FACTORS"] = list(DOWNSCALE_FACTORS)
CONFIG["LOAD_CACHE"] = bool(ARGS.load_cache)
CONFIG["SAVE_CACHE"] = bool(ARGS.save_cache)
RUN_CONFIG = scale_config_to_resolution(copy.deepcopy(CONFIG))

MAX_THRESHOLD_PERCENTILE = CONFIG.get("MAX_THRESHOLD_PERCENTILE", 99.7)
BASE_MAX_THRESHOLD_PERCENTILE = (
    MAX_THRESHOLD_PERCENTILE
    if FIXED_MAX_THRESHOLD_PERCENTILE is None
    else float(FIXED_MAX_THRESHOLD_PERCENTILE)
)

SHARED_IMPROVED_REGION_GROWING_CONFIG = {
    "use_seed_refinement": True,
    "use_multi_seed": True,
    "use_priority_queue": False,
    "use_adaptive_acceptance": False,
    "use_distance_limit": False,
    "grow_each_ostium_separately": True,
    "seed_search_radius": 2,
    "seed_candidate_radius": 1,
    "max_seed_candidates": 6,
    "seed_min_vesselness_fraction": 0.02,
}


def rg_config_with(**overrides: object) -> dict:
    config = copy.deepcopy(SHARED_IMPROVED_REGION_GROWING_CONFIG)
    config.update(overrides)
    return config


def pipeline_variant(
    label: str,
    *,
    stage: str,
    region_growing_method: str = "standard",
    config_overrides: dict | None = None,
    region_growing: dict | None = None,
    max_threshold_percentile: float | None = None,
    lcc_per_slice: bool = True,
) -> dict:
    return {
        "label": label,
        "approach_family": "normal_pipeline_sweep",
        "pipeline_stage": stage,
        "fuzzy_output_mode": "none",
        "apply_weight_to": "none",
        "region_growing_method": region_growing_method,
        "config_overrides": copy.deepcopy(config_overrides or {}),
        "region_growing": copy.deepcopy(region_growing or {}),
        "max_threshold_percentile": max_threshold_percentile,
        "lcc_per_slice": bool(lcc_per_slice),
    }


PIPELINE_STAGE_VARIANTS = {
    BASELINE_APPROACH: pipeline_variant(
        "normal_standard_baseline",
        stage="baseline",
    ),
    # Rodada focada em ostios/aorta, escolhida a partir dos 12 casos iniciais.
    "normal_ostia_lower086": pipeline_variant(
        "normal_ostia_lower_fraction086",
        stage="ostia_detection",
        config_overrides={"OSTIA_DETECTION": {"lower_fraction": 0.86}},
    ),
    "normal_ostia_lower088": pipeline_variant(
        "normal_ostia_lower_fraction088",
        stage="ostia_detection",
        config_overrides={"OSTIA_DETECTION": {"lower_fraction": 0.88}},
    ),
    "normal_ostia_lower090": pipeline_variant(
        "normal_ostia_lower_fraction090",
        stage="ostia_detection",
        config_overrides={"OSTIA_DETECTION": {"lower_fraction": 0.90}},
    ),
    "normal_ostia_lower092": pipeline_variant(
        "normal_ostia_lower_fraction092",
        stage="ostia_detection",
        config_overrides={"OSTIA_DETECTION": {"lower_fraction": 0.92}},
    ),
    "normal_ostia_lower094": pipeline_variant(
        "normal_ostia_lower_fraction094",
        stage="ostia_detection",
        config_overrides={"OSTIA_DETECTION": {"lower_fraction": 0.94}},
    ),
    "normal_level_iter40": pipeline_variant(
        "normal_aorta_level_set_iter40",
        stage="aorta_segmentation",
        config_overrides={"LEVEL_SET": {"num_iter": 40}},
    ),
    "normal_level_iter45": pipeline_variant(
        "normal_aorta_level_set_iter45",
        stage="aorta_segmentation",
        config_overrides={"LEVEL_SET": {"num_iter": 45}},
    ),
    "normal_level_iter50": pipeline_variant(
        "normal_aorta_level_set_iter50",
        stage="aorta_segmentation",
        config_overrides={"LEVEL_SET": {"num_iter": 50}},
    ),
    "normal_lower090_iter45": pipeline_variant(
        "normal_ostia_lower090_level_iter45",
        stage="combined_ostia_aorta",
        config_overrides={
            "OSTIA_DETECTION": {"lower_fraction": 0.90},
            "LEVEL_SET": {"num_iter": 45},
        },
    ),
    "normal_lower088_iter45": pipeline_variant(
        "normal_ostia_lower088_level_iter45",
        stage="combined_ostia_aorta",
        config_overrides={
            "OSTIA_DETECTION": {"lower_fraction": 0.88},
            "LEVEL_SET": {"num_iter": 45},
        },
    ),
    "normal_lower092_iter45": pipeline_variant(
        "normal_ostia_lower092_level_iter45",
        stage="combined_ostia_aorta",
        config_overrides={
            "OSTIA_DETECTION": {"lower_fraction": 0.92},
            "LEVEL_SET": {"num_iter": 45},
        },
    ),
    "normal_threshold_p999": pipeline_variant(
        "normal_threshold_percentile_999",
        stage="thresholding",
        max_threshold_percentile=99.9,
    ),
    "normal_threshold_p9995": pipeline_variant(
        "normal_threshold_percentile_9995",
        stage="thresholding",
        max_threshold_percentile=99.95,
    ),
    "normal_lcc_per_slice": pipeline_variant(
        "normal_lcc_per_slice",
        stage="lcc_mode",
        lcc_per_slice=True,
    ),
    "normal_lcc_per_volume": pipeline_variant(
        "normal_lcc_per_volume",
        stage="lcc_mode",
        lcc_per_slice=False,
    ),
    "normal_circle_score": pipeline_variant(
        "normal_aorta_circle_candidate_score",
        stage="aorta_localization",
        config_overrides={
            "CIRCLE_DETECTION": {"candidate_selection_strategy": "score"}
        },
    ),
    "normal_circle_miss_tolerated": pipeline_variant(
        "normal_aorta_circle_miss_tolerated",
        stage="aorta_localization",
        config_overrides={
            "CIRCLE_DETECTION": {"out_of_tolerance_as_miss": True}
        },
    ),
    "normal_circle_score_miss_tolerated": pipeline_variant(
        "normal_aorta_circle_score_miss_tolerated",
        stage="aorta_localization",
        config_overrides={
            "CIRCLE_DETECTION": {
                "candidate_selection_strategy": "score",
                "out_of_tolerance_as_miss": True,
            }
        },
    ),
    # Variantes mantidas para a fase posterior de segmentacao arterial.
    "normal_rg_improved_baseline": pipeline_variant(
        "normal_region_growing_improved_baseline",
        stage="region_growing",
        region_growing_method="standard_improved",
        region_growing=rg_config_with(),
    ),
    "normal_rg_improved_min0090": pipeline_variant(
        "normal_region_growing_improved_min0090",
        stage="region_growing",
        region_growing_method="standard_improved",
        region_growing=rg_config_with(min_vesselness_fraction=0.09),
    ),
    "normal_rg_improved_threshold6": pipeline_variant(
        "normal_region_growing_improved_threshold6",
        stage="region_growing",
        region_growing_method="standard_improved",
        region_growing=rg_config_with(threshold_divisor=6),
    ),
    "normal_rg_improved_min0090_threshold6": pipeline_variant(
        "normal_region_growing_improved_min0090_threshold6",
        stage="region_growing",
        region_growing_method="standard_improved",
        region_growing=rg_config_with(
            min_vesselness_fraction=0.09,
            threshold_divisor=6,
        ),
    ),
    "normal_rg_improved_seed_radius3": pipeline_variant(
        "normal_region_growing_improved_seed_radius3",
        stage="region_growing",
        region_growing_method="standard_improved",
        region_growing=rg_config_with(seed_search_radius=3),
    ),
    "normal_rg_improved_multiseed8": pipeline_variant(
        "normal_region_growing_improved_multiseed8",
        stage="region_growing",
        region_growing_method="standard_improved",
        region_growing=rg_config_with(max_seed_candidates=8),
    ),
    "normal_standard_rg_min0070": pipeline_variant(
        "normal_standard_region_growing_min0070",
        stage="region_growing",
        region_growing={"min_vesselness_fraction": 0.07},
    ),
    "normal_standard_rg_min0090": pipeline_variant(
        "normal_standard_region_growing_min0090",
        stage="region_growing",
        region_growing={"min_vesselness_fraction": 0.09},
    ),
    "normal_standard_rg_threshold6": pipeline_variant(
        "normal_standard_region_growing_threshold6",
        stage="region_growing",
        region_growing={"threshold_divisor": 6},
    ),
    "normal_standard_rg_percentile90": pipeline_variant(
        "normal_standard_region_growing_positive_percentile90",
        stage="region_growing",
        region_growing={
            "min_vesselness_mode": "positive_percentile",
            "min_vesselness_percentile": 90.0,
        },
    ),
    "normal_vessel_artery_gamma65": pipeline_variant(
        "normal_artery_vesselness_gamma65",
        stage="artery_vesselness",
        config_overrides={"VESSELNESS_ARTERY": {"gamma": 65}},
    ),
    "normal_vessel_artery_smooth03": pipeline_variant(
        "normal_artery_vesselness_smooth03",
        stage="artery_vesselness",
        config_overrides={"VESSELNESS_ARTERY": {"smooth_sigma": 0.3}},
    ),
    "normal_vessel_artery_sigmas_15_35": pipeline_variant(
        "normal_artery_vesselness_sigmas_15_35",
        stage="artery_vesselness",
        config_overrides={"VESSELNESS_ARTERY": {"sigmas": [1.5, 2.0, 2.5, 3.0, 3.5]}},
    ),
    "normal_post_close2": pipeline_variant(
        "normal_postprocessing_closing2",
        stage="postprocessing",
        config_overrides={"POSTPROCESSING": {"closing_radius": 2}},
    ),
    "normal_post_dilate1": pipeline_variant(
        "normal_postprocessing_dilation1",
        stage="postprocessing",
        config_overrides={"POSTPROCESSING": {"dilation_radius": 1}},
    ),
}

PHASE_STAGE_GROUPS = {
    "ostia": {
        "baseline",
        "thresholding",
        "aorta_localization",
        "aorta_segmentation",
        "ostia_vesselness",
        "ostia_detection",
        "combined_ostia_aorta",
        "lcc_mode",
    },
    "segmentation": {
        "baseline",
        "artery_vesselness",
        "region_growing",
        "postprocessing",
    },
    "all": None,
}


def is_variant_active_for_phase(approach: str, spec: dict) -> bool:
    active_stages = PHASE_STAGE_GROUPS.get(OPTIMIZATION_PHASE)
    if active_stages is None:
        return True
    if approach == BASELINE_APPROACH:
        return True
    return spec.get("pipeline_stage") in active_stages


ACTIVE_PIPELINE_STAGES = PHASE_STAGE_GROUPS.get(OPTIMIZATION_PHASE)
THRESHOLD_APPROACHES = [
    approach
    for approach, spec in PIPELINE_STAGE_VARIANTS.items()
    if is_variant_active_for_phase(approach, spec)
]

train_ids, val_ids, test_ids, all_ids = get_data_splits(str(BASE_PATH))
TRAIN_SAMPLE_IDS = train_ids[:TRAIN_SAMPLE_SIZE]
VAL_SAMPLE_IDS = val_ids[:VAL_SAMPLE_SIZE]
SAMPLE_RECORDS = [
    {"img_id": img_id, "split": "train"} for img_id in TRAIN_SAMPLE_IDS
] + [
    {"img_id": img_id, "split": "val"} for img_id in VAL_SAMPLE_IDS
]
SAMPLE_IMAGE_IDS = [record["img_id"] for record in SAMPLE_RECORDS]
SAMPLE_SPLITS_BY_ID = {record["img_id"]: record["split"] for record in SAMPLE_RECORDS}

config_df = pd.DataFrame(
    [
        {
            "train_sample_size": TRAIN_SAMPLE_SIZE,
            "val_sample_size": VAL_SAMPLE_SIZE,
            "total_sample_size": len(SAMPLE_IMAGE_IDS),
            "train_sample_ids": TRAIN_SAMPLE_IDS,
            "val_sample_ids": VAL_SAMPLE_IDS,
            "sample_image_ids": SAMPLE_IMAGE_IDS,
            "optimization_phase": OPTIMIZATION_PHASE,
            "run_segmentation": RUN_SEGMENTATION,
            "active_pipeline_stages": ACTIVE_PIPELINE_STAGES,
            "threshold_approaches": THRESHOLD_APPROACHES,
            "pipeline_stage_variants": PIPELINE_STAGE_VARIANTS,
            "fixed_detection_overrides": FIXED_DETECTION_OVERRIDES,
            "fixed_max_threshold_percentile": FIXED_MAX_THRESHOLD_PERCENTILE,
            "baseline_approach": BASELINE_APPROACH,
            "shared_improved_region_growing_config": SHARED_IMPROVED_REGION_GROWING_CONFIG,
            "base_path": str(BASE_PATH),
            "downscale_factors": DOWNSCALE_FACTORS,
            "min_hu": MIN_HU,
            "normal_max_threshold_percentile": BASE_MAX_THRESHOLD_PERCENTILE,
            "load_cache": RUN_CONFIG["LOAD_CACHE"],
            "save_cache": RUN_CONFIG["SAVE_CACHE"],
        }
    ]
)
config_df

# %% [notebook cell 7]
def load_sample_image(img_id: int) -> dict:
    """Carrega a imagem, a label e os metadados reduzidos de um caso."""
    img_path = BASE_PATH / f"{img_id}.img.nii.gz"
    label_path = BASE_PATH / f"{img_id}.label.nii.gz"
    nii_img, nii_label = load_raw_img_and_label(str(img_path), str(label_path))
    image = nii_img.get_fdata(dtype=np.float32)
    label = nii_label.get_fdata(dtype=np.float32).astype(np.uint8)
    spacing = tuple(float(value) for value in nii_img.header.get_zooms()[:3])
    down_label = downscale_image_ndi(label, DOWNSCALE_FACTORS, order=0).astype(np.uint8)
    scaled_spacing = tuple(
        spacing[idx] * DOWNSCALE_FACTORS[idx] for idx in range(len(DOWNSCALE_FACTORS))
    )
    return {
        "img_id": img_id,
        "image": image,
        "down_label": down_label,
        "spacing": spacing,
        "scaled_spacing": scaled_spacing,
        "image_shape": image.shape,
        "label_shape": label.shape,
        "down_label_shape": down_label.shape,
    }


def estimate_contextual_fuzzy_centers(
    volume: np.ndarray,
    object_percentile: float = CONTEXTUAL_FUZZY_OBJECT_PERCENTILE,
    dense_percentile: float = CONTEXTUAL_FUZZY_DENSE_PERCENTILE,
) -> tuple[np.ndarray, dict]:
    """Estima centros HU para fundo mole, objeto e fundo denso via percentis."""
    values = np.asarray(volume, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Nao ha voxels finitos para estimar os centros fuzzy.")

    valid_values = values[values >= MIN_HU]
    if valid_values.size == 0:
        raise ValueError("Nao ha voxels acima de MIN_HU para estimar os centros fuzzy.")

    soft_center = float(MIN_HU - CONTEXTUAL_FUZZY_SOFT_MARGIN_HU)
    object_center = float(np.percentile(valid_values, object_percentile))
    dense_center = float(np.percentile(valid_values, dense_percentile))

    min_gap = np.finfo(np.float32).eps
    object_center = max(object_center, MIN_HU + min_gap)
    dense_center = max(dense_center, object_center + min_gap)
    details = {
        "percentile_object_center_hu": object_center,
        "percentile_dense_center_hu": dense_center,
    }
    return np.array([soft_center, object_center, dense_center], dtype=np.float32), details


def three_class_membership(volume: np.ndarray, centers: np.ndarray) -> np.ndarray:
    soft_center, object_center, dense_center = map(float, centers)
    soft_width = max(MIN_HU - soft_center, np.finfo(np.float32).eps)
    dense_width = max(dense_center - object_center, np.finfo(np.float32).eps)

    soft = np.clip((MIN_HU - volume) / soft_width, 0.0, 1.0)
    dense = np.clip((volume - object_center) / dense_width, 0.0, 1.0)
    object_membership = np.minimum(1.0 - soft, 1.0 - dense)

    memberships = np.stack([soft, object_membership, dense], axis=0).astype(np.float32)
    membership_sum = memberships.sum(axis=0, keepdims=True)
    return memberships / np.maximum(membership_sum, np.finfo(np.float32).eps)


def aggregate_memberships(
    memberships: np.ndarray,
    radius: int,
    mode: str,
) -> np.ndarray:
    if radius <= 0:
        return memberships
    size = 2 * radius + 1
    aggregated = np.empty_like(memberships, dtype=np.float32)
    for class_idx in range(memberships.shape[0]):
        if mode == "median":
            aggregated[class_idx] = median_filter(memberships[class_idx], size=size)
        else:
            aggregated[class_idx] = uniform_filter(memberships[class_idx], size=size)
    membership_sum = aggregated.sum(axis=0, keepdims=True)
    return aggregated / np.maximum(membership_sum, np.finfo(np.float32).eps)


def contextual_fuzzy_3class_outputs(
    volume: np.ndarray,
    weight_floor: float = 0.25,
    dense_power: float = 1.0,
    weight_mode: str = "object_dense",
    object_percentile: float = CONTEXTUAL_FUZZY_OBJECT_PERCENTILE,
    dense_percentile: float = CONTEXTUAL_FUZZY_DENSE_PERCENTILE,
    smooth_radius: int | None = None,
    smooth_mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    centers, center_details = estimate_contextual_fuzzy_centers(
        volume,
        object_percentile=object_percentile,
        dense_percentile=dense_percentile,
    )
    memberships = three_class_membership(volume, centers)
    smooth_radius = CONTEXTUAL_FUZZY_SMOOTH_RADIUS if smooth_radius is None else int(smooth_radius)
    smooth_mode = CONTEXTUAL_FUZZY_SMOOTH_MODE if smooth_mode is None else smooth_mode
    aggregated = aggregate_memberships(
        memberships,
        radius=smooth_radius,
        mode=smooth_mode,
    )
    object_membership = aggregated[1]
    dense_membership = aggregated[2]
    labels = np.argmax(aggregated, axis=0)
    object_mask = (labels == 1) & (volume >= MIN_HU)
    dense_mask = labels == 2

    if weight_mode == "dense_only":
        raw_weight_map = 1.0 - np.power(dense_membership, dense_power)
    else:
        raw_weight_map = object_membership * np.power(1.0 - dense_membership, dense_power)

    weight_map = weight_floor + (1.0 - weight_floor) * raw_weight_map
    weight_map = np.clip(weight_map, weight_floor, 1.0).astype(np.float32)
    details = {
        "soft_center_hu": float(centers[0]),
        "object_center_hu": float(centers[1]),
        "dense_center_hu": float(centers[2]),
        "dense_voxels": int(dense_mask.sum()),
        "mean_weight": float(weight_map.mean()),
        "min_weight": float(weight_map.min()),
        "max_weight": float(weight_map.max()),
        "weight_floor": float(weight_floor),
        "dense_power": float(dense_power),
        "weight_mode": weight_mode,
        "object_percentile": float(object_percentile),
        "dense_percentile": float(dense_percentile),
        "smooth_radius": smooth_radius,
        "smooth_mode": smooth_mode,
        **center_details,
    }
    return object_mask, weight_map, details


def build_artery_vesselness_config(
    config: dict,
    overrides: dict | None = None,
) -> dict:
    vesselness_config = copy.deepcopy(config["VESSELNESS_ARTERY"])
    overrides = copy.deepcopy(overrides or {})
    modified_overrides = overrides.pop("modified", None)
    vesselness_config.update(overrides)
    if modified_overrides:
        vesselness_config.setdefault("modified", {})
        vesselness_config["modified"].update(modified_overrides)
    return vesselness_config


def apply_vesselness_weight_map(
    vesselness: np.ndarray,
    weight_map: np.ndarray | None,
) -> np.ndarray:
    if weight_map is None:
        return vesselness
    if vesselness.shape != weight_map.shape:
        raise ValueError(
            f"Shape diferente entre vesselness e weight_map: {vesselness.shape} vs {weight_map.shape}"
        )
    return (vesselness * weight_map).astype(vesselness.dtype, copy=False)




def collect_seed_candidates_by_score(
    seed: tuple[int, int, int] | None,
    score_map: np.ndarray,
    search_radius: int,
    min_seed_score: float,
    max_candidates: int = 1,
    min_candidate_distance_voxels: float = 0.0,
) -> list[tuple[int, int, int]]:
    if seed is None:
        return []
    y, x, z = map(int, seed)
    if not (0 <= y < score_map.shape[0] and 0 <= x < score_map.shape[1] and 0 <= z < score_map.shape[2]):
        return []

    radius = max(int(search_radius), 0)
    y0, y1 = max(0, y - radius), min(score_map.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(score_map.shape[1], x + radius + 1)
    z0, z1 = max(0, z - radius), min(score_map.shape[2], z + radius + 1)
    local_scores = score_map[y0:y1, x0:x1, z0:z1]
    if local_scores.size == 0:
        return [(y, x, z)] if float(score_map[y, x, z]) >= min_seed_score else []

    candidate_mask = local_scores >= min_seed_score
    if not np.any(candidate_mask):
        return [(y, x, z)] if float(score_map[y, x, z]) >= min_seed_score else []

    candidate_indices = np.argwhere(candidate_mask)
    candidate_scores = local_scores[candidate_mask]
    order = np.argsort(candidate_scores)[::-1]
    selected = []
    min_distance = float(min_candidate_distance_voxels)
    for idx in order:
        ly, lx, lz = candidate_indices[idx]
        candidate = (int(y0 + ly), int(x0 + lx), int(z0 + lz))
        if min_distance > 0 and selected:
            candidate_arr = np.asarray(candidate, dtype=np.float32)
            selected_arr = np.asarray(selected, dtype=np.float32)
            distances = np.sqrt(np.sum((selected_arr - candidate_arr) ** 2, axis=1))
            if np.any(distances < min_distance):
                continue
        selected.append(candidate)
        if len(selected) >= max(int(max_candidates), 1):
            break
    return selected


def is_within_seed_distance(
    coord: tuple[int, int, int],
    seed_coords: np.ndarray,
    spacing: tuple[float, float, float] | None,
    max_distance_mm: float | None,
    max_distance_voxels: float | None,
) -> bool:
    if seed_coords.size == 0:
        return True
    coord_arr = np.asarray(coord, dtype=np.float32)
    deltas = seed_coords - coord_arr
    if max_distance_mm is not None and spacing is not None:
        spacing_arr = np.asarray(spacing, dtype=np.float32)
        distances = np.sqrt(np.sum((deltas * spacing_arr) ** 2, axis=1))
        return bool(np.min(distances) <= float(max_distance_mm))
    if max_distance_voxels is not None:
        distances = np.sqrt(np.sum(deltas ** 2, axis=1))
        return bool(np.min(distances) <= float(max_distance_voxels))
    return True


def contextual_fuzzy_kwargs(config: dict) -> dict:
    return {
        "weight_floor": config.get("weight_floor", 0.25),
        "dense_power": config.get("dense_power", 1.0),
        "weight_mode": config.get("weight_mode", "object_dense"),
        "object_percentile": config.get("object_percentile", CONTEXTUAL_FUZZY_OBJECT_PERCENTILE),
        "dense_percentile": config.get("dense_percentile", CONTEXTUAL_FUZZY_DENSE_PERCENTILE),
        "smooth_radius": config.get("smooth_radius"),
        "smooth_mode": config.get("smooth_mode"),
    }



def approach_config_for(approach: str) -> dict:
    spec = PIPELINE_STAGE_VARIANTS[approach]
    config = deep_update_config(RUN_CONFIG, FIXED_DETECTION_OVERRIDES)
    return deep_update_config(config, spec.get("config_overrides", {}))


def approach_threshold_settings(spec: dict) -> tuple[float, float]:
    min_hu = float(spec.get("min_hu", MIN_HU))
    max_percentile = float(
        spec.get("max_threshold_percentile")
        if spec.get("max_threshold_percentile") is not None
        else BASE_MAX_THRESHOLD_PERCENTILE
    )
    return min_hu, max_percentile


def has_detection_specific_changes(
    spec: dict,
    min_hu: float,
    max_percentile: float,
    lcc_per_slice: bool,
) -> bool:
    overrides = spec.get("config_overrides", {})
    has_detection_override = any(key in overrides for key in DETECTION_CONFIG_KEYS)
    threshold_changed = (min_hu != float(MIN_HU)) or (
        max_percentile != float(BASE_MAX_THRESHOLD_PERCENTILE)
    )
    lcc_changed = bool(lcc_per_slice) is not True
    return has_detection_override or threshold_changed or lcc_changed


def build_threshold_inputs(img_id: int, image: np.ndarray) -> tuple[dict, dict, pd.DataFrame]:
    """Gera a entrada threshold/LCC para cada configuracao normal ativa."""
    down_image = downscale_image_ndi(image, DOWNSCALE_FACTORS, order=3).astype(np.float32)

    approach_inputs = {}
    approach_metadata = {}
    records = []
    threshold_cache = {}

    for approach in THRESHOLD_APPROACHES:
        spec = PIPELINE_STAGE_VARIANTS[approach]
        approach_config = approach_config_for(approach)
        min_hu, max_percentile = approach_threshold_settings(spec)
        lcc_per_slice = bool(spec.get("lcc_per_slice", True))
        lcc_mode = "per_slice" if lcc_per_slice else "per_volume"
        threshold_key = (min_hu, max_percentile, lcc_per_slice)

        if threshold_key not in threshold_cache:
            max_hu = float(np.percentile(down_image, max_percentile))
            _, threshold_mask, _ = threshold_image_with_offset(
                down_image,
                min_val=int(min_hu),
                max_val=int(max_hu),
            )
            lcc_image, lcc_mask = build_lcc_image_from_mask(
                down_image,
                threshold_mask,
                offset=abs(int(min_hu)),
                per_slice=lcc_per_slice,
            )
            threshold_cache[threshold_key] = {
                "lcc_image": lcc_image,
                "max_hu": max_hu,
                "threshold_voxels": int(threshold_mask.sum()),
                "lcc_voxels": int(lcc_mask.sum()),
            }

        threshold_record = threshold_cache[threshold_key]
        rg_overrides = copy.deepcopy(spec.get("region_growing", {}))
        region_growing_method = spec.get("region_growing_method", "standard")
        artery_config = build_artery_vesselness_config(approach_config, {})
        detection_changed = has_detection_specific_changes(
            spec,
            min_hu,
            max_percentile,
            lcc_per_slice,
        )
        reuse_detection_from = None
        if approach != BASELINE_APPROACH and not detection_changed:
            reuse_detection_from = BASELINE_APPROACH

        approach_inputs[approach] = threshold_record["lcc_image"]
        approach_metadata[approach] = {
            "config": approach_config,
            "ostios": None,
            "artery": None,
            "region_growing": rg_overrides,
            "region_growing_method": region_growing_method,
            "reuse_detection_from": reuse_detection_from,
            "vesselness_artery_overrides": {},
        }
        records.append(
            {
                "img_id": img_id,
                "sample_split": SAMPLE_SPLITS_BY_ID.get(img_id),
                "approach": approach,
                "threshold_mode": spec["label"],
                "approach_family": spec.get("approach_family", "normal_pipeline_sweep"),
                "pipeline_stage": spec.get("pipeline_stage"),
                "config_overrides": spec.get("config_overrides", {}),
                "fuzzy_output_mode": spec.get("fuzzy_output_mode", "none"),
                "apply_weight_to": spec.get("apply_weight_to", "none"),
                "region_growing_method": region_growing_method,
                "rg_ablation_step": spec.get("pipeline_stage") if spec.get("pipeline_stage") == "region_growing" else None,
                "vesselness_ablation_step": spec.get("pipeline_stage") if "vesselness" in str(spec.get("pipeline_stage")) else None,
                "reuse_detection_from": reuse_detection_from,
                "base_weighted_approach": None,
                "min_hu": min_hu,
                "max_hu": threshold_record["max_hu"],
                "max_threshold_percentile": max_percentile,
                "lcc_per_slice": lcc_per_slice,
                "lcc_mode": lcc_mode,
                "threshold_voxels": threshold_record["threshold_voxels"],
                "lcc_voxels": threshold_record["lcc_voxels"],
                "artery_vesselness_method": artery_config.get("method", "normal"),
                "artery_sigmas": artery_config.get("sigmas"),
                "artery_alpha": artery_config.get("alpha"),
                "artery_beta": artery_config.get("beta"),
                "artery_gamma": artery_config.get("gamma"),
                "artery_normalization": artery_config.get("normalization"),
                "artery_smooth_sigma": artery_config.get("smooth_sigma", 0.0),
                "artery_vesselness_label": spec["label"],
                "aorta_vesselness_method": approach_config["VESSELNESS_AORTA"].get("method", "normal"),
                "aorta_sigmas": approach_config["VESSELNESS_AORTA"].get("sigmas"),
                "aorta_alpha": approach_config["VESSELNESS_AORTA"].get("alpha"),
                "aorta_beta": approach_config["VESSELNESS_AORTA"].get("beta"),
                "aorta_gamma": approach_config["VESSELNESS_AORTA"].get("gamma"),
                "aorta_smooth_sigma": approach_config["VESSELNESS_AORTA"].get("smooth_sigma", 0.0),
                "circle_canny_sigma": approach_config["CIRCLE_DETECTION"].get("canny_sigma"),
                "circle_total_num_peaks_initial": approach_config["CIRCLE_DETECTION"].get("total_num_peaks_initial"),
                "circle_total_num_peaks": approach_config["CIRCLE_DETECTION"].get("total_num_peaks"),
                "circle_local_roi_padding": approach_config["CIRCLE_DETECTION"].get("local_roi_padding"),
                "circle_candidate_selection_strategy": approach_config["CIRCLE_DETECTION"].get("candidate_selection_strategy", "closest"),
                "circle_out_of_tolerance_as_miss": approach_config["CIRCLE_DETECTION"].get("out_of_tolerance_as_miss", False),
                "circle_candidate_score_accum_weight": approach_config["CIRCLE_DETECTION"].get("candidate_score_accum_weight", 1.0),
                "circle_candidate_score_distance_weight": approach_config["CIRCLE_DETECTION"].get("candidate_score_distance_weight", 1.0),
                "circle_candidate_score_radius_weight": approach_config["CIRCLE_DETECTION"].get("candidate_score_radius_weight", 1.0),
                "level_set_num_iter": approach_config["LEVEL_SET"].get("num_iter"),
                "level_set_balloon": approach_config["LEVEL_SET"].get("balloon"),
                "level_set_smoothing": approach_config["LEVEL_SET"].get("smoothing"),
                "ostia_top_n": approach_config["OSTIA_DETECTION"].get("top_n"),
                "ostia_lower_fraction": approach_config["OSTIA_DETECTION"].get("lower_fraction"),
                "ostia_min_center_distance_factor": approach_config["OSTIA_DETECTION"].get("min_center_distance_factor"),
                "ostia_min_lateral_factor": approach_config["OSTIA_DETECTION"].get("min_lateral_factor"),
                "ostia_erosion_radius": approach_config["OSTIA_DETECTION"].get("erosion_radius"),
                "post_closing_radius": approach_config["POSTPROCESSING"].get("closing_radius"),
                "post_dilation_radius": approach_config["POSTPROCESSING"].get("dilation_radius"),
                "use_seed_refinement": rg_overrides.get("use_seed_refinement", False),
                "use_multi_seed": rg_overrides.get("use_multi_seed", False),
                "use_priority_queue": rg_overrides.get("use_priority_queue", False),
                "use_adaptive_acceptance": rg_overrides.get("use_adaptive_acceptance", False),
                "use_distance_limit": rg_overrides.get("use_distance_limit", False),
                "grow_each_ostium_separately": rg_overrides.get("grow_each_ostium_separately", False),
                "seed_search_radius": rg_overrides.get("seed_search_radius"),
                "seed_candidate_radius": rg_overrides.get("seed_candidate_radius"),
                "max_seed_candidates": rg_overrides.get("max_seed_candidates"),
                "seed_min_vesselness_fraction": rg_overrides.get("seed_min_vesselness_fraction"),
                "min_seed_candidate_distance_voxels": rg_overrides.get("min_seed_candidate_distance_voxels"),
                "adaptive_threshold_std_factor": rg_overrides.get("adaptive_threshold_std_factor"),
                "min_threshold_fraction": rg_overrides.get("min_threshold_fraction"),
                "max_distance_mm": rg_overrides.get("max_distance_mm"),
                "max_distance_voxels": rg_overrides.get("max_distance_voxels"),
                "rg_comparison_window": rg_overrides.get("comparison_window", approach_config["REGION_GROWING"]["comparison_window"]),
                "rg_threshold_divisor": rg_overrides.get("threshold_divisor", approach_config["REGION_GROWING"]["threshold_divisor"]),
                "rg_threshold_mode": rg_overrides.get("threshold_mode", "range_fraction"),
                "rg_threshold_percentiles": rg_overrides.get("threshold_percentiles"),
                "rg_threshold_std_factor": rg_overrides.get("threshold_std_factor"),
                "rg_threshold_scale": rg_overrides.get("threshold_scale", 1.0),
                "rg_min_vesselness_fraction": rg_overrides.get("min_vesselness_fraction", approach_config["REGION_GROWING"]["min_vesselness_fraction"]),
                "rg_min_vesselness_mode": rg_overrides.get("min_vesselness_mode", "max_fraction"),
                "rg_min_vesselness_percentile": rg_overrides.get("min_vesselness_percentile"),
                "rg_min_vesselness_std_factor": rg_overrides.get("min_vesselness_std_factor"),
                "rg_min_vesselness_mad_factor": rg_overrides.get("min_vesselness_mad_factor"),
                "rg_min_vesselness_scale": rg_overrides.get("min_vesselness_scale", 1.0),
            }
        )

    return approach_inputs, approach_metadata, pd.DataFrame(records)

def run_detection_stage(
    img_id: int,
    approach: str,
    lcc_image: np.ndarray,
    label: np.ndarray,
    scaled_spacing: tuple[float, float, float],
    config: dict,
    vesselness_weight_map: np.ndarray | None = None,
) -> dict:
    """Executa aorta + ostios sem criar cache."""
    stage_root = RUN_OUTPUT_DIR / "cache" / str(img_id)
    result = {
        "img_id": img_id,
        "approach": approach,
        "ostia_found": False,
        "ostia_status": "not_evaluated",
        "both_correct": False,
        "both_tolerable": False,
        "left_dist_mm": np.inf,
        "right_dist_mm": np.inf,
        "ostia_left": None,
        "ostia_right": None,
        "num_circles": 0,
        "aorta_voxels": 0,
        "ostia_error": None,
    }
    vesselness_ostios = get_or_compute_vesselness(
        str(img_id),
        lcc_image,
        cache_dir=str(stage_root / approach / "vesselness_ostios_cache"),
        vesselness_config=config["VESSELNESS_AORTA"],
        load_cache=config["LOAD_CACHE"],
        save_cache=config["SAVE_CACHE"],
        use_gpu=config.get("USE_GPU", False),
    )
    vesselness_ostios = apply_vesselness_weight_map(
        vesselness_ostios,
        vesselness_weight_map,
    )
    detected_circles = get_or_detect_aorta_circles(
        str(img_id),
        lcc_image,
        DOWNSCALE_FACTORS,
        scaled_spacing,
        config["CIRCLE_DETECTION"],
        stage_root / approach,
        load_cache=config["LOAD_CACHE"],
        save_cache=config["SAVE_CACHE"],
    )
    result["num_circles"] = len(detected_circles)
    aorta_mask = get_or_segment_aorta(
        str(img_id),
        lcc_image,
        detected_circles,
        config["LEVEL_SET"],
        stage_root / approach,
        load_cache=config["LOAD_CACHE"],
        save_cache=config["SAVE_CACHE"],
        use_gpu=config.get("USE_GPU", False),
    )
    result["aorta_voxels"] = int(aorta_mask.sum())
    try:
        ostia_eval = detect_and_evaluate_ostia(
            aorta_mask,
            vesselness_ostios,
            label,
            scaled_spacing,
            config,
        )
    except ValueError as exc:
        result["ostia_status"] = "not_found"
        result["ostia_error"] = str(exc)
        return result

    missing_ostia = [
        side
        for side, ostium in (
            ("left", ostia_eval.get("ostia_left")),
            ("right", ostia_eval.get("ostia_right")),
        )
        if ostium is None
    ]
    if missing_ostia:
        result["ostia_status"] = "not_found"
        result["ostia_error"] = "Ostio(s) nao encontrado(s): " + ", ".join(missing_ostia)
        return result

    if ostia_eval["both_correct"]:
        ostia_status = "both_correct"
    elif ostia_eval["both_tolerable"]:
        ostia_status = "both_tolerable"
    else:
        ostia_status = "found_but_wrong"

    result.update(
        {
            "ostia_found": True,
            "ostia_status": ostia_status,
            "both_correct": bool(ostia_eval["both_correct"]),
            "both_tolerable": bool(ostia_eval["both_tolerable"]),
            "left_dist_mm": ostia_eval["left_info"]["physical_dist"],
            "right_dist_mm": ostia_eval["right_info"]["physical_dist"],
            "ostia_left": tuple(map(int, ostia_eval["ostia_left"])),
            "ostia_right": tuple(map(int, ostia_eval["ostia_right"])),
            "label_artery": ostia_eval["label_artery"],
        }
    )
    return result


def postprocess_artery_mask(mask: np.ndarray, config: dict) -> np.ndarray:
    """Aplica apenas o pos-processamento morfologico usado no pipeline."""
    post_config = config["POSTPROCESSING"]
    closed_mask = binary_closing(
        mask > 0,
        structure=ball(post_config["closing_radius"]),
        gpu=config.get("USE_GPU", False),
    )
    return binary_dilation(
        closed_mask,
        structure=ball(post_config["dilation_radius"]),
        gpu=config.get("USE_GPU", False),
    )


def finite_vesselness_values(vesselness_artery: np.ndarray) -> np.ndarray:
    values = np.asarray(vesselness_artery, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([0.0], dtype=np.float32)
    positive_values = values[values > 0]
    return positive_values if positive_values.size else values


def robust_mad(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return mad if mad > 0 else float(np.std(values))


def compute_region_growing_threshold(
    values: np.ndarray,
    vesselness_artery: np.ndarray,
    rg_config: dict,
    overrides: dict,
) -> float:
    threshold_divisor = float(overrides.get("threshold_divisor", rg_config["threshold_divisor"]))
    threshold_mode = overrides.get("threshold_mode", "range_fraction")
    if threshold_mode == "percentile_range_fraction":
        low_percentile, high_percentile = overrides.get("threshold_percentiles", (50.0, 99.5))
        low_value, high_value = np.percentile(values, [low_percentile, high_percentile])
        threshold = max(float(high_value - low_value), 0.0) / threshold_divisor
    elif threshold_mode == "std_fraction":
        threshold = float(np.std(values)) * float(overrides.get("threshold_std_factor", 0.35))
    else:
        threshold = (float(np.max(vesselness_artery)) - float(np.min(vesselness_artery))) / threshold_divisor
    threshold *= float(overrides.get("threshold_scale", 1.0))
    return max(threshold, float(np.finfo(np.float32).eps))


def compute_region_growing_min_vesselness(
    values: np.ndarray,
    vesselness_artery: np.ndarray,
    rg_config: dict,
    overrides: dict,
) -> float:
    mode = overrides.get("min_vesselness_mode", "max_fraction")
    if mode == "positive_percentile":
        min_vesselness = float(
            np.percentile(values, float(overrides.get("min_vesselness_percentile", 85.0)))
        )
    elif mode == "mean_std":
        min_vesselness = float(np.mean(values)) + float(
            overrides.get("min_vesselness_std_factor", 0.5)
        ) * float(np.std(values))
    elif mode == "median_mad":
        min_vesselness = float(np.median(values)) + float(
            overrides.get("min_vesselness_mad_factor", 3.0)
        ) * robust_mad(values)
    else:
        min_vesselness_fraction = overrides.get(
            "min_vesselness_fraction", rg_config["min_vesselness_fraction"]
        )
        min_vesselness = float(np.max(vesselness_artery)) * float(min_vesselness_fraction)
    min_vesselness *= float(overrides.get("min_vesselness_scale", 1.0))
    max_vesselness = float(np.max(vesselness_artery))
    return min(max(min_vesselness, 0.0), max_vesselness)


def build_region_growing_params(
    vesselness_artery: np.ndarray,
    config: dict,
    overrides: dict | None = None,
) -> dict:
    rg_config = config["REGION_GROWING"]
    overrides = overrides or {}
    values = finite_vesselness_values(vesselness_artery)
    threshold = compute_region_growing_threshold(values, vesselness_artery, rg_config, overrides)
    min_vesselness = compute_region_growing_min_vesselness(
        values,
        vesselness_artery,
        rg_config,
        overrides,
    )
    return {
        "threshold": threshold,
        "max_volume": overrides.get("max_volume", rg_config["max_volume"]),
        "min_vesselness": min_vesselness,
        "relaxed_floor_factor": overrides.get(
            "relaxed_floor_factor", rg_config["relaxed_floor_factor"]
        ),
        "switch_at_voxels": overrides.get("switch_at_voxels", rg_config["switch_at_voxels"]),
        "comparison_window": overrides.get("comparison_window", rg_config["comparison_window"]),
        "smooth_relaxation": overrides.get("smooth_relaxation", rg_config["smooth_relaxation"]),
        "verbose": False,
    }



def standard_region_growing_from_seeds(
    vesselness_artery: np.ndarray,
    seeds,
    config: dict,
    params_reference_vesselness: np.ndarray | None = None,
    region_growing_overrides: dict | None = None,
) -> np.ndarray:
    """Executa o region growing original para cada ostio e junta as mascaras."""
    overrides = region_growing_overrides or {}
    params_source = params_reference_vesselness if params_reference_vesselness is not None else vesselness_artery
    params = build_region_growing_params(params_source, config, overrides=overrides)
    combined_mask = np.zeros_like(vesselness_artery, dtype=np.uint8)
    for seed in seeds:
        if seed is None:
            continue
        seed_mask = region_growing_segmentation(
            vesselness_artery,
            seed,
            threshold=params["threshold"],
            min_vesselness=params["min_vesselness"],
            max_volume=params["max_volume"],
            relaxed_floor_factor=params["relaxed_floor_factor"],
            switch_at_voxels=params["switch_at_voxels"],
            comparison_window=params["comparison_window"],
            smooth_relaxation=params["smooth_relaxation"],
            verbose=False,
        )
        combined_mask |= seed_mask.astype(np.uint8)
    return combined_mask.astype(np.uint8)


def standard_region_growing_improved_from_seeds(
    vesselness_artery: np.ndarray,
    seeds,
    config: dict,
    params_reference_vesselness: np.ndarray | None = None,
    region_growing_overrides: dict | None = None,
    spacing: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Region growing experimental com melhorias ligadas/desligadas por parametro."""
    if vesselness_artery.ndim != 3:
        raise ValueError(f"vesselness_artery deve ser 3D, recebido shape={vesselness_artery.shape}")

    overrides = region_growing_overrides or {}
    params_source = params_reference_vesselness if params_reference_vesselness is not None else vesselness_artery
    params = build_region_growing_params(params_source, config, overrides=overrides)
    max_vesselness = float(np.max(vesselness_artery))
    if max_vesselness <= 0:
        return np.zeros_like(vesselness_artery, dtype=np.uint8)

    use_seed_refinement = bool(overrides.get("use_seed_refinement", True))
    use_multi_seed = bool(overrides.get("use_multi_seed", False))
    use_priority_queue = bool(overrides.get("use_priority_queue", False))
    use_adaptive_acceptance = bool(overrides.get("use_adaptive_acceptance", False))
    use_distance_limit = bool(overrides.get("use_distance_limit", False))
    grow_each_ostium_separately = bool(overrides.get("grow_each_ostium_separately", False))

    seed_radius = int(overrides.get("seed_search_radius", 2))
    candidate_radius = int(overrides.get("seed_candidate_radius", seed_radius))
    max_seed_candidates = int(overrides.get("max_seed_candidates", 1 if not use_multi_seed else 6))
    if not use_multi_seed:
        max_seed_candidates = 1
    seed_min_fraction = float(
        overrides.get(
            "seed_min_vesselness_fraction",
            config["REGION_GROWING"]["min_vesselness_fraction"] * 0.5,
        )
    )
    seed_min_vesselness = max_vesselness * seed_min_fraction
    min_candidate_distance = float(overrides.get("min_seed_candidate_distance_voxels", 0.0))

    def collect_for_seed(seed) -> list[tuple[int, int, int]]:
        if seed is None:
            return []
        if use_seed_refinement or use_multi_seed:
            return collect_seed_candidates_by_score(
                seed,
                vesselness_artery,
                candidate_radius if use_multi_seed else seed_radius,
                seed_min_vesselness,
                max_candidates=max_seed_candidates,
                min_candidate_distance_voxels=min_candidate_distance,
            )
        y, x, z = map(int, seed)
        if 0 <= y < vesselness_artery.shape[0] and 0 <= x < vesselness_artery.shape[1] and 0 <= z < vesselness_artery.shape[2]:
            return [(y, x, z)]
        return []

    seed_groups = [collect_for_seed(seed) for seed in seeds]
    seed_groups = [list(dict.fromkeys(group)) for group in seed_groups if group]
    if not seed_groups:
        return np.zeros_like(vesselness_artery, dtype=np.uint8)

    min_start = min(float(params["min_vesselness"]), max_vesselness)
    min_end = min_start * float(params["relaxed_floor_factor"])
    base_threshold = float(params["threshold"])
    min_threshold = max_vesselness * float(overrides.get("min_threshold_fraction", 0.0))
    adaptive_std_factor = float(overrides.get("adaptive_threshold_std_factor", 0.0))
    max_volume = int(params["max_volume"])
    switch_at_voxels = int(params["switch_at_voxels"])
    max_distance_mm = overrides.get("max_distance_mm") if use_distance_limit else None
    max_distance_voxels = overrides.get("max_distance_voxels") if use_distance_limit else None
    dims = vesselness_artery.shape
    neighbors = tuple(
        (dy, dx, dz)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dy, dx, dz) != (0, 0, 0)
    )

    def grow_from_group(refined_seeds: list[tuple[int, int, int]]) -> np.ndarray:
        seed_coords = np.asarray(refined_seeds, dtype=np.float32)
        mask = np.zeros_like(vesselness_artery, dtype=np.uint8)
        visited = np.zeros_like(vesselness_artery, dtype=bool)
        queue = []
        queue_head = 0
        running_sum = 0.0
        running_sq_sum = 0.0
        count = 0

        def push(coord: tuple[int, int, int], value: float) -> None:
            if use_priority_queue:
                heapq.heappush(queue, (-value, coord))
            else:
                queue.append(coord)

        def pop():
            nonlocal queue_head
            if use_priority_queue:
                neg_value, coord = heapq.heappop(queue)
                return coord, -float(neg_value)
            coord = queue[queue_head]
            queue_head += 1
            return coord, float(vesselness_artery[coord])

        def has_items() -> bool:
            return bool(queue) if use_priority_queue else queue_head < len(queue)

        for seed in refined_seeds:
            if visited[seed]:
                continue
            seed_value = float(vesselness_artery[seed])
            if seed_value < seed_min_vesselness:
                continue
            visited[seed] = True
            mask[seed] = 1
            running_sum += seed_value
            running_sq_sum += seed_value * seed_value
            count += 1
            push(seed, seed_value)

        if count == 0:
            return np.zeros_like(vesselness_artery, dtype=np.uint8)

        while has_items() and count < max_volume:
            current, current_value = pop()
            progress = min(count / max(switch_at_voxels, 1), 1.0)
            current_floor = min_start + (min_end - min_start) * progress
            if use_priority_queue and current_value < current_floor:
                break

            region_mean = running_sum / max(count, 1)
            variance = max((running_sq_sum / max(count, 1)) - region_mean**2, 0.0)
            adaptive_threshold = max(base_threshold, adaptive_std_factor * float(np.sqrt(variance)), min_threshold)

            cy, cx, cz = current
            for dy, dx, dz in neighbors:
                ny, nx, nz = cy + dy, cx + dx, cz + dz
                if not (0 <= ny < dims[0] and 0 <= nx < dims[1] and 0 <= nz < dims[2]):
                    continue
                candidate = (ny, nx, nz)
                if visited[candidate]:
                    continue
                if use_distance_limit and not is_within_seed_distance(
                    candidate,
                    seed_coords,
                    spacing,
                    max_distance_mm,
                    max_distance_voxels,
                ):
                    continue

                candidate_value = float(vesselness_artery[candidate])
                if candidate_value < current_floor:
                    continue
                if use_adaptive_acceptance:
                    if candidate_value < (region_mean - adaptive_threshold):
                        continue
                elif abs(candidate_value - current_value) > base_threshold:
                    continue

                visited[candidate] = True
                mask[candidate] = 1
                running_sum += candidate_value
                running_sq_sum += candidate_value * candidate_value
                count += 1
                push(candidate, candidate_value)
                if count >= max_volume:
                    break

        return mask.astype(np.uint8)

    if grow_each_ostium_separately:
        combined = np.zeros_like(vesselness_artery, dtype=np.uint8)
        for seed_group in seed_groups:
            combined |= grow_from_group(seed_group)
        return combined.astype(np.uint8)

    all_refined_seeds = list(dict.fromkeys(seed for group in seed_groups for seed in group))
    return grow_from_group(all_refined_seeds)



def run_segmentation_stage(
    img_id: int,
    detection_result: dict,
    lcc_image: np.ndarray,
    config: dict,
    vesselness_weight_map: np.ndarray | None = None,
    region_growing_overrides: dict | None = None,
    vesselness_artery_overrides: dict | None = None,
    scaled_spacing: tuple[float, float, float] | None = None,
    region_growing_method: str = "standard",
) -> dict:
    """Executa region growing normal ou modificado, com e sem morfologia."""
    result = {
        "img_id": img_id,
        "sample_split": SAMPLE_SPLITS_BY_ID.get(img_id),
        "approach": detection_result["approach"],
        "segmentation_attempted": False,
        "proceeded_with_bad_ostia": False,
        "dice_artery": np.nan,
        "artery_voxels": 0,
        "segmentation_error": detection_result["ostia_error"],
        "case_rows": [],
    }
    if not detection_result["ostia_found"]:
        return result

    result["segmentation_attempted"] = True
    result["proceeded_with_bad_ostia"] = not (
        detection_result["both_correct"] or detection_result["both_tolerable"]
    )
    stage_root = RUN_OUTPUT_DIR / "cache" / str(img_id)
    vesselness_artery_config = build_artery_vesselness_config(
        config,
        vesselness_artery_overrides,
    )
    vesselness_artery = get_or_compute_vesselness(
        str(img_id),
        lcc_image,
        cache_dir=str(stage_root / detection_result["approach"] / "vesselness_artery_cache"),
        vesselness_config=vesselness_artery_config,
        load_cache=config["LOAD_CACHE"],
        save_cache=config["SAVE_CACHE"],
        use_gpu=config.get("USE_GPU", False),
        spacing=scaled_spacing,
    )
    original_vesselness_artery = vesselness_artery
    vesselness_artery = apply_vesselness_weight_map(
        vesselness_artery,
        vesselness_weight_map,
    )
    seeds = [detection_result["ostia_left"], detection_result["ostia_right"]]
    if region_growing_method == "standard_improved":
        raw_mask = standard_region_growing_improved_from_seeds(
            vesselness_artery,
            seeds,
            config,
            params_reference_vesselness=original_vesselness_artery,
            region_growing_overrides=region_growing_overrides,
            spacing=scaled_spacing,
        )
        segmentation_case_prefix = "standard_region_growing_improved_pipeline"
    else:
        raw_mask = standard_region_growing_from_seeds(
            vesselness_artery,
            seeds,
            config,
            params_reference_vesselness=original_vesselness_artery,
            region_growing_overrides=region_growing_overrides,
        )
        segmentation_case_prefix = "standard_region_growing_pipeline"

    masks = {
        "without_morphology": raw_mask,
        "with_morphology": postprocess_artery_mask(raw_mask, config),
    }
    for suffix, mask in masks.items():
        result["case_rows"].append(
            {
                "img_id": img_id,
                "sample_split": SAMPLE_SPLITS_BY_ID.get(img_id),
                "approach": detection_result["approach"],
                "segmentation_case": f"{segmentation_case_prefix}_{suffix}",
                "region_growing_method": region_growing_method,
                "uses_morphology": suffix == "with_morphology",
                "dice_artery": float(dice_score(mask, detection_result["label_artery"])),
                "artery_voxels": int(mask.sum()),
                "segmentation_attempted": result["segmentation_attempted"],
                "proceeded_with_bad_ostia": result["proceeded_with_bad_ostia"],
                "segmentation_error": result["segmentation_error"],
            }
        )

    final_row = next(row for row in result["case_rows"] if row["uses_morphology"])
    result["dice_artery"] = final_row["dice_artery"]
    result["artery_voxels"] = final_row["artery_voxels"]
    return result


def run_sample_experiment(img_id: int) -> dict:
    sample = load_sample_image(img_id)
    approach_inputs, approach_metadata, threshold_df = build_threshold_inputs(img_id, sample["image"])
    detection_results = {}
    for approach, lcc_image in approach_inputs.items():
        approach_config = approach_metadata.get(approach, {})
        reuse_from = approach_config.get("reuse_detection_from")
        if reuse_from in detection_results:
            detection_results[approach] = copy.deepcopy(detection_results[reuse_from])
            detection_results[approach]["approach"] = approach
            continue

        detection_results[approach] = run_detection_stage(
            img_id,
            approach,
            lcc_image,
            sample["down_label"],
            sample["scaled_spacing"],
            approach_config.get("config", RUN_CONFIG),
            vesselness_weight_map=approach_config.get("ostios"),
        )
    ostia_df = pd.DataFrame(
        [
            {
                "img_id": result["img_id"],
                "sample_split": SAMPLE_SPLITS_BY_ID.get(img_id),
                "approach": result["approach"],
                "ostia_status": result["ostia_status"],
                "ostia_found": result["ostia_found"],
                "both_correct": result["both_correct"],
                "both_tolerable": result["both_tolerable"],
                "ostia_acceptable": bool(result["both_correct"] or result["both_tolerable"]),
                "left_dist_mm": result["left_dist_mm"],
                "right_dist_mm": result["right_dist_mm"],
                "ostia_left": result["ostia_left"],
                "ostia_right": result["ostia_right"],
                "num_circles": result["num_circles"],
                "aorta_voxels": result["aorta_voxels"],
                "ostia_error": result["ostia_error"],
            }
            for result in detection_results.values()
        ]
    )
    if RUN_SEGMENTATION:
        segmentation_results = {
            approach: run_segmentation_stage(
                img_id,
                detection_result,
                approach_inputs[approach],
                approach_metadata.get(approach, {}).get("config", RUN_CONFIG),
                vesselness_weight_map=approach_metadata.get(approach, {}).get("artery"),
                region_growing_overrides=approach_metadata.get(approach, {}).get("region_growing"),
                vesselness_artery_overrides=approach_metadata.get(approach, {}).get("vesselness_artery_overrides"),
                scaled_spacing=sample["scaled_spacing"],
                region_growing_method=approach_metadata.get(approach, {}).get("region_growing_method", "standard"),
            )
            for approach, detection_result in detection_results.items()
        }
        segmentation_df = pd.DataFrame(
            [
                {key: value for key, value in result.items() if key != "case_rows"}
                for result in segmentation_results.values()
            ]
        )
        segmentation_comparison_df = pd.DataFrame(
            row for result in segmentation_results.values() for row in result["case_rows"]
        )
    else:
        segmentation_df = pd.DataFrame()
        segmentation_comparison_df = pd.DataFrame()
    sample_info_df = pd.DataFrame(
        [
            {
                "img_id": img_id,
                "sample_split": SAMPLE_SPLITS_BY_ID.get(img_id),
                "image_shape": sample["image_shape"],
                "label_shape": sample["label_shape"],
                "down_label_shape": sample["down_label_shape"],
                "spacing_mm": sample["spacing"],
                "scaled_spacing_mm": sample["scaled_spacing"],
            }
        ]
    )
    return {
        "sample_info_df": sample_info_df,
        "threshold_df": threshold_df,
        "ostia_df": ostia_df,
        "segmentation_df": segmentation_df,
        "segmentation_comparison_df": segmentation_comparison_df,
    }

def concat_outputs(outputs: list[dict], key: str) -> pd.DataFrame:
    frames = [output[key] for output in outputs if not output[key].empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_partial_outputs(outputs: list[dict]) -> None:
    """Salva tabelas parciais para recuperação simples se a execução cair."""
    partial_dir = RUN_OUTPUT_DIR / "partial"
    for key in (
        "sample_info_df",
        "threshold_df",
        "ostia_df",
        "segmentation_df",
        "segmentation_comparison_df",
    ):
        save_csv(concat_outputs(outputs, key), key.replace("_df", ""), partial_dir)


# %% [notebook cell 9]

sample_ids_df = pd.DataFrame(
    [
        {
            "sample_order": idx,
            "img_id": record["img_id"],
            "split": record["split"],
        }
        for idx, record in enumerate(SAMPLE_RECORDS, start=1)
    ]
)
save_csv(sample_ids_df, "sample_ids")

# %% [notebook cell 11]

print(f"Saídas: {RUN_OUTPUT_DIR}")
print(
    f"Rodando phase={OPTIMIZATION_PHASE}, imagens={len(SAMPLE_IMAGE_IDS)}, "
    f"variantes={len(THRESHOLD_APPROACHES)}"
)
sample_outputs = []
for sample_idx, img_id in enumerate(SAMPLE_IMAGE_IDS, start=1):
    print(f"[{sample_idx}/{len(SAMPLE_IMAGE_IDS)}] imagem {img_id}")
    sample_outputs.append(run_sample_experiment(img_id))
    if not ARGS.no_partial_csv:
        save_partial_outputs(sample_outputs)

sample_info_df = concat_outputs(sample_outputs, "sample_info_df")
threshold_df = concat_outputs(sample_outputs, "threshold_df")
ostia_df = concat_outputs(sample_outputs, "ostia_df")
segmentation_df = concat_outputs(sample_outputs, "segmentation_df")
segmentation_comparison_df = concat_outputs(sample_outputs, "segmentation_comparison_df")

optional_threshold_cols = [
    "sample_split",
    "approach_family",
    "pipeline_stage",
    "config_overrides",
    "fuzzy_output_mode",
    "base_weighted_approach",
    "max_threshold_percentile",
    "lcc_per_slice",
    "lcc_mode",
    "soft_center_hu",
    "object_center_hu",
    "dense_center_hu",
    "percentile_object_center_hu",
    "percentile_dense_center_hu",
    "dense_voxels",
    "mean_weight",
    "min_weight",
    "max_weight",
    "weight_floor",
    "dense_power",
    "weight_mode",
    "object_percentile",
    "dense_percentile",
    "smooth_radius",
    "smooth_mode",
    "apply_weight_to",
    "aorta_vesselness_method",
    "aorta_sigmas",
    "aorta_alpha",
    "aorta_beta",
    "aorta_gamma",
    "aorta_smooth_sigma",
    "artery_vesselness_label",
    "artery_vesselness_method",
    "artery_sigmas",
    "artery_alpha",
    "artery_beta",
    "artery_gamma",
    "artery_normalization",
    "artery_smooth_sigma",
    "vesselness_ablation_step",
    "circle_canny_sigma",
    "circle_total_num_peaks_initial",
    "circle_total_num_peaks",
    "circle_local_roi_padding",
    "circle_candidate_selection_strategy",
    "circle_out_of_tolerance_as_miss",
    "circle_candidate_score_accum_weight",
    "circle_candidate_score_distance_weight",
    "circle_candidate_score_radius_weight",
    "level_set_num_iter",
    "level_set_balloon",
    "level_set_smoothing",
    "ostia_top_n",
    "ostia_lower_fraction",
    "ostia_min_center_distance_factor",
    "ostia_min_lateral_factor",
    "ostia_erosion_radius",
    "post_closing_radius",
    "post_dilation_radius",
    "region_growing_method",
    "rg_ablation_step",
    "reuse_detection_from",
    "use_seed_refinement",
    "use_multi_seed",
    "use_priority_queue",
    "use_adaptive_acceptance",
    "use_distance_limit",
    "grow_each_ostium_separately",
    "seed_search_radius",
    "seed_candidate_radius",
    "max_seed_candidates",
    "seed_min_vesselness_fraction",
    "min_seed_candidate_distance_voxels",
    "adaptive_threshold_std_factor",
    "max_distance_voxels",
    "max_distance_mm",
    "min_threshold_fraction",
    "weight_gate",
    "rg_comparison_window",
    "rg_threshold_divisor",
    "rg_threshold_mode",
    "rg_threshold_percentiles",
    "rg_threshold_std_factor",
    "rg_threshold_scale",
    "rg_min_vesselness_fraction",
    "rg_min_vesselness_mode",
    "rg_min_vesselness_percentile",
    "rg_min_vesselness_std_factor",
    "rg_min_vesselness_mad_factor",
    "rg_min_vesselness_scale",
]
for col in optional_threshold_cols:
    if col not in threshold_df.columns:
        threshold_df[col] = np.nan

threshold_meta_cols = [
    "img_id",
    "sample_split",
    "approach",
    "threshold_mode",
    "pipeline_stage",
    "min_hu",
    "max_hu",
    "max_threshold_percentile",
    "lcc_per_slice",
    "lcc_mode",
    "threshold_voxels",
    "lcc_voxels",
    *optional_threshold_cols,
]
threshold_meta_cols = list(dict.fromkeys(threshold_meta_cols))

if not segmentation_comparison_df.empty:
    segmentation_comparison_df = threshold_df[threshold_meta_cols].merge(
        segmentation_comparison_df,
        on=["img_id", "sample_split", "approach"],
        how="right",
    )
    segmentation_comparison_df = segmentation_comparison_df.sort_values(
        ["img_id", "approach", "uses_morphology"],
        na_position="first",
    ).reset_index(drop=True)

ostia_df = threshold_df[
    [
        "img_id",
        "sample_split",
        "approach",
        "threshold_mode",
        "pipeline_stage",
        "approach_family",
        "fuzzy_output_mode",
        "min_hu",
        "max_hu",
        "max_threshold_percentile",
        "lcc_per_slice",
        "lcc_mode",
    ]
].merge(ostia_df, on=["img_id", "sample_split", "approach"], how="right")
ostia_df["ostia_acceptable"] = ostia_df["both_correct"] | ostia_df["both_tolerable"]

threshold_df.sort_values(["img_id", "approach"])[
    [
        "img_id",
        "sample_split",
        "approach",
        "pipeline_stage",
        "threshold_mode",
        "min_hu",
        "max_hu",
        "max_threshold_percentile",
        "lcc_per_slice",
        "lcc_mode",
        "threshold_voxels",
        "lcc_voxels",
        "region_growing_method",
        "reuse_detection_from",
        "rg_min_vesselness_fraction",
        "rg_min_vesselness_mode",
        "rg_min_vesselness_percentile",
        "rg_threshold_divisor",
        "rg_comparison_window",
        "use_seed_refinement",
        "use_multi_seed",
        "grow_each_ostium_separately",
        "seed_search_radius",
        "max_seed_candidates",
        "aorta_gamma",
        "aorta_sigmas",
        "artery_gamma",
        "artery_sigmas",
        "artery_smooth_sigma",
        "circle_canny_sigma",
        "circle_total_num_peaks",
        "level_set_num_iter",
        "level_set_balloon",
        "ostia_top_n",
        "ostia_lower_fraction",
        "ostia_erosion_radius",
        "post_closing_radius",
        "post_dilation_radius",
    ]
]

# %% [notebook cell 13]

ostia_df.sort_values(["img_id", "approach"])[
    [
        "img_id",
        "sample_split",
        "approach",
        "pipeline_stage",
        "ostia_status",
        "ostia_found",
        "ostia_acceptable",
        "both_correct",
        "both_tolerable",
        "left_dist_mm",
        "right_dist_mm",
        "num_circles",
        "aorta_voxels",
        "ostia_error",
    ]
]

# %% [notebook cell 15]

if segmentation_comparison_df.empty:
    morphology_dice_df = pd.DataFrame()
    dice_by_image_df = pd.DataFrame()
else:
    morphology_dice_df = segmentation_comparison_df[
        segmentation_comparison_df["uses_morphology"]
    ].copy()

    dice_by_image_df = (
        morphology_dice_df.pivot_table(
            index=["sample_split", "img_id"],
            columns="approach",
            values="dice_artery",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["sample_split", "img_id"])
    )
    dice_by_image_df.columns.name = None

dice_by_image_df

# %% [notebook cell 17]

if morphology_dice_df.empty:
    dice_summary_df = pd.DataFrame(
        columns=[
            "approach",
            "threshold_mode",
            "pipeline_stage",
            "mean_dice",
            "std_dice",
            "min_dice",
            "max_dice",
            "evaluated_images",
            "mean_dice_acceptable_ostia",
            "acceptable_ostia_dice_images",
            "mean_threshold_hu",
            "mean_threshold_voxels",
            "mean_lcc_voxels",
        ]
    )
else:
    dice_summary_df = (
        morphology_dice_df.groupby(["approach", "threshold_mode", "pipeline_stage"], as_index=False)
        .agg(
            mean_dice=("dice_artery", "mean"),
            std_dice=("dice_artery", "std"),
            min_dice=("dice_artery", "min"),
            max_dice=("dice_artery", "max"),
            evaluated_images=("img_id", "nunique"),
            mean_threshold_hu=("max_hu", "mean"),
            mean_threshold_voxels=("threshold_voxels", "mean"),
            mean_lcc_voxels=("lcc_voxels", "mean"),
        )
    )
    acceptable_dice_df = morphology_dice_df.merge(
        ostia_df[["img_id", "approach", "ostia_acceptable"]],
        on=["img_id", "approach"],
        how="left",
    )
    acceptable_dice_summary_df = (
        acceptable_dice_df[acceptable_dice_df["ostia_acceptable"].fillna(False)]
        .groupby(["approach"], as_index=False)
        .agg(
            mean_dice_acceptable_ostia=("dice_artery", "mean"),
            acceptable_ostia_dice_images=("img_id", "nunique"),
        )
    )
    dice_summary_df = dice_summary_df.merge(
        acceptable_dice_summary_df,
        on="approach",
        how="left",
    )

ostia_eval_df = ostia_df.copy()
for distance_col in ["left_dist_mm", "right_dist_mm"]:
    ostia_eval_df[distance_col] = ostia_eval_df[distance_col].replace(np.inf, np.nan)

ostia_summary_df = (
    ostia_eval_df.groupby(["approach", "threshold_mode", "pipeline_stage"], as_index=False)
    .agg(
        ostia_evaluated_images=("img_id", "nunique"),
        ostia_found_rate=("ostia_found", "mean"),
        both_correct_rate=("both_correct", "mean"),
        both_tolerable_rate=("both_tolerable", "mean"),
        ostia_acceptable_rate=("ostia_acceptable", "mean"),
        mean_left_dist_mm=("left_dist_mm", "mean"),
        mean_right_dist_mm=("right_dist_mm", "mean"),
        mean_num_circles=("num_circles", "mean"),
        mean_aorta_voxels=("aorta_voxels", "mean"),
    )
)
ostia_summary_df["wrong_ostia_rate"] = 1.0 - ostia_summary_df["ostia_acceptable_rate"]
ostia_summary_df["mean_ostia_dist_mm"] = (
    ostia_summary_df["mean_left_dist_mm"] + ostia_summary_df["mean_right_dist_mm"]
) / 2
ostia_summary_df["max_mean_side_dist_mm"] = ostia_summary_df[
    ["mean_left_dist_mm", "mean_right_dist_mm"]
].max(axis=1)

if dice_summary_df.empty:
    pipeline_summary_df = ostia_summary_df.copy()
    for col in [
        "mean_dice",
        "std_dice",
        "min_dice",
        "max_dice",
        "evaluated_images",
        "mean_dice_acceptable_ostia",
        "acceptable_ostia_dice_images",
        "mean_threshold_hu",
        "mean_threshold_voxels",
        "mean_lcc_voxels",
    ]:
        pipeline_summary_df[col] = np.nan
else:
    pipeline_summary_df = dice_summary_df.merge(
        ostia_summary_df,
        on=["approach", "threshold_mode", "pipeline_stage"],
        how="left",
    )

pipeline_summary_df["ostia_priority_score"] = (
    0.70 * pipeline_summary_df["ostia_acceptable_rate"].fillna(0)
    + 0.20 * pipeline_summary_df["both_correct_rate"].fillna(0)
    + 0.10 * pipeline_summary_df["ostia_found_rate"].fillna(0)
)
pipeline_summary_df["segmentation_priority_score"] = (
    pipeline_summary_df["mean_dice_acceptable_ostia"]
    .fillna(pipeline_summary_df["mean_dice"])
    .fillna(0)
    * pipeline_summary_df["ostia_acceptable_rate"].fillna(0)
)
pipeline_summary_df["selection_score"] = np.where(
    OPTIMIZATION_PHASE == "ostia",
    pipeline_summary_df["ostia_priority_score"],
    pipeline_summary_df["segmentation_priority_score"],
)

ranking_columns = [
    "approach",
    "pipeline_stage",
    "selection_score",
    "ostia_acceptable_rate",
    "both_correct_rate",
    "both_tolerable_rate",
    "ostia_found_rate",
    "mean_left_dist_mm",
    "mean_right_dist_mm",
    "mean_dice_acceptable_ostia",
    "mean_dice",
    "std_dice",
    "min_dice",
    "evaluated_images",
    "ostia_evaluated_images",
    "threshold_mode",
    "lcc_mode",
]
if OPTIMIZATION_PHASE == "ostia":
    sort_columns = [
        "ostia_acceptable_rate",
        "both_correct_rate",
        "mean_ostia_dist_mm",
        "max_mean_side_dist_mm",
        "ostia_found_rate",
    ]
    sort_ascending = [False, False, True, True, False]
else:
    sort_columns = [
        "selection_score",
        "ostia_acceptable_rate",
        "both_correct_rate",
        "mean_dice_acceptable_ostia",
        "mean_dice",
    ]
    sort_ascending = [False, False, False, False, False]

optimization_ranking_df = (
    pipeline_summary_df[ranking_columns]
    .sort_values(
        sort_columns,
        ascending=sort_ascending,
        na_position="last",
    )
    .reset_index(drop=True)
)
save_csv(config_df, "config")
save_csv(sample_info_df, "sample_info")
save_csv(threshold_df, "threshold")
save_csv(ostia_df, "ostia")
save_csv(segmentation_df, "segmentation")
save_csv(segmentation_comparison_df, "segmentation_comparison")
save_csv(dice_by_image_df, "dice_by_image")
save_csv(dice_summary_df, "dice_summary")
save_csv(ostia_summary_df, "ostia_summary")
save_csv(pipeline_summary_df, "pipeline_summary")
save_csv(optimization_ranking_df, "optimization_ranking")

run_finished_at = datetime.now()
save_json(
    {
        "run_name": RUN_NAME,
        "started_at": RUN_STARTED_AT.isoformat(timespec="seconds"),
        "finished_at": run_finished_at.isoformat(timespec="seconds"),
        "duration_seconds": (run_finished_at - RUN_STARTED_AT).total_seconds(),
        "repo_root": str(REPO_ROOT),
        "config_path": str(CONFIG_PATH),
        "base_path": str(BASE_PATH),
        "output_dir": str(RUN_OUTPUT_DIR),
        "optimization_phase": OPTIMIZATION_PHASE,
        "run_segmentation": RUN_SEGMENTATION,
        "train_sample_size": TRAIN_SAMPLE_SIZE,
        "val_sample_size": VAL_SAMPLE_SIZE,
        "total_sample_size": len(SAMPLE_IMAGE_IDS),
        "threshold_approaches": THRESHOLD_APPROACHES,
        "load_cache": CONFIG["LOAD_CACHE"],
        "save_cache": CONFIG["SAVE_CACHE"],
    },
    "metadata",
)

print(f"CSV final salvo em: {RUN_OUTPUT_DIR}")
print(optimization_ranking_df.head(20).to_string(index=False))
