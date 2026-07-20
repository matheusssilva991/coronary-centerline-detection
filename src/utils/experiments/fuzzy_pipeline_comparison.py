"""Helpers para comparar variantes fuzzy no pipeline coronário.

Este módulo concentra a lógica que o notebook de comparação usa para alternar
entre threshold normal, threshold fuzzy, region growing e fuzzy connectedness.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from utils.experiments.sweep_common import (
    csv_safe,
    get_nested,
    make_json_safe,
    resolve_cli_path,
    set_nested,
)
from utils.project.config import load_config_json
from utils.processing.preprocessing import build_lcc_image_from_mask, downscale_image
from utils.segmentation.artery_segmentation import normal_region_growing_from_ostia
from utils.segmentation.fuzzy_connectedness import segment_artery_fuzzy_connectedness
from utils.segmentation.fuzzy_threshold import fuzzy_threshold_outputs
from utils.segmentation.lower_threshold import resolve_lower_threshold
from utils.segmentation.pipeline_arteries import postprocess_artery_mask
from utils.segmentation.pipeline_detection import (
    detect_and_evaluate_ostia,
    get_or_detect_aorta_circles,
    get_or_segment_aorta,
)
from utils.segmentation.pipeline_preprocessing import get_or_compute_vesselness
from utils.utils.metrics import dice_score
from utils.utils.nifti_io import load_raw_img_and_label


IMAGE_COLUMNS = [
    "variant",
    "split",
    "IMG_ID",
    "threshold_mode",
    "artery_method",
    "dice_artery",
    "artery_voxels",
    "ostia_success",
    "ostia_found",
    "ostia_status",
    "both_correct",
    "both_tolerable",
    "left_dist_mm",
    "right_dist_mm",
    "segmentation_attempted",
    "threshold_voxels",
    "lcc_voxels",
    "fc_processed_voxels",
    "fc_effective_alpha",
    "error",
]

PARAMETER_COLUMNS = [
    "variant",
    "overrides_json",
    "threshold_mode",
    "fuzzy.soft_margin_hu",
    "fuzzy.object_percentile",
    "fuzzy.dense_percentile",
    "fuzzy.smooth_radius",
    "fuzzy.smooth_mode",
    "artery_method",
    "fc.alpha",
    "fc.sigma_hu",
    "fc.neighborhood",
    "fc.candidate_min_vesselness",
    "fc.seed_min_vesselness",
    "fc.vesselness_weight",
    "fc.vesselness_floor",
    "MAX_THRESHOLD_PERCENTILE",
    "REGION_GROWING.min_vesselness_fraction",
    "REGION_GROWING.threshold_divisor",
]

EXPERIMENT_KEYS = {
    "threshold_mode",
    "artery_method",
    "max_candidate_voxels",
    "max_processed_voxels",
}


def split_overrides(overrides: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separa overrides da config do pipeline e parâmetros experimentais."""
    config_overrides: dict[str, Any] = {}
    experiment: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in EXPERIMENT_KEYS or key.startswith(("fc.", "fuzzy.")):
            set_nested(experiment, key, value)
        else:
            set_nested(config_overrides, key, value)
    return config_overrides, experiment


def opencv_interpolation_value(name: str) -> int:
    """Converte nome textual de interpolação para constante do OpenCV."""
    return {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos4": cv2.INTER_LANCZOS4,
    }.get(name, cv2.INTER_LINEAR)


def load_downsampled_case(
    img_id: int,
    base_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Carrega imagem/label e aplica o downsampling definido na config."""
    img_path = base_path / f"{img_id}.img.nii.gz"
    label_path = base_path / f"{img_id}.label.nii.gz"
    nii_img, nii_label = load_raw_img_and_label(str(img_path), str(label_path))
    image = nii_img.get_fdata(dtype=np.float32)
    label = nii_label.get_fdata(dtype=np.float32).astype(np.uint8)
    spacing = tuple(float(value) for value in nii_img.header.get_zooms()[:3])
    factors = tuple(config["DOWNSCALE_FACTORS"])

    down_image = downscale_image(
        image,
        factors,
        order=3,
        use_opencv=config.get("DOWNSCALE_METHOD") == "opencv",
        opencv_interpolation=opencv_interpolation_value(
            config.get("OPENCV_INTERPOLATION", "linear")
        ),
    ).astype(np.float32)
    down_label = downscale_image(
        label,
        factors,
        order=0,
        use_opencv=False,
    ).astype(np.uint8)
    scaled_spacing = tuple(spacing[idx] * factors[idx] for idx in range(3))
    return {
        "down_image": down_image,
        "down_label": down_label,
        "scaled_spacing": scaled_spacing,
        "downscale_factors": factors,
    }


def build_preprocessed_inputs(
    down_image: np.ndarray,
    config: dict[str, Any],
    experiment: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Monta threshold/LCC para a variante."""
    min_hu, lower_details = resolve_lower_threshold(down_image, config)
    max_hu = float(np.percentile(down_image, float(config["MAX_THRESHOLD_PERCENTILE"])))
    threshold_mode = experiment.get("threshold_mode", "normal")
    fuzzy_cfg = experiment.get("fuzzy", {})

    fuzzy_mask, fuzzy_details = fuzzy_threshold_outputs(
        down_image,
        min_hu=min_hu,
        soft_margin_hu=float(fuzzy_cfg.get("soft_margin_hu", 100)),
        object_percentile=float(fuzzy_cfg.get("object_percentile", 99.8)),
        dense_percentile=float(fuzzy_cfg.get("dense_percentile", 99.96)),
        smooth_radius=int(fuzzy_cfg.get("smooth_radius", 0)),
        smooth_mode=str(fuzzy_cfg.get("smooth_mode", "mean")),
    )

    if threshold_mode == "fuzzy":
        mask = fuzzy_mask
    else:
        mask = (down_image >= min_hu) & (down_image <= max_hu)

    lcc_image, lcc_mask = build_lcc_image_from_mask(
        down_image,
        mask,
        offset=abs(int(min_hu)),
        per_slice=True,
    )
    return lcc_image, lcc_mask, {
        "threshold_mode": threshold_mode,
        **lower_details,
        "max_hu": max_hu,
        "threshold_voxels": int(mask.sum()),
        "lcc_voxels": int(lcc_mask.sum()),
        **fuzzy_details,
    }


def build_base_config(args: Any) -> dict[str, Any]:
    """Carrega a config base com ajustes simples de resolução/cache/GPU."""
    config = load_config_json(str(resolve_cli_path(args.config_path)), {})
    if args.resolution == "high":
        config["DOWNSCALE_FACTORS"] = [1, 1, 1]
    config["LOAD_CACHE"] = bool(args.load_cache)
    config["SAVE_CACHE"] = bool(args.save_cache)
    if args.use_gpu is not None:
        config["USE_GPU"] = bool(args.use_gpu)
    return config


def compute_vesselness_spacing(
    scaled_spacing: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Converte spacing da imagem para a convenção usada pelo Frangi."""
    dx, dy, dz = scaled_spacing
    return (dy, dx, dz)


def run_image(
    img_id: int,
    variant_name: str,
    split_name: str,
    base_path: Path,
    cache_dir: Path,
    config: dict[str, Any],
    experiment: dict[str, Any],
) -> dict[str, Any]:
    """Executa uma imagem em uma variante completa do pipeline."""
    row = {
        "variant": variant_name,
        "split": split_name,
        "IMG_ID": img_id,
        "threshold_mode": experiment.get("threshold_mode", "normal"),
        "artery_method": experiment.get("artery_method", "region_growing"),
        "dice_artery": np.nan,
        "dice_artery_before_morphology": np.nan,
        "dice_artery_after_morphology": np.nan,
        "dice_artery_morphology_delta": np.nan,
        "artery_voxels": 0,
        "artery_voxels_before_morphology": 0,
        "artery_voxels_after_morphology": 0,
        "ostia_success": False,
        "ostia_found": False,
        "ostia_status": "not_evaluated",
        "both_correct": False,
        "both_tolerable": False,
        "left_dist_mm": np.nan,
        "right_dist_mm": np.nan,
        "segmentation_attempted": False,
        "threshold_voxels": 0,
        "lcc_voxels": 0,
        "fc_processed_voxels": np.nan,
        "fc_effective_alpha": np.nan,
        "error": None,
    }
    try:
        case = load_downsampled_case(img_id, base_path, config)
        lcc_image, lcc_mask, prep_details = build_preprocessed_inputs(
            case["down_image"],
            config,
            experiment,
        )
        row.update(
            {
                key: prep_details.get(key)
                for key in ("threshold_voxels", "lcc_voxels")
            }
        )
        spacing = case["scaled_spacing"]
        vesselness_spacing = compute_vesselness_spacing(spacing)

        vesselness_ostios = get_or_compute_vesselness(
            str(img_id),
            lcc_image,
            cache_dir=str(cache_dir / variant_name / "vesselness_ostios"),
            vesselness_config=config["VESSELNESS_AORTA"],
            load_cache=config["LOAD_CACHE"],
            save_cache=config["SAVE_CACHE"],
            use_gpu=config.get("USE_GPU", False),
            spacing=vesselness_spacing,
        )
        detected_circles = get_or_detect_aorta_circles(
            str(img_id),
            lcc_image,
            case["downscale_factors"],
            spacing,
            config["CIRCLE_DETECTION"],
            str(cache_dir / variant_name),
            load_cache=config["LOAD_CACHE"],
            save_cache=False,
        )
        aorta_mask = get_or_segment_aorta(
            str(img_id),
            lcc_image,
            detected_circles,
            config["LEVEL_SET"],
            str(cache_dir / variant_name),
            load_cache=config["LOAD_CACHE"],
            save_cache=False,
            use_gpu=config.get("USE_GPU", False),
        )
        ostia_eval = detect_and_evaluate_ostia(
            aorta_mask,
            vesselness_ostios,
            case["down_label"],
            spacing,
            config,
        )
        both_correct = bool(ostia_eval["both_correct"])
        both_tolerable = bool(ostia_eval["both_tolerable"])
        row.update(
            {
                "ostia_found": True,
                "both_correct": both_correct,
                "both_tolerable": both_tolerable,
                "ostia_success": both_correct or both_tolerable,
                "ostia_status": (
                    "both_correct"
                    if both_correct
                    else "both_tolerable"
                    if both_tolerable
                    else "found_but_wrong"
                ),
                "left_dist_mm": ostia_eval["left_info"]["physical_dist"],
                "right_dist_mm": ostia_eval["right_info"]["physical_dist"],
            }
        )

        vesselness_artery = get_or_compute_vesselness(
            str(img_id),
            lcc_image,
            cache_dir=str(cache_dir / variant_name / "vesselness_artery"),
            vesselness_config=config["VESSELNESS_ARTERY"],
            load_cache=config["LOAD_CACHE"],
            save_cache=config["SAVE_CACHE"],
            use_gpu=config.get("USE_GPU", False),
            spacing=vesselness_spacing,
        )
        row["segmentation_attempted"] = True
        label_artery = ostia_eval["label_artery"]

        if experiment.get("artery_method", "region_growing") == "fuzzy_connectedness":
            fc_params = {
                "alpha": 0.18,
                "sigma_hu": 80,
                "neighborhood": 26,
                **experiment.get("fc", {}),
            }
            fc_result = segment_artery_fuzzy_connectedness(
                lcc_image,
                vesselness_artery,
                [ostia_eval["ostia_left"], ostia_eval["ostia_right"]],
                lcc_mask,
                config,
                params=fc_params,
                max_candidate_voxels=experiment.get("max_candidate_voxels", 500_000),
                max_processed_voxels=experiment.get("max_processed_voxels", 500_000),
            )
            artery_mask = fc_result["artery_mask"]
            raw_mask = fc_result["raw_mask"]
            row["fc_processed_voxels"] = fc_result["details"].get("processed_voxels")
            row["fc_effective_alpha"] = fc_result["details"].get("effective_alpha")
        else:
            raw_mask = normal_region_growing_from_ostia(
                vesselness_artery,
                ostia_eval["ostia_left"],
                ostia_eval["ostia_right"],
                config,
            )
            artery_mask = postprocess_artery_mask(raw_mask, config)

        dice_before = float(dice_score(raw_mask, label_artery))
        dice_after = float(dice_score(artery_mask, label_artery))
        row.update(
            {
                "artery_voxels": int(np.sum(artery_mask)),
                "artery_voxels_before_morphology": int(np.sum(raw_mask)),
                "artery_voxels_after_morphology": int(np.sum(artery_mask)),
                "dice_artery": dice_after,
                "dice_artery_before_morphology": dice_before,
                "dice_artery_after_morphology": dice_after,
                "dice_artery_morphology_delta": dice_after - dice_before,
            }
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        if row["ostia_status"] == "not_evaluated":
            row["ostia_status"] = "error"
    return row


def summarize_variant(
    variant_name: str,
    rows: list[dict[str, Any]],
    runtime_seconds: float,
) -> dict[str, Any]:
    """Agrega resultados por variante."""
    df = pd.DataFrame(rows)
    if df.empty:
        return {"variant": variant_name, "images": 0}

    dice = pd.to_numeric(df["dice_artery"], errors="coerce")
    dice_before = pd.to_numeric(
        df["dice_artery_before_morphology"], errors="coerce"
    )
    morphology_delta = pd.to_numeric(
        df["dice_artery_morphology_delta"], errors="coerce"
    )
    success = df["ostia_success"].fillna(False).astype(bool)
    success_dice = dice[success]
    score_dice = success_dice.mean() if success_dice.notna().any() else dice.mean()
    score_dice = 0.0 if pd.isna(score_dice) else float(score_dice)
    return {
        "variant": variant_name,
        "images": int(len(df)),
        "ostia_success_rate": float(success.mean()),
        "ostia_found_rate": float(df["ostia_found"].fillna(False).astype(bool).mean()),
        "both_correct_rate": float(df["both_correct"].fillna(False).astype(bool).mean()),
        "both_tolerable_rate": float(
            df["both_tolerable"].fillna(False).astype(bool).mean()
        ),
        "mean_dice": float(dice.mean()) if dice.notna().any() else None,
        "mean_dice_before_morphology": (
            float(dice_before.mean()) if dice_before.notna().any() else None
        ),
        "mean_dice_after_morphology": (
            float(dice.mean()) if dice.notna().any() else None
        ),
        "mean_dice_morphology_delta": (
            float(morphology_delta.mean())
            if morphology_delta.notna().any()
            else None
        ),
        "median_dice": float(dice.median()) if dice.notna().any() else None,
        "mean_dice_success_ostia": (
            float(success_dice.mean()) if success_dice.notna().any() else None
        ),
        "mean_left_dist_mm": pd.to_numeric(
            df["left_dist_mm"],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).mean(),
        "mean_right_dist_mm": pd.to_numeric(
            df["right_dist_mm"],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).mean(),
        "error_count": int(df["error"].notna().sum()),
        "runtime_seconds": float(runtime_seconds),
        "runtime_minutes": float(runtime_seconds / 60),
        "selection_score": float(success.mean()) * score_dice,
    }


def parameter_row(
    variant_name: str,
    overrides: dict[str, Any],
    config: dict[str, Any],
    experiment: dict[str, Any],
) -> dict[str, Any]:
    """Cria uma linha compacta de parâmetros para a variante."""
    merged = copy.deepcopy(config)
    merged.update(experiment)
    row = {
        "variant": variant_name,
        "overrides_json": json.dumps(make_json_safe(overrides), ensure_ascii=False),
    }
    for key in PARAMETER_COLUMNS:
        if key in row:
            continue
        row[key] = make_json_safe(get_nested(merged, key, experiment.get(key)))
    return row


def save_outputs(
    run_dir: Path,
    summaries: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    parameter_rows: list[dict[str, Any]],
) -> None:
    """Salva CSVs compactos de ranking, resultados por imagem e parâmetros."""
    summary_dir = run_dir / "summary"
    results_dir = run_dir / "results"
    parameters_dir = run_dir / "parameters"
    summary_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    parameters_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            [
                "selection_score",
                "ostia_success_rate",
                "mean_dice_success_ostia",
                "mean_dice",
            ],
            ascending=[False, False, False, False],
            na_position="last",
        )
    csv_safe(summary_df).to_csv(summary_dir / "ranking.csv", index=False)
    csv_safe(pd.DataFrame(image_rows, columns=IMAGE_COLUMNS)).to_csv(
        results_dir / "image_results.csv",
        index=False,
    )
    csv_safe(pd.DataFrame(parameter_rows).reindex(columns=PARAMETER_COLUMNS)).to_csv(
        parameters_dir / "variant_parameters.csv",
        index=False,
    )


__all__ = [
    "EXPERIMENT_KEYS",
    "IMAGE_COLUMNS",
    "PARAMETER_COLUMNS",
    "build_base_config",
    "build_preprocessed_inputs",
    "compute_vesselness_spacing",
    "fuzzy_threshold_outputs",
    "load_downsampled_case",
    "parameter_row",
    "run_image",
    "save_outputs",
    "split_overrides",
    "summarize_variant",
]
