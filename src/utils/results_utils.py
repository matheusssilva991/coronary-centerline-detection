"""Utilities para criação de relatórios e persistência de resultados."""

import json
import os
import platform
from datetime import datetime

import pandas as pd


def create_timestamped_output_dir(base_output_dir, experiment_name="segmentation"):
    """Cria diretório de saída com timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(base_output_dir, experiment_name, timestamp)
    os.makedirs(output_path, exist_ok=True)
    return output_path


def make_result_dataframe(results):
    """Converte lista de resultados em DataFrame formatado."""
    rows = []
    for result in results:
        row = {
            "IMG_ID": result.get("IMG_ID"),
            "dice_artery": result.get("dice_artery"),
            "artery_voxels": result.get("artery_voxels"),
            "ostia_found": result.get("ostia_found", False),
            "ostia_status": result.get("ostia_status"),
            "segmentation_attempted": result.get("segmentation_attempted", False),
            "proceeded_with_bad_ostia": result.get("proceeded_with_bad_ostia", False),
            "skip_reason": result.get("skip_reason"),
            "ostia_error": result.get("ostia_error"),
            "both_correct": result.get("both_correct", False),
            "both_tolerable": result.get("both_tolerable", False),
            "left_intersects": result.get("left_intersects", False),
            "right_intersects": result.get("right_intersects", False),
            "left_dist_mm": result.get("left_dist_mm"),
            "right_dist_mm": result.get("right_dist_mm"),
            "ostia_left": result.get("ostia_left"),
            "ostia_right": result.get("ostia_right"),
            "error": result.get("error", None),
        }

        if result.get("ostia_status") == "not_found":
            row["status"] = "óstios não encontrados"
        elif result.get("both_correct", False):
            row["status"] = "ambos corretos"
        elif result.get("both_tolerable", False):
            row["status"] = "ambos toleráveis"
        elif result.get("left_intersects", False) or result.get(
            "right_intersects", False
        ):
            row["status"] = "um correto"
        elif result.get("error"):
            row["status"] = "erro"
        else:
            row["status"] = "nenhum correto"

        rows.append(row)

    return pd.DataFrame(rows)


def save_results(results, split_name, output_dir, config=None):
    """Salva resultados em CSV."""
    df = make_result_dataframe(results)

    if config is not None:
        df["downscale_method"] = config.get("DOWNSCALE_METHOD", "N/A")
        df["opencv_interpolation"] = (
            config.get("OPENCV_INTERPOLATION", "N/A")
            if config.get("DOWNSCALE_METHOD") == "opencv"
            else "N/A"
        )
        df["downscale_factors"] = str(config.get("DOWNSCALE_FACTORS", "N/A"))
        df["max_threshold_percentile"] = config.get("MAX_THRESHOLD_PERCENTILE", "N/A")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"ostios_{split_name}_summary.csv")
    df.to_csv(output_path, index=False)
    return output_path


def save_metadata(
    split_name,
    output_dir,
    config,
    ids,
    results,
    execution_time=None,
    base_path=None,
    base_save_path=None,
    root_output_dir=None,
):
    """Salva metadados da execução em arquivo JSON."""
    df = make_result_dataframe(results)

    both_correct_series = df["both_correct"].fillna(False)
    both_tolerable_series = df["both_tolerable"].fillna(False)
    ostia_found_series = df["ostia_found"].fillna(False)
    segmentation_attempted_series = df["segmentation_attempted"].fillna(False)
    proceeded_with_bad_ostia_series = df["proceeded_with_bad_ostia"].fillna(False)
    ostia_not_found_series = df["ostia_status"].eq("not_found")

    metadata = {
        "execution_info": {
            "timestamp": datetime.now().isoformat(),
            "split_name": split_name,
            "num_images": len(ids),
            "image_ids": ids,
            "execution_time_seconds": execution_time,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "state_counters": {
                "ostia_found": int(ostia_found_series.sum()),
                "ostia_status_not_found": int(ostia_not_found_series.sum()),
                "segmentation_attempted": int(segmentation_attempted_series.sum()),
                "proceeded_with_bad_ostia": int(proceeded_with_bad_ostia_series.sum()),
                "error_not_null": int(df["error"].notna().sum()),
            },
        },
        "preprocessing_config": {
            "downscale_method": config.get("DOWNSCALE_METHOD"),
            "opencv_interpolation": config.get("OPENCV_INTERPOLATION")
            if config.get("DOWNSCALE_METHOD") == "opencv"
            else None,
            "downscale_factors": config.get("DOWNSCALE_FACTORS"),
            "max_threshold_percentile": config.get("MAX_THRESHOLD_PERCENTILE"),
        },
        "vesselness_config": {
            "ostios": {
                "sigmas": config["VESSELNESS_AORTA"]["sigmas"].tolist()
                if hasattr(config["VESSELNESS_AORTA"]["sigmas"], "tolist")
                else list(config["VESSELNESS_AORTA"]["sigmas"]),
                "alpha": config["VESSELNESS_AORTA"]["alpha"],
                "beta": config["VESSELNESS_AORTA"]["beta"],
                "gamma": config["VESSELNESS_AORTA"]["gamma"],
            },
            "artery": {
                "sigmas": config["VESSELNESS_ARTERY"]["sigmas"].tolist()
                if hasattr(config["VESSELNESS_ARTERY"]["sigmas"], "tolist")
                else list(config["VESSELNESS_ARTERY"]["sigmas"]),
                "alpha": config["VESSELNESS_ARTERY"]["alpha"],
                "beta": config["VESSELNESS_ARTERY"]["beta"],
                "gamma": config["VESSELNESS_ARTERY"]["gamma"],
            },
        },
        "circle_detection_config": config.get("CIRCLE_DETECTION"),
        "level_set_config": config.get("LEVEL_SET"),
        "ostia_detection_config": config.get("OSTIA_DETECTION"),
        "region_growing_config": config.get("REGION_GROWING"),
        "postprocessing_config": config.get("POSTPROCESSING"),
        "cache_config": {
            "load_cache": config.get("LOAD_CACHE"),
            "save_cache": config.get("SAVE_CACHE"),
        },
        "evaluation_config": {
            "tolerable_distance_mm": config["OSTIA_VALIDATION"][
                "distance_threshold_mm"
            ],
        },
        "results_summary": {
            "total_processed": len(df),
            "ostia_found": int(ostia_found_series.sum()),
            "ostia_found_percent": float(ostia_found_series.mean() * 100),
            "ostia_status_not_found": int(ostia_not_found_series.sum()),
            "ostia_status_not_found_percent": float(
                ostia_not_found_series.mean() * 100
            ),
            "both_correct": int(both_correct_series.sum()),
            "both_correct_percent": float(both_correct_series.mean() * 100),
            "both_tolerable": int(both_tolerable_series.sum()),
            "both_tolerable_percent": float(both_tolerable_series.mean() * 100),
            "segmentation_attempted": int(segmentation_attempted_series.sum()),
            "segmentation_attempted_percent": float(
                segmentation_attempted_series.mean() * 100
            ),
            "proceeded_with_bad_ostia": int(proceeded_with_bad_ostia_series.sum()),
            "proceeded_with_bad_ostia_percent": float(
                proceeded_with_bad_ostia_series.mean() * 100
            ),
            "total_success": int((both_correct_series | both_tolerable_series).sum()),
            "total_success_percent": float(
                (both_correct_series | both_tolerable_series).mean() * 100
            ),
            "left_correct": int(df["left_intersects"].sum()),
            "right_correct": int(df["right_intersects"].sum()),
            "error_not_null": int(df["error"].notna().sum()),
            "dice_artery_mean": float(df["dice_artery"].mean())
            if df["dice_artery"].notna().any()
            else None,
            "dice_artery_std": float(df["dice_artery"].std())
            if df["dice_artery"].notna().any()
            else None,
            "dice_artery_median": float(df["dice_artery"].median())
            if df["dice_artery"].notna().any()
            else None,
        },
    }

    if (
        base_path is not None
        or base_save_path is not None
        or root_output_dir is not None
    ):
        metadata["paths"] = {
            "base_path": base_path,
            "base_save_path": base_save_path,
            "output_dir": root_output_dir,
        }

    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, f"ostios_{split_name}_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2, ensure_ascii=False)

    return metadata_path
