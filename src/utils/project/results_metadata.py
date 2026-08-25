"""Construção e persistência dos metadados de uma execução."""

from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

from .results_schema import make_result_dataframe, summarize_results_df
from .results_timing import duration_breakdown


def make_json_safe(value: Any) -> Any:
    """Converte valores comuns de pandas/numpy/pathlib para JSON nativo."""
    # Percorre estruturas aninhadas antes de converter objetos escalares.
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "as_posix"):
        return value.as_posix()
    if hasattr(value, "tolist"):
        return make_json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def _vesselness_metadata(config: dict[str, Any], key: str) -> dict[str, Any]:
    vesselness_config = config[key]
    sigmas = vesselness_config["sigmas"]
    return {
        "method": vesselness_config.get("method", "normal"),
        "sigmas": sigmas.tolist() if hasattr(sigmas, "tolist") else list(sigmas),
        "black_ridges": vesselness_config.get("black_ridges", False),
        "alpha": vesselness_config["alpha"],
        "beta": vesselness_config["beta"],
        "gamma": vesselness_config["gamma"],
        "normalization": vesselness_config.get("normalization", "none"),
        "smooth_sigma": vesselness_config.get("smooth_sigma", 0.0),
    }


def _runtime_config_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Resume os parâmetros efetivos que diferenciam runs completos."""
    circle_config = config.get("CIRCLE_DETECTION", {})
    artery_method = config.get("ARTERY_SEGMENTATION", {}).get(
        "method", "region_growing"
    )
    return {
        "use_gpu": config.get("USE_GPU"),
        "save_segmentation_visuals": config.get("SAVE_SEGMENTATION_VISUALS", False),
        "visual_output_dir": config.get("VISUAL_OUTPUT_DIR"),
        "downscale_method": config.get("DOWNSCALE_METHOD"),
        "opencv_interpolation": config.get("OPENCV_INTERPOLATION")
        if config.get("DOWNSCALE_METHOD") == "opencv"
        else None,
        "downscale_factors": config.get("DOWNSCALE_FACTORS"),
        "min_threshold": config.get("MIN_THRESHOLD"),
        "max_threshold_percentile": config.get("MAX_THRESHOLD_PERCENTILE"),
        "thresholding": config.get("THRESHOLDING"),
        "lcc_per_slice": True,
        "lcc_mode": "per_slice",
        "aorta_miss_count": circle_config.get("max_slice_miss_threshold"),
        "aorta_interpolate_missed_circles": circle_config.get(
            "interpolate_missed_circles"
        ),
        "artery_segmentation_method": str(artery_method),
        "aorta_level_set_mode": config.get("LEVEL_SET", {}).get(
            "iteration_mode", "fixed"
        ),
    }


def build_metadata(
    split_name,
    config,
    ids,
    results,
    execution_time=None,
    current_run_execution_time=None,
    batch_timings=None,
    batch_timing_summary=None,
    base_path=None,
    root_output_dir=None,
):
    """Monta a estrutura JSON de metadados sem gravar arquivo."""
    # Recalcula os agregados a partir das linhas efetivamente persistidas.
    df = make_result_dataframe(results)
    results_summary = summarize_results_df(df)

    execution_duration = duration_breakdown(execution_time)
    current_run_duration = duration_breakdown(current_run_execution_time)
    # Agrupa informações de execução, configuração e desempenho em seções.
    metadata = {
        "execution_info": {
            "timestamp": datetime.now().isoformat(),
            "split_name": split_name,
            "num_images": len(ids),
            "image_ids": ids,
            "execution_time_seconds": execution_time,
            "execution_time_minutes": execution_duration["minutes"],
            "execution_time_hours": execution_duration["hours"],
            "current_run_execution_time_seconds": current_run_execution_time,
            "current_run_execution_time_minutes": current_run_duration["minutes"],
            "current_run_execution_time_hours": current_run_duration["hours"],
            "batch_timing_summary": batch_timing_summary,
            "batch_timings": batch_timings or [],
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "state_counters": {
                "ostia_found": results_summary["ostia_found"],
                "ostia_status_not_found": results_summary["ostia_status_not_found"],
                "segmentation_attempted": results_summary["segmentation_attempted"],
                "proceeded_with_bad_ostia": results_summary["proceeded_with_bad_ostia"],
                "error_not_null": results_summary["error_not_null"],
            },
        },
        "preprocessing_config": {
            "downscale_method": config.get("DOWNSCALE_METHOD"),
            "opencv_interpolation": config.get("OPENCV_INTERPOLATION")
            if config.get("DOWNSCALE_METHOD") == "opencv"
            else None,
            "downscale_factors": config.get("DOWNSCALE_FACTORS"),
            "min_threshold": config.get("MIN_THRESHOLD"),
            "max_threshold_percentile": config.get("MAX_THRESHOLD_PERCENTILE"),
            "thresholding": config.get("THRESHOLDING"),
            "lcc_per_slice": True,
            "lcc_mode": "per_slice",
        },
        "runtime_config": _runtime_config_metadata(config),
        "vesselness_config": {
            "ostios": _vesselness_metadata(config, "VESSELNESS_AORTA"),
            "artery": _vesselness_metadata(config, "VESSELNESS_ARTERY"),
        },
        "circle_detection_config": config.get("CIRCLE_DETECTION"),
        "level_set_config": config.get("LEVEL_SET"),
        "ostia_detection_config": config.get("OSTIA_DETECTION"),
        "artery_segmentation_config": config.get("ARTERY_SEGMENTATION"),
        "thresholding_config": config.get("THRESHOLDING"),
        "region_growing_config": config.get("REGION_GROWING"),
        "fuzzy_connectedness_config": config.get("FUZZY_CONNECTEDNESS"),
        "postprocessing_config": config.get("POSTPROCESSING"),
        "evaluation_config": {
            "tolerable_distance_mm": config["OSTIA_VALIDATION"][
                "distance_threshold_mm"
            ],
        },
        "results_summary": results_summary,
    }

    # Caminhos são opcionais para manter o helper útil em testes isolados.
    if base_path is not None or root_output_dir is not None:
        metadata["paths"] = {
            "base_path": base_path,
            "output_dir": root_output_dir,
        }
    return metadata


def save_metadata(
    split_name,
    output_dir,
    config,
    ids,
    results,
    execution_time=None,
    current_run_execution_time=None,
    batch_timings=None,
    batch_timing_summary=None,
    base_path=None,
    root_output_dir=None,
):
    """Salva metadados da execução em arquivo JSON."""
    # Mantém a construção separada da persistência para facilitar testes.
    metadata = build_metadata(
        split_name,
        config,
        ids,
        results,
        execution_time=execution_time,
        current_run_execution_time=current_run_execution_time,
        batch_timings=batch_timings,
        batch_timing_summary=batch_timing_summary,
        base_path=base_path,
        root_output_dir=root_output_dir,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / f"ostios_{split_name}_metadata.json"
    # A conversão final trata arrays NumPy e Paths presentes na configuração.
    with metadata_path.open("w", encoding="utf-8") as file_handle:
        json.dump(make_json_safe(metadata), file_handle, indent=2, ensure_ascii=False)

    return str(metadata_path)
