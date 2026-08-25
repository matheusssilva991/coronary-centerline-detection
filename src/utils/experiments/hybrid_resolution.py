"""Pipeline híbrido com localização em mid e segmentação em high resolution.

O fluxo preserva as etapas validadas do projeto. A aorta e os óstios são
localizados no volume reduzido; somente as coordenadas dos óstios atravessam a
fronteira de resolução. O volume high resolution é então pré-processado de
forma independente e usado na segmentação arterial e no cálculo das métricas.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..segmentation.ostia_detection import check_ostium_intersection
from ..segmentation.pipeline_arteries import segment_arteries_from_vesselness
from ..segmentation.pipeline_detection import (
    detect_and_evaluate_ostia,
    locate_aorta_circles,
    segment_aorta,
)
from ..segmentation.pipeline_orchestration import summarize_aorta_circles
from ..segmentation.pipeline_preprocessing import (
    compute_vesselness,
    load_and_preprocess_image,
)


VoxelCoordinate = tuple[int, int, int]


@dataclass
class HybridResolutionPreparedImage:
    """Dados compartilhados pelas variantes arteriais de uma imagem."""

    result: dict[str, Any]
    high_lcc: np.ndarray
    high_label_artery: np.ndarray
    high_ostia_left: VoxelCoordinate | None
    high_ostia_right: VoxelCoordinate | None


def rescale_voxel_coordinate(
    coordinate: Sequence[int | float] | None,
    source_factors: Sequence[int | float],
    target_factors: Sequence[int | float],
    target_shape: Sequence[int],
) -> VoxelCoordinate | None:
    """Converte uma coordenada ``(y, x, z)`` entre fatores de downsampling.

    A transformação preserva a posição no volume original:

    ``target = round(source * source_factor / target_factor)``.

    O resultado é limitado ao shape de destino para absorver diferenças de um
    voxel causadas pelo arredondamento de dimensões ímpares no downsampling.
    """
    if coordinate is None:
        return None

    source = np.asarray(coordinate, dtype=float).reshape(-1)
    source_scale = np.asarray(source_factors, dtype=float).reshape(-1)
    target_scale = np.asarray(target_factors, dtype=float).reshape(-1)
    shape = np.asarray(target_shape, dtype=int).reshape(-1)
    if any(values.size != 3 for values in (source, source_scale, target_scale, shape)):
        raise ValueError("Coordenada, fatores e shape devem possuir três elementos.")
    if np.any(source_scale <= 0) or np.any(target_scale <= 0):
        raise ValueError("Os fatores de downsampling devem ser positivos.")
    if np.any(shape <= 0):
        raise ValueError("O shape de destino deve ser positivo.")

    mapped = np.rint(source * source_scale / target_scale).astype(int)
    mapped = np.clip(mapped, 0, shape - 1)
    return tuple(int(value) for value in mapped)


def rescale_ostia_pair(
    ostia_left: Sequence[int | float] | None,
    ostia_right: Sequence[int | float] | None,
    source_factors: Sequence[int | float],
    target_factors: Sequence[int | float],
    target_shape: Sequence[int],
) -> tuple[VoxelCoordinate | None, VoxelCoordinate | None]:
    """Reescala os dois óstios usando a mesma transformação espacial."""
    return (
        rescale_voxel_coordinate(
            ostia_left, source_factors, target_factors, target_shape
        ),
        rescale_voxel_coordinate(
            ostia_right, source_factors, target_factors, target_shape
        ),
    )


def evaluate_ostia_coordinates(
    label_artery: Any,
    ostia_left: Sequence[int] | None,
    ostia_right: Sequence[int] | None,
    scaled_spacing: Sequence[float],
    tolerance_mm: float,
) -> dict[str, Any]:
    """Avalia um par de óstios já conhecido contra um label arterial."""
    dx, dy, dz = (float(value) for value in scaled_spacing[:3])
    metric_spacing = (dy, dx, dz)
    left_info = check_ostium_intersection(
        ostia_left,
        np.asarray(label_artery),
        spacing=metric_spacing,
        ostium_name="Óstio esquerdo",
    )
    right_info = check_ostium_intersection(
        ostia_right,
        np.asarray(label_artery),
        spacing=metric_spacing,
        ostium_name="Óstio direito",
    )

    pair_complete = ostia_left is not None and ostia_right is not None
    both_correct = bool(
        pair_complete and left_info["intersects"] and right_info["intersects"]
    )
    both_acceptable = bool(
        pair_complete
        and (left_info["intersects"] or left_info["physical_dist"] <= tolerance_mm)
        and (right_info["intersects"] or right_info["physical_dist"] <= tolerance_mm)
    )
    both_tolerable = both_acceptable and not both_correct
    if not pair_complete:
        status = "incomplete_pair"
    elif both_correct:
        status = "both_correct"
    elif both_tolerable:
        status = "both_tolerable"
    else:
        status = "found_but_wrong"

    return {
        "left_info": left_info,
        "right_info": right_info,
        "pair_complete": pair_complete,
        "both_correct": both_correct,
        "both_tolerable": both_tolerable,
        "ostia_success": both_correct or both_tolerable,
        "ostia_status": status,
    }


def _ostia_fields(prefix: str, evaluation: dict[str, Any]) -> dict[str, Any]:
    """Converte a avaliação dos óstios em colunas escalares para o CSV."""
    return {
        f"{prefix}_ostia_pair_complete": evaluation["pair_complete"],
        f"{prefix}_ostia_status": evaluation["ostia_status"],
        f"{prefix}_ostia_success": evaluation["ostia_success"],
        f"{prefix}_both_correct": evaluation["both_correct"],
        f"{prefix}_both_tolerable": evaluation["both_tolerable"],
        f"{prefix}_left_intersects": evaluation["left_info"]["intersects"],
        f"{prefix}_right_intersects": evaluation["right_info"]["intersects"],
        f"{prefix}_left_dist_mm": evaluation["left_info"]["physical_dist"],
        f"{prefix}_right_dist_mm": evaluation["right_info"]["physical_dist"],
    }


def _detected_ostia_evaluation(ostia_results: dict[str, Any]) -> dict[str, Any]:
    """Normaliza o retorno da detecção mid para o contrato de avaliação comum."""
    pair_complete = (
        ostia_results["ostia_left"] is not None
        and ostia_results["ostia_right"] is not None
    )
    both_correct = bool(ostia_results["both_correct"])
    both_tolerable = bool(ostia_results["both_tolerable"])
    if not pair_complete:
        status = "incomplete_pair"
    elif both_correct:
        status = "both_correct"
    elif both_tolerable:
        status = "both_tolerable"
    else:
        status = "found_but_wrong"
    return {
        "left_info": ostia_results["left_info"],
        "right_info": ostia_results["right_info"],
        "pair_complete": pair_complete,
        "both_correct": both_correct,
        "both_tolerable": both_tolerable,
        "ostia_success": both_correct or both_tolerable,
        "ostia_status": status,
    }


def _prepare_hybrid_resolution_image(
    img_id: int | str,
    mid_config: dict[str, Any],
    high_config: dict[str, Any],
    base_path: str | Path,
) -> HybridResolutionPreparedImage:
    """Executa uma vez as etapas compartilhadas pelas variantes high."""
    image_id = int(img_id)
    base_path = str(base_path)
    result: dict[str, Any] = {
        "IMG_ID": image_id,
        "pipeline_mode": "mid_ostia_high_artery",
        "mid_downscale_factors": tuple(mid_config["DOWNSCALE_FACTORS"]),
        "high_downscale_factors": tuple(high_config["DOWNSCALE_FACTORS"]),
        "artery_segmentation_method": high_config.get("ARTERY_SEGMENTATION", {}).get(
            "method", "region_growing"
        ),
        "mid_ostia_success": False,
        "high_ostia_success": False,
        "segmentation_attempted": False,
        "dice_artery": 0.0,
        "dice_artery_before_morphology": np.nan,
        "dice_artery_after_morphology": 0.0,
        "dice_artery_morphology_delta": np.nan,
        "error_stage": None,
        "error": None,
    }

    stage_started = time.perf_counter()
    mid_data = load_and_preprocess_image(str(image_id), base_path, mid_config)
    result["mid_preprocessing_seconds"] = time.perf_counter() - stage_started
    mid_lcc = mid_data["lcc_image"]
    mid_label = mid_data["label"]
    mid_spacing = mid_data["scaled_spacing"]
    mid_factors = mid_data["downscale_factors"]
    mid_details = mid_data.get("preprocessing_details", {})
    result.update(
        {
            "mid_image_shape": tuple(int(value) for value in mid_lcc.shape),
            "mid_min_threshold": mid_details.get("min_threshold"),
            "mid_max_threshold": mid_details.get("max_threshold"),
            "mid_threshold_voxels": mid_details.get("threshold_voxels"),
            "mid_lcc_voxels": mid_details.get("lcc_voxels"),
        }
    )

    stage_started = time.perf_counter()
    vesselness_ostia = compute_vesselness(
        mid_lcc,
        vesselness_config=mid_config["VESSELNESS_AORTA"],
        use_gpu=mid_config.get("USE_GPU", False),
    )
    detected_circles = locate_aorta_circles(
        mid_lcc,
        mid_factors,
        mid_spacing,
        mid_config["CIRCLE_DETECTION"],
    )
    circle_summary = summarize_aorta_circles(detected_circles, mid_lcc.shape[2])
    result.update({f"mid_{key}": value for key, value in circle_summary.items()})
    aorta_mask = segment_aorta(
        mid_lcc,
        detected_circles,
        mid_config["LEVEL_SET"],
        use_gpu=mid_config.get("USE_GPU", False),
    )
    result["mid_aorta_mask_voxels"] = int(np.sum(aorta_mask))
    ostia_results = detect_and_evaluate_ostia(
        aorta_mask,
        vesselness_ostia,
        mid_label,
        mid_spacing,
        mid_config,
    )
    result["mid_aorta_ostia_seconds"] = time.perf_counter() - stage_started
    mid_ostia_left = tuple(int(value) for value in ostia_results["ostia_left"])
    mid_ostia_right = (
        tuple(int(value) for value in ostia_results["ostia_right"])
        if ostia_results["ostia_right"] is not None
        else None
    )
    result["mid_ostia_left"] = mid_ostia_left
    result["mid_ostia_right"] = mid_ostia_right
    result.update(_ostia_fields("mid", _detected_ostia_evaluation(ostia_results)))

    # Os volumes mid deixam de ser necessários após obter as sementes.
    del mid_data, mid_lcc, mid_label, vesselness_ostia, aorta_mask

    stage_started = time.perf_counter()
    high_data = load_and_preprocess_image(str(image_id), base_path, high_config)
    result["high_preprocessing_seconds"] = time.perf_counter() - stage_started
    high_lcc = np.asarray(high_data["lcc_image"])
    high_label_artery = (np.asarray(high_data["label"]) == 1).astype(np.uint8)
    high_spacing = high_data["scaled_spacing"]
    high_factors = high_data["downscale_factors"]
    high_details = high_data.get("preprocessing_details", {})
    result.update(
        {
            "high_image_shape": tuple(int(value) for value in high_lcc.shape),
            "high_min_threshold": high_details.get("min_threshold"),
            "high_max_threshold": high_details.get("max_threshold"),
            "high_threshold_voxels": high_details.get("threshold_voxels"),
            "high_lcc_voxels": high_details.get("lcc_voxels"),
            "high_label_artery_voxels": int(np.sum(high_label_artery)),
        }
    )

    # Somente as coordenadas são transferidas; a máscara não é interpolada.
    high_ostia_left, high_ostia_right = rescale_ostia_pair(
        mid_ostia_left,
        mid_ostia_right,
        mid_factors,
        high_factors,
        high_lcc.shape,
    )
    result["high_ostia_left"] = high_ostia_left
    result["high_ostia_right"] = high_ostia_right
    result["ostia_coordinate_scale"] = tuple(
        float(source) / float(target)
        for source, target in zip(mid_factors, high_factors)
    )
    high_ostia_evaluation = evaluate_ostia_coordinates(
        high_label_artery,
        high_ostia_left,
        high_ostia_right,
        high_spacing,
        float(high_config["OSTIA_VALIDATION"]["distance_threshold_mm"]),
    )
    result.update(_ostia_fields("high", high_ostia_evaluation))
    return HybridResolutionPreparedImage(
        result=result,
        high_lcc=high_lcc,
        high_label_artery=high_label_artery,
        high_ostia_left=high_ostia_left,
        high_ostia_right=high_ostia_right,
    )


def _freeze_config_value(value: Any) -> Any:
    """Converte configurações aninhadas em uma chave imutável de cache."""
    if isinstance(value, dict):
        return tuple(
            (key, _freeze_config_value(item)) for key, item in sorted(value.items())
        )
    if isinstance(value, np.ndarray):
        return tuple(_freeze_config_value(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config_value(item) for item in value)
    return value


def process_hybrid_resolution_variants(
    img_id: int | str,
    mid_config: dict[str, Any],
    high_variants: dict[str, dict[str, Any]],
    base_path: str | Path,
) -> list[dict[str, Any]]:
    """Avalia variantes high reutilizando localização, volume e vesselness.

    Variantes com a mesma configuração ``VESSELNESS_ARTERY`` compartilham o
    mesmo mapa. Cada linha ainda registra o tempo estimado da configuração
    isolada, além do tempo incremental efetivamente gasto no sweep.
    """
    if not high_variants:
        raise ValueError("high_variants deve conter ao menos uma configuração.")

    image_started = time.perf_counter()
    reference_config = next(iter(high_variants.values()))
    try:
        prepared = _prepare_hybrid_resolution_image(
            img_id,
            mid_config,
            reference_config,
            base_path,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - image_started
        return [
            {
                "IMG_ID": int(img_id),
                "variant": variant,
                "pipeline_mode": "mid_ostia_high_artery",
                "segmentation_attempted": False,
                "dice_artery": 0.0,
                "error_stage": "shared_preparation",
                "error": f"{type(exc).__name__}: {exc}",
                "total_seconds": elapsed,
            }
            for variant in high_variants
        ]

    shared_seconds = time.perf_counter() - image_started
    vesselness_cache: dict[Any, tuple[np.ndarray, float]] = {}
    vesselness_keys = {
        variant: _freeze_config_value(config.get("VESSELNESS_ARTERY", {}))
        for variant, config in high_variants.items()
    }
    remaining_vesselness_uses = Counter(vesselness_keys.values())
    rows: list[dict[str, Any]] = []
    for variant, config in high_variants.items():
        row = dict(prepared.result)
        row["variant"] = variant
        row["segmentation_attempted"] = True
        row["shared_preparation_seconds"] = shared_seconds
        rg_config = config.get("REGION_GROWING", {})
        post_config = config.get("POSTPROCESSING", {})
        vesselness_config = config.get("VESSELNESS_ARTERY", {})
        row["rg_threshold_divisor"] = rg_config.get("threshold_divisor")
        row["rg_min_vesselness_fraction"] = rg_config.get(
            "min_vesselness_fraction"
        )
        row["post_closing_radius"] = post_config.get("closing_radius")
        row["post_dilation_radius"] = post_config.get("dilation_radius")
        row["artery_sigmas"] = tuple(vesselness_config.get("sigmas", ()))

        key = vesselness_keys[variant]
        vesselness_reused = key in vesselness_cache
        if not vesselness_reused:
            vesselness_started = time.perf_counter()
            vesselness = compute_vesselness(
                prepared.high_lcc,
                vesselness_config=vesselness_config,
                use_gpu=config.get("USE_GPU", False),
            )
            vesselness_cache[key] = (
                np.asarray(vesselness),
                time.perf_counter() - vesselness_started,
            )
        vesselness, vesselness_seconds = vesselness_cache[key]
        row["high_vesselness_seconds"] = vesselness_seconds
        row["high_vesselness_reused"] = vesselness_reused

        segmentation_started = time.perf_counter()
        try:
            artery_results = segment_arteries_from_vesselness(
                prepared.high_lcc,
                prepared.high_label_artery,
                vesselness,
                prepared.high_ostia_left,
                prepared.high_ostia_right,
                config,
            )
            row["high_artery_segmentation_seconds"] = (
                time.perf_counter() - segmentation_started
            )
            for name, value in artery_results.items():
                if name not in {"artery_mask", "raw_artery_mask"}:
                    row[name] = value
        except Exception as exc:
            row["error_stage"] = "high_artery_segmentation"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["high_artery_segmentation_seconds"] = (
                time.perf_counter() - segmentation_started
            )

        row["total_seconds"] = (
            shared_seconds
            + vesselness_seconds
            + row["high_artery_segmentation_seconds"]
        )
        row["incremental_sweep_seconds"] = (
            row["high_artery_segmentation_seconds"]
            + (0.0 if vesselness_reused else vesselness_seconds)
        )
        rows.append(row)
        remaining_vesselness_uses[key] -= 1
        if remaining_vesselness_uses[key] == 0:
            # Evita manter dois mapas high volumosos quando os sigmas mudam.
            vesselness_cache.pop(key, None)
    return rows


def process_hybrid_resolution_image(
    img_id: int | str,
    mid_config: dict[str, Any],
    high_config: dict[str, Any],
    base_path: str | Path,
) -> dict[str, Any]:
    """Localiza os óstios em mid e segmenta as artérias em high resolution."""
    return process_hybrid_resolution_variants(
        img_id,
        mid_config,
        {"baseline_high_scaled": high_config},
        base_path,
    )[0]


__all__ = [
    "evaluate_ostia_coordinates",
    "process_hybrid_resolution_image",
    "process_hybrid_resolution_variants",
    "rescale_ostia_pair",
    "rescale_voxel_coordinate",
]
