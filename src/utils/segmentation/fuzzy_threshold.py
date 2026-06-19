"""Threshold fuzzy e ponderação contextual reutilizáveis no pipeline.

O mesmo modelo fuzzy contextual pode ser usado de duas formas:

- ``contextual_object``: mantém apenas voxels cuja maior pertinência é a classe
  de objeto, substituindo o threshold superior por percentis fuzzy;
- ``contextual_apply_to``: usa a pertinência contextual como mapa de peso para
  penalizar vesselness em voxels densos ou pouco compatíveis com artéria.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import median_filter, uniform_filter

from ..processing import largest_connected_component


def fuzzy_trapezoid_threshold(
    volume: np.ndarray,
    min_hu: float,
    max_hu: float,
    margin_hu: float,
) -> np.ndarray:
    """Retorna a pertinência fuzzy para uma faixa HU trapezoidal.

    Voxels dentro de `[min_hu, max_hu]` recebem pertinência alta. A margem cria
    transições lineares nas bordas para evitar um corte totalmente rígido.
    """
    if margin_hu <= 0:
        return ((volume >= min_hu) & (volume <= max_hu)).astype(np.float32)

    rising = (volume - (min_hu - margin_hu)) / margin_hu
    falling = ((max_hu + margin_hu) - volume) / margin_hu
    membership = np.minimum(rising, falling)
    return np.clip(membership, 0.0, 1.0).astype(np.float32)


def build_lcc_image_from_mask(
    volume: np.ndarray,
    mask: np.ndarray,
    offset: float,
    per_slice: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Replica a maior componente conectada do pipeline usando máscara externa."""
    thresholded = np.zeros_like(volume, dtype=np.float32)
    thresholded[mask] = volume[mask] + offset

    if per_slice:
        # Mantém a mesma lógica do pipeline: LCC por fatia para reduzir vazamentos.
        lcc_image = np.zeros_like(thresholded, dtype=np.float32)
        lcc_mask = np.zeros_like(mask, dtype=bool)
        for z_idx in range(thresholded.shape[2]):
            lcc_slice, lcc_mask_slice = largest_connected_component(
                thresholded[:, :, z_idx], mask[:, :, z_idx]
            )
            lcc_image[:, :, z_idx] = lcc_slice
            lcc_mask[:, :, z_idx] = lcc_mask_slice
    else:
        # Alternativa 3D global para experimentos específicos.
        lcc_image, lcc_mask = largest_connected_component(thresholded, mask)

    return (lcc_image - offset).astype(np.float32), lcc_mask.astype(bool)


def normalize_threshold_mode(mode: Any) -> str:
    """Normaliza o método de threshold selecionado por config/CLI."""
    normalized = str(mode or "normal").strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    aliases = {
        "normal": "normal",
        "classic": "normal",
        "threshold": "normal",
        "fuzzy": "contextual_object",
        "contextual": "contextual_object",
        "contextual_3class": "contextual_object",
        "contextual_object": "contextual_object",
        "object": "contextual_object",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Método de threshold inválido: {mode!r}. Use: {valid}.")
    return aliases[normalized]


def normalize_contextual_apply_to(value: Any) -> str:
    """Normaliza a etapa onde o mapa contextual deve ponderar vesselness."""
    normalized = str(value or "none").strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    aliases = {
        "none": "none",
        "off": "none",
        "false": "none",
        "no": "none",
        "artery": "artery",
        "arteries": "artery",
        "ostia": "ostia",
        "ostios": "ostia",
        "both": "both",
        "all": "both",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(
            f"Alvo contextual inválido: {value!r}. Use: {valid}."
        )
    return aliases[normalized]


def get_thresholding_config(config: dict[str, Any]) -> dict[str, Any]:
    """Retorna a config de threshold com defaults compatíveis com o pipeline."""
    thresholding = dict(config.get("THRESHOLDING", {}))
    thresholding.setdefault("method", "normal")
    thresholding.setdefault("contextual_apply_to", "none")
    thresholding.setdefault("contextual", {})
    return thresholding


def estimate_contextual_centers(
    volume: np.ndarray,
    min_hu: float,
    soft_margin_hu: float,
    object_percentile: float,
    dense_percentile: float,
) -> np.ndarray:
    """Estima centros HU para fundo mole, objeto e fundo denso."""
    values = np.asarray(volume, dtype=np.float32)
    values = values[np.isfinite(values)]
    valid = values[values >= min_hu]
    if valid.size == 0:
        valid = values

    soft_center = float(min_hu - soft_margin_hu)
    object_center = float(np.percentile(valid, object_percentile))
    dense_center = float(np.percentile(valid, dense_percentile))
    object_center = max(object_center, min_hu + np.finfo(np.float32).eps)
    dense_center = max(dense_center, object_center + np.finfo(np.float32).eps)
    return np.array([soft_center, object_center, dense_center], dtype=np.float32)


def contextual_fuzzy_outputs(
    volume: np.ndarray,
    *,
    min_hu: float,
    soft_margin_hu: float = 160,
    object_percentile: float = 99.8,
    dense_percentile: float = 99.95,
    smooth_radius: int = 1,
    smooth_mode: str = "mean",
    weight_floor: float = 0.15,
    dense_power: float = 2.0,
    weight_mode: str = "dense_only",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Gera máscara fuzzy de objeto e mapa de peso contextual.

    A classe de objeto fica entre fundo mole (abaixo do threshold HU mínimo) e
    fundo denso (percentis altos). A agregação local suaviza pertinências para
    reduzir decisões isoladas por voxel.
    """
    centers = estimate_contextual_centers(
        volume,
        min_hu,
        soft_margin_hu,
        object_percentile,
        dense_percentile,
    )
    soft_center, object_center, dense_center = map(float, centers)
    soft_width = max(min_hu - soft_center, np.finfo(np.float32).eps)
    dense_width = max(dense_center - object_center, np.finfo(np.float32).eps)

    soft = np.clip((min_hu - volume) / soft_width, 0.0, 1.0)
    dense = np.clip((volume - object_center) / dense_width, 0.0, 1.0)
    obj = np.minimum(1.0 - soft, 1.0 - dense)
    memberships = np.stack([soft, obj, dense], axis=0).astype(np.float32)
    memberships /= np.maximum(
        memberships.sum(axis=0, keepdims=True),
        np.finfo(np.float32).eps,
    )

    if smooth_radius > 0:
        size = 2 * int(smooth_radius) + 1
        aggregated = np.empty_like(memberships)
        for idx in range(memberships.shape[0]):
            if smooth_mode == "median":
                aggregated[idx] = median_filter(memberships[idx], size=size)
            else:
                aggregated[idx] = uniform_filter(memberships[idx], size=size)
        memberships = aggregated / np.maximum(
            aggregated.sum(axis=0, keepdims=True),
            np.finfo(np.float32).eps,
        )

    object_membership = memberships[1]
    dense_membership = memberships[2]
    object_mask = (np.argmax(memberships, axis=0) == 1) & (volume >= min_hu)
    if weight_mode == "dense_only":
        raw_weight = 1.0 - np.power(dense_membership, dense_power)
    else:
        raw_weight = object_membership * np.power(1.0 - dense_membership, dense_power)
    weight = weight_floor + (1.0 - weight_floor) * raw_weight
    weight = np.clip(weight, weight_floor, 1.0).astype(np.float32)

    return object_mask.astype(bool), weight, {
        "soft_center_hu": soft_center,
        "object_center_hu": object_center,
        "dense_center_hu": dense_center,
        "mean_contextual_weight": float(weight.mean()),
        "min_contextual_weight": float(weight.min()),
        "max_contextual_weight": float(weight.max()),
    }


def contextual_fuzzy_from_config(
    volume: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Executa o fuzzy contextual usando a seção ``THRESHOLDING`` da config."""
    thresholding = get_thresholding_config(config)
    contextual = dict(thresholding.get("contextual", {}))
    return contextual_fuzzy_outputs(
        np.asarray(volume, dtype=np.float32),
        min_hu=float(config.get("MIN_THRESHOLD", -300)),
        soft_margin_hu=float(contextual.get("soft_margin_hu", 160)),
        object_percentile=float(contextual.get("object_percentile", 99.8)),
        dense_percentile=float(contextual.get("dense_percentile", 99.95)),
        smooth_radius=int(contextual.get("smooth_radius", 1)),
        smooth_mode=str(contextual.get("smooth_mode", "mean")),
        weight_floor=float(contextual.get("weight_floor", 0.15)),
        dense_power=float(contextual.get("dense_power", 2.0)),
        weight_mode=str(contextual.get("weight_mode", "dense_only")),
    )


def maybe_apply_contextual_weight(
    vesselness: np.ndarray,
    weight_map: np.ndarray | None,
    apply_to: Any,
    stage: str,
) -> np.ndarray:
    """Pondera vesselness com o mapa contextual quando a etapa foi selecionada."""
    target = normalize_contextual_apply_to(apply_to)
    if target not in {stage, "both"} or weight_map is None:
        return vesselness
    if vesselness.shape != weight_map.shape:
        raise ValueError(
            f"Mapa contextual incompatível: {vesselness.shape} vs {weight_map.shape}"
        )
    return (np.asarray(vesselness) * np.asarray(weight_map)).astype(
        np.asarray(vesselness).dtype,
        copy=False,
    )
