"""Limiar inferior adaptativo para o pré-processamento da segmentação."""

from __future__ import annotations

from typing import Any

import numpy as np


def get_lower_threshold_config(config: dict[str, Any]) -> dict[str, Any]:
    """Retorna a configuração do limiar inferior com defaults compatíveis."""
    lower_config = dict(config.get("LOWER_THRESHOLD", {}))
    lower_config.setdefault("method", "fixed")
    lower_config.setdefault("fixed_hu", config.get("MIN_THRESHOLD", -300))
    lower_config.setdefault("percentile", 5.0)
    lower_config.setdefault("clip_min_hu", -700.0)
    lower_config.setdefault("clip_max_hu", 500.0)
    lower_config.setdefault("object_percentile", 99.5)
    return lower_config


def normalize_lower_threshold_method(method: Any) -> str:
    """Normaliza aliases dos métodos de limiar inferior."""
    normalized = str(method or "fixed").strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    aliases = {
        "fixed": "fixed",
        "constant": "fixed",
        "percentile": "percentile",
        "low_percentile": "percentile",
        "object_relative": "object_relative_percentile",
        "object_relative_percentile": "object_relative_percentile",
        "relative": "object_relative_percentile",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(
            f"Método de limiar inferior inválido: {method!r}. Use: {valid}."
        )
    return aliases[normalized]


def _finite_clipped_values(
    volume: np.ndarray,
    clip_min_hu: float | None,
    clip_max_hu: float | None,
) -> np.ndarray:
    """Seleciona voxels finitos dentro da faixa HU configurada."""
    values = np.asarray(volume, dtype=np.float32)
    values = values[np.isfinite(values)]
    if clip_min_hu is not None:
        values = values[values >= float(clip_min_hu)]
    if clip_max_hu is not None:
        values = values[values <= float(clip_max_hu)]
    return values


def resolve_lower_threshold(
    volume: np.ndarray,
    config: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Calcula o limiar inferior efetivo para uma imagem.

    Métodos suportados:
    - ``fixed``: usa o valor histórico, normalmente -300 HU.
    - ``percentile``: percentil baixo dos voxels finitos em uma faixa HU.
    - ``object_relative_percentile``: percentil baixo dos voxels abaixo de um
      centro de objeto estimado por percentil alto.
    """
    lower_config = get_lower_threshold_config(config)
    method = normalize_lower_threshold_method(lower_config.get("method"))
    fixed_hu = float(lower_config.get("fixed_hu", config.get("MIN_THRESHOLD", -300)))

    details: dict[str, Any] = {
        "lower_threshold_method": method,
        "lower_threshold_percentile": None,
        "lower_threshold_object_percentile": None,
        "lower_threshold_object_center_hu": None,
        "lower_threshold_clip_min_hu": lower_config.get("clip_min_hu"),
        "lower_threshold_clip_max_hu": lower_config.get("clip_max_hu"),
    }

    if method == "fixed":
        details["min_threshold"] = fixed_hu
        return fixed_hu, details

    values = _finite_clipped_values(
        volume,
        lower_config.get("clip_min_hu"),
        lower_config.get("clip_max_hu"),
    )
    if values.size == 0:
        details["min_threshold"] = fixed_hu
        details["lower_threshold_fallback"] = "empty_valid_values"
        return fixed_hu, details

    percentile = float(lower_config.get("percentile", 5.0))
    details["lower_threshold_percentile"] = percentile

    if method == "object_relative_percentile":
        object_percentile = float(lower_config.get("object_percentile", 99.5))
        object_center = float(np.percentile(values, object_percentile))
        relative_values = values[values < object_center]
        if relative_values.size == 0:
            relative_values = values
        threshold = float(np.percentile(relative_values, percentile))
        details["lower_threshold_object_percentile"] = object_percentile
        details["lower_threshold_object_center_hu"] = object_center
    else:
        threshold = float(np.percentile(values, percentile))

    details["min_threshold"] = threshold
    return threshold, details


__all__ = [
    "get_lower_threshold_config",
    "normalize_lower_threshold_method",
    "resolve_lower_threshold",
]
