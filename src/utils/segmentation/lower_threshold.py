"""Limiar inferior adaptativo para o pré-processamento da segmentação."""

from __future__ import annotations

from typing import Any

import numpy as np


def get_lower_threshold_config(config: dict[str, Any]) -> dict[str, Any]:
    """Retorna a configuração do limiar inferior com defaults compatíveis."""
    # Trabalha sobre uma cópia para não alterar a configuração efetiva da execução.
    lower_config = dict(config.get("LOWER_THRESHOLD", {}))
    lower_config.setdefault("method", "fixed")
    lower_config.setdefault("fixed_hu", config.get("MIN_THRESHOLD", -300))
    lower_config.setdefault("percentile", 5.0)
    lower_config.setdefault("clip_min_hu", -700.0)
    lower_config.setdefault("clip_max_hu", 500.0)
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
    # Remove NaN/inf antes de restringir a distribuição à faixa HU relevante.
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
    """
    lower_config = get_lower_threshold_config(config)
    method = normalize_lower_threshold_method(lower_config.get("method"))
    fixed_hu = float(lower_config.get("fixed_hu", config.get("MIN_THRESHOLD", -300)))

    details: dict[str, Any] = {
        "lower_threshold_method": method,
        "lower_threshold_percentile": None,
        "lower_threshold_clip_min_hu": lower_config.get("clip_min_hu"),
        "lower_threshold_clip_max_hu": lower_config.get("clip_max_hu"),
    }

    # O método fixo preserva o comportamento histórico de -300 HU.
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

    # No modo adaptativo, cada exame obtém seu próprio piso pela distribuição HU.
    threshold = float(np.percentile(values, percentile))

    details["min_threshold"] = threshold
    return threshold, details


__all__ = [
    "get_lower_threshold_config",
    "normalize_lower_threshold_method",
    "resolve_lower_threshold",
]
