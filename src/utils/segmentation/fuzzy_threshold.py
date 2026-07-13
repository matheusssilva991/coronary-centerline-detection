"""Threshold fuzzy reutilizável no pipeline de segmentação.

Este módulo concentra a etapa fuzzy usada antes da detecção de aorta/óstios e
da segmentação arterial. O modelo divide cada voxel em três pertinências:
fundo mole, objeto e fundo denso. A máscara fuzzy mantém os voxels cuja classe
dominante é objeto.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import median_filter, uniform_filter


def normalize_threshold_mode(mode: Any) -> str:
    """Normaliza aliases de configuração para os modos suportados."""
    # Permite nomes curtos na config sem espalhar aliases pelo pipeline.
    normalized = str(mode or "normal").strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    aliases = {
        "normal": "normal",
        "classic": "normal",
        "threshold": "normal",
        "fuzzy": "fuzzy",
        "object": "fuzzy",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Método de threshold inválido: {mode!r}. Use: {valid}.")
    return aliases[normalized]


def get_thresholding_config(config: dict[str, Any]) -> dict[str, Any]:
    """Retorna a seção ``THRESHOLDING`` com defaults do pipeline."""
    # Mantém compatibilidade com configs antigas que ainda não tinham THRESHOLDING.
    thresholding = dict(config.get("THRESHOLDING", {}))
    thresholding.setdefault("method", "normal")
    thresholding.setdefault("fuzzy", {})
    return thresholding


def estimate_fuzzy_centers(
    volume: np.ndarray,
    min_hu: float,
    soft_margin_hu: float,
    object_percentile: float,
    dense_percentile: float,
) -> np.ndarray:
    """Estima centros HU das três classes fuzzy.

    O centro de fundo mole é ancorado abaixo de ``min_hu``. Os centros de objeto
    e fundo denso são estimados por percentis da própria imagem, considerando
    apenas voxels acima do limiar mínimo sempre que possível.
    """
    # Usa apenas valores finitos e, quando possível, ignora fundo abaixo do HU mínimo.
    values = np.asarray(volume, dtype=np.float32)
    values = values[np.isfinite(values)]
    valid = values[values >= min_hu]
    if valid.size == 0:
        valid = values

    # O fundo mole é fixado por parâmetro; objeto/denso se adaptam à imagem.
    soft_center = float(min_hu - soft_margin_hu)
    object_center = float(np.percentile(valid, object_percentile))
    dense_center = float(np.percentile(valid, dense_percentile))

    # Garante ordem estrita dos centros para evitar divisões degeneradas.
    object_center = max(object_center, min_hu + np.finfo(np.float32).eps)
    dense_center = max(dense_center, object_center + np.finfo(np.float32).eps)
    return np.array([soft_center, object_center, dense_center], dtype=np.float32)


def fuzzy_threshold_outputs(
    volume: np.ndarray,
    *,
    min_hu: float,
    soft_margin_hu: float = 160,
    object_percentile: float = 99.8,
    dense_percentile: float = 99.95,
    smooth_radius: int = 1,
    smooth_mode: str = "mean",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Gera máscara fuzzy de objeto e metadados dos centros.

    A pertinência de objeto é alta entre o fundo mole e o fundo denso. Após a
    normalização das três classes, a máscara seleciona voxels cuja classe
    dominante é objeto.

    Args:
        volume: Volume 3D usado para estimar e aplicar as pertinências.
        min_hu: Limiar mínimo que ancora a separação entre fundo mole e objeto.
        soft_margin_hu: Distância abaixo de ``min_hu`` usada como centro de
            fundo mole.
        object_percentile: Percentil usado como centro robusto da classe objeto.
        dense_percentile: Percentil usado como centro robusto da classe densa.
        smooth_radius: Raio da janela cúbica de suavização das pertinências.
        smooth_mode: Tipo de suavização local, ``mean`` ou ``median``.

    Returns:
        Máscara de objeto e metadados dos centros/pertinências.
    """
    centers = estimate_fuzzy_centers(
        volume,
        min_hu,
        soft_margin_hu,
        object_percentile,
        dense_percentile,
    )
    soft_center, object_center, dense_center = map(float, centers)
    soft_width = max(min_hu - soft_center, np.finfo(np.float32).eps)
    dense_width = max(dense_center - object_center, np.finfo(np.float32).eps)

    # Constrói rampas fuzzy: fundo mole cai, fundo denso sobe, objeto fica no meio.
    soft = np.clip((min_hu - volume) / soft_width, 0.0, 1.0)
    dense = np.clip((volume - object_center) / dense_width, 0.0, 1.0)
    obj = np.minimum(1.0 - soft, 1.0 - dense)

    # Normaliza as três pertinências para cada voxel somar 1.
    memberships = np.stack([soft, obj, dense], axis=0).astype(np.float32)
    memberships /= np.maximum(
        memberships.sum(axis=0, keepdims=True),
        np.finfo(np.float32).eps,
    )

    if smooth_radius > 0:
        # Suaviza cada classe separadamente e renormaliza a distribuição fuzzy.
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

    valid_min = volume >= min_hu
    object_mask = (np.argmax(memberships, axis=0) == 1) & valid_min

    return object_mask.astype(bool), {
        "soft_center_hu": soft_center,
        "object_center_hu": object_center,
        "dense_center_hu": dense_center,
        "fuzzy_mask_strategy": "object_argmax",
    }


def fuzzy_threshold_from_config(
    volume: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Executa o fuzzy threshold a partir da configuração completa."""
    # Isola os parâmetros do bloco THRESHOLDING.fuzzy e aplica defaults locais.
    thresholding = get_thresholding_config(config)
    fuzzy = dict(thresholding.get("fuzzy", {}))
    return fuzzy_threshold_outputs(
        np.asarray(volume, dtype=np.float32),
        min_hu=float(config.get("MIN_THRESHOLD", -300)),
        soft_margin_hu=float(fuzzy.get("soft_margin_hu", 100)),
        object_percentile=float(fuzzy.get("object_percentile", 99.8)),
        dense_percentile=float(fuzzy.get("dense_percentile", 99.96)),
        smooth_radius=int(fuzzy.get("smooth_radius", 0)),
        smooth_mode=str(fuzzy.get("smooth_mode", "mean")),
    )
