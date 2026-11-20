"""Threshold adaptativo inferior e segmentação fuzzy para pré-processamento."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import median_filter, uniform_filter


def get_lower_threshold_config(config: dict[str, Any]) -> dict[str, Any]:
    """Retorna a configuração do limiar inferior com defaults compatíveis.

    Args:
        config (dict[str, Any]): Dicionário de configuração do pipeline (pode conter
            ``LOWER_THRESHOLD``).

    Returns:
        dict[str, Any]: Cópia normalizada da seção ``LOWER_THRESHOLD`` preenchida
            com defaults.
    """
    # Trabalha sobre uma cópia para não alterar a configuração efetiva da execução.
    lower_config = dict(config.get("LOWER_THRESHOLD", {}))
    lower_config.setdefault("method", "fixed")
    lower_config.setdefault("fixed_hu", config.get("MIN_THRESHOLD", -300))
    lower_config.setdefault("percentile", 5.0)
    lower_config.setdefault("clip_min_hu", -700.0)
    lower_config.setdefault("clip_max_hu", 500.0)
    return lower_config


def normalize_lower_threshold_method(method: Any) -> str:
    """Normaliza aliases dos métodos de limiar inferior.

    Args:
        method (Any): Nome ou alias do método (``fixed``/``constant`` ou
            ``percentile``/``low_percentile``).

    Returns:
        str: Identificador canônico (``"fixed"`` ou ``"percentile"``).
    """
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
    """Seleciona voxels finitos dentro da faixa HU configurada.

    Args:
        volume (np.ndarray): Volume 3D em unidades Hounsfield (HU).
        clip_min_hu (float | None): Limite inferior opcional em HU para filtrar
            amostras; ``None`` não aplica corte inferior.
        clip_max_hu (float | None): Limite superior opcional em HU para filtrar
            amostras; ``None`` não aplica corte superior.

    Returns:
        np.ndarray: Vetor 1D ``float32`` com os valores finitos dentro da faixa.
    """
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

    Suporta ``fixed`` (valor constante em HU) e ``percentile`` (percentil inferior
    dos voxels finitos dentro da faixa de recorte configurada).

    Args:
        volume (np.ndarray): Volume 3D em HU usado quando o método é adaptativo por
            percentil.
        config (dict[str, Any]): Configuração efetiva contendo ``LOWER_THRESHOLD`` e/ou
            ``MIN_THRESHOLD``.

    Returns:
        tuple[float, dict[str, Any]]: Tupla ``(min_threshold_hu, details)`` com o
            limiar calculado e metadados do método.
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

    # Extrai amostras HU válidas e retorna o piso fixo caso não haja voxels na faixa.
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


def normalize_threshold_mode(mode: Any) -> str:
    """Normaliza aliases de configuração para os modos de threshold suportados.

    Args:
        mode (Any): Nome ou alias do modo (``normal``/``classic``,
            ``fuzzy``/``object`` ou ``none``).

    Returns:
        str: Identificador canônico (``"normal"``, ``"fuzzy"`` ou ``"none"``).
    """
    # Permite nomes curtos na config sem espalhar aliases pelo pipeline.
    normalized = str(mode or "normal").strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    aliases = {
        "normal": "normal",
        "classic": "normal",
        "threshold": "normal",
        "fuzzy": "fuzzy",
        "object": "fuzzy",
        "none": "none",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Método de threshold inválido: {mode!r}. Use: {valid}.")
    return aliases[normalized]


def get_thresholding_config(config: dict[str, Any]) -> dict[str, Any]:
    """Retorna a seção ``THRESHOLDING`` com defaults do pipeline.

    Args:
        config (dict[str, Any]): Dicionário de configuração efetiva do pipeline.

    Returns:
        dict[str, Any]: Cópia da seção ``THRESHOLDING`` com ``method`` e ``fuzzy``
            preenchidos.
    """
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
    """Estima centros HU das três classes fuzzy (fundo mole, objeto e fundo denso).

    Args:
        volume (np.ndarray): Volume 3D em HU usado para estimar os percentis da imagem.
        min_hu (float): Piso de intensidade em HU que ancora o limite inferior do
            objeto.
        soft_margin_hu (float): Margem em HU abaixo de ``min_hu`` que define o centro
            de fundo mole.
        object_percentile (float): Percentil em ``[0, 100]`` usado como centro robusto
            da classe objeto.
        dense_percentile (float): Percentil em ``[0, 100]`` usado como centro robusto
            da classe densa.

    Returns:
        np.ndarray: Array ``float32`` ``[soft_center, object_center, dense_center]`` em
            ordem estritamente crescente.
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
        volume (np.ndarray): Volume 3D em HU usado para estimar e aplicar pertinências.
        min_hu (float): Limiar mínimo em HU que ancora a separação entre fundo mole e
            objeto.
        soft_margin_hu (float): Distância em HU abaixo de ``min_hu`` usada como centro
            de fundo mole.
        object_percentile (float): Percentil em ``[0, 100]`` usado como centro robusto
            da classe objeto.
        dense_percentile (float): Percentil em ``[0, 100]`` usado como centro robusto
            da classe densa.
        smooth_radius (int): Raio em voxels da janela cúbica de suavização das
            pertinências; ``0`` desativa.
        smooth_mode (str): Tipo de suavização local das pertinências (``"mean"`` ou
            ``"median"``).

    Returns:
        tuple[np.ndarray, dict[str, Any]]: Tupla ``(object_mask, metadata)`` com
            máscara booleana 3D e dicionário dos centros HU estimados.
    """
    # Estima centros HU e larguras das transições entre fundo mole, objeto e denso.
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

    # Seleciona apenas voxels acima de min_hu cujo maior grau de pertinência é objeto.
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
    """Executa o fuzzy threshold a partir da configuração completa.

    Args:
        volume (np.ndarray): Volume 3D em HU a limiarizar.
        config (dict[str, Any]): Configuração efetiva contendo ``MIN_THRESHOLD`` e a
            seção ``THRESHOLDING.fuzzy``.

    Returns:
        tuple[np.ndarray, dict[str, Any]]: Tupla ``(object_mask, metadata)`` com
            máscara booleana 3D e centros HU estimados.
    """
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


__all__ = [
    "estimate_fuzzy_centers",
    "fuzzy_threshold_from_config",
    "fuzzy_threshold_outputs",
    "get_lower_threshold_config",
    "get_thresholding_config",
    "normalize_lower_threshold_method",
    "normalize_threshold_mode",
    "resolve_lower_threshold",
]
