"""Segmentação coronária por fuzzy connectedness em volumes 3D.

Este módulo reúne a implementação reutilizável do experimento de fuzzy
connectedness para substituir/contrastar o crescimento de região arterial.
A abordagem usa os óstios detectados como sementes, combina similaridade de
vesselness e de intensidade HU para formar afinidades fuzzy entre voxels
vizinhos, propaga conectividade pelo critério max-min e gera uma máscara
binária a partir de um limiar de conectividade.

Fluxo principal:
1. normalizar o mapa de vesselness;
2. restringir a propagação a uma região candidata;
3. refinar sementes locais ao redor dos óstios;
4. propagar fuzzy connectedness;
5. pós-processar a máscara arterial com as mesmas operações do pipeline.
"""

from __future__ import annotations

import heapq
import math
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .pipeline_arteries import postprocess_artery_mask
from ..utils.normalization import normalize_vesselness


# =============================================================================
# Funções auxiliares
# =============================================================================


def neighbor_offsets_3d(neighborhood: int) -> tuple[tuple[int, int, int], ...]:
    """Retorna offsets 3D para vizinhança 6, 18 ou 26."""
    if neighborhood not in {6, 18, 26}:
        raise ValueError("neighborhood must be one of: 6, 18, 26")

    # Monta todos os deslocamentos ao redor do voxel central.
    offsets = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dy, dx, dz) == (0, 0, 0):
                    continue
                manhattan = abs(dy) + abs(dx) + abs(dz)
                if neighborhood == 6 and manhattan != 1:
                    continue
                if neighborhood == 18 and manhattan > 2:
                    continue
                offsets.append((dy, dx, dz))
    return tuple(offsets)


def vesselness_affinity(
    current_vessel: float,
    neighbor_vessel: float,
    mode: str = "mean",
    power: float = 1.0,
    floor: float = 0.0,
) -> float:
    """Calcula a afinidade de vesselness entre dois voxels vizinhos.

    Args:
        current_vessel: Valor normalizado de vesselness do voxel atual.
        neighbor_vessel: Valor normalizado de vesselness do voxel vizinho.
        mode: Estratégia de combinação (`mean`, `geometric_mean`,
            `max_relaxed`).
        power: Expoente aplicado à afinidade resultante.
        floor: Piso mínimo suave para evitar afinidades exatamente nulas.

    Returns:
        Afinidade no intervalo [0, 1].
    """
    current_vessel = float(np.clip(current_vessel, 0.0, 1.0))
    neighbor_vessel = float(np.clip(neighbor_vessel, 0.0, 1.0))

    # Mede compatibilidade entre o voxel atual e o vizinho no mapa de vasos.
    if mode == "geometric_mean":
        affinity = math.sqrt(current_vessel * neighbor_vessel)
    elif mode == "max_relaxed":
        affinity = (
            0.7 * max(current_vessel, neighbor_vessel)
            + 0.3 * min(current_vessel, neighbor_vessel)
        )
    else:
        affinity = 0.5 * (current_vessel + neighbor_vessel)

    if power != 1.0:
        affinity = float(np.power(np.clip(affinity, 0.0, 1.0), power))

    # O piso evita que pequenas falhas do vesselness quebrem totalmente o caminho.
    floor = float(np.clip(floor, 0.0, 1.0))
    affinity = floor + (1.0 - floor) * affinity
    return float(np.clip(affinity, 0.0, 1.0))


def edge_affinity(
    mu_vessel: float,
    mu_hu: float,
    mode: str = "min",
    vesselness_weight: float = 0.7,
) -> float:
    """Combina afinidade de vesselness e similaridade HU em uma aresta.

    A combinação `weighted_product` foi a melhor nos experimentos recentes,
    pois penaliza discordâncias sem ser tão rígida quanto o mínimo puro.
    """
    mu_vessel = float(np.clip(mu_vessel, 0.0, 1.0))
    mu_hu = float(np.clip(mu_hu, 0.0, 1.0))
    weight = float(np.clip(vesselness_weight, 0.0, 1.0))

    if mode == "weighted_sum":
        return float(np.clip(weight * mu_vessel + (1.0 - weight) * mu_hu, 0.0, 1.0))
    if mode == "weighted_product":
        # Produto ponderado preserva arestas boas em ambos os critérios.
        return float(np.power(mu_vessel, weight) * np.power(mu_hu, 1.0 - weight))
    return min(mu_vessel, mu_hu)


def valid_seed(
    seed: Sequence[int] | None,
    shape: Sequence[int],
    candidate_mask: NDArray[Any] | None = None,
) -> tuple[int, int, int] | None:
    """Valida uma semente e, opcionalmente, inclui o voxel na máscara candidata."""
    if seed is None:
        return None

    y, x, z = map(int, seed)
    if not (0 <= y < shape[0] and 0 <= x < shape[1] and 0 <= z < shape[2]):
        return None

    if candidate_mask is not None:
        candidate_mask[y, x, z] = True
    return (y, x, z)


def limit_candidate_mask_by_vesselness(
    candidate_mask: NDArray[Any],
    vesselness_norm: NDArray[Any],
    max_candidate_voxels: int | None,
    min_candidate_vesselness: float = 0.01,
) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    """Limita a propagação aos voxels candidatos com maior vesselness.

    Args:
        candidate_mask: Máscara binária da região onde a FC pode crescer.
        vesselness_norm: Vesselness já normalizado para [0, 1].
        max_candidate_voxels: Número máximo de voxels permitidos na região
            candidata. Se `None`, não limita.
        min_candidate_vesselness: Corte mínimo usado antes da limitação.

    Returns:
        Tupla com a máscara candidata final e metadados do corte aplicado.
    """
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    initial_voxels = int(candidate_mask.sum())

    if max_candidate_voxels is None or initial_voxels <= int(max_candidate_voxels):
        return candidate_mask, {
            "candidate_voxels_initial": initial_voxels,
            "candidate_voxels_final": initial_voxels,
            "candidate_vesselness_cutoff": float(min_candidate_vesselness),
        }

    values = np.asarray(vesselness_norm)[candidate_mask]
    kth = max(values.size - int(max_candidate_voxels), 0)
    cutoff = float(np.partition(values, kth)[kth])
    limited_mask = candidate_mask & (vesselness_norm >= cutoff)
    return limited_mask.astype(bool), {
        "candidate_voxels_initial": initial_voxels,
        "candidate_voxels_final": int(limited_mask.sum()),
        "candidate_vesselness_cutoff": cutoff,
    }


def collect_local_object_seeds(
    seeds: Iterable[Sequence[int] | None],
    vesselness_norm: NDArray[Any],
    candidate_mask: NDArray[Any],
    search_radius: int,
    max_seeds_per_ostium: int,
    min_seed_vesselness: float,
    min_seed_distance_voxels: float = 0.0,
) -> tuple[list[tuple[int, int, int]], dict[str, Any]]:
    """Seleciona sementes locais de alto vesselness ao redor dos óstios.

    A função procura em uma janela cúbica ao redor de cada óstio e mantém os
    melhores voxels candidatos. Isso reduz a dependência de uma única semente,
    especialmente quando o óstio detectado cai em borda ou em voxel ruidoso.
    """
    selected: list[tuple[int, int, int]] = []
    per_ostium_counts = []
    dims = vesselness_norm.shape
    radius = max(int(search_radius), 0)
    max_per_ostium = max(int(max_seeds_per_ostium), 1)
    min_score = float(min_seed_vesselness)
    min_distance = float(min_seed_distance_voxels)

    for seed in seeds:
        coord = valid_seed(seed, dims, candidate_mask=candidate_mask)
        if coord is None:
            per_ostium_counts.append(0)
            continue

        y, x, z = coord
        y0, y1 = max(0, y - radius), min(dims[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(dims[1], x + radius + 1)
        z0, z1 = max(0, z - radius), min(dims[2], z + radius + 1)

        # Busca as melhores sementes em uma vizinhança pequena do óstio.
        local_mask = candidate_mask[y0:y1, x0:x1, z0:z1]
        local_scores = vesselness_norm[y0:y1, x0:x1, z0:z1]
        candidate_indices = np.argwhere(local_mask & (local_scores >= min_score))
        if candidate_indices.size == 0:
            selected.append(coord)
            per_ostium_counts.append(1)
            continue

        # Ordena os candidatos por vesselness para iniciar no centro provável do vaso.
        candidate_scores = local_scores[tuple(candidate_indices.T)]
        order = np.argsort(candidate_scores)[::-1]
        local_selected: list[tuple[int, int, int]] = []

        for idx in order:
            ly, lx, lz = candidate_indices[idx]
            candidate = (int(y0 + ly), int(x0 + lx), int(z0 + lz))
            if min_distance > 0 and local_selected:
                candidate_arr = np.asarray(candidate, dtype=np.float32)
                selected_arr = np.asarray(local_selected, dtype=np.float32)
                distances = np.sqrt(np.sum((selected_arr - candidate_arr) ** 2, axis=1))
                if np.any(distances < min_distance):
                    continue

            local_selected.append(candidate)
            if len(local_selected) >= max_per_ostium:
                break

        selected.extend(local_selected)
        per_ostium_counts.append(len(local_selected))

    selected = list(dict.fromkeys(selected))
    return selected, {
        "object_seed_count": len(selected),
        "object_seed_counts_per_ostium": per_ostium_counts,
    }


def fuzzy_connectedness_map(
    image: NDArray[Any],
    vesselness: NDArray[Any],
    seeds: Iterable[Sequence[int] | None],
    *,
    sigma_hu: float,
    neighborhood: int = 26,
    candidate_mask: NDArray[Any] | None = None,
    max_processed_voxels: int | None = None,
    min_connectivity_to_process: float = 0.0,
    vesselness_affinity_mode: str = "mean",
    vesselness_power: float = 1.0,
    vesselness_floor: float = 0.0,
    edge_affinity_mode: str = "min",
    vesselness_weight: float = 0.7,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Propaga conectividade fuzzy max-min a partir das sementes do objeto.

    Cada voxel recebe a maior conectividade possível entre ele e alguma semente,
    onde a força de um caminho é o mínimo das afinidades ao longo do caminho. A
    fila de prioridade processa primeiro os voxels com maior conectividade, de
    modo semelhante a um crescimento best-first.

    Args:
        image: Volume 3D em HU ou imagem pré-processada usada na similaridade HU.
        vesselness: Mapa de vesselness arterial com o mesmo shape de `image`.
        seeds: Sementes do objeto em coordenadas `(y, x, z)`.
        sigma_hu: Escala da similaridade Gaussiana de intensidade.
        neighborhood: Conectividade espacial 6, 18 ou 26.
        candidate_mask: Região permitida para a propagação.
        max_processed_voxels: Limite de segurança para evitar expansão excessiva.
        min_connectivity_to_process: Menor conectividade que ainda entra na fila.
        vesselness_affinity_mode: Estratégia de afinidade baseada em vesselness.
        vesselness_power: Expoente aplicado à afinidade de vesselness.
        vesselness_floor: Piso suave da afinidade de vesselness.
        edge_affinity_mode: Como combinar vesselness e HU.
        vesselness_weight: Peso do vesselness na combinação ponderada.

    Returns:
        Mapa de conectividade e metadados da propagação.
    """
    if image.shape != vesselness.shape:
        raise ValueError(f"image and vesselness must have the same shape: {image.shape} vs {vesselness.shape}")
    if sigma_hu <= 0:
        raise ValueError("sigma_hu must be positive")

    image = np.asarray(image, dtype=np.float32)
    vesselness_norm = normalize_vesselness(vesselness)
    candidate_mask = (
        np.ones(image.shape, dtype=bool)
        if candidate_mask is None
        else np.asarray(candidate_mask, dtype=bool).copy()
    )
    connectivity = np.zeros(image.shape, dtype=np.float32)
    finalized = np.zeros(image.shape, dtype=bool)
    queue: list[tuple[float, tuple[int, int, int]]] = []

    # Inicializa a fila de prioridade com conectividade máxima nas sementes.
    valid_seeds = []
    for seed in seeds:
        coord = valid_seed(seed, image.shape, candidate_mask=candidate_mask)
        if coord is None:
            continue
        valid_seeds.append(coord)
        connectivity[coord] = 1.0
        heapq.heappush(queue, (-1.0, coord))

    if not valid_seeds:
        return connectivity, {
            "valid_seed_count": 0,
            "processed_voxels": 0,
            "max_connectivity": 0.0,
        }

    offsets = neighbor_offsets_3d(neighborhood)
    sigma_term = 2.0 * float(sigma_hu) ** 2
    dims = image.shape
    processed_voxels = 0

    while queue:
        neg_priority, current = heapq.heappop(queue)
        current_connectivity = -float(neg_priority)
        if finalized[current]:
            continue
        if current_connectivity < float(connectivity[current]) - 1e-7:
            continue

        finalized[current] = True
        processed_voxels += 1
        if max_processed_voxels is not None and processed_voxels > max_processed_voxels:
            break

        cy, cx, cz = current
        current_hu = float(image[current])
        current_vessel = float(vesselness_norm[current])

        # Expande primeiro pelos vizinhos com maior conectividade acumulada.
        for dy, dx, dz in offsets:
            ny, nx, nz = cy + dy, cx + dx, cz + dz
            if not (0 <= ny < dims[0] and 0 <= nx < dims[1] and 0 <= nz < dims[2]):
                continue

            neighbor = (ny, nx, nz)
            if finalized[neighbor] or not candidate_mask[neighbor]:
                continue

            # Afinidade local: compatibilidade de vaso + similaridade HU.
            mu_vessel = vesselness_affinity(
                current_vessel,
                float(vesselness_norm[neighbor]),
                mode=vesselness_affinity_mode,
                power=vesselness_power,
                floor=vesselness_floor,
            )
            hu_diff = current_hu - float(image[neighbor])
            mu_hu = math.exp(-((hu_diff * hu_diff) / sigma_term))
            affinity = edge_affinity(
                mu_vessel,
                mu_hu,
                mode=edge_affinity_mode,
                vesselness_weight=vesselness_weight,
            )
            # Critério max-min: a força do caminho é limitada pela pior aresta.
            new_connectivity = min(current_connectivity, affinity)
            if new_connectivity < float(min_connectivity_to_process):
                continue

            if new_connectivity > float(connectivity[neighbor]):
                connectivity[neighbor] = new_connectivity
                heapq.heappush(queue, (-new_connectivity, neighbor))

    return connectivity, {
        "valid_seed_count": len(valid_seeds),
        "processed_voxels": processed_voxels,
        "max_connectivity": float(connectivity.max()),
        "vesselness_affinity_mode": vesselness_affinity_mode,
        "vesselness_power": float(vesselness_power),
        "vesselness_floor": float(vesselness_floor),
        "edge_affinity_mode": edge_affinity_mode,
        "vesselness_weight": float(vesselness_weight),
    }


def build_background_seeds(
    candidate_mask: NDArray[Any],
    vesselness: NDArray[Any],
    max_count: int = 300,
    low_vesselness_percentile: float = 10.0,
) -> list[tuple[int, int, int]]:
    """Gera sementes simples de fundo a partir de regiões com baixo vesselness."""
    vesselness_norm = normalize_vesselness(vesselness)
    coords = np.argwhere(candidate_mask)
    if coords.size == 0:
        return []

    values = vesselness_norm[np.asarray(candidate_mask, dtype=bool)]
    cutoff = np.percentile(values, low_vesselness_percentile)
    low_coords = coords[values <= cutoff]
    if len(low_coords) == 0:
        return []

    if len(low_coords) > max_count:
        step = max(1, len(low_coords) // max_count)
        low_coords = low_coords[::step][:max_count]
    return [tuple(map(int, coord)) for coord in low_coords]


def fuzzy_connectedness_segmentation(
    image: NDArray[Any],
    vesselness: NDArray[Any],
    object_seeds: Iterable[Sequence[int] | None],
    *,
    alpha: float,
    sigma_hu: float,
    neighborhood: int,
    candidate_mask: NDArray[Any],
    background_seeds: Iterable[Sequence[int] | None] | None = None,
    max_processed_voxels: int | None = None,
    vesselness_affinity_mode: str = "mean",
    vesselness_power: float = 1.0,
    vesselness_floor: float = 0.0,
    edge_affinity_mode: str = "min",
    vesselness_weight: float = 0.7,
    mask_strategy: str = "alpha",
    connectivity_percentile: float | None = None,
) -> dict[str, Any]:
    """Executa fuzzy connectedness e converte o mapa em máscara binária.

    Se `background_seeds` for informado, a máscara é definida por competição
    entre conectividade do objeto e do fundo (`Ko > Kb`). Caso contrário, usa o
    limiar absoluto `alpha` ou, opcionalmente, um percentil das conectividades.

    Returns:
        Dicionário com:
        - `connectivity`: mapa de conectividade do objeto;
        - `background_connectivity`: mapa do fundo, quando usado;
        - `mask`: máscara binária antes do pós-processamento;
        - `details`: metadados da propagação e dos parâmetros efetivos.
    """
    background_seeds = list(background_seeds or [])
    # Calcula o mapa de conectividade a partir das sementes do objeto.
    object_connectivity, object_details = fuzzy_connectedness_map(
        image,
        vesselness,
        object_seeds,
        sigma_hu=sigma_hu,
        neighborhood=neighborhood,
        candidate_mask=candidate_mask,
        max_processed_voxels=max_processed_voxels,
        min_connectivity_to_process=0.0 if background_seeds else float(alpha),
        vesselness_affinity_mode=vesselness_affinity_mode,
        vesselness_power=vesselness_power,
        vesselness_floor=vesselness_floor,
        edge_affinity_mode=edge_affinity_mode,
        vesselness_weight=vesselness_weight,
    )

    background_connectivity = None
    background_details = None
    if background_seeds:
        # Modo competitivo: objeto vence quando Ko é maior que Kb.
        background_connectivity, background_details = fuzzy_connectedness_map(
            image,
            vesselness,
            background_seeds,
            sigma_hu=sigma_hu,
            neighborhood=neighborhood,
            candidate_mask=candidate_mask,
            max_processed_voxels=max_processed_voxels,
            min_connectivity_to_process=0.05,
            vesselness_affinity_mode=vesselness_affinity_mode,
            vesselness_power=vesselness_power,
            vesselness_floor=vesselness_floor,
            edge_affinity_mode=edge_affinity_mode,
            vesselness_weight=vesselness_weight,
        )
        mask = object_connectivity > background_connectivity
        effective_alpha = np.nan
    elif mask_strategy == "connectivity_percentile":
        # Alternativa experimental: escolhe o corte por percentil da conectividade.
        positive_values = object_connectivity[object_connectivity > 0]
        if positive_values.size == 0:
            effective_alpha = float(alpha)
        else:
            percentile = 70.0 if connectivity_percentile is None else float(connectivity_percentile)
            percentile = float(np.clip(percentile, 0.0, 100.0))
            effective_alpha = max(float(np.percentile(positive_values, percentile)), float(alpha))
        mask = object_connectivity >= effective_alpha
    else:
        effective_alpha = float(alpha)
        mask = object_connectivity >= effective_alpha

    return {
        "connectivity": object_connectivity,
        "background_connectivity": background_connectivity,
        "mask": mask.astype(np.uint8),
        "details": {
            **object_details,
            "alpha": float(alpha),
            "sigma_hu": float(sigma_hu),
            "neighborhood": int(neighborhood),
            "vesselness_affinity_mode": vesselness_affinity_mode,
            "vesselness_power": float(vesselness_power),
            "vesselness_floor": float(vesselness_floor),
            "edge_affinity_mode": edge_affinity_mode,
            "vesselness_weight": float(vesselness_weight),
            "mask_strategy": mask_strategy,
            "connectivity_percentile": connectivity_percentile,
            "effective_alpha": effective_alpha,
            "uses_background_seeds": bool(background_seeds),
            "background_seed_count": len(background_seeds),
            "background_processed_voxels": (
                None if background_details is None else background_details["processed_voxels"]
            ),
        },
    }


def segment_artery_fuzzy_connectedness(
    image: NDArray[Any],
    vesselness_artery: NDArray[Any],
    ostia_seeds: Iterable[Sequence[int] | None],
    lcc_mask: NDArray[Any],
    config: dict[str, Any],
    *,
    params: dict[str, Any],
    max_candidate_voxels: int | None = 500_000,
    max_processed_voxels: int | None = 500_000,
) -> dict[str, Any]:
    """Segmenta artérias com parâmetros externos de fuzzy connectedness.

    Esta é a função de mais alto nível do módulo. Ela recebe o volume já
    reduzido/pré-processado, o vesselness arterial, os óstios detectados e a
    máscara da maior componente conectada. Em seguida, monta a região candidata,
    refina as sementes, executa fuzzy connectedness e aplica o pós-processamento
    final.

    Args:
        image: Volume 3D usado na similaridade HU.
        vesselness_artery: Mapa de vesselness arterial.
        ostia_seeds: Óstios detectados em coordenadas `(y, x, z)`.
        lcc_mask: Máscara da maior componente conectada usada como limite
            anatômico/candidato.
        config: Configuração do pipeline, principalmente `POSTPROCESSING`.
        params: Parâmetros da abordagem. No momento, os valores de referência
            ficam no notebook experimental e futuramente podem ir para
            `pipeline_config.json`.
        max_candidate_voxels: Limite máximo da região candidata.
        max_processed_voxels: Limite máximo de voxels processados pela FC.

    Returns:
        Dicionário com máscara bruta, máscara pós-processada, mapa de
        conectividade e metadados úteis para análise.
    """
    params = dict(params)
    # Normaliza vesselness antes de definir região candidata e sementes.
    vesselness_norm = normalize_vesselness(vesselness_artery)
    candidate_min = float(params.get("candidate_min_vesselness", 0.02))
    # Restringe a FC à LCC e aos voxels com vesselness mínimo.
    candidate_mask = (np.asarray(lcc_mask) > 0) & (vesselness_norm >= candidate_min)
    candidate_mask, candidate_details = limit_candidate_mask_by_vesselness(
        candidate_mask,
        vesselness_norm,
        max_candidate_voxels,
        min_candidate_vesselness=candidate_min,
    )
    # Refina as sementes ao redor dos óstios detectados.
    object_seeds, seed_details = collect_local_object_seeds(
        ostia_seeds,
        vesselness_norm,
        candidate_mask,
        search_radius=int(params.get("seed_search_radius", 2)),
        max_seeds_per_ostium=int(params.get("max_seeds_per_ostium", 4)),
        min_seed_vesselness=float(params.get("seed_min_vesselness", 0.02)),
        min_seed_distance_voxels=float(params.get("min_seed_distance_voxels", 1.0)),
    )
    # Propaga conectividade fuzzy e gera a máscara bruta.
    fc_result = fuzzy_connectedness_segmentation(
        image,
        vesselness_artery,
        object_seeds,
        alpha=float(params["alpha"]),
        sigma_hu=float(params["sigma_hu"]),
        neighborhood=int(params["neighborhood"]),
        candidate_mask=candidate_mask,
        background_seeds=None,
        max_processed_voxels=max_processed_voxels,
        vesselness_affinity_mode=str(params.get("vesselness_affinity_mode", "geometric_mean")),
        vesselness_power=float(params.get("vesselness_power", 1.0)),
        vesselness_floor=float(params.get("vesselness_floor", 0.02)),
        edge_affinity_mode=str(params.get("edge_affinity_mode", "weighted_product")),
        vesselness_weight=float(params.get("vesselness_weight", 0.90)),
        mask_strategy=str(params.get("mask_strategy", "alpha")),
        connectivity_percentile=params.get("connectivity_percentile"),
    )
    raw_mask = fc_result["mask"].astype(np.uint8)
    # Aplica o mesmo pós-processamento morfológico da segmentação arterial.
    artery_mask = postprocess_artery_mask(
        raw_mask,
        config,
        closing_radius=params.get("postprocess_closing_radius"),
        dilation_radius=params.get("postprocess_dilation_radius"),
    )
    details = {
        **fc_result["details"],
        **candidate_details,
        **seed_details,
        "raw_artery_voxels": int(raw_mask.sum()),
        "artery_voxels": int(artery_mask.sum()),
    }
    return {
        "raw_mask": raw_mask,
        "artery_mask": artery_mask,
        "connectivity": fc_result["connectivity"],
        "details": details,
    }


__all__ = [
    "build_background_seeds",
    "collect_local_object_seeds",
    "edge_affinity",
    "fuzzy_connectedness_map",
    "fuzzy_connectedness_segmentation",
    "limit_candidate_mask_by_vesselness",
    "neighbor_offsets_3d",
    "segment_artery_fuzzy_connectedness",
    "valid_seed",
    "vesselness_affinity",
]
