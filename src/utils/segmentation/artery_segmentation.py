"""Crescimento de região para segmentação de artérias coronárias.

O módulo mantém apenas duas entradas públicas:

- ``normal_region_growing_from_ostia``: método usado pelo pipeline principal;
- ``region_growing_article``: variante simples baseada no método do artigo.

As demais funções são helpers privados usados para validar sementes, selecionar
sementes locais e controlar os critérios de aceitação do crescimento de região.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


NEIGHBORS_26 = tuple(
    (dy, dx, dz)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dy, dx, dz) != (0, 0, 0)
)


def _validate_seed(
    seed_point: Sequence[int] | None,
    volume_shape: Sequence[int],
) -> tuple[int, int, int] | None:
    """Retorna a semente como tupla se ela estiver dentro do volume."""
    if seed_point is None:
        return None

    sy, sx, sz = map(int, seed_point)
    height, width, depth = volume_shape
    if 0 <= sy < height and 0 <= sx < width and 0 <= sz < depth:
        return (sy, sx, sz)
    return None


def _adaptive_floor(
    count: int,
    min_start: float,
    min_end: float,
    switch_at_voxels: int,
    smooth_relaxation: bool,
) -> float:
    """Calcula o piso mínimo de vesselness no estágio atual do crescimento."""
    if switch_at_voxels <= 0 or count >= switch_at_voxels:
        return min_end
    if not smooth_relaxation:
        return min_start

    progress = count / switch_at_voxels
    return min_start - (min_start - min_end) * progress


def _comparison_mean(
    comparison_window: int,
    use_running_mean: bool,
    current_value: float,
    running_sum: float,
    running_count: int,
    value_history: deque[float] | None,
) -> float:
    """Define a referência usada para comparar o próximo voxel candidato."""
    if comparison_window == 1:
        return current_value
    if use_running_mean:
        return running_sum / max(running_count, 1)
    return float(np.mean(value_history))


def _neighbor_is_acceptable(
    neighbor_val: float,
    comparison_value: float,
    current_floor: float,
    threshold: float,
) -> bool:
    """Verifica piso de vesselness e similaridade com a região atual."""
    if neighbor_val < current_floor:
        return False
    return abs(neighbor_val - comparison_value) <= threshold


def _collect_seed_candidates_by_score(
    seed: Sequence[int] | None,
    score_map: NDArray[Any],
    search_radius: int,
    min_seed_score: float,
    max_candidates: int,
) -> list[tuple[int, int, int]]:
    """Seleciona os melhores voxels locais para iniciar o crescimento."""
    validated_seed = _validate_seed(seed, score_map.shape)
    if validated_seed is None:
        return []

    y, x, z = validated_seed
    radius = max(int(search_radius), 0)
    y0, y1 = max(0, y - radius), min(score_map.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(score_map.shape[1], x + radius + 1)
    z0, z1 = max(0, z - radius), min(score_map.shape[2], z + radius + 1)

    local_scores = score_map[y0:y1, x0:x1, z0:z1]
    candidate_mask = local_scores >= float(min_seed_score)
    if not np.any(candidate_mask):
        seed_score = float(score_map[validated_seed])
        return [validated_seed] if seed_score >= float(min_seed_score) else []

    candidate_indices = np.argwhere(candidate_mask)
    candidate_scores = local_scores[candidate_mask]
    order = np.argsort(candidate_scores)[::-1]

    selected = []
    for idx in order:
        ly, lx, lz = candidate_indices[idx]
        selected.append((int(y0 + ly), int(x0 + lx), int(z0 + lz)))
        if len(selected) >= max(1, int(max_candidates)):
            break
    return selected


def _region_growing_from_seeds(
    vesselness_map: NDArray[Any],
    seeds: Iterable[Sequence[int]],
    *,
    threshold: float,
    min_vesselness: float,
    max_volume: int,
    relaxed_floor_factor: float,
    switch_at_voxels: int,
    comparison_window: int,
    smooth_relaxation: bool,
    verbose: bool = False,
) -> NDArray[np.uint8]:
    """Cresce uma região usando uma ou mais sementes iniciais."""
    if vesselness_map.ndim != 3:
        raise ValueError(
            f"vesselness_map deve ser 3D, recebido shape={vesselness_map.shape}"
        )

    use_running_mean = False
    value_history: deque[float] | None = None
    if comparison_window == -1 or isinstance(comparison_window, str):
        comparison_window = -1
        use_running_mean = True
    elif comparison_window > 1:
        value_history = deque(maxlen=comparison_window)
    else:
        comparison_window = 1

    min_start = float(min_vesselness)
    min_end = min_start * float(relaxed_floor_factor)
    mask = np.zeros_like(vesselness_map, dtype=np.uint8)
    visited = np.zeros_like(vesselness_map, dtype=bool)
    queue: deque[tuple[int, int, int]] = deque()
    running_sum = 0.0
    running_count = 0

    # Inicializa a fila com as sementes válidas e acima do piso mínimo.
    for seed in seeds:
        coord = _validate_seed(seed, vesselness_map.shape)
        if coord is None or visited[coord]:
            continue

        seed_val = float(vesselness_map[coord])
        if seed_val < min_start:
            continue

        visited[coord] = True
        mask[coord] = 1
        queue.append(coord)
        running_sum += seed_val
        running_count += 1
        if value_history is not None:
            value_history.append(seed_val)

    if running_count == 0:
        if verbose:
            print("Nenhuma semente válida acima do piso de vesselness.")
        return mask

    count = running_count
    dims = vesselness_map.shape
    while queue:
        if count >= int(max_volume):
            if verbose:
                print(f"Volume máximo atingido: {max_volume}")
            break

        cy, cx, cz = queue.popleft()
        current_value = float(vesselness_map[cy, cx, cz])
        reference = _comparison_mean(
            comparison_window,
            use_running_mean,
            current_value,
            running_sum,
            running_count,
            value_history,
        )
        current_floor = _adaptive_floor(
            count,
            min_start,
            min_end,
            int(switch_at_voxels),
            bool(smooth_relaxation),
        )

        # Testa a vizinhança 26 para seguir ramos arteriais em 3D.
        for dy, dx, dz in NEIGHBORS_26:
            ny, nx, nz = cy + dy, cx + dx, cz + dz
            if not (0 <= ny < dims[0] and 0 <= nx < dims[1] and 0 <= nz < dims[2]):
                continue
            if visited[ny, nx, nz]:
                continue

            neighbor_value = float(vesselness_map[ny, nx, nz])
            if not _neighbor_is_acceptable(
                neighbor_value,
                reference,
                current_floor,
                float(threshold),
            ):
                continue

            visited[ny, nx, nz] = True
            mask[ny, nx, nz] = 1
            queue.append((ny, nx, nz))
            count += 1

            if use_running_mean:
                running_sum += neighbor_value
                running_count += 1
            elif value_history is not None:
                value_history.append(neighbor_value)

    if verbose:
        print(f"Voxels segmentados: {count}")
    return mask


def _initialize_article_region(
    vesselness_map: NDArray[Any],
    seeds: Iterable[Sequence[int]],
    min_vesselness: float | None,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], deque[tuple[int, int, int]], float, int]:
    """Inicializa a variante do artigo com múltiplas sementes."""
    mask = np.zeros_like(vesselness_map, dtype=bool)
    visited = np.zeros_like(vesselness_map, dtype=bool)
    queue: deque[tuple[int, int, int]] = deque()
    initial_sum = 0.0
    initial_count = 0

    for seed in seeds:
        coord = _validate_seed(seed, vesselness_map.shape)
        if coord is None:
            continue

        seed_val = float(vesselness_map[coord])
        if min_vesselness is not None and seed_val < float(min_vesselness):
            continue

        visited[coord] = True
        mask[coord] = True
        queue.append(coord)
        initial_sum += seed_val
        initial_count += 1

    return mask, visited, queue, initial_sum, initial_count


def region_growing_article(
    vesselness_map: NDArray[Any],
    seeds: Iterable[Sequence[int]],
    threshold: float | None = None,
    min_vesselness: float | None = None,
    max_volume: int | None = None,
) -> NDArray[np.bool_]:
    """Executa a variante de crescimento de região baseada no artigo."""
    if vesselness_map.ndim != 3:
        raise ValueError(
            f"vesselness_map deve ser 3D, recebido shape={vesselness_map.shape}"
        )

    if threshold is None:
        threshold = (float(vesselness_map.max()) - float(vesselness_map.min())) / 10.0

    mask, visited, queue, current_sum, current_count = _initialize_article_region(
        vesselness_map,
        seeds,
        None if min_vesselness is None else float(min_vesselness),
    )
    if current_count == 0:
        print("Nenhuma semente válida fornecida.")
        return mask

    region_mean = float(current_sum / current_count)
    dims = vesselness_map.shape
    while queue:
        if max_volume and current_count >= int(max_volume):
            print(f"Limite de voxels atingido: {max_volume}")
            break

        cy, cx, cz = queue.popleft()
        for dy, dx, dz in NEIGHBORS_26:
            ny, nx, nz = cy + dy, cx + dx, cz + dz
            if not (0 <= ny < dims[0] and 0 <= nx < dims[1] and 0 <= nz < dims[2]):
                continue
            if visited[ny, nx, nz]:
                continue

            visited[ny, nx, nz] = True
            neighbor_val = float(vesselness_map[ny, nx, nz])
            if min_vesselness is not None and neighbor_val < float(min_vesselness):
                continue

            if abs(neighbor_val - region_mean) < float(threshold):
                mask[ny, nx, nz] = True
                queue.append((ny, nx, nz))
                current_sum += neighbor_val
                current_count += 1
                region_mean = float(current_sum / current_count)

    return mask


def normal_region_growing_from_ostia(
    vesselness_artery: NDArray[Any],
    ostia_left: Sequence[int] | None,
    ostia_right: Sequence[int] | None,
    config: dict[str, Any],
) -> NDArray[np.uint8]:
    """Segmenta artérias coronárias a partir dos óstios detectados.

    O método usa o crescimento de região padrão do pipeline, com refinamento de
    semente e multiseed local quando esses parâmetros estão ligados na config.
    """
    if vesselness_artery.ndim != 3:
        raise ValueError(
            f"vesselness_artery deve ser 3D, recebido shape={vesselness_artery.shape}"
        )

    max_vesselness = float(np.max(vesselness_artery))
    min_vesselness_map = float(np.min(vesselness_artery))
    if max_vesselness <= 0:
        return np.zeros_like(vesselness_artery, dtype=np.uint8)

    rg_config = config["REGION_GROWING"]
    params = {
        "threshold": (max_vesselness - min_vesselness_map)
        / float(rg_config["threshold_divisor"]),
        "max_volume": int(rg_config["max_volume"]),
        "min_vesselness": max_vesselness
        * float(rg_config["min_vesselness_fraction"]),
        "relaxed_floor_factor": float(rg_config["relaxed_floor_factor"]),
        "switch_at_voxels": int(rg_config["switch_at_voxels"]),
        "comparison_window": rg_config["comparison_window"],
        "smooth_relaxation": bool(rg_config["smooth_relaxation"]),
        "verbose": bool(rg_config.get("verbose", False)),
    }

    use_seed_refinement = bool(rg_config.get("use_seed_refinement", False))
    use_multi_seed = bool(rg_config.get("use_multi_seed", False))
    grow_each_ostium_separately = bool(
        rg_config.get("grow_each_ostium_separately", True)
    )
    seed_min_score = max_vesselness * float(
        rg_config.get("seed_min_vesselness_fraction", 0.02)
    )
    seed_search_radius = int(rg_config.get("seed_search_radius", 2))
    seed_candidate_radius = int(rg_config.get("seed_candidate_radius", 1))
    max_seed_candidates = int(rg_config.get("max_seed_candidates", 1))

    def seeds_for_ostium(ostium: Sequence[int] | None) -> list[tuple[int, int, int]]:
        if ostium is None:
            return []
        if not (use_seed_refinement or use_multi_seed):
            seed = _validate_seed(ostium, vesselness_artery.shape)
            return [] if seed is None else [seed]

        radius = seed_candidate_radius if use_multi_seed else seed_search_radius
        candidate_count = max_seed_candidates if use_multi_seed else 1
        return _collect_seed_candidates_by_score(
            ostium,
            vesselness_artery,
            radius,
            seed_min_score,
            candidate_count,
        )

    seed_groups = [
        seeds_for_ostium(ostia_left),
        seeds_for_ostium(ostia_right),
    ]
    seed_groups = [group for group in seed_groups if group]
    if not seed_groups:
        return np.zeros_like(vesselness_artery, dtype=np.uint8)

    if grow_each_ostium_separately:
        combined = np.zeros_like(vesselness_artery, dtype=np.uint8)
        for seed_group in seed_groups:
            combined |= _region_growing_from_seeds(
                vesselness_artery,
                seed_group,
                **params,
            )
        return combined.astype(np.uint8)

    all_seeds = [seed for seed_group in seed_groups for seed in seed_group]
    return _region_growing_from_seeds(
        vesselness_artery,
        all_seeds,
        **params,
    ).astype(np.uint8)


__all__ = [
    "NEIGHBORS_26",
    "normal_region_growing_from_ostia",
    "region_growing_article",
]
