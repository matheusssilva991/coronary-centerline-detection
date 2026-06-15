"""Rotinas de segmentação de artérias por crescimento de região.

Este módulo inclui duas estratégias de crescimento de região para segmentação
de artérias coronárias a partir de volumes de vesselness.
"""

from collections import deque

import numpy as np
from typing import Any, Deque, Iterable, Optional, Sequence, Tuple
from numpy.typing import NDArray


NEIGHBORS_26 = [
    (dy, dx, dz)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dy == 0 and dx == 0 and dz == 0)
]


def _validate_seed(
    seed_point: Sequence[int], volume_shape: Sequence[int]
) -> Optional[Tuple[int, int, int]]:
    """Valida se a semente está dentro dos limites do volume."""
    sy, sx, sz = map(int, seed_point)
    height, width, depth = volume_shape

    if 0 <= sy < height and 0 <= sx < width and 0 <= sz < depth:
        return (sy, sx, sz)
    return None


def _calculate_adaptive_floor(
    count: int,
    min_start: float,
    min_end: float,
    switch_at_voxels: int,
    smooth_relaxation: bool,
) -> float:
    """Calcula o piso adaptativo usado na aceitação de vizinhos."""
    if switch_at_voxels <= 0:
        return min_end
    if count >= switch_at_voxels:
        return min_end

    if smooth_relaxation:
        progress = count / switch_at_voxels
        return min_start - (min_start - min_end) * progress
    return min_start


def _calculate_comparison_mean(
    comparison_window: int,
    use_running_mean: bool,
    current_value: float,
    running_sum: float,
    running_count: int,
    value_history: Optional[Deque[float]],
) -> float:
    """Calcula a média de referência para comparar novos voxels."""
    if comparison_window == 1:
        return current_value
    if use_running_mean:
        return running_sum / running_count
    return np.mean(value_history)


def _is_neighbor_acceptable(
    neighbor_val: float, comparison_mean: float, current_floor: float, threshold: float
) -> bool:
    """Verifica se um voxel vizinho atende aos critérios de inclusão."""
    if neighbor_val < current_floor:
        return False
    if abs(neighbor_val - comparison_mean) > threshold:
        return False
    return True


def _initialize_seed_region(
    vesselness_map: NDArray[Any],
    seeds: Iterable[Sequence[int]],
    min_vesselness: Optional[float] = None,
):
    """Inicializa máscara, visitados e fila a partir das sementes válidas."""
    mask = np.zeros_like(vesselness_map, dtype=bool)
    visited = np.zeros_like(vesselness_map, dtype=bool)
    queue = deque()

    initial_sum = 0.0
    initial_count = 0

    for seed in seeds:
        validated_seed = _validate_seed(seed, vesselness_map.shape)
        if validated_seed is None:
            continue

        sy, sx, sz = validated_seed
        seed_val = float(vesselness_map[sy, sx, sz])

        if min_vesselness is not None and seed_val < min_vesselness:
            continue

        queue.append((sy, sx, sz))
        visited[sy, sx, sz] = True
        mask[sy, sx, sz] = True

        initial_sum += seed_val
        initial_count += 1

    return mask, visited, queue, initial_sum, initial_count


def region_growing_segmentation(
    vesselness_map: NDArray[Any],
    seed_point: Sequence[int],
    threshold: Optional[float] = None,
    min_vesselness: Optional[float] = None,
    max_volume: int = 100000,
    relaxed_floor_factor: float = 0.40,
    switch_at_voxels: int = 1000,
    comparison_window: int = 1,
    smooth_relaxation: bool = False,
    verbose: bool = False,
) -> NDArray[Any]:
    """Segmenta vasos por crescimento de região com controle adaptativo.

    A região cresce a partir de uma semente no mapa de vesselness. Um vizinho é
    aceito quando mantém vesselness acima de um piso mínimo e é suficientemente
    parecido com o voxel/região de referência.
    """
    if vesselness_map.ndim != 3:
        raise ValueError(
            f"vesselness_map deve ser 3D, recebido shape={vesselness_map.shape}"
        )

    height, width, depth = vesselness_map.shape
    v_max, v_min = np.max(vesselness_map), np.min(vesselness_map)

    if threshold is None:
        threshold = (v_max - v_min) / 10
    if min_vesselness is None:
        min_vesselness = v_max * 0.05

    use_running_mean = False
    value_history = None

    if comparison_window == -1 or isinstance(comparison_window, str):
        comparison_window = -1
        use_running_mean = True
    elif comparison_window > 1:
        value_history = deque(maxlen=comparison_window)
    else:
        comparison_window = 1

    # O piso de vesselness pode relaxar ao longo do crescimento para evitar
    # parar cedo demais em trechos arteriais mais fracos.
    min_start = float(min_vesselness)
    min_end = min_start * relaxed_floor_factor

    # Seleciona e valida a semente inicial.
    validated_seed = _validate_seed(seed_point, vesselness_map.shape)
    if validated_seed is None:
        if verbose:
            print(f"Semente fora dos limites: {seed_point}")
        return np.zeros_like(vesselness_map, dtype=np.uint8)

    sy, sx, sz = validated_seed
    seed_val = vesselness_map[sy, sx, sz]

    if seed_val < min_start:
        if verbose:
            print(f"Semente abaixo do piso: {seed_val:.4f} < {min_start:.4f}")
        return np.zeros_like(vesselness_map, dtype=np.uint8)

    # Inicializa a fila de crescimento com a semente aceita.
    mask = np.zeros_like(vesselness_map, dtype=np.uint8)
    visited = np.zeros_like(vesselness_map, dtype=bool)
    queue = deque([(sy, sx, sz)])

    visited[sy, sx, sz] = True
    mask[sy, sx, sz] = 1
    count = 1

    running_sum = float(seed_val)
    running_count = 1

    if comparison_window > 1:
        value_history.append(seed_val)

    if verbose:
        window_desc = "todos" if use_running_mean else str(comparison_window)
        print(
            f"RG | Seed={seed_val:.4f} | Floor: {min_start:.4f}→{min_end:.4f} "
            f"@ {switch_at_voxels} | Window={window_desc}"
        )

    while queue:
        if count >= max_volume:
            if verbose:
                print(f"Volume máximo atingido: {max_volume}")
            break

        cy, cx, cz = queue.popleft()
        current_val = vesselness_map[cy, cx, cz]

        # Calcula a referência local usada para comparar o próximo vizinho.
        comparison_mean = _calculate_comparison_mean(
            comparison_window,
            use_running_mean,
            current_val,
            running_sum,
            running_count,
            value_history,
        )

        # Atualiza o piso mínimo de vesselness de acordo com o estágio do crescimento.
        current_floor = _calculate_adaptive_floor(
            count, min_start, min_end, switch_at_voxels, smooth_relaxation
        )

        for dy, dx, dz in NEIGHBORS_26:
            ny, nx, nz = cy + dy, cx + dx, cz + dz

            if not (0 <= ny < height and 0 <= nx < width and 0 <= nz < depth):
                continue
            if visited[ny, nx, nz]:
                continue

            neighbor_val = vesselness_map[ny, nx, nz]
            # Aceita o vizinho se ele respeita piso mínimo e similaridade local.
            if _is_neighbor_acceptable(
                neighbor_val, comparison_mean, current_floor, threshold
            ):
                mask[ny, nx, nz] = 1
                visited[ny, nx, nz] = True
                queue.append((ny, nx, nz))
                count += 1

                if use_running_mean:
                    running_sum += neighbor_val
                    running_count += 1
                elif comparison_window > 1:
                    value_history.append(neighbor_val)

    if verbose:
        print(f"Voxels segmentados: {count}")

    return mask


def region_growing_article(
    vesselness_map: NDArray[Any],
    seeds: Iterable[Sequence[int]],
    threshold: Optional[float] = None,
    min_vesselness: Optional[float] = None,
    max_volume: Optional[int] = None,
) -> NDArray[Any]:
    """Executa variante de crescimento de região baseada no método do artigo."""
    if vesselness_map.ndim != 3:
        raise ValueError(
            f"vesselness_map deve ser 3D, recebido shape={vesselness_map.shape}"
        )

    if threshold is None:
        threshold = (vesselness_map.max() - vesselness_map.min()) / 10.0

    if min_vesselness is not None:
        min_vesselness = float(min_vesselness)

    mask, visited, queue, current_sum, current_count = _initialize_seed_region(
        vesselness_map, seeds, min_vesselness
    )

    if current_count == 0:
        print("Nenhuma semente válida fornecida.")
        return mask

    region_mean = float(current_sum / current_count)
    print(
        f"Iniciando crescimento de região com {current_count} sementes. "
        f"Média inicial: {region_mean:.4f}"
    )

    dims = vesselness_map.shape

    while queue:
        if max_volume and current_count >= max_volume:
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

            if min_vesselness is not None and neighbor_val < min_vesselness:
                continue

            delta_vm = abs(neighbor_val - region_mean)
            if delta_vm < threshold:
                mask[ny, nx, nz] = True
                queue.append((ny, nx, nz))
                current_sum += neighbor_val
                current_count += 1
                region_mean = float(current_sum / current_count)

    print(
        f"Segmentação concluída. Voxels: {current_count}. "
        f"Média final: {region_mean:.4f}\n"
    )

    return mask


def normal_region_growing_from_ostia(
    vesselness_artery: NDArray[Any],
    ostia_left: Optional[Sequence[int]],
    ostia_right: Optional[Sequence[int]],
    config: dict[str, Any],
) -> NDArray[np.uint8]:
    """Executa o crescimento de região padrão separadamente por óstio.

    Cada óstio gera uma máscara independente; a máscara arterial final é a união
    binária das duas, evitando que uma falha em um lado apague o outro.
    """
    rg_config = config["REGION_GROWING"]
    params = {
        "threshold": (
            float(np.max(vesselness_artery)) - float(np.min(vesselness_artery))
        )
        / rg_config["threshold_divisor"],
        "max_volume": rg_config["max_volume"],
        "min_vesselness": float(np.max(vesselness_artery))
        * rg_config["min_vesselness_fraction"],
        "relaxed_floor_factor": rg_config["relaxed_floor_factor"],
        "switch_at_voxels": rg_config["switch_at_voxels"],
        "comparison_window": rg_config["comparison_window"],
        "smooth_relaxation": rg_config["smooth_relaxation"],
        "verbose": False,
    }
    # Cresce a artéria esquerda a partir do óstio esquerdo detectado.
    left_mask = (
        region_growing_segmentation(vesselness_artery, seed_point=ostia_left, **params)
        if ostia_left is not None
        else np.zeros_like(vesselness_artery, dtype=np.uint8)
    )
    # Cresce a artéria direita a partir do óstio direito detectado.
    right_mask = (
        region_growing_segmentation(vesselness_artery, seed_point=ostia_right, **params)
        if ostia_right is not None
        else np.zeros_like(vesselness_artery, dtype=np.uint8)
    )
    return ((left_mask > 0) | (right_mask > 0)).astype(np.uint8)


__all__ = [
    "NEIGHBORS_26",
    "normal_region_growing_from_ostia",
    "region_growing_article",
    "region_growing_segmentation",
]
