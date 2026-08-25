"""Utilitários para detecção de óstios coronários na superfície da aorta.

Este módulo fornece funções auxiliares para:
- extrair a superfície da aorta,
- selecionar candidatos com alto vesselness,
- validar restrições anatômicas,
- classificar óstios esquerdo/direito,
- validar interseção do óstio com máscaras arteriais.
"""

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    maximum_filter,
    percentile_filter,
    uniform_filter,
)
from skimage.morphology import ball
from typing import Any, Optional, Sequence, Tuple, cast
from numpy.typing import ArrayLike, NDArray

from ..processing.binary_operations import binary_erosion


def _validate_coordinates(coords: ArrayLike, volume_shape: Sequence[int]) -> bool:
    """Valida se uma coordenada (y, x, z) está dentro dos limites do volume."""
    coords_array = np.asarray(coords).ravel()
    if coords_array.size != 3:
        raise ValueError(f"Coordenada deve ter três elementos: {coords_array}")
    y, x, z = (int(coords_array[index]) for index in range(3))
    height, width, depth = volume_shape

    if y < 0 or x < 0 or z < 0 or y >= height or x >= width or z >= depth:
        raise ValueError(
            f"Coordenadas fora dos limites do volume: "
            f"(y={y}, x={x}, z={z}), shape={volume_shape}"
        )
    return True


def _extract_lower_region(
    surface_mask: NDArray[Any], lower_fraction: float = 0.3
) -> Tuple[NDArray[Any], int, int]:
    """Extrai a região inferior em z da superfície da aorta onde os óstios são esperados."""
    if not 0 < lower_fraction <= 1:
        raise ValueError("lower_fraction deve estar no intervalo (0, 1]")

    z_indices = np.where(np.any(surface_mask, axis=(0, 1)))[0]
    if len(z_indices) == 0:
        raise ValueError("Nenhuma superfície de aorta encontrada!")

    # Os óstios são buscados na porção inicial/inferior da aorta segmentada.
    z_min, z_max = z_indices.min(), z_indices.max()
    z_stop = z_min + int((z_max - z_min) * lower_fraction)
    z_stop = min(z_max + 1, max(z_min + 1, z_stop))

    lower_region_mask = np.zeros_like(surface_mask)
    lower_region_mask[:, :, z_min:z_stop] = surface_mask[:, :, z_min:z_stop]

    return lower_region_mask, z_min, z_max


def _get_top_candidates(
    surface_mask: NDArray[Any],
    score_map: NDArray[Any],
    top_n: int = 50,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    suppression_radius_mm: float = 0.0,
) -> NDArray[Any]:
    """Retorna candidatos ordenados, opcionalmente limitados a máximos locais."""
    if top_n <= 0:
        raise ValueError("top_n deve ser maior que 0")

    candidate_mask = surface_mask > 0
    if suppression_radius_mm > 0:
        # Mantém somente máximos locais fisicamente separados para evitar que
        # milhares de voxels da mesma região dominem a lista de candidatos.
        radii = np.maximum(
            1,
            np.ceil(suppression_radius_mm / np.asarray(spacing, dtype=float)).astype(
                int
            ),
        )
        window = tuple(int(2 * radius + 1) for radius in radii)
        local_maximum = maximum_filter(score_map, size=window, mode="nearest")
        candidate_mask &= score_map >= local_maximum
        positive_candidates = candidate_mask & (score_map > 0)
        if np.any(positive_candidates):
            candidate_mask = positive_candidates

    surface_coords = np.argwhere(candidate_mask)
    if len(surface_coords) == 0:
        raise ValueError("Nenhum voxel encontrado na superfície!")

    # Quanto maior o vesselness na superfície, mais provável o ponto de óstio.
    surface_values = score_map[candidate_mask]
    sorted_indices = np.argsort(surface_values)[::-1][:top_n]
    return surface_coords[sorted_indices]


def _candidate_score_map(
    aorta_mask: NDArray[Any],
    vesselness_map: NDArray[Any],
    mode: str,
    radius: int,
    local_percentile: float,
    point_weight: float,
    evaluation_mask: Optional[NDArray[Any]] = None,
) -> NDArray[np.float32]:
    """Calcula o score pontual, local ou externo usado para ordenar candidatos."""
    vesselness = np.asarray(vesselness_map, dtype=np.float32)
    if mode == "voxel":
        return vesselness
    if radius < 1:
        raise ValueError("candidate_score_radius deve ser >= 1")

    filter_size = 2 * int(radius) + 1
    if mode == "local_mean":
        return np.asarray(
            uniform_filter(vesselness, size=filter_size, mode="nearest"),
            dtype=np.float32,
        )
    if mode == "robust_percentile":
        if not 0 <= local_percentile <= 100:
            raise ValueError("candidate_local_percentile deve estar em [0, 100]")
        if not 0 <= point_weight <= 1:
            raise ValueError("candidate_point_weight deve estar em [0, 1]")
        if evaluation_mask is None or not np.any(evaluation_mask):
            evaluation_mask = np.ones_like(vesselness, dtype=bool)
        coords = np.argwhere(evaluation_mask)
        lower = np.maximum(coords.min(axis=0) - radius, 0)
        upper = np.minimum(coords.max(axis=0) + radius + 1, vesselness.shape)
        slices = tuple(
            slice(int(start), int(stop)) for start, stop in zip(lower, upper)
        )
        vesselness_roi = vesselness[slices]
        local_score_roi = percentile_filter(
            vesselness_roi,
            percentile=local_percentile,
            size=filter_size,
            mode="nearest",
        )
        score = np.zeros_like(vesselness)
        score[slices] = (
            point_weight * vesselness_roi + (1.0 - point_weight) * local_score_roi
        ).astype(np.float32)
        return score
    if mode == "external_mean":
        outside = (~np.asarray(aorta_mask, dtype=bool)).astype(np.float32)
        weighted_sum = uniform_filter(
            vesselness * outside,
            size=filter_size,
            mode="nearest",
        )
        outside_fraction = uniform_filter(outside, size=filter_size, mode="nearest")
        return np.divide(
            weighted_sum,
            outside_fraction,
            out=np.zeros_like(weighted_sum),
            where=outside_fraction > 0,
        )
    raise ValueError(
        "candidate_score_mode deve ser 'voxel', 'local_mean', "
        "'external_mean' ou 'robust_percentile'"
    )


def _validate_ostium_pair(
    ostium_1: ArrayLike,
    ostium_2: ArrayLike,
    min_center_dist: float,
    max_z_diff_mm: float,
    min_lateral_sep: float,
    spacing: Sequence[float],
    distance_mode: str,
) -> bool:
    """Verifica restrições anatômicas para um par candidato de óstios."""
    first = np.asarray(ostium_1, dtype=float)
    second = np.asarray(ostium_2, dtype=float)
    delta = first - second
    if distance_mode == "voxel_xyz":
        dist = np.linalg.norm(delta)
    elif distance_mode == "physical_xy":
        dist = np.linalg.norm(delta[:2] * np.asarray(spacing[:2], dtype=float))
    else:
        raise ValueError("pair_distance_mode deve ser 'voxel_xyz' ou 'physical_xy'")
    z_diff_voxels = abs(first[2] - second[2])
    z_diff_mm = z_diff_voxels * spacing[2]
    x_diff = abs(first[1] - second[1])
    if distance_mode == "physical_xy":
        x_diff *= spacing[1]

    return bool(
        dist >= min_center_dist
        and z_diff_mm <= max_z_diff_mm
        and x_diff >= min_lateral_sep
    )


def _find_second_ostium(
    first_ostium: Sequence[float],
    candidates: NDArray[Any],
    min_center_dist: float,
    max_z_diff_mm: float,
    min_lateral_sep: float,
    spacing: Sequence[float],
    distance_mode: str,
) -> Optional[NDArray[Any]]:
    """Busca o segundo óstio entre candidatos com base em restrições anatômicas."""
    for candidate in candidates[1:]:
        # O segundo óstio precisa estar separado e em fatia anatomicamente plausível.
        if _validate_ostium_pair(
            first_ostium,
            candidate,
            min_center_dist,
            max_z_diff_mm,
            min_lateral_sep,
            spacing,
            distance_mode,
        ):
            return candidate
    return None


def _find_best_ostium_pair(
    candidates: NDArray[Any],
    score_map: NDArray[Any],
    aorta_mask: NDArray[Any],
    min_center_distance_factor: float,
    max_z_diff_mm: float,
    min_lateral_factor: float,
    spacing: Sequence[float],
    distance_mode: str,
    top_k: int,
) -> Tuple[Optional[NDArray[Any]], Optional[NDArray[Any]]]:
    """Escolhe globalmente o par anatômico com maior soma de scores."""
    if top_k < 2:
        raise ValueError("joint_pair_top_k deve ser >= 2")
    limited = candidates[:top_k]
    best_pair: tuple[NDArray[Any], NDArray[Any]] | None = None
    best_score = -np.inf

    for index, first in enumerate(limited[:-1]):
        for second in limited[index + 1 :]:
            diameter_first = calculate_robust_diameter(aorta_mask[:, :, int(first[2])])
            diameter_second = calculate_robust_diameter(
                aorta_mask[:, :, int(second[2])]
            )
            diameter_ref = 0.5 * (diameter_first + diameter_second)
            min_center_dist = diameter_ref * min_center_distance_factor
            min_lateral_sep = min_center_dist * min_lateral_factor
            if distance_mode == "physical_xy":
                xy_spacing = float(np.mean(spacing[:2]))
                min_center_dist *= xy_spacing
                min_lateral_sep *= xy_spacing
            if not _validate_ostium_pair(
                first,
                second,
                min_center_dist,
                max_z_diff_mm,
                min_lateral_sep,
                spacing,
                distance_mode,
            ):
                continue

            pair_score = float(score_map[tuple(first)]) + float(
                score_map[tuple(second)]
            )
            if pair_score > best_score:
                best_score = pair_score
                best_pair = (first, second)

    if best_pair is None:
        return None, None
    return best_pair[0].copy(), best_pair[1].copy()


def _classify_left_right(
    ostium_1: NDArray[Any], ostium_2: NDArray[Any]
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """Classifica o óstio esquerdo/direito com base na convenção da coordenada x."""
    if ostium_1[1] < ostium_2[1]:
        return ostium_2.copy(), ostium_1.copy()
    return ostium_1.copy(), ostium_2.copy()


def find_aorta_surface(
    aorta_mask: NDArray[Any],
    erosion_radius: int = 2,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    surface_mode: str = "erosion",
    surface_thickness_mm: float = 2.0,
) -> NDArray[Any]:
    """Extrai a casca da aorta por erosão ou distância física ao fundo."""
    mask = aorta_mask.astype(bool)
    if surface_mode == "physical_distance":
        if surface_thickness_mm <= 0:
            raise ValueError("surface_thickness_mm deve ser maior que zero")
        distance_inside = np.asarray(distance_transform_edt(mask, sampling=spacing))
        return (mask & (distance_inside <= surface_thickness_mm)).astype(np.uint8)
    if surface_mode != "erosion":
        raise ValueError("surface_mode deve ser 'erosion' ou 'physical_distance'")

    struct_elem = np.asarray(ball(erosion_radius))
    eroded = binary_erosion(
        mask,
        structure=struct_elem,
        # A superfície entra diretamente na escolha dos candidatos de óstio.
        # Mantê-la na CPU reduz diferenças discretas entre execuções CPU/GPU.
        gpu=False,
    )
    surface = mask & (~eroded)  # pyright: ignore[reportOperatorIssue]
    return surface.astype(np.uint8)


def calculate_robust_diameter(mask_slice: NDArray[Any]) -> float:
    """Estima o diâmetro a partir da área circular equivalente em uma fatia 2D."""
    area = np.sum(mask_slice)
    if area == 0:
        return 0.0
    return 2 * np.sqrt(area / np.pi)


def check_ostium_intersection(
    ostium_coords: Optional[ArrayLike],
    label_mask: NDArray[Any],
    spacing: Sequence[float],
    ostium_name: str = "Óstio",
    distance_threshold_mm: float = 5.0,
    verbose: bool = False,
) -> dict:
    """Verifica se o óstio intersecta a máscara arterial ou está suficientemente próximo em mm."""

    if ostium_coords is None:
        return {
            "intersects": False,
            "euclidean_dist": float("inf"),
            "physical_dist": float("inf"),
            "nearest_voxel": (0, 0, 0),
            "is_acceptable": False,
        }

    _validate_coordinates(ostium_coords, label_mask.shape)
    coords_array = np.asarray(ostium_coords).ravel()
    y, x, z = (int(coords_array[index]) for index in range(3))
    dy, dx, dz = spacing

    if label_mask[y, x, z] == 1:
        if verbose:
            print(f"✓ {ostium_name} intersecta o label")
        return {
            "intersects": True,
            "euclidean_dist": 0.0,
            "physical_dist": 0.0,
            "nearest_voxel": (y, x, z),
            "is_acceptable": True,
        }

    if not np.any(label_mask > 0):
        raise ValueError("label_mask não possui voxels positivos")

    dist_mm, indices = cast(
        tuple[NDArray[Any], NDArray[Any]],
        distance_transform_edt(
            label_mask == 0,
            sampling=(dy, dx, dz),
            return_indices=True,
        ),
    )

    physical_dist = float(dist_mm[y, x, z])
    nearest_voxel = (
        int(indices[0, y, x, z]),
        int(indices[1, y, x, z]),
        int(indices[2, y, x, z]),
    )

    euclidean_dist = float(
        np.linalg.norm(
            np.array([y, x, z], dtype=float) - np.array(nearest_voxel, dtype=float)
        )
    )
    is_acceptable = physical_dist <= distance_threshold_mm

    if verbose:
        status_symbol = "✓" if is_acceptable else "✗"
        print(f"{status_symbol} {ostium_name} NÃO intersecta o label")
        print(f"  Distância euclidiana: {euclidean_dist:.2f} voxels")
        print(f"  Distância física: {physical_dist:.2f} mm")
        print(f"  Voxel mais próximo: {nearest_voxel}")
        if is_acceptable:
            print(f"  ✓ Distância aceitável (< {distance_threshold_mm} mm)")
        else:
            print(f"  ✗ Distância excede o threshold ({distance_threshold_mm} mm)")
        print()

    return {
        "intersects": False,
        "euclidean_dist": euclidean_dist,
        "physical_dist": physical_dist,
        "nearest_voxel": nearest_voxel,
        "is_acceptable": is_acceptable,
    }


def find_ostia(
    aorta_mask: NDArray[Any],
    vesselness_map: NDArray[Any],
    spacing: Sequence[float],
    top_n: int = 50,
    max_z_diff_mm: float = 40.0,
    lower_fraction: float = 0.3,
    min_center_distance_factor: float = 0.8,
    min_lateral_factor: float = 0.5,
    erosion_radius: int = 2,
    surface_mode: str = "erosion",
    surface_thickness_mm: float = 2.0,
    candidate_score_mode: str = "voxel",
    candidate_score_radius: int = 2,
    candidate_local_percentile: float = 90.0,
    candidate_point_weight: float = 0.7,
    candidate_suppression_radius_mm: float = 0.0,
    pair_selection_mode: str = "greedy",
    joint_pair_top_k: int = 100,
    pair_distance_mode: str = "voxel_xyz",
    verbose: bool = True,
) -> Tuple[NDArray[Any], Optional[NDArray[Any]]]:
    """Detecta óstios coronários esquerdo/direito na superfície da aorta.

    O primeiro óstio é o candidato de maior vesselness na região inferior da
    superfície. O segundo é escolhido entre os próximos candidatos respeitando
    distância mínima, separação lateral e diferença máxima em z.
    """
    if aorta_mask.shape != vesselness_map.shape:
        raise ValueError(
            f"aorta_mask e vesselness_map devem ter o mesmo shape: "
            f"{aorta_mask.shape} vs {vesselness_map.shape}"
        )

    # Extrai a casca da aorta e mantém apenas a região onde os óstios são esperados.
    aorta_surface = find_aorta_surface(
        aorta_mask,
        erosion_radius=erosion_radius,
        spacing=spacing,
        surface_mode=surface_mode,
        surface_thickness_mm=surface_thickness_mm,
    )
    lower_region_mask, _, _ = _extract_lower_region(aorta_surface, lower_fraction)
    # Agrega o vesselness conforme a estratégia e ordena os candidatos.
    score_map = _candidate_score_map(
        aorta_mask,
        vesselness_map,
        mode=candidate_score_mode,
        radius=candidate_score_radius,
        local_percentile=candidate_local_percentile,
        point_weight=candidate_point_weight,
        evaluation_mask=lower_region_mask,
    )
    top_candidates = _get_top_candidates(
        lower_region_mask,
        score_map,
        top_n,
        spacing=spacing,
        suppression_radius_mm=candidate_suppression_radius_mm,
    )

    if pair_selection_mode == "joint":
        ostium_1, ostium_2 = _find_best_ostium_pair(
            top_candidates,
            score_map,
            aorta_mask,
            min_center_distance_factor,
            max_z_diff_mm,
            min_lateral_factor,
            spacing,
            pair_distance_mode,
            joint_pair_top_k,
        )
        if ostium_1 is None or ostium_2 is None:
            return top_candidates[0].copy(), None
        return _classify_left_right(ostium_1, ostium_2)
    if pair_selection_mode != "greedy":
        raise ValueError("pair_selection_mode deve ser 'greedy' ou 'joint'")

    # Primeiro óstio: maior vesselness na região candidata.
    ostium_1 = top_candidates[0]
    diameter_ref = calculate_robust_diameter(aorta_mask[:, :, ostium_1[2]])
    min_center_dist = diameter_ref * min_center_distance_factor
    min_lateral_sep = min_center_dist * min_lateral_factor
    if pair_distance_mode == "physical_xy":
        xy_spacing = float(np.mean(spacing[:2]))
        min_center_dist *= xy_spacing
        min_lateral_sep *= xy_spacing

    # Segundo óstio: próximo candidato que respeita restrições anatômicas.
    ostium_2 = _find_second_ostium(
        ostium_1,
        top_candidates,
        min_center_dist,
        max_z_diff_mm,
        min_lateral_sep,
        spacing,
        pair_distance_mode,
    )

    if ostium_2 is None:
        if verbose:
            print(
                "⚠️ AVISO: Segundo óstio não encontrado. Retornando None para a coronária direita."
            )
        return ostium_1.copy(), None

    # Classifica esquerda/direita pela convenção de coordenada x usada no projeto.
    ostia_left, ostia_right = _classify_left_right(ostium_1, ostium_2)
    return ostia_left, ostia_right


__all__ = [
    "calculate_robust_diameter",
    "check_ostium_intersection",
    "find_aorta_surface",
    "find_ostia",
]
