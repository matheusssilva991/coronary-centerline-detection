"""Utilitários para detecção de óstios coronários na superfície da aorta.

Este módulo fornece funções auxiliares para:
- extrair a superfície da aorta,
- selecionar candidatos com alto vesselness,
- validar restrições anatômicas,
- classificar óstios esquerdo/direito,
- validar interseção do óstio com máscaras arteriais.
"""

from typing import Any, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from skimage.morphology import ball

from ..processing.binary_operations import binary_dilation, binary_erosion
from ..utils.roi import mask_voxel_bbox_slices


def _validate_coordinates(coords: ArrayLike, volume_shape: Sequence[int]) -> bool:
    """Valida se uma coordenada ``(y, x, z)`` está dentro dos limites do volume.

    Args:
        coords (ArrayLike): Coordenada 3D candidata com três elementos ``(y, x, z)``.
        volume_shape (Sequence[int]): Dimensões ``(height, width, depth)`` do volume 3D.

    Returns:
        bool: ``True`` quando a coordenada pertence aos limites do volume.
    """
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
    """Extrai a região inferior em ``z`` da superfície onde os óstios são esperados.

    Args:
        surface_mask (NDArray[Any]): Máscara binária 3D da superfície da aorta.
        lower_fraction (float): Fração inferior da extensão axial em ``(0, 1]``.

    Returns:
        tuple[NDArray[Any], int, int]: Tupla ``(lower_region_mask, z_min, z_max)`` com
            a região filtrada e os índices axiais mínimo e máximo da aorta.
    """
    if not 0 < lower_fraction <= 1:
        raise ValueError("lower_fraction deve estar no intervalo (0, 1]")

    z_indices = np.where(np.any(surface_mask, axis=(0, 1)))[0]
    if len(z_indices) == 0:
        raise ValueError("Nenhuma superfície de aorta encontrada!")

    # Os óstios são buscados na porção inicial/inferior da aorta segmentada.
    z_min, z_max = z_indices.min(), z_indices.max()
    z_stop = z_min + int((z_max - z_min) * lower_fraction)
    z_stop = min(z_max + 1, max(z_min + 1, z_stop))
    if lower_fraction == 1.0:
        z_stop = z_max + 1  # O limite superior do slice Python e exclusivo.

    # Copia apenas as fatias inferiores da superfície onde nascem as coronárias.
    lower_region_mask = np.zeros_like(surface_mask)
    lower_region_mask[:, :, z_min:z_stop] = surface_mask[:, :, z_min:z_stop]

    return lower_region_mask, z_min, z_max


def _get_top_candidates(
    surface_mask: NDArray[Any],
    vesselness_map: NDArray[Any],
    top_n: int = 50,
) -> NDArray[Any]:
    """Retorna os candidatos de maior vesselness na superfície.

    Args:
        surface_mask (NDArray[Any]): Máscara binária 3D da superfície candidata.
        vesselness_map (NDArray[Any]): Mapa 3D de vesselness alinhado à superfície.
        top_n (int): Quantidade máxima de voxels candidatos ordenados por vesselness.

    Returns:
        NDArray[Any]: Array ``(N, 3)`` com coordenadas ``(y, x, z)`` em ordem
            decrescente de vesselness.
    """
    if top_n <= 0:
        raise ValueError("top_n deve ser maior que 0")

    candidate_mask = surface_mask > 0
    surface_coords = np.argwhere(candidate_mask)
    if len(surface_coords) == 0:
        raise ValueError("Nenhum voxel encontrado na superfície!")

    # Quanto maior o vesselness na superfície, mais provável o ponto de óstio.
    surface_values = vesselness_map[candidate_mask]
    sorted_indices = np.argsort(surface_values)[::-1][:top_n]
    return surface_coords[sorted_indices]


def _validate_ostium_pair(
    ostium_1: ArrayLike,
    ostium_2: ArrayLike,
    min_center_dist: float,
    max_z_diff_mm: float,
    min_lateral_sep: float,
    spacing: Sequence[float],
    distance_mode: str,
) -> bool:
    """Verifica restrições anatômicas para um par candidato de óstios.

    Args:
        ostium_1 (ArrayLike): Coordenada ``(y, x, z)`` do primeiro óstio candidato.
        ostium_2 (ArrayLike): Coordenada ``(y, x, z)`` do segundo óstio candidato.
        min_center_dist (float): Distância mínima exigida entre os dois óstios.
        max_z_diff_mm (float): Diferença axial máxima permitida em ``mm``.
        min_lateral_sep (float): Separação mínima exigida ao longo do eixo ``x``.
        spacing (Sequence[float]): Espaçamento físico em ``mm`` no formato
            ``(dy, dx, dz)``.
        distance_mode (str): Métrica de separação (``"voxel_xyz"`` ou
            ``"physical_xy"``).

    Returns:
        bool: ``True`` se o par satisfaz todas as restrições anatômicas.
    """
    # Calcula a distância euclidiana entre candidatos no modo voxel 3D ou físico XY.
    first = np.asarray(ostium_1, dtype=float)
    second = np.asarray(ostium_2, dtype=float)
    delta = first - second
    if distance_mode == "voxel_xyz":
        dist = np.linalg.norm(delta)
    elif distance_mode == "physical_xy":
        dist = np.linalg.norm(delta[:2] * np.asarray(spacing[:2], dtype=float))
    else:
        raise ValueError("pair_distance_mode deve ser 'voxel_xyz' ou 'physical_xy'")

    # Mede separação axial em mm e separação lateral no eixo x para validação.
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
    """Busca o segundo óstio entre candidatos com base em restrições anatômicas.

    Args:
        first_ostium (Sequence[float]): Coordenada ``(y, x, z)`` do primeiro óstio.
        candidates (NDArray[Any]): Array ``(N, 3)`` de candidatos ordenados por
            vesselness.
        min_center_dist (float): Distância mínima exigida entre os dois óstios.
        max_z_diff_mm (float): Diferença axial máxima permitida em ``mm``.
        min_lateral_sep (float): Separação mínima exigida no eixo ``x``.
        spacing (Sequence[float]): Espaçamento físico em ``mm`` no formato
            ``(dy, dx, dz)``.
        distance_mode (str): Métrica de separação (``"voxel_xyz"`` ou
            ``"physical_xy"``).

    Returns:
        NDArray[Any] | None: Coordenada ``(y, x, z)`` do segundo óstio válido, ou
            ``None`` se nenhum candidato satisfizer as restrições.
    """
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


def _classify_left_right(
    ostium_1: NDArray[Any], ostium_2: NDArray[Any]
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """Classifica o óstio esquerdo e direito pela convenção da coordenada ``x``.

    Args:
        ostium_1 (NDArray[Any]): Coordenada ``(y, x, z)`` do primeiro óstio.
        ostium_2 (NDArray[Any]): Coordenada ``(y, x, z)`` do segundo óstio.

    Returns:
        tuple[NDArray[Any], NDArray[Any]]: Tupla ``(ostia_left, ostia_right)`` com
            cópias das coordenadas ordenadas por lado anatômico.
    """
    if ostium_1[1] < ostium_2[1]:
        return ostium_2.copy(), ostium_1.copy()
    return ostium_1.copy(), ostium_2.copy()


def find_aorta_surface(
    aorta_mask: NDArray[Any],
    erosion_radius: int = 2,
    surface_padding_radius: int = 0,
) -> NDArray[Any]:
    """Extrai a casca da máscara dilatada, sem modificar a aorta original.

    Calcula ``D_p(M) - E_e(D_p(M))`` com elementos esféricos 3D;
    ``surface_padding_radius=0`` preserva a superfície interna original da aorta.
    Em volumes anisotrópicos, o alcance físico difere entre eixos.

    Args:
        aorta_mask (NDArray[Any]): Máscara binária 3D da aorta segmentada.
        erosion_radius (int): Raio em voxels da erosão esférica que define a espessura
            da casca.
        surface_padding_radius (int): Raio inteiro não negativo em voxels de dilatação
            externa prévia.

    Returns:
        NDArray[Any]: Máscara binária ``uint8`` com os voxels da superfície no mesmo
            shape de ``aorta_mask``.
    """
    if (
        surface_padding_radius < 0
        or int(surface_padding_radius) != surface_padding_radius
    ):
        raise ValueError("surface_padding_radius deve ser um inteiro nao negativo")

    # Restringe as operações morfológicas à bounding box da aorta para poupar memória.
    mask = aorta_mask.astype(bool)
    pad = int(surface_padding_radius) + int(erosion_radius) + 1
    bbox = mask_voxel_bbox_slices(mask, pad)
    if bbox is None:
        return np.zeros(mask.shape, dtype=np.uint8)

    # Aplica dilatação externa opcional e erosão esférica para isolar a casca da aorta.
    sub_mask = mask[bbox]
    if surface_padding_radius:
        sub_mask = binary_dilation(
            sub_mask,
            structure=np.asarray(ball(surface_padding_radius)),
            gpu=False,
        ).astype(bool)
    struct_elem = np.asarray(ball(erosion_radius))
    eroded = binary_erosion(
        sub_mask,
        structure=struct_elem,
        # A superfície entra diretamente na escolha dos candidatos de óstio.
        # Mantê-la na CPU reduz diferenças discretas entre execuções CPU/GPU.
        gpu=False,
    )
    surface = np.zeros(mask.shape, dtype=np.uint8)
    surface[bbox] = (sub_mask & (~eroded.astype(bool))).astype(np.uint8)
    return surface


def calculate_robust_diameter(mask_slice: NDArray[Any]) -> float:
    """Estima o diâmetro a partir da área circular equivalente em uma fatia 2D.

    Args:
        mask_slice (NDArray[Any]): Máscara binária 2D de uma fatia axial da aorta.

    Returns:
        float: Diâmetro equivalente em pixels (``0.0`` se a fatia estiver vazia).
    """
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
    """Verifica interseção do óstio ou proximidade dentro de ``distance_threshold_mm``.

    Args:
        ostium_coords (ArrayLike | None): Coordenadas ``(y, x, z)`` do óstio, ou
            ``None`` se não detectado.
        label_mask (NDArray[Any]): Máscara 3D de referência arterial no mesmo espaço de
            ``ostium_coords``.
        spacing (Sequence[float]): Espaçamento físico em ``mm`` no formato
            ``(dy, dx, dz)``.
        ostium_name (str): Rótulo textual exibido nos logs quando ``verbose=True``.
        distance_threshold_mm (float): Limite máximo em ``mm`` para considerar o óstio
            aceitável.
        verbose (bool): Se ``True``, imprime resumo detalhado das distâncias calculadas.

    Returns:
        dict: Dicionário com ``intersects``, ``euclidean_dist`` em voxels,
            ``physical_dist`` em mm, ``nearest_voxel`` e ``is_acceptable``.
    """

    if ostium_coords is None:
        return {
            "intersects": False,
            "euclidean_dist": float("inf"),
            "physical_dist": float("inf"),
            "nearest_voxel": (0, 0, 0),
            "is_acceptable": False,
        }

    # Caminho rápido: acerto exato no voxel elimina busca pelos positivos do label.
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

    # Quando não há interseção, mede a menor distância física até a máscara de referência.
    fg_coords = np.argwhere(label_mask > 0)
    if fg_coords.size == 0:
        raise ValueError("label_mask não possui voxels positivos")

    # Calcula distância física diretamente contra os voxels positivos sem alocar EDT 3D.
    origin = np.array([y, x, z], dtype=float)
    step = np.array([dy, dx, dz], dtype=float)
    deltas = fg_coords - origin
    scaled_deltas = deltas * step
    physical_sq = np.einsum("ij,ij->i", scaled_deltas, scaled_deltas)
    nearest_idx = int(np.argmin(physical_sq))

    physical_dist = float(np.sqrt(physical_sq[nearest_idx]))
    nearest_voxel = (
        int(fg_coords[nearest_idx, 0]),
        int(fg_coords[nearest_idx, 1]),
        int(fg_coords[nearest_idx, 2]),
    )
    euclidean_dist = float(np.linalg.norm(deltas[nearest_idx]))
    is_acceptable = physical_dist <= distance_threshold_mm

    # Registra no console o diagnóstico detalhado de distância quando solicitado.
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
    surface_padding_radius: int = 0,
    pair_distance_mode: str = "voxel_xyz",
    verbose: bool = True,
) -> Tuple[NDArray[Any], Optional[NDArray[Any]]]:
    """Detecta óstios coronários esquerdo e direito na superfície da aorta.

    Args:
        aorta_mask (NDArray[Any]): Máscara binária 3D da aorta em ordem ``(y, x, z)``.
        vesselness_map (NDArray[Any]): Mapa 3D de vesselness alinhado a ``aorta_mask``
            (``VESSELNESS_AORTA``).
        spacing (Sequence[float]): Espaçamento físico em ``mm`` no formato
            ``(dy, dx, dz)``.
        top_n (int): Número de candidatos de maior vesselness avaliados na superfície.
        max_z_diff_mm (float): Diferença axial máxima em ``mm`` permitida entre os
            dois óstios.
        lower_fraction (float): Fração inferior em ``(0, 1]`` da extensão axial da
            aorta considerada na busca.
        min_center_distance_factor (float): Múltiplo do diâmetro da aorta exigido como
            distância mínima entre óstios.
        min_lateral_factor (float): Fração de ``min_center_distance`` exigida como
            separação no eixo ``x``.
        erosion_radius (int): Raio em voxels da erosão esférica na extração da casca.
        surface_padding_radius (int): Raio em voxels da dilatação externa prévia da
            casca da aorta.
        pair_distance_mode (str): Métrica de separação do par (``voxel_xyz`` ou
            ``physical_xy``).
        verbose (bool): Se ``True``, imprime aviso quando o segundo óstio não é
            encontrado.

    Returns:
        tuple[NDArray[Any], NDArray[Any] | None]: Tupla ``(ostia_left, ostia_right)``
            com coordenadas inteiras ``(y, x, z)``; ``ostia_right`` pode ser ``None``.
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
        surface_padding_radius=surface_padding_radius,
    )
    lower_region_mask, _, _ = _extract_lower_region(aorta_surface, lower_fraction)
    # Ordena diretamente pelo valor pontual do mapa de vesselness.
    top_candidates = _get_top_candidates(
        lower_region_mask,
        vesselness_map,
        top_n,
    )

    # Primeiro óstio: maior vesselness na região candidata.
    ostium_1 = top_candidates[0]
    diameter_ref = calculate_robust_diameter(aorta_mask[:, :, ostium_1[2]])
    min_center_dist = diameter_ref * min_center_distance_factor
    min_lateral_sep = min_center_dist * min_lateral_factor
    # Converte separações para mm quando pair_distance_mode usa distância física XY.
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
