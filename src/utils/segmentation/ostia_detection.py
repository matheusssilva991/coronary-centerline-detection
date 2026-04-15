"""Utilitários para detecção de óstios coronários na superfície da aorta.

Este módulo fornece funções auxiliares para:
- extrair a superfície da aorta,
- selecionar candidatos com alto vesselness,
- validar restrições anatômicas,
- classificar óstios esquerdo/direito,
- validar interseção do óstio com máscaras arteriais.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import ball

from ..processing.binary_operations import binary_erosion


def _validate_coordinates(coords, volume_shape):
    """Valida se uma coordenada (y, x, z) está dentro dos limites do volume."""
    y, x, z = map(int, coords)
    height, width, depth = volume_shape

    if y < 0 or x < 0 or z < 0 or y >= height or x >= width or z >= depth:
        raise ValueError(
            f"Coordenadas fora dos limites do volume: "
            f"(y={y}, x={x}, z={z}), shape={volume_shape}"
        )
    return True


def _extract_lower_region(surface_mask, lower_fraction=0.3):
    """Extrai a região inferior em z da superfície da aorta onde os óstios são esperados."""
    z_indices = np.where(np.any(surface_mask, axis=(0, 1)))[0]
    if len(z_indices) == 0:
        raise ValueError("Nenhuma superfície de aorta encontrada!")

    z_min, z_max = z_indices.min(), z_indices.max()
    z_threshold = z_min + int((z_max - z_min) * lower_fraction)

    lower_region_mask = np.zeros_like(surface_mask)
    lower_region_mask[:, :, z_min:z_threshold] = surface_mask[:, :, z_min:z_threshold]

    return lower_region_mask, z_min, z_max


def _get_top_candidates(surface_mask, vesselness_map, top_n=50):
    """Retorna os top-N candidatos de superfície ordenados por vesselness decrescente."""
    surface_coords = np.argwhere(surface_mask > 0)
    if len(surface_coords) == 0:
        raise ValueError("Nenhum voxel encontrado na superfície!")

    surface_values = vesselness_map[surface_mask > 0]
    sorted_indices = np.argsort(surface_values)[::-1][:top_n]
    return surface_coords[sorted_indices]


def _validate_ostium_pair(
    ostium_1, ostium_2, min_center_dist, max_z_diff_mm, min_lateral_sep, spacing_dz
):
    """Verifica restrições anatômicas para um par candidato de óstios."""
    dist = np.linalg.norm(ostium_1 - ostium_2)
    z_diff_voxels = abs(ostium_1[2] - ostium_2[2])
    z_diff_mm = z_diff_voxels * spacing_dz
    x_diff = abs(ostium_1[1] - ostium_2[1])

    return (
        dist >= min_center_dist
        and z_diff_mm <= max_z_diff_mm
        and x_diff >= min_lateral_sep
    )


def _find_second_ostium(
    first_ostium,
    candidates,
    min_center_dist,
    max_z_diff_mm,
    min_lateral_sep,
    spacing_dz,
):
    """Busca o segundo óstio entre candidatos com base em restrições anatômicas."""
    for candidate in candidates[1:]:
        if _validate_ostium_pair(
            first_ostium,
            candidate,
            min_center_dist,
            max_z_diff_mm,
            min_lateral_sep,
            spacing_dz,
        ):
            return candidate
    return None


def _classify_left_right(ostium_1, ostium_2):
    """Classifica o óstio esquerdo/direito com base na convenção da coordenada x."""
    if ostium_1[1] < ostium_2[1]:
        return ostium_2.copy(), ostium_1.copy()
    return ostium_1.copy(), ostium_2.copy()


def find_aorta_surface(aorta_mask, erosion_radius=2):
    """Extrai a casca da aorta como (máscara - máscara erodida)."""
    struct_elem = ball(erosion_radius)
    eroded = binary_erosion(aorta_mask.astype(bool), structure=struct_elem)
    surface = aorta_mask.astype(bool) & (~eroded)  # pyright: ignore[reportOperatorIssue]
    return surface.astype(np.uint8)


def calculate_robust_diameter(mask_slice):
    """Estima o diâmetro a partir da área circular equivalente em uma fatia 2D."""
    area = np.sum(mask_slice)
    if area == 0:
        return 0
    return 2 * np.sqrt(area / np.pi)


def check_ostium_intersection(
    ostium_coords,
    label_mask,
    spacing,
    ostium_name="Óstio",
    distance_threshold_mm=5.0,
    verbose=False,
):
    """Verifica se o óstio intersecta a máscara arterial ou está suficientemente próximo em mm."""
    _validate_coordinates(ostium_coords, label_mask.shape)
    y, x, z = map(int, ostium_coords)
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

    dist_mm, indices = distance_transform_edt(
        label_mask == 0,
        sampling=(dy, dx, dz),
        return_indices=True,
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
    aorta_mask,
    vesselness_map,
    spacing,
    top_n=50,
    max_z_diff_mm=40.0,
    lower_fraction=0.3,
    min_center_distance_factor=0.8,
    min_lateral_factor=0.5,
    erosion_radius=2,
):
    """Detecta óstios coronários esquerdo e direito a partir da superfície da aorta e do vesselness."""
    if aorta_mask.shape != vesselness_map.shape:
        raise ValueError(
            f"aorta_mask e vesselness_map devem ter o mesmo shape: "
            f"{aorta_mask.shape} vs {vesselness_map.shape}"
        )

    aorta_surface = find_aorta_surface(aorta_mask, erosion_radius=erosion_radius)
    lower_region_mask, _, _ = _extract_lower_region(aorta_surface, lower_fraction)
    top_candidates = _get_top_candidates(lower_region_mask, vesselness_map, top_n)

    ostium_1 = top_candidates[0]
    diameter_ref = calculate_robust_diameter(aorta_mask[:, :, ostium_1[2]])
    min_center_dist = diameter_ref * min_center_distance_factor
    min_lateral_sep = min_center_dist * min_lateral_factor

    ostium_2 = _find_second_ostium(
        ostium_1,
        top_candidates,
        min_center_dist,
        max_z_diff_mm,
        min_lateral_sep,
        spacing[2],
    )

    if ostium_2 is None:
        raise ValueError(
            f"Segundo óstio não encontrado! Tentou {top_n} candidatos. "
            f"Restrições: dist>={min_center_dist:.1f}, z_diff<={max_z_diff_mm}mm, "
            f"x_sep>={min_lateral_sep:.1f}. "
            f"Tente aumentar 'top_n' ou relaxar os fatores."
        )

    ostia_left, ostia_right = _classify_left_right(ostium_1, ostium_2)
    return ostia_left, ostia_right


__all__ = [
    "calculate_robust_diameter",
    "check_ostium_intersection",
    "find_aorta_surface",
    "find_ostia",
]
