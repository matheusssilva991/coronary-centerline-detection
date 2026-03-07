"""
Módulo para detecção de óstios das artérias coronárias na aorta.

Este módulo implementa algoritmos para localizar automaticamente os óstios (pontos
de origem) das artérias coronárias esquerda e direita na superfície da aorta,
usando mapas de vesselness e restrições anatômicas.
"""

import numpy as np
from skimage.morphology import ball
from scipy.ndimage import distance_transform_edt

# Importa operações morfológicas com suporte GPU
from .binary_operations import binary_erosion


# =============================================================================
# Funções Auxiliares Privadas
# =============================================================================


def _validate_coordinates(coords, volume_shape):
    """
    Valida se as coordenadas estão dentro dos limites do volume.

    Args:
        coords (tuple): Coordenadas (y, x, z)
        volume_shape (tuple): Shape do volume (altura, largura, profundidade)

    Returns:
        bool: True se coordenadas são válidas

    Raises:
        ValueError: Se coordenadas estão fora dos limites
    """
    y, x, z = map(int, coords)
    height, width, depth = volume_shape

    if y < 0 or x < 0 or z < 0 or y >= height or x >= width or z >= depth:
        raise ValueError(
            f"Coordenadas fora dos limites do volume: "
            f"(y={y}, x={x}, z={z}), shape={volume_shape}"
        )
    return True


def _extract_lower_region(surface_mask, lower_fraction=0.3):
    """
    Extrai a região inferior da superfície da aorta onde os óstios estão localizados.

    Args:
        surface_mask (np.ndarray): Máscara binária da superfície da aorta
        lower_fraction (float): Fração inferior a extrair (0.3 = 30% inferior)

    Returns:
        tuple: (lower_region_mask, z_min, z_max) contendo:
            - lower_region_mask: Máscara da região inferior
            - z_min: Índice z mínimo da superfície
            - z_max: Índice z máximo da superfície

    Raises:
        ValueError: Se nenhuma superfície for encontrada
    """
    z_indices = np.where(np.any(surface_mask, axis=(0, 1)))[0]
    if len(z_indices) == 0:
        raise ValueError("Nenhuma superfície de aorta encontrada!")

    z_min, z_max = z_indices.min(), z_indices.max()
    z_threshold = z_min + int((z_max - z_min) * lower_fraction)

    lower_region_mask = np.zeros_like(surface_mask)
    lower_region_mask[:, :, z_min:z_threshold] = surface_mask[:, :, z_min:z_threshold]

    return lower_region_mask, z_min, z_max


def _get_top_candidates(surface_mask, vesselness_map, top_n=50):
    """
    Obtém os candidatos com maior vesselness na superfície.

    Args:
        surface_mask (np.ndarray): Máscara binária da região de busca
        vesselness_map (np.ndarray): Mapa de vesselness
        top_n (int): Número de candidatos a retornar

    Returns:
        np.ndarray: Array de coordenadas (N, 3) dos top_n candidatos

    Raises:
        ValueError: Se nenhum voxel for encontrado
    """
    surface_coords = np.argwhere(surface_mask > 0)  # (y, x, z)
    if len(surface_coords) == 0:
        raise ValueError("Nenhum voxel encontrado na superfície!")

    surface_values = vesselness_map[surface_mask > 0]
    sorted_indices = np.argsort(surface_values)[::-1][:top_n]

    return surface_coords[sorted_indices]


def _validate_ostium_pair(
    ostium_1, ostium_2, min_center_dist, max_z_diff_mm, min_lateral_sep, spacing_dz
):
    """
    Valida se dois pontos satisfazem os critérios anatômicos para serem óstios.

    Args:
        ostium_1 (np.ndarray): Coordenadas (y, x, z) do primeiro óstio
        ostium_2 (np.ndarray): Coordenadas (y, x, z) do segundo óstio
        min_center_dist (float): Distância mínima entre centros em voxels
        max_z_diff_mm (float): Diferença máxima em z em milímetros
        min_lateral_sep (float): Separação lateral mínima em x
        spacing_dz (float): Espaçamento físico em z (mm/voxel)

    Returns:
        bool: True se o par é válido
    """
    dist = np.linalg.norm(ostium_1 - ostium_2)
    z_diff_voxels = abs(ostium_1[2] - ostium_2[2])
    z_diff_mm = z_diff_voxels * spacing_dz
    x_diff = abs(ostium_1[1] - ostium_2[1])

    return (
        dist >= min_center_dist and z_diff_mm <= max_z_diff_mm and x_diff >= min_lateral_sep
    )


def _find_second_ostium(
    first_ostium, candidates, min_center_dist, max_z_diff_mm, min_lateral_sep, spacing_dz
):
    """
    Busca o segundo óstio com base em restrições anatômicas.

    Args:
        first_ostium (np.ndarray): Coordenadas do primeiro óstio
        candidates (np.ndarray): Array de candidatos (N, 3)
        min_center_dist (float): Distância mínima entre centros
        max_z_diff_mm (float): Diferença máxima em z em milímetros
        min_lateral_sep (float): Separação lateral mínima
        spacing_dz (float): Espaçamento físico em z (mm/voxel)

    Returns:
        np.ndarray or None: Coordenadas do segundo óstio ou None se não encontrado
    """
    for candidate in candidates[1:]:
        if _validate_ostium_pair(
            first_ostium, candidate, min_center_dist, max_z_diff_mm, min_lateral_sep, spacing_dz
        ):
            return candidate
    return None


def _classify_left_right(ostium_1, ostium_2):
    """
    Classifica os óstios em esquerdo e direito pela posição X.

    Por convenção, óstios com maior X são considerados à esquerda
    (assumindo orientação padrão radiológica).

    Args:
        ostium_1 (np.ndarray): Coordenadas (y, x, z) do primeiro óstio
        ostium_2 (np.ndarray): Coordenadas (y, x, z) do segundo óstio

    Returns:
        tuple: (ostia_left, ostia_right) com coordenadas classificadas
    """
    if ostium_1[1] < ostium_2[1]:
        return ostium_2.copy(), ostium_1.copy()
    else:
        return ostium_1.copy(), ostium_2.copy()


# =============================================================================
# Funções Públicas
# =============================================================================


def find_aorta_surface(aorta_mask, erosion_radius=2):
    """
    Extrai a superfície externa (casca) da aorta usando erosão morfológica.

    A superfície é obtida pela diferença entre a máscara original e sua versão
    erodida, resultando em uma fina camada que representa a borda externa da aorta.

    Args:
        aorta_mask (np.ndarray): Máscara binária 3D da aorta (y, x, z)
        erosion_radius (int): Raio do elemento estruturante esférico para erosão.
            Valores maiores resultam em superfície mais interna. Default: 2

    Returns:
        np.ndarray: Máscara binária 3D (dtype=uint8) representando apenas a
            superfície da aorta

    Example:
        >>> aorta = segment_aorta(volume)
        >>> surface = find_aorta_surface(aorta, erosion_radius=3)
        >>> print(f"Superfície: {surface.sum()} voxels")

    Note:
        - Aumentar erosion_radius cria superfície mais interna e fina
        - Útil para reduzir busca apenas à casca externa onde os óstios estão
    """
    struct_elem = ball(erosion_radius)
    eroded = binary_erosion(aorta_mask.astype(bool), structure=struct_elem)  # Erosão com suporte GPU

    # Superfície = aorta original - aorta erodida
    surface = aorta_mask.astype(bool) & (~eroded)  # pyright: ignore[reportOperatorIssue]
    return surface.astype(np.uint8)


def calculate_robust_diameter(mask_slice):
    """
    Estima o diâmetro da aorta a partir da área de um corte transversal.

    Assume que a seção transversal da aorta é aproximadamente circular e
    calcula o diâmetro de um círculo com área equivalente.

    Args:
        mask_slice (np.ndarray): Fatia 2D da máscara binária (y, x)

    Returns:
        float: Diâmetro estimado em pixels, calculado como d = 2√(área/π)

    Example:
        >>> slice_2d = aorta_mask[:, :, 50]
        >>> diameter = calculate_robust_diameter(slice_2d)
        >>> print(f"Diâmetro estimado: {diameter:.1f} pixels")

    Note:
        - Retorna 0 se a fatia não contiver pixels da aorta
        - Mais robusto que medir diâmetro direto (invariante a orientação)
        - Útil para estabelecer distâncias mínimas entre óstios
    """
    area = np.sum(mask_slice)
    if area == 0:
        return 0

    # Diâmetro de um círculo com área equivalente: A = πr² → d = 2√(A/π)
    return 2 * np.sqrt(area / np.pi)


def check_ostium_intersection(
    ostium_coords,
    label_mask,
    spacing,
    ostium_name="Óstio",
    distance_threshold_mm=5.0,
    verbose=False,
):
    """
    Verifica se um ponto (óstio) intersecta a máscara de artéria segmentada.

    Se o óstio não intersectar diretamente a artéria, calcula a distância até
    o voxel mais próximo da artéria e verifica se está dentro de um threshold
    aceitável (pode haver pequenos gaps devido a limitações da segmentação).

    Args:
        ostium_coords (tuple): Coordenadas (y, x, z) do óstio
        label_mask (np.ndarray): Máscara binária 3D da artéria (dtype=uint8 ou bool)
        spacing (tuple): Espaçamento físico (dy, dx, dz) em mm/voxel
        ostium_name (str): Nome do óstio para mensagens. Default: "Óstio"
        distance_threshold_mm (float): Distância máxima aceitável em mm.
            Se a distância for menor que isso, considera-se "aceitável" mesmo
            sem interseção direta. Default: 5.0
        verbose (bool): Se True, imprime informações detalhadas. Default: False

    Returns:
        dict: Dicionário com informações da verificação:
            - 'intersects' (bool): True se há interseção direta
            - 'euclidean_dist' (float): Distância euclidiana em voxels
            - 'physical_dist' (float): Distância física em mm
            - 'nearest_voxel' (tuple): Coordenadas (y, x, z) do voxel mais próximo
            - 'is_acceptable' (bool): True se dist <= threshold

    Raises:
        ValueError: Se coordenadas estiverem fora dos limites ou label vazio

    Example:
        >>> ostium = (250, 200, 45)
        >>> artery_mask = segment_artery(volume)
        >>> spacing = (0.5, 0.5, 0.625)  # mm
        >>> result = check_ostium_intersection(
        ...     ostium, artery_mask, spacing, "Óstio Esquerdo", verbose=True
        ... )
        >>> if result['is_acceptable']:
        ...     print("✓ Óstio validado")

    Note:
        - Usa transformada de distância euclidiana para encontrar voxel mais próximo
        - Threshold de 5mm é razoável considerando resolução CCTA típica (~0.5mm)
        - Interseção direta sempre retorna is_acceptable=True
    """
    _validate_coordinates(ostium_coords, label_mask.shape)
    y, x, z = map(int, ostium_coords)
    dy, dx, dz = spacing

    # Caso 1: Interseção direta
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

    # Caso 2: Sem interseção - calcular distância ao voxel mais próximo
    if not np.any(label_mask > 0):
        raise ValueError("label_mask não possui voxels positivos")

    # Transformada de distância euclidiana
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

    # Distância euclidiana em voxels
    euclidean_dist = float(
        np.linalg.norm(
            np.array([y, x, z], dtype=float) - np.array(nearest_voxel, dtype=float)
        )
    )

    # Verificar se está dentro do threshold aceitável
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
    """
    Localiza os óstios esquerdo e direito das artérias coronárias na superfície da aorta.

    Este algoritmo detecta automaticamente os pontos de origem (óstios) das artérias
    coronárias usando uma estratégia baseada em:
    1. Análise da superfície externa da aorta
    2. Seleção de pontos com alto vesselness (indicando início de vasos)
    3. Aplicação de restrições anatômicas (distância mínima, mesmo nível vertical)

    O algoritmo funciona em 7 etapas principais:
    1. Extrai superfície da aorta por erosão morfológica
    2. Define ROI na porção inferior da aorta (onde os óstios estão)
    3. Identifica candidatos com maior vesselness
    4. Seleciona primeiro óstio (maior vesselness)
    5. Calcula distâncias mínimas baseadas no diâmetro da aorta
    6. Busca segundo ósticom restrições anatômicas
    7. Classifica óstios em esquerdo/direito por posição X

    Args:
        aorta_mask (np.ndarray): Máscara binária 3D da aorta (y, x, z)
        vesselness_map (np.ndarray): Mapa 3D de vesselness (y, x, z). Valores
            mais altos indicam maior probabilidade de ser vaso sanguíneo
        spacing (tuple): Espaçamento físico (dy, dx, dz) em mm/voxel.
            Usado para converter critérios de mm para voxels
        top_n (int): Número de candidatos com maior vesselness a analisar.
            Default: 50
        max_z_diff_mm (float): Diferença máxima em z permitida entre os
            dois óstios em milímetros. Garante que estão aproximadamente
            no mesmo nível vertical. Critério fisiológico: 40mm típico.
            Default: 40.0
        lower_fraction (float): Fração inferior da aorta para buscar óstios.
            0.3 significa buscar apenas nos 30% inferiores. Default: 0.3
        min_center_distance_factor (float): Fator multiplicativo do diâmetro
            da aorta para calcular distância mínima entre óstios.
            Se diameter=40 e factor=0.7, então min_dist=28 voxels. Default: 0.8
        min_lateral_factor (float): Fração da distância mínima entre centros
            exigida como separação lateral em X. Garante que os óstios estão
            em lados opostos da aorta. Default: 0.5
        erosion_radius (int): Raio do elemento estruturante para extrair
            superfície da aorta. Default: 2

    Returns:
        tuple: (ostia_left, ostia_right) onde cada elemento é um array numpy
            com coordenadas (y, x, z) do respectivo óstio

    Raises:
        ValueError: Se:
            - Shapes de aorta_mask e vesselness_map não coincidirem
            - Nenhuma superfície for encontrada
            - Nenhum candidato for encontrado
            - Segundo óstio não satisfizer restrições anatômicas

    Example:
        >>> aorta = segment_aorta(volume)
        >>> vesselness = compute_frangi_filter(volume)
        >>> spacing = (0.5, 0.5, 0.625)  # mm/voxel
        >>> left, right = find_ostia(
        ...     aorta, vesselness, spacing,
        ...     top_n=100,
        ...     max_z_diff_mm=40.0,
        ...     lower_fraction=0.25,
        ...     min_center_distance_factor=0.7
        ... )
        >>> print(f"Óstio esquerdo: {left}")
        >>> print(f"Óstio direito: {right}")

    Note:
        - A classificação esquerdo/direito assume orientação radiológica padrão
        - Ajuste lower_fraction se óstios estiverem muito acima/abaixo
        - Aumente top_n se não encontrar segundo óstio
        - Valores típicos de distância: 0.5-1.0 × diâmetro da aorta

    See Also:
        find_aorta_surface: Para entender como a superfície é extraída
        calculate_robust_diameter: Para entender o cálculo de diâmetro
    """
    # Validação de entrada
    if aorta_mask.shape != vesselness_map.shape:
        raise ValueError(
            f"aorta_mask e vesselness_map devem ter o mesmo shape: "
            f"{aorta_mask.shape} vs {vesselness_map.shape}"
        )

    # Etapa 1: Extrair superfície da aorta
    aorta_surface = find_aorta_surface(aorta_mask, erosion_radius=erosion_radius)

    # Etapa 2: Definir ROI na porção inferior da aorta
    lower_region_mask, z_min, z_max = _extract_lower_region(
        aorta_surface, lower_fraction
    )

    # Etapa 3: Extrair candidatos ordenados por vesselness
    top_candidates = _get_top_candidates(lower_region_mask, vesselness_map, top_n)

    # Etapa 4: Primeiro óstio = candidato com maior vesselness
    ostium_1 = top_candidates[0]

    # Etapa 5: Calcular distâncias mínimas baseadas no diâmetro da aorta
    diameter_ref = calculate_robust_diameter(aorta_mask[:, :, ostium_1[2]])
    min_center_dist = diameter_ref * min_center_distance_factor
    min_lateral_sep = min_center_dist * min_lateral_factor

    # Etapa 6: Buscar segundo óstio com restrições anatômicas
    ostium_2 = _find_second_ostium(
        ostium_1, top_candidates, min_center_dist, max_z_diff_mm, min_lateral_sep, spacing[2]
    )

    if ostium_2 is None:
        raise ValueError(
            f"Segundo óstio não encontrado! Tentou {top_n} candidatos. "
            f"Restrições: dist>={min_center_dist:.1f}, z_diff<={max_z_diff_mm}mm, "
            f"x_sep>={min_lateral_sep:.1f}. "
            f"Tente aumentar 'top_n' ou relaxar os fatores."
        )

    # Etapa 7: Classificar esquerdo/direito por posição X
    ostia_left, ostia_right = _classify_left_right(ostium_1, ostium_2)

    return ostia_left, ostia_right
