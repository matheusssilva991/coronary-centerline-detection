"""
Módulo para segmentação de artérias usando crescimento de região (Region Growing).

Este módulo implementa algoritmos de segmentação baseados em crescimento de região
guiado por mapas de vesselness (Frangi filter), partindo de sementes (óstios) e
expandindo para vizinhos com características similares.
"""

from collections import deque

import numpy as np


# =============================================================================
# Constantes
# =============================================================================

# 26-vizinhança para volumes 3D
NEIGHBORS_26 = [
    (dy, dx, dz)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dy == 0 and dx == 0 and dz == 0)
]


# =============================================================================
# Funções Auxiliares Privadas
# =============================================================================


def _validate_seed(seed_point, volume_shape):
    """
    Valida se uma semente está dentro dos limites do volume.

    Args:
        seed_point (tuple): Coordenadas (y, x, z) da semente
        volume_shape (tuple): Shape do volume (altura, largura, profundidade)

    Returns:
        tuple or None: Coordenadas validadas como inteiros ou None se inválida
    """
    sy, sx, sz = map(int, seed_point)
    height, width, depth = volume_shape

    if 0 <= sy < height and 0 <= sx < width and 0 <= sz < depth:
        return (sy, sx, sz)
    return None


def _initialize_seed_region(vesselness_map, seeds, min_vesselness=None):
    """
    Inicializa a região de crescimento com as sementes fornecidas.

    Args:
        vesselness_map (np.ndarray): Mapa de vesselness 3D
        seeds (list): Lista de coordenadas (y, x, z) das sementes
        min_vesselness (float, optional): Valor mínimo de vesselness para aceitar semente

    Returns:
        tuple: (mask, visited, queue, initial_sum, initial_count) contendo:
            - mask: Máscara binária inicializada
            - visited: Máscara de visitados
            - queue: Fila de processamento com sementes válidas
            - initial_sum: Soma dos valores das sementes
            - initial_count: Contagem de sementes válidas
    """
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

        # Verificar piso mínimo se especificado
        if min_vesselness is not None and seed_val < min_vesselness:
            continue

        # Adicionar semente válida
        queue.append((sy, sx, sz))
        visited[sy, sx, sz] = True
        mask[sy, sx, sz] = True

        initial_sum += seed_val
        initial_count += 1

    return mask, visited, queue, initial_sum, initial_count


def _calculate_adaptive_floor(
    count, min_start, min_end, switch_at_voxels, smooth_relaxation
):
    """
    Calcula o piso (floor) adaptativo baseado no número de voxels processados.

    Args:
        count (int): Número atual de voxels na região
        min_start (float): Piso inicial (mais restritivo)
        min_end (float): Piso final relaxado (menos restritivo)
        switch_at_voxels (int): Número de voxels para trocar de piso
        smooth_relaxation (bool): Se True, usa transição suave; se False, degrau

    Returns:
        float: Valor do piso atual
    """
    if count >= switch_at_voxels:
        return min_end

    if smooth_relaxation:
        # Transição linear suave
        progress = count / switch_at_voxels
        return min_start - (min_start - min_end) * progress
    else:
        # Transição em degrau
        return min_start


def _calculate_comparison_mean(
    comparison_window,
    use_running_mean,
    current_value,
    running_sum,
    running_count,
    value_history,
):
    """
    Calcula a média de comparação baseada na estratégia escolhida.

    Args:
        comparison_window (int): Janela de comparação
        use_running_mean (bool): Se True, usa média global
        current_value (float): Valor do voxel atual (para window=1)
        running_sum (float): Soma acumulada para média global
        running_count (int): Contagem para média global
        value_history (deque): Histórico para janela fixa

    Returns:
        float: Média de comparação calculada
    """
    if comparison_window == 1:
        return current_value  # Comparação com pai direto

    if use_running_mean:
        return running_sum / running_count  # Média global

    # Média de janela fixa
    return np.mean(value_history)


def _is_neighbor_acceptable(neighbor_val, comparison_mean, current_floor, threshold):
    """
    Verifica se um vizinho deve ser aceito na região.

    Args:
        neighbor_val (float): Valor de vesselness do vizinho
        comparison_mean (float): Média de comparação
        current_floor (float): Piso mínimo atual
        threshold (float): Diferença máxima permitida

    Returns:
        bool: True se o vizinho deve ser aceito
    """
    if neighbor_val < current_floor:
        return False

    if abs(neighbor_val - comparison_mean) > threshold:
        return False

    return True


# =============================================================================
# Funções Públicas
# =============================================================================


def region_growing_segmentation(
    vesselness_map,
    seed_point,
    threshold=None,
    min_vesselness=None,
    max_volume=100000,
    relaxed_floor_factor=0.40,
    switch_at_voxels=1000,
    comparison_window=1,
    smooth_relaxation=False,
    verbose=False,
):
    """
    Segmentação por crescimento de região guiada por vesselness com piso adaptativo.

    Este algoritmo expande uma região a partir de uma semente única, aceitando
    vizinhos quando:
    1. O valor de vesselness está acima do piso (floor) atual
    2. A diferença para a média de comparação é <= threshold

    O piso começa restritivo e relaxa progressivamente após processar um número
    especificado de voxels, permitindo capturar regiões com vesselness mais baixo.

    Args:
        vesselness_map (np.ndarray): Volume 3D com valores de vesselness (Frangi)
        seed_point (tuple): Coordenadas (y, x, z) da semente inicial
        threshold (float, optional): Delta máximo de vesselness para aceitar vizinho.
            Se None, usa (max - min) / 10. Default: None
        min_vesselness (float, optional): Piso inicial de vesselness.
            Se None, usa 5% do valor máximo. Default: None
        max_volume (int): Limite máximo de voxels a aceitar. Default: 100000
        relaxed_floor_factor (float): Fator multiplicativo para o piso relaxado.
            O piso final será min_vesselness * relaxed_floor_factor. Default: 0.40
        switch_at_voxels (int): Número de voxels após o qual relaxar o piso.
            Default: 1000
        comparison_window (int or str): Estratégia para calcular média de comparação:
            - 1: Compara com valor do pai direto
            - N > 1: Média dos últimos N voxels aceitos
            - -1 ou "all": Média global de todos os voxels aceitos
            Default: 1
        smooth_relaxation (bool): Se True, relaxa o piso de forma linear e suave.
            Se False, muda de min_start para min_end em degrau. Default: False
        verbose (bool): Se True, imprime informações durante o crescimento. Default: False

    Returns:
        np.ndarray: Máscara binária 3D (dtype=uint8) com a segmentação

    Example:
        >>> vesselness = compute_frangi_filter(volume)
        >>> seed = (256, 256, 50)  # Centro da aorta
        >>> mask = region_growing_segmentation(
        ...     vesselness, seed, threshold=0.05, max_volume=50000
        ... )
        >>> print(f"Segmentados {mask.sum()} voxels")

    Note:
        - Use comparison_window=1 para crescimento mais conservador
        - Use comparison_window=-1 para regiões mais homogêneas
        - Ajuste relaxed_floor_factor para controlar quão permissivo fica
    """
    HEIGHT, WIDTH, DEPTH = vesselness_map.shape
    v_max, v_min = np.max(vesselness_map), np.min(vesselness_map)

    # Configurar parâmetros padrão
    if threshold is None:
        threshold = (v_max - v_min) / 10
    if min_vesselness is None:
        min_vesselness = v_max * 0.05

    # Configurar janela de comparação
    use_running_mean = False
    value_history = None

    if comparison_window == -1 or isinstance(comparison_window, str):
        comparison_window = -1
        use_running_mean = True
    elif comparison_window > 1:
        value_history = deque(maxlen=comparison_window)

    # Configurar pisos inicial e relaxado
    min_start = float(min_vesselness)
    min_end = min_start * relaxed_floor_factor

    # Validar e inicializar semente
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

    # Inicializar estruturas
    mask = np.zeros_like(vesselness_map, dtype=np.uint8)
    visited = np.zeros_like(vesselness_map, dtype=bool)
    queue = deque([(sy, sx, sz)])

    visited[sy, sx, sz] = True
    mask[sy, sx, sz] = 1
    count = 1

    # Estatísticas para média global
    running_sum = float(seed_val)
    running_count = 1

    # Histórico para janela fixa
    if comparison_window > 1:
        value_history.append(seed_val)

    if verbose:
        window_desc = "todos" if use_running_mean else str(comparison_window)
        print(
            f"RG | Seed={seed_val:.4f} | Floor: {min_start:.4f}→{min_end:.4f} "
            f"@ {switch_at_voxels} | Window={window_desc}"
        )

    # Loop principal de crescimento
    while queue:
        if count >= max_volume:
            if verbose:
                print(f"Volume máximo atingido: {max_volume}")
            break

        cy, cx, cz = queue.popleft()
        current_val = vesselness_map[cy, cx, cz]

        # Calcular média de comparação
        comparison_mean = _calculate_comparison_mean(
            comparison_window,
            use_running_mean,
            current_val,
            running_sum,
            running_count,
            value_history,
        )

        # Calcular piso adaptativo
        current_floor = _calculate_adaptive_floor(
            count, min_start, min_end, switch_at_voxels, smooth_relaxation
        )

        # Processar vizinhos
        for dy, dx, dz in NEIGHBORS_26:
            ny, nx, nz = cy + dy, cx + dx, cz + dz

            # Validar limites
            if not (0 <= ny < HEIGHT and 0 <= nx < WIDTH and 0 <= nz < DEPTH):
                continue
            if visited[ny, nx, nz]:
                continue

            neighbor_val = vesselness_map[ny, nx, nz]

            # Verificar aceitação do vizinho
            if _is_neighbor_acceptable(
                neighbor_val, comparison_mean, current_floor, threshold
            ):
                mask[ny, nx, nz] = 1
                visited[ny, nx, nz] = True
                queue.append((ny, nx, nz))
                count += 1

                # Atualizar estatísticas
                if use_running_mean:
                    running_sum += neighbor_val
                    running_count += 1
                elif comparison_window > 1:
                    value_history.append(neighbor_val)

    if verbose:
        print(f"Voxels segmentados: {count}")

    return mask


def region_growing_article(
    vesselness_map, seeds, threshold=None, min_vesselness=None, max_volume=None
):
    """
    Crescimento de região baseado no artigo de Sukanya et al. (2020).

    Este algoritmo implementa segmentação de artérias coronárias expandindo a partir
    de múltiplas sementes (óstios). Vizinhos são aceitos quando a diferença entre
    seu valor de vesselness e a média global da região é menor que um threshold.

    A diferença principal em relação ao `region_growing_segmentation` é que este
    usa sempre a média global de TODOS os voxels aceitos como comparação, enquanto
    o outro permite diferentes estratégias de comparação.

    Algoritmo:
    1. Inicializa região com todas as sementes (óstios)
    2. Calcula média inicial (R_mean)
    3. Para cada vizinho não visitado:
       a. Verifica se vesselness >= min_vesselness (se especificado)
       b. Calcula diferença |vizinho - R_mean|
       c. Se diferença < threshold, aceita vizinho
       d. Atualiza R_mean dinamicamente

    Args:
        vesselness_map (np.ndarray): Mapa de Vesselness 3D (Frangi), normalizado [0, 1]
        seeds (list): Lista de coordenadas dos óstios [(y, x, z), ...]
        threshold (float, optional): Threshold de diferença de vesselness (T_vm).
            Como o mapa é normalizado 0-1, valores típicos são baixos (ex: 0.05).
            Se None, usa (max - min) / 10. Default: None
        min_vesselness (float, optional): Valor mínimo de vesselness para aceitar voxel.
            Útil para evitar expansão em regiões de baixo contraste.
            Se None, não aplica filtro de valor mínimo. Default: None
        max_volume (int, optional): Limite de segurança para número máximo de voxels.
            Previne expansão descontrolada. Se None, sem limite. Default: None

    Returns:
        np.ndarray: Máscara binária 3D (dtype=bool) com a segmentação

    Example:
        >>> vesselness = compute_frangi_filter(volume)
        >>> ostia_coords = [(250, 200, 45), (260, 210, 45)]  # Óstios L/R
        >>> mask = region_growing_article(
        ...     vesselness, ostia_coords, threshold=0.05, min_vesselness=0.1
        ... )
        >>> print(f"Artérias: {mask.sum()} voxels")

    Note:
        - Projetado para mapas de vesselness normalizados [0, 1]
        - Thresholds típicos: 0.03-0.10 para vesselness normalizado
        - A média é atualizada dinamicamente após aceitar cada voxel
        - Use min_vesselness para evitar vazamento em tecidos não vasculares

    References:
        Sukanya et al. (2020) - Segmentação de artérias coronárias em CCTA
    """
    # Configurar parâmetros padrão
    if threshold is None:
        threshold = (vesselness_map.max() - vesselness_map.min()) / 10.0

    if min_vesselness is not None:
        min_vesselness = float(min_vesselness)

    # Inicializar região com sementes
    mask, visited, queue, current_sum, current_count = _initialize_seed_region(
        vesselness_map, seeds, min_vesselness
    )

    if current_count == 0:
        print("Nenhuma semente válida fornecida.")
        return mask

    # Calcular média inicial
    region_mean = float(current_sum / current_count)

    print(
        f"Iniciando crescimento de região com {current_count} sementes. "
        f"Média inicial: {region_mean:.4f}"
    )

    dims = vesselness_map.shape

    # Loop principal de crescimento
    while queue:
        # Verificar limite de voxels
        if max_volume and current_count >= max_volume:
            print(f"Limite de voxels atingido: {max_volume}")
            break

        # Processar voxel atual
        cy, cx, cz = queue.popleft()

        # Processar vizinhos em 26-vizinhança
        for dy, dx, dz in NEIGHBORS_26:
            ny, nx, nz = cy + dy, cx + dx, cz + dz

            # Validar limites
            if not (0 <= ny < dims[0] and 0 <= nx < dims[1] and 0 <= nz < dims[2]):
                continue

            # Pular se já visitado
            if visited[ny, nx, nz]:
                continue

            visited[ny, nx, nz] = True

            # Obter valor do vizinho
            neighbor_val = float(vesselness_map[ny, nx, nz])

            # Verificar piso mínimo (se definido)
            if min_vesselness is not None and neighbor_val < min_vesselness:
                continue

            # Calcular diferença com a média da região
            delta_vm = abs(neighbor_val - region_mean)

            # Aceitar se diferença for pequena
            if delta_vm < threshold:
                mask[ny, nx, nz] = True
                queue.append((ny, nx, nz))

                # Atualizar média da região dinamicamente
                current_sum += neighbor_val
                current_count += 1
                region_mean = float(current_sum / current_count)

    print(
        f"Segmentação concluída. Voxels: {current_count}. "
        f"Média final: {region_mean:.4f}\n"
    )

    return mask
