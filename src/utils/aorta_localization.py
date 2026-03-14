"""
Módulo para localização e detecção de círculos da aorta em volumes 3D.

Este módulo fornece funções para detectar automaticamente a aorta em imagens
CCTA usando transformada de Hough para detecção de círculos.
"""

import numpy as np
from skimage import feature
from skimage.transform import hough_circle, hough_circle_peaks


# =============================================================================
# Funções Auxiliares Privadas
# =============================================================================


def _calculate_distance(x1, y1, x2, y2):
    """
    Calcula a distância euclidiana entre dois pontos no plano 2D.

    Args:
        x1 (float): Coordenada x do primeiro ponto
        y1 (float): Coordenada y do primeiro ponto
        x2 (float): Coordenada x do segundo ponto
        y2 (float): Coordenada y do segundo ponto

    Returns:
        float: Distância euclidiana entre os dois pontos

    Example:
        >>> _calculate_distance(0, 0, 3, 4)
        5.0
    """
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _detect_circles_in_slice(img_slice, hough_radii, total_num_peaks, canny_sigma):
    """
    Detecta círculos em uma fatia 2D usando Canny edge detection e Hough Transform.

    Esta função aplica o filtro de Canny para detectar bordas na imagem e então
    usa a transformada de Hough para identificar círculos nas bordas detectadas.

    Args:
        img_slice (np.ndarray): Fatia 2D da imagem (altura x largura)
        hough_radii (np.ndarray): Array com os raios (em pixels) a serem testados
        total_num_peaks (int): Número máximo de círculos a detectar
        canny_sigma (float): Desvio padrão do filtro Gaussiano usado no Canny

    Returns:
        tuple: Tupla contendo (accums, cx, cy, radii) onde:
            - accums (np.ndarray): Valores de acumulação de cada círculo
            - cx (np.ndarray): Coordenadas x dos centros dos círculos
            - cy (np.ndarray): Coordenadas y dos centros dos círculos
            - radii (np.ndarray): Raios dos círculos detectados
    """
    edges = feature.canny(img_slice.astype(float), sigma=canny_sigma)
    hough_res = hough_circle(edges, hough_radii)
    return hough_circle_peaks(hough_res, hough_radii, total_num_peaks=total_num_peaks)


def _find_closest_circle(cx, cy, radii, ref_x, ref_y):
    """
    Encontra o círculo mais próximo de um ponto de referência.

    Calcula a distância euclidiana entre cada círculo detectado e o ponto de
    referência, retornando o índice e distância do círculo mais próximo.

    Args:
        cx (np.ndarray): Array com coordenadas x dos centros dos círculos
        cy (np.ndarray): Array com coordenadas y dos centros dos círculos
        radii (np.ndarray): Array com os raios dos círculos (não utilizado no cálculo)
        ref_x (float): Coordenada x do ponto de referência
        ref_y (float): Coordenada y do ponto de referência

    Returns:
        tuple: Tupla (min_idx, min_dist) contendo:
            - min_idx (int): Índice do círculo mais próximo
            - min_dist (float): Distância do círculo mais próximo à referência
    """
    distances = [
        _calculate_distance(cx[i], cy[i], ref_x, ref_y) for i in range(len(cx))
    ]
    min_idx = np.argmin(distances)
    return min_idx, distances[min_idx]


def _is_circle_within_tolerance(
    circle_radius, circle_distance, ref_radius, radius_tolerance, distance_tolerance
):
    """
    Verifica se um círculo está dentro das tolerâncias especificadas.

    Um círculo é considerado válido se tanto a diferença de raio quanto a
    distância do centro estiverem dentro dos limites especificados.

    Args:
        circle_radius (float): Raio do círculo a validar (em pixels)
        circle_distance (float): Distância do círculo à referência (em pixels)
        ref_radius (float): Raio de referência (em pixels)
        radius_tolerance (float): Tolerância máxima de diferença de raio (em pixels)
        distance_tolerance (float): Tolerância máxima de distância (em pixels)

    Returns:
        bool: True se ambas as condições forem satisfeitas, False caso contrário

    Note:
        Esta função usa AND lógico: ambas as condições devem ser satisfeitas.
    """
    radius_diff = abs(circle_radius - ref_radius)
    return circle_distance <= distance_tolerance and radius_diff <= radius_tolerance


def _compute_local_roi_bounds(
    img_shape,
    ref_x,
    ref_y,
    ref_radius,
    distance_tolerance,
    radius_tolerance,
    local_roi_padding,
):
    """
    Calcula limites de uma ROI quadrada local centrada no círculo de referência.

    A ROI é dimensionada para cobrir deslocamentos esperados do centro e
    variações de raio entre fatias consecutivas.
    """
    height, width = img_shape

    search_radius = ref_radius + distance_tolerance + radius_tolerance + local_roi_padding
    half_size = int(np.ceil(max(8.0, search_radius)))

    cx = int(round(ref_x))
    cy = int(round(ref_y))

    x_min = max(0, cx - half_size)
    x_max = min(width, cx + half_size)
    y_min = max(0, cy - half_size)
    y_max = min(height, cy + half_size)

    return x_min, x_max, y_min, y_max


def _process_initial_circle(
    img_slice,
    hough_radii,
    initial_circle,
    neighbor_distance_threshold,
    total_num_peaks,
    canny_sigma,
):
    """
    Processa e refina o círculo inicial detectado usando círculos vizinhos.

    Detecta todos os círculos na fatia inicial e refina o círculo inicial
    calculando a média dos círculos próximos a ele.

    Args:
        img_slice (np.ndarray): Fatia 2D da imagem
        hough_radii (np.ndarray): Array de raios para Hough transform
        initial_circle (dict): Dicionário com o círculo inicial contendo
            'center_x', 'center_y', 'radius' e 'accum'
        neighbor_distance_threshold (float): Distância máxima (em pixels) para
            considerar círculos como vizinhos
        total_num_peaks (int): Número máximo de picos a detectar
        canny_sigma (float): Sigma para o filtro Canny

    Returns:
        dict: Dicionário com círculo refinado contendo:
            - 'center_x' (float): Coordenada x refinada
            - 'center_y' (float): Coordenada y refinada
            - 'radius' (float): Raio refinado
            - 'accum' (float): Valor de acumulação original
    """
    accums, cx, cy, radii = _detect_circles_in_slice(
        img_slice, hough_radii, total_num_peaks, canny_sigma
    )

    ref_x, ref_y, ref_radius = refine_circle_with_neighbors(
        cx,
        cy,
        radii,
        initial_circle["center_x"],
        initial_circle["center_y"],
        neighbor_distance_threshold,
    )

    if ref_radius is None:
        ref_radius = initial_circle["radius"]

    return {
        "center_x": ref_x,
        "center_y": ref_y,
        "radius": ref_radius,
        "accum": initial_circle["accum"],
    }


def _process_slice(
    img_slice,
    hough_radii,
    reference_circle,
    radius_tolerance,
    distance_tolerance,
    neighbor_distance_threshold,
    total_num_peaks,
    canny_sigma,
    use_local_roi=True,
    local_roi_padding=20,
):
    """
    Processa uma fatia individual, detectando e validando círculos.

    Esta função detecta círculos na fatia, encontra o mais próximo da referência,
    valida se está dentro das tolerâncias e refina o resultado com base em vizinhos.

    Args:
        img_slice (np.ndarray): Fatia 2D da imagem
        hough_radii (np.ndarray): Array de raios para Hough transform
        reference_circle (dict): Círculo de referência da fatia anterior contendo
            'center_x', 'center_y', 'radius' e 'slice_index'
        radius_tolerance (float): Tolerância máxima de raio em pixels
        distance_tolerance (float): Tolerância máxima de distância em pixels
        neighbor_distance_threshold (float): Distância para considerar círculos
            como vizinhos durante o refinamento
        total_num_peaks (int): Número de picos para detecção
        canny_sigma (float): Sigma para o filtro Canny
        use_local_roi (bool): Se True, busca círculos primeiro em ROI local
            centrada no círculo de referência. Default: True
        local_roi_padding (int): Margem extra em pixels usada na ROI local.
            Default: 20

    Returns:
        dict or None or str: Retorna:
            - dict: Círculo detectado com 'center_x', 'center_y', 'radius', 'accum'
            - None: Nenhum círculo foi detectado na fatia
            - 'out_of_tolerance': Círculo detectado mas fora das tolerâncias

    Note:
        A string 'out_of_tolerance' é usada como sinal para interromper o processamento
        sequencial de fatias quando a geometria da aorta muda significativamente.
    """
    # Encontrar círculo mais próximo da referência
    ref_x = reference_circle["center_x"]
    ref_y = reference_circle["center_y"]
    ref_radius = reference_circle["radius"]

    # Detectar círculos priorizando ROI local para reduzir custo computacional
    if use_local_roi:
        x_min, x_max, y_min, y_max = _compute_local_roi_bounds(
            img_slice.shape,
            ref_x,
            ref_y,
            ref_radius,
            distance_tolerance,
            radius_tolerance,
            local_roi_padding,
        )

        roi_slice = img_slice[y_min:y_max, x_min:x_max]
        accums, cx, cy, radii = _detect_circles_in_slice(
            roi_slice, hough_radii, total_num_peaks, canny_sigma
        )

        # Converter coordenadas da ROI para coordenadas da fatia completa
        if len(radii) > 0:
            cx = cx + x_min
            cy = cy + y_min
        else:
            # Fallback para robustez quando ROI não retorna candidatos
            accums, cx, cy, radii = _detect_circles_in_slice(
                img_slice, hough_radii, total_num_peaks, canny_sigma
            )
    else:
        accums, cx, cy, radii = _detect_circles_in_slice(
            img_slice, hough_radii, total_num_peaks, canny_sigma
        )

    if len(radii) == 0:
        return None

    min_idx, min_dist = _find_closest_circle(cx, cy, radii, ref_x, ref_y)

    # Verificar se está dentro das tolerâncias
    if not _is_circle_within_tolerance(
        radii[min_idx], min_dist, ref_radius, radius_tolerance, distance_tolerance
    ):
        slice_idx = reference_circle.get("slice_index", "N/A")
        print(
            f"Parada na fatia {slice_idx - 1}: Δr={abs(radii[min_idx] - ref_radius):.2f} ou dist={min_dist:.2f}"
        )
        return "out_of_tolerance"

    # Refinar círculo com vizinhos
    cx_mean, cy_mean, radius_mean = refine_circle_with_neighbors(
        cx,
        cy,
        radii,
        float(cx[min_idx]),
        float(cy[min_idx]),
        neighbor_distance_threshold,
    )

    if radius_mean is None:
        radius_mean = float(radii[min_idx])

    return {
        "center_x": cx_mean,
        "center_y": cy_mean,
        "radius": radius_mean,
        "accum": float(accums[min_idx]),
    }


# =============================================================================
# Funções Públicas
# =============================================================================


def detect_initial_circle(
    img_slice, hough_radii, quadrant_offset=(30, 30), total_num_peaks=10, canny_sigma=3
):
    """
    Detecta o círculo inicial no primeiro quadrante da fatia.

    Esta função é usada para encontrar o círculo da aorta na fatia inicial
    (tipicamente a última fatia do volume). Ela restringe a busca ao primeiro
    quadrante da imagem para focar na região onde a aorta ascendente geralmente
    está localizada.

    Args:
        img_slice (np.ndarray): Fatia 2D da imagem (altura x largura)
        hough_radii (np.ndarray): Array de raios em pixels a serem testados
        quadrant_offset (tuple): Tupla (offset_x, offset_y) em pixels para ajustar
            a divisão dos quadrantes. Valores positivos deslocam o centro usado
            para definir quadrantes. Default: (30, 30)
        total_num_peaks (int): Número máximo de picos/círculos a detectar. Default: 10
        canny_sigma (float): Desvio padrão do filtro Gaussiano no Canny. Default: 3

    Returns:
        dict or None: Retorna None se nenhum círculo for encontrado. Caso contrário,
            retorna dicionário com:
            - 'center_x' (float): Coordenada x do centro do círculo
            - 'center_y' (float): Coordenada y do centro do círculo
            - 'radius' (float): Raio do círculo em pixels
            - 'accum' (float): Valor de acumulação do detector de Hough

    Note:
        O primeiro quadrante é definido como: x > center_x e y < center_y,
        onde center_x e center_y são calculados com base no offset fornecido.
    """
    accums, cx, cy, radii = _detect_circles_in_slice(
        img_slice, hough_radii, total_num_peaks, canny_sigma
    )

    if len(accums) == 0:
        return None

    # Calcular centro da imagem e definir primeiro quadrante
    height, width = img_slice.shape
    center_x = (width // 2) - quadrant_offset[0]
    center_y = (height // 2) + quadrant_offset[1]

    # Filtrar círculos no primeiro quadrante
    first_quad_indices = [
        i for i in range(len(cx)) if cx[i] > center_x and cy[i] < center_y
    ]

    if not first_quad_indices:
        return None

    # Selecionar o círculo com maior acumulador no primeiro quadrante
    idx = first_quad_indices[0]
    return {
        "center_x": float(cx[idx]),
        "center_y": float(cy[idx]),
        "radius": float(radii[idx]),
        "accum": float(accums[idx]),
    }


def refine_circle_with_neighbors(cx, cy, radii, ref_x, ref_y, distance_threshold=5):
    """
    Refina um círculo calculando a média ponderada dos círculos vizinhos próximos.

    Esta função melhora a precisão da detecção encontrando todos os círculos
    dentro de uma distância especificada do círculo de referência e calculando
    a média de suas posições e raios.

    Args:
        cx (np.ndarray): Array com coordenadas x dos centros dos círculos detectados
        cy (np.ndarray): Array com coordenadas y dos centros dos círculos detectados
        radii (np.ndarray): Array com os raios dos círculos detectados
        ref_x (float): Coordenada x do círculo de referência
        ref_y (float): Coordenada y do círculo de referência
        distance_threshold (float): Distância máxima em pixels para considerar
            um círculo como vizinho. Default: 5

    Returns:
        tuple: Tupla (x_mean, y_mean, radius_mean) contendo:
            - x_mean (float): Coordenada x refinada (média dos vizinhos)
            - y_mean (float): Coordenada y refinada (média dos vizinhos)
            - radius_mean (float or None): Raio refinado (média dos vizinhos),
              ou None se nenhum vizinho foi encontrado

    Note:
        Se nenhum círculo vizinho for encontrado dentro do threshold, a função
        retorna as coordenadas de referência originais com radius_mean = None.
    """
    nearest_circles = []

    # Calcular distâncias e selecionar círculos próximos
    for i in range(len(cx)):
        dist = _calculate_distance(cx[i], cy[i], ref_x, ref_y)
        if dist <= distance_threshold:
            nearest_circles.append((float(radii[i]), float(cx[i]), float(cy[i])))

    if not nearest_circles:
        return ref_x, ref_y, None

    # Calcular média dos círculos próximos
    radius_mean = np.mean([c[0] for c in nearest_circles])
    x_mean = np.mean([c[1] for c in nearest_circles])
    y_mean = np.mean([c[2] for c in nearest_circles])

    return x_mean, y_mean, radius_mean


def detect_aorta_circles(
    img_volume,
    hough_radii,
    pixel_spacing,
    tol_radius_mm=9.0,
    tol_distance_mm=18.0,
    max_slice_miss_threshold=5,
    neighbor_distance_threshold=5,
    quadrant_offset=(30, 30),
    total_num_peaks_initial=10,
    total_num_peaks=20,
    canny_sigma=3,
    use_local_roi=True,
    local_roi_padding=20,
):
    """
    Detecta círculos da aorta em um volume 3D de forma sequencial.

    Esta é a função principal do módulo. Ela detecta automaticamente a aorta
    em um volume CCTA começando pela última fatia (inferior) e progredindo
    slice por slice em direção ao topo, rastreando a geometria circular da aorta.

    O algoritmo funciona em três etapas:
    1. Detecta o círculo inicial na última fatia (região da aorta ascendente)
    2. Para cada fatia subsequente, detecta círculos e escolhe o mais próximo
    3. Valida continuidade usando tolerâncias de raio e distância

    Args:
        img_volume (np.ndarray): Volume 3D da imagem com shape (altura, largura, num_fatias)
        hough_radii (np.ndarray): Array de raios em pixels a serem testados pela
            transformada de Hough (ex: np.arange(15, 30))
        pixel_spacing (float): Espaçamento de pixels em milímetros, usado para
            converter tolerâncias de mm para pixels
        tol_radius_mm (float): Tolerância máxima de variação de raio entre fatias
            consecutivas, em milímetros. Default: 9.0
        tol_distance_mm (float): Tolerância máxima de distância entre centros de
            círculos em fatias consecutivas, em milímetros. Default: 18.0
        max_slice_miss_threshold (int): Número máximo de fatias consecutivas sem
            detecção antes de interromper o processamento. Default: 5
        neighbor_distance_threshold (float): Distância em pixels para considerar
            círculos como vizinhos durante refinamento. Default: 5
        quadrant_offset (tuple): Offset (x, y) em pixels para definir o primeiro
            quadrante na detecção inicial. Default: (30, 30)
        total_num_peaks_initial (int): Número de picos a detectar na fatia inicial.
            Default: 10
        total_num_peaks (int): Número de picos a detectar nas fatias subsequentes.
            Default: 20
        canny_sigma (float): Desvio padrão do filtro Gaussiano usado no detector
            de bordas Canny. Default: 3
        use_local_roi (bool): Se True, usa busca local por ROI nas fatias
            subsequentes para reduzir custo computacional. Default: True
        local_roi_padding (int): Margem extra (pixels) na ROI local.
            Default: 20

    Returns:
        list: Lista de dicionários, cada um representando um círculo detectado
            em uma fatia. Cada dicionário contém:
            - 'slice_index' (int): Índice da fatia no volume (0-based)
            - 'center_x' (float): Coordenada x do centro em pixels
            - 'center_y' (float): Coordenada y do centro em pixels
            - 'radius' (float): Raio do círculo em pixels
            - 'accum' (float): Valor de acumulação do detector de Hough

            Retorna lista vazia [] se nenhum círculo inicial for detectado.

    Raises:
        IndexError: Se img_volume não tiver 3 dimensões
        ValueError: Se pixel_spacing for <= 0

    Example:
        >>> import numpy as np
        >>> volume = np.random.rand(512, 512, 100)
        >>> radii = np.arange(20, 35)
        >>> circles = detect_aorta_circles(volume, radii, pixel_spacing=0.5)
        >>> print(f"Detectados {len(circles)} círculos")

    Note:
        - O processamento para quando círculos consecutivos excedem as tolerâncias
        - A lista retornada está em ordem decrescente de slice_index
        - As tolerâncias em mm são automaticamente convertidas para pixels
    """
    num_slices = img_volume.shape[2]
    first_slice_idx = num_slices - 1

    # Conversão de tolerâncias de mm para pixels
    radius_tolerance = tol_radius_mm / pixel_spacing
    distance_tolerance = tol_distance_mm / pixel_spacing

    # Detectar e processar círculo inicial
    initial_circle = detect_initial_circle(
        img_volume[:, :, first_slice_idx],
        hough_radii,
        quadrant_offset,
        total_num_peaks_initial,
        canny_sigma,
    )

    if initial_circle is None:
        print("Nenhum círculo inicial detectado.")
        return []

    refined_initial = _process_initial_circle(
        img_volume[:, :, first_slice_idx],
        hough_radii,
        initial_circle,
        neighbor_distance_threshold,
        total_num_peaks_initial,
        canny_sigma,
    )

    detected_circles = [{"slice_index": first_slice_idx, **refined_initial}]

    # Processar fatias restantes (de baixo para cima)
    miss_counter = 0

    for slice_idx in range(first_slice_idx - 1, -1, -1):
        result = _process_slice(
            img_volume[:, :, slice_idx],
            hough_radii,
            detected_circles[-1],
            radius_tolerance,
            distance_tolerance,
            neighbor_distance_threshold,
            total_num_peaks,
            canny_sigma,
            use_local_roi,
            local_roi_padding,
        )

        if result is None:
            miss_counter += 1
            if miss_counter >= max_slice_miss_threshold:
                print(
                    f"Parada: {max_slice_miss_threshold} fatias consecutivas sem detecção."
                )
                break
            continue

        if result == "out_of_tolerance":
            break

        detected_circles.append({"slice_index": slice_idx, **result})
        miss_counter = 0

    return detected_circles
