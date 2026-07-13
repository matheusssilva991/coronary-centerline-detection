"""Segmentação da aorta em volumes 3D usando Level Set.

Este módulo implementa segmentação baseada em contornos ativos geodésicos
morfológicos (Morphological Geodesic Active Contour - MGAC) usando círculos
detectados como inicialização.
"""

import numpy as np
from skimage.draw import disk
from skimage.segmentation import (
    inverse_gaussian_gradient,
    morphological_geodesic_active_contour,
)
from skimage.morphology import ball
from typing import Any, Dict, Sequence
from numpy.typing import NDArray

# Operação morfológica reutilizada do pacote de processamento.
from ..processing.binary_operations import binary_opening

# Utilitários de GPU usados em trechos pontuais do level set/morfologia.
from ..processing.gpu_utils import GPU_AVAILABLE, to_gpu, to_cpu, cu_ndi, cp


# =============================================================================
# Funções Auxiliares Privadas
# =============================================================================


def _calculate_roi_bounds(
    detected_circles: Sequence[Dict[str, Any]],
    volume_shape: Sequence[int],
    roi_margin: int,
) -> Dict[str, int]:
    """
    Calcula os limites da região de interesse (ROI) baseado nos círculos detectados.

    Args:
        detected_circles (list): Lista de dicionários com círculos detectados
        volume_shape (tuple): Shape do volume 3D (altura, largura, profundidade)
        roi_margin (int): Margem extra em voxels ao redor da ROI

    Returns:
        dict: Dicionário com os limites da ROI contendo:
            - 'x_min', 'x_max': Limites no eixo x
            - 'y_min', 'y_max': Limites no eixo y
            - 'z_min', 'z_max': Limites no eixo z
    """
    # A ROI acompanha as fatias onde os círculos foram detectados.
    slice_indices = [int(c["slice_index"]) for c in detected_circles]
    z_min = max(0, min(slice_indices) - roi_margin)
    z_max = min(volume_shape[2], max(slice_indices) + roi_margin + 1)

    # Expande x/y pelo maior raio detectado e pela margem configurada.
    x_coords = [c["center_x"] for c in detected_circles]
    y_coords = [c["center_y"] for c in detected_circles]
    radii = [c["radius"] for c in detected_circles]
    max_radius = max(radii) if radii else 50

    x_min = max(0, int(min(x_coords) - max_radius - roi_margin))
    x_max = min(volume_shape[1], int(max(x_coords) + max_radius + roi_margin))
    y_min = max(0, int(min(y_coords) - max_radius - roi_margin))
    y_max = min(volume_shape[0], int(max(y_coords) + max_radius + roi_margin))

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "z_min": z_min,
        "z_max": z_max,
    }


def _adjust_circles_to_roi(
    detected_circles: Sequence[Dict[str, Any]], roi_bounds: Dict[str, int]
) -> Sequence[Dict[str, Any]]:
    """
    Ajusta as coordenadas dos círculos para o sistema de coordenadas da ROI.

    Args:
        detected_circles (list): Lista de círculos em coordenadas globais
        roi_bounds (dict): Limites da ROI com 'x_min', 'y_min', 'z_min', etc.

    Returns:
        list: Lista de círculos com coordenadas ajustadas para a ROI
    """
    roi_circles = []
    z_min = roi_bounds["z_min"]
    z_max = roi_bounds["z_max"]
    x_min = roi_bounds["x_min"]
    y_min = roi_bounds["y_min"]

    for c in detected_circles:
        if z_min <= c["slice_index"] < z_max:
            # Desloca centro/fatia para o sistema local da ROI.
            roi_c = {
                "slice_index": int(c["slice_index"]) - z_min,
                "center_x": c["center_x"] - x_min,
                "center_y": c["center_y"] - y_min,
                "radius": c["radius"],
            }
            roi_circles.append(roi_c)

    return roi_circles


def _initialize_level_set_from_circles(
    volume_shape: Sequence[int],
    circles: Sequence[Dict[str, Any]],
    radius_reduction_factor: float = 0.8,
) -> NDArray[Any]:
    """
    Inicializa a máscara do level set usando círculos detectados como sementes.

    Cria uma máscara binária onde cada círculo é desenhado como um disco
    preenchido na fatia correspondente.

    Args:
        volume_shape (tuple): Shape do volume (altura, largura, profundidade)
        circles (list): Lista de dicionários com círculos contendo
            'slice_index', 'center_x', 'center_y', 'radius'
        radius_reduction_factor (float): Fator para reduzir o raio inicial.
            Valores < 1.0 criam sementes menores que os círculos detectados.
            Default: 0.8

    Returns:
        np.ndarray: Máscara binária 3D (dtype=int8) com as sementes inicializadas
    """
    init_level_set = np.zeros(volume_shape, dtype=np.int8)
    height, width = volume_shape[:2]

    for circle in circles:
        # Cada círculo vira uma semente circular na fatia correspondente.
        slice_idx = int(circle["slice_index"])
        cx = circle["center_x"]
        cy = circle["center_y"]
        r = max(1, circle["radius"] * radius_reduction_factor)

        # Desenhar disco preenchido na fatia
        rr, cc = disk((cy, cx), r, shape=(height, width))
        init_level_set[rr, cc, slice_idx] = 1

    return init_level_set


def build_circle_trajectory_envelope(
    volume_shape: Sequence[int],
    detected_circles: Sequence[Dict[str, Any]],
    radius_factor: float = 1.5,
) -> NDArray[np.uint8]:
    """Cria um envelope 3D interpolado ao redor da trajetória dos círculos.

    O envelope limita a máscara da aorta à região anatomicamente acompanhada
    pelo rastreamento circular. Centros e raios são interpolados nas fatias sem
    círculo explícito entre a primeira e a última detecção.
    """
    if radius_factor <= 0:
        raise ValueError("radius_factor deve ser maior que zero")
    if not detected_circles:
        return np.zeros(volume_shape, dtype=np.uint8)

    circles_by_slice: dict[int, Dict[str, Any]] = {}
    for circle in detected_circles:
        slice_index = int(circle["slice_index"])
        if 0 <= slice_index < volume_shape[2]:
            circles_by_slice[slice_index] = circle
    if not circles_by_slice:
        return np.zeros(volume_shape, dtype=np.uint8)

    source_slices = np.array(sorted(circles_by_slice), dtype=float)
    centers_x = np.array(
        [circles_by_slice[int(z)]["center_x"] for z in source_slices], dtype=float
    )
    centers_y = np.array(
        [circles_by_slice[int(z)]["center_y"] for z in source_slices], dtype=float
    )
    radii = np.array(
        [circles_by_slice[int(z)]["radius"] for z in source_slices], dtype=float
    )
    target_slices = np.arange(int(source_slices[0]), int(source_slices[-1]) + 1)
    interp_x = np.interp(target_slices, source_slices, centers_x)
    interp_y = np.interp(target_slices, source_slices, centers_y)
    interp_radii = np.interp(target_slices, source_slices, radii)

    envelope = np.zeros(volume_shape, dtype=np.uint8)
    height, width = volume_shape[:2]
    for z, center_x, center_y, radius in zip(
        target_slices, interp_x, interp_y, interp_radii
    ):
        rr, cc = disk(
            (center_y, center_x),
            max(1.0, radius * radius_factor),
            shape=(height, width),
        )
        envelope[rr, cc, int(z)] = 1
    return envelope


def restrict_mask_to_circle_trajectory(
    aorta_mask: NDArray[Any],
    detected_circles: Sequence[Dict[str, Any]],
    radius_factor: float,
) -> NDArray[np.uint8]:
    """Remove da máscara voxels fora do envelope da trajetória circular."""
    envelope = build_circle_trajectory_envelope(
        aorta_mask.shape,
        detected_circles,
        radius_factor=radius_factor,
    )
    return (aorta_mask.astype(bool) & envelope.astype(bool)).astype(np.uint8)


def correct_anomalous_aorta_slices(
    aorta_mask: NDArray[Any],
    detected_circles: Sequence[Dict[str, Any]],
    area_ratio_threshold: float,
    radius_factor: float = 1.75,
) -> NDArray[np.uint8]:
    """Restringe somente fatias cuja área excede muito o círculo rastreado.

    Ao contrário do envelope global, fatias compatíveis com a trajetória são
    preservadas integralmente. A correção atua apenas quando a razão entre a
    área segmentada e a área do círculo ultrapassa ``area_ratio_threshold``.
    """
    if area_ratio_threshold <= 0:
        raise ValueError("area_ratio_threshold deve ser maior que zero")
    if radius_factor <= 0:
        raise ValueError("radius_factor deve ser maior que zero")

    corrected = np.asarray(aorta_mask, dtype=np.uint8).copy()
    height, width, depth = corrected.shape
    for circle in detected_circles:
        z = int(circle["slice_index"])
        if not 0 <= z < depth:
            continue
        radius = max(float(circle["radius"]), 1.0)
        expected_area = np.pi * radius**2
        segmented_area = float(corrected[:, :, z].sum())
        if segmented_area <= area_ratio_threshold * expected_area:
            continue

        # Recorta apenas a fatia anômala, preservando uma margem sobre o círculo.
        allowed = np.zeros((height, width), dtype=bool)
        rr, cc = disk(
            (float(circle["center_y"]), float(circle["center_x"])),
            radius * radius_factor,
            shape=(height, width),
        )
        allowed[rr, cc] = True
        corrected[:, :, z] = (
            corrected[:, :, z].astype(bool) & allowed
        ).astype(np.uint8)
    return corrected


# =============================================================================
# Funções Públicas
# =============================================================================


def level_set_segmentation(
    volume_ccta: NDArray[Any],
    detected_circles: Sequence[Dict[str, Any]],
    num_iter: int = 50,
    smoothing: int = 1,
    balloon: int = 1,
    threshold: Any = "auto",
    radius_reduction_factor: float = 0.8,
    roi_margin: int = 10,
    use_roi: bool = True,
    alpha: float = 1000,
    sigma: float = 2,
    use_gpu: bool = False,
) -> NDArray[Any]:
    """
    Segmenta a aorta usando Level Set 3D inicializado com círculos detectados.

    Esta função implementa segmentação por contorno ativo geodésico morfológico,
    usando os círculos detectados pela transformada de Hough como inicialização.
    Opcionalmente pode processar apenas uma região de interesse (ROI) para
    maior eficiência computacional.

    O algoritmo segue estas etapas:
    1. Calcula ROI baseada nos círculos detectados (se use_roi=True)
    2. Inicializa level set desenhando discos nos círculos detectados
    3. Calcula gradiente inverso para guiar a evolução do contorno
    4. Aplica contorno ativo geodésico morfológico
    5. Retorna máscara no volume completo (se usou ROI)

    Args:
        volume_ccta (np.ndarray): Volume 3D original (altura, largura, profundidade),
            já pré-processado/normalizado
        detected_circles (list): Lista de dicionários, cada um contendo:
            - 'slice_index' (int): Índice da fatia
            - 'center_x' (float): Coordenada x do centro
            - 'center_y' (float): Coordenada y do centro
            - 'radius' (float): Raio do círculo
        num_iter (int): Número de iterações do algoritmo Level Set. Default: 50
        smoothing (int): Número de iterações de suavização do contorno a cada
            passo. Valores maiores = contornos mais suaves. Default: 1
        balloon (int): Força de expansão (+) ou contração (-) do contorno.
            Default: 1 (leve expansão)
        threshold (str or float): Critério de parada. 'auto' usa critério
            automático baseado no gradiente. Default: 'auto'
        radius_reduction_factor (float): Fator multiplicativo para reduzir o
            raio inicial dos círculos (0.0-1.0). Sementes menores permitem que
            o contorno expanda até as bordas. Default: 0.8
        roi_margin (int): Margem extra em voxels ao redor da ROI para incluir
            contexto adicional. Default: 10
        use_roi (bool): Se True, processa apenas ROI ao redor dos círculos.
            Se False, processa volume completo. Default: True
        alpha (float): Sensibilidade às bordas no cálculo do gradiente.
            Valores maiores = bordas mais fracas também influenciam o contorno.
            Default: 1000
        sigma (float): Desvio padrão da suavização Gaussiana antes do gradiente.
            Valores maiores = mais suavização, menos sensibilidade a ruído.
            Default: 2

    Returns:
        np.ndarray: Máscara binária 3D (dtype=int8) com a segmentação da aorta,
            com o mesmo shape do volume_ccta de entrada. Valores: 0 (fundo) e 1 (aorta)

    Example:
        >>> volume = load_ccta_volume()  # shape: (512, 512, 200)
        >>> circles = detect_aorta_circles(volume, ...)
        >>> mask = level_set_segmentation(
        ...     volume, circles, num_iter=100, balloon=2
        ... )
        >>> print(f"Volume aorta: {mask.sum()} voxels")

    Note:
        - Para volumes grandes, use_roi=True é altamente recomendado
        - O parâmetro balloon controla se o contorno expande ou contrai
        - Ajuste alpha e sigma se houver muito ruído ou bordas fracas
    """
    if volume_ccta.ndim != 3:
        raise ValueError(f"volume_ccta deve ser 3D, recebido shape={volume_ccta.shape}")
    if not detected_circles:
        return np.zeros_like(volume_ccta, dtype=np.int8)

    # Determinar ROI e volume de trabalho
    if use_roi and len(detected_circles) > 0:
        roi_bounds = _calculate_roi_bounds(
            detected_circles, volume_ccta.shape, roi_margin
        )

        # Recorta a ROI ao redor dos círculos para reduzir custo do level set.
        roi_volume = volume_ccta[
            roi_bounds["y_min"] : roi_bounds["y_max"],
            roi_bounds["x_min"] : roi_bounds["x_max"],
            roi_bounds["z_min"] : roi_bounds["z_max"],
        ]

        # Converte os círculos globais para coordenadas locais da ROI.
        work_circles = _adjust_circles_to_roi(detected_circles, roi_bounds)
        work_volume = roi_volume
    else:
        work_volume = volume_ccta
        work_circles = detected_circles
        roi_bounds = {"x_min": 0, "y_min": 0, "z_min": 0}

    # Inicializa o contorno ativo com discos nas fatias onde a aorta foi localizada.
    init_level_set = _initialize_level_set_from_circles(
        work_volume.shape, work_circles, radius_reduction_factor
    )

    # Calcula o mapa de bordas que guia a evolução do contorno.
    if use_gpu and GPU_AVAILABLE:
        # Acelera blur + Sobel na GPU e transfere uma única vez para CPU.
        vol_gpu = to_gpu(work_volume.astype(np.float32))
        blurred = cu_ndi.gaussian_filter(vol_gpu, sigma=sigma)
        gx = cu_ndi.sobel(blurred, axis=1)
        gy = cu_ndi.sobel(blurred, axis=0)
        gmag = cp.sqrt(gx ** 2 + gy ** 2)
        g_gpu = 1.0 / (1.0 + alpha * (gmag ** 2))
        gimage = to_cpu(g_gpu)
    else:
        gimage = inverse_gaussian_gradient(work_volume, alpha=alpha, sigma=sigma)

    # Evolui o contorno ativo até a máscara refinada da aorta.
    refined_segmentation = morphological_geodesic_active_contour(
        gimage,
        num_iter=num_iter,
        init_level_set=init_level_set,
        smoothing=smoothing,
        balloon=balloon,
        threshold=threshold,
    )

    # Reinsere a máscara da ROI no volume completo.
    if use_roi and len(detected_circles) > 0:
        full_mask = np.zeros_like(volume_ccta, dtype=np.int8)
        full_mask[
            roi_bounds["y_min"] : roi_bounds["y_max"],
            roi_bounds["x_min"] : roi_bounds["x_max"],
            roi_bounds["z_min"] : roi_bounds["z_max"],
        ] = refined_segmentation
        return full_mask

    return refined_segmentation


def remove_leaks_morphology(
    mask_3d: NDArray[Any],
    radius: int = 3,
    use_gpu: bool = False,
) -> NDArray[Any]:
    """
    Remove vazamentos e ruído da máscara 3D usando abertura morfológica.

    A abertura morfológica (erosão seguida de dilatação) é eficaz para:
    - Remover pequenas conexões espúrias (vazamentos)
    - Eliminar ruído e pequenos componentes isolados
    - Suavizar o contorno da máscara
    - Separar objetos que estão fracamente conectados

    O tamanho do elemento estruturante (raio) controla a escala dos artefatos
    removidos. Vazamentos maiores que o raio serão preservados.

    Args:
        mask_3d (np.ndarray): Máscara binária 3D a ser limpa (dtype=bool ou int)
        radius (int): Raio do elemento estruturante esférico (ball) em voxels.
            Valores maiores removem estruturas maiores mas podem alterar
            significativamente a geometria. Default: 3

    Returns:
        np.ndarray: Máscara limpa com o mesmo shape e dtype da entrada

    Example:
        >>> noisy_mask = segment_aorta(volume)
        >>> clean_mask = remove_leaks_morphology(noisy_mask, radius=2)
        >>> print(f"Removed {noisy_mask.sum() - clean_mask.sum()} voxels")

    Note:
        - A operação preserva o tipo de dado da entrada
        - Para máscaras muito ruidosas, considere aplicar múltiplas vezes
          com raios diferentes ou usar outras técnicas de pós-processamento
        - O custo computacional cresce com o raio (O(r³))

    See Also:
        skimage.morphology.opening: Documentação da operação morfológica
        skimage.morphology.ball: Elemento estruturante esférico 3D
    """
    radius = int(radius)
    if radius <= 0:
        return mask_3d.copy()

    kernel = ball(radius)

    # Remove conexões finas/vazamentos com abertura morfológica.
    mask_cleaned = binary_opening(mask_3d, structure=kernel, gpu=bool(use_gpu))

    return mask_cleaned
