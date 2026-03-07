"""
Módulo para pré-processamento de imagens médicas CCTA.

Este módulo implementa operações de pré-processamento essenciais para volumes CCTA,
incluindo normalização, redimensionamento, thresholding e extração de componentes
conectados para isolar estruturas anatômicas de interesse.
"""

import numpy as np
import scipy.ndimage as ndi
import cv2
import warnings

# Importa utilitários de GPU
from .gpu_utils import (
    use_gpu,
    to_gpu,
    to_cpu,
    GPU_AVAILABLE,
    cp,
    cu_ndi,
)

# Importa operações binárias (morfológicas + componentes conectados)
from .binary_operations import label, keep_largest_component

# Importa normalização
from .utils import normalize_image


# =============================================================================
# Funções Auxiliares Privadas
# =============================================================================


def _find_largest_component_label(labeled_array):
    """
    Encontra o rótulo do maior componente conectado.

    Args:
        labeled_array (np.ndarray): Array com componentes rotulados

    Returns:
        int or None: Rótulo do maior componente, ou None se não houver componentes
    """
    comp_sizes = np.bincount(labeled_array.ravel())
    comp_sizes[0] = 0  # Ignora o fundo (rótulo 0)

    if len(comp_sizes) <= 1:
        return None

    return np.argmax(comp_sizes)


# =============================================================================
# Funções Públicas - Downscaling
# =============================================================================


def downscale_image_ndi(image, factors, order=3):
    """
    Reduz a resolução espacial da imagem pelos fatores especificados.

    Usa interpolação spline (ordem 3 por padrão) para suavizar a imagem
    durante o redimensionamento, preservando características importantes.

    Args:
        image (np.ndarray): Imagem de entrada de qualquer dimensionalidade
        factors (tuple): Fatores de downscale para cada dimensão.
            Valores > 1 reduzem a resolução, valores < 1 aumentam.
            Ex: (2, 2, 1) reduz x e y por 2, mantém z inalterado
        order (int): Ordem da interpolação spline (0-5).
            0 = nearest, 1 = linear, 3 = cúbica (padrão). Default: 3

    Returns:
        np.ndarray: Imagem redimensionada com shape reduzido pelos fatores

    Example:
        >>> volume = np.random.rand(512, 512, 100)
        >>> downscaled = downscale_image(volume, factors=(2, 2, 1))
        >>> print(f"Shape original: {volume.shape}")
        >>> print(f"Shape reduzido: {downscaled.shape}")
        Shape original: (512, 512, 100)
        Shape reduzido: (256, 256, 100)

    Note:
        - order=3 (cúbica) é bom equilíbrio entre qualidade e velocidade
        - order=1 (linear) é mais rápido mas pode perder detalhes
        - Útil para reduzir custo computacional de processamentos pesados
        - Lembre-se de ajustar pixel_spacing após downscale
    """
    zoom_factors = tuple(1.0 / f for f in factors)
    down_img = ndi.zoom(image, zoom=zoom_factors, order=order)
    return down_img


def downscale_image_opencv(image, factors, interpolation=cv2.INTER_LINEAR):
    """
    Reduz a resolução espacial da imagem pelos fatores especificados usando OpenCV.

    Usa interpolação do OpenCV (linear por padrão) para redimensionar a imagem.
    Para volumes 3D, aplica o redimensionamento slice por slice no plano XY.

    Args:
        image (np.ndarray): Imagem de entrada (2D ou 3D)
        factors (tuple): Fatores de downscale para cada dimensão.
            Valores > 1 reduzem a resolução, valores < 1 aumentam.
            Para 3D: (factor_x, factor_y, factor_z)
            Para 2D: (factor_x, factor_y)
        interpolation (int): Método de interpolação do OpenCV.
            cv2.INTER_LINEAR (padrão) - interpolação bilinear
            cv2.INTER_NEAREST - vizinho mais próximo
            cv2.INTER_CUBIC - interpolação bicúbica
            cv2.INTER_AREA - reamostragem usando área de pixel

    Returns:
        np.ndarray: Imagem redimensionada com shape reduzido pelos fatores

    Example:
        >>> volume = np.random.rand(512, 512, 100)
        >>> downscaled = downscale_image_opencv(volume, factors=(2, 2, 1))
        >>> print(f"Shape original: {volume.shape}")
        >>> print(f"Shape reduzido: {downscaled.shape}")
        Shape original: (512, 512, 100)
        Shape reduzido: (256, 256, 100)

    Note:
        - cv2.INTER_AREA é recomendado para downscaling (redução de resolução)
        - cv2.INTER_CUBIC é recomendado para upscaling (aumento de resolução)
        - cv2.INTER_LINEAR oferece bom equilíbrio entre qualidade e velocidade
        - Para volumes 3D, processa cada slice independentemente no eixo Z
    """
    if image.ndim == 2:
        # Imagem 2D
        new_shape = (
            int(image.shape[1] / factors[1]),  # width
            int(image.shape[0] / factors[0]),  # height
        )
        return cv2.resize(image, new_shape, interpolation=interpolation)

    elif image.ndim == 3:
        # Volume 3D - processa slice por slice
        factor_x, factor_y, factor_z = factors
        new_shape_xy = (
            int(image.shape[1] / factor_y),  # width
            int(image.shape[0] / factor_x),  # height
        )
        new_shape_z = int(image.shape[2] / factor_z)

        # Redimensiona no plano XY
        volume_resized_xy = np.zeros(
            (new_shape_xy[1], new_shape_xy[0], image.shape[2]), dtype=image.dtype
        )

        for z in range(image.shape[2]):
            volume_resized_xy[:, :, z] = cv2.resize(
                image[:, :, z], new_shape_xy, interpolation=interpolation
            )

        # Se factor_z != 1, redimensiona também no eixo Z
        if factor_z != 1:
            volume_final = np.zeros(
                (new_shape_xy[1], new_shape_xy[0], new_shape_z), dtype=image.dtype
            )

            for y in range(new_shape_xy[1]):
                slice_xz = volume_resized_xy[y, :, :]
                resized_xz = cv2.resize(
                    slice_xz,
                    (new_shape_z, new_shape_xy[0]),
                    interpolation=interpolation,
                )
                volume_final[y, :, :] = resized_xz

            return volume_final
        else:
            return volume_resized_xy

    else:
        raise ValueError(f"Imagem deve ser 2D ou 3D, recebido shape: {image.shape}")


def threshold_image(image, min_val=-300, max_val=675):
    """
    Aplica thresholding por faixa de valores na imagem.

    Cria uma máscara onde apenas voxels dentro do intervalo [min_val, max_val]
    são preservados. Útil para isolar tecidos com densidade específica em CCTA.

    Args:
        image (np.ndarray): Imagem de entrada (tipicamente em unidades Hounsfield)
        min_val (int): Valor mínimo do threshold (inclusivo). Default: -300
        max_val (int): Valor máximo do threshold (inclusivo). Default: 675

    Returns:
        tuple: (thresh_img, thresh_mask) contendo:
            - thresh_img (np.ndarray): Imagem com valores fora do range zerados
            - thresh_mask (np.ndarray): Máscara binária (True onde dentro do range)

    Example:
        >>> ccta = load_volume()  # Unidades Hounsfield
        >>> thresh_img, mask = threshold_image(ccta, min_val=-200, max_val=800)
        >>> print(f"Voxels preservados: {mask.sum()}")

    Note:
        - Valores padrão (-300, 675) são típicos para tecidos vasculares/cardíacos
        - min_val=-300 remove ar/pulmão
        - max_val=675 remove calcificações muito densas
        - Máscara pode ser usada para operações morfológicas subsequentes
    """
    thresh_mask = (image >= min_val) & (image <= max_val)
    thresh_img = thresh_mask * image
    return thresh_img, thresh_mask


def threshold_image_with_offset(image, min_val=-300, max_val=675):
    """
    Aplica threshold e desloca valores para garantir que todos sejam positivos.

    Similar a threshold_image(), mas adiciona offset para evitar valores negativos,
    útil para algoritmos que requerem valores não-negativos.

    Args:
        image (np.ndarray): Imagem de entrada
        min_val (int): Valor mínimo do threshold. Default: -300
        max_val (int): Valor máximo do threshold. Default: 675

    Returns:
        tuple: (thresh_img, thresh_mask, offset) contendo:
            - thresh_img (np.ndarray): Imagem thresholded com offset aplicado
            - thresh_mask (np.ndarray): Máscara binária
            - offset (float): Valor do offset adicionado (abs(min_val))

    Example:
        >>> ccta = load_volume()
        >>> thresh_img, mask, offset = threshold_image_with_offset(ccta)
        >>> original = thresh_img - offset  # Recuperar valores originais
        >>> print(f"Offset aplicado: {offset}")

    Note:
        - offset = abs(min_val) garante que min_val → 0
        - Lembre-se de subtrair offset para recuperar valores originais
        - Útil antes de algoritmos que assumem valores não-negativos
    """
    offset = np.abs(min_val)
    image_offset = image + offset

    thresh_mask = (image >= min_val) & (image <= max_val)
    thresh_img = thresh_mask * image_offset

    return thresh_img, thresh_mask, offset


def largest_connected_component(image, mask):
    """
    Extrai o maior componente conectado da máscara e aplica na imagem.

    Identifica regiões conectadas na máscara binária, encontra a maior região
    e mascara a imagem para manter apenas essa região.

    Args:
        image (np.ndarray): Imagem de entrada
        mask (np.ndarray): Máscara binária para análise de componentes

    Returns:
        tuple: (lcc_img, lcc_mask) contendo:
            - lcc_img (np.ndarray): Imagem mascarada com apenas o maior componente
            - lcc_mask (np.ndarray): Máscara binária do maior componente

    Example:
        >>> thresh_img, mask = threshold_image(volume)
        >>> lcc_img, lcc_mask = largest_connected_component(thresh_img, mask)
        >>> print(f"Voxels do maior componente: {lcc_mask.sum()}")

    Note:
        - Usa conectividade completa (26-vizinhança em 3D)
        - Útil para remover ruído e regiões espúrias
        - Se nenhum componente for encontrado, retorna entrada inalterada
    """
    labeled_array, num_features = ndi.label(mask)
    if num_features == 0:
        return image, mask

    largest_comp_label = _find_largest_component_label(labeled_array)
    if largest_comp_label is None:
        return image, mask

    largest_comp_mask = labeled_array == largest_comp_label
    lcc_img = image * largest_comp_mask

    return lcc_img, largest_comp_mask


def run_core_preprocessing_pipeline(
    image,
    downscale_factors,
    min_threshold=-300,
    max_threshold_percentile=99.5,
    lcc_per_slice=True,
    order=3,
    use_opencv=False,
    opencv_interpolation=None,
):
    """
    Executa pipeline completo de pré-processamento em volume CCTA.

    Este pipeline aplica uma sequência de transformações para preparar o volume
    para segmentação: redimensionamento, thresholding adaptativo e extração do
    maior componente conectado para isolar o corpo do paciente.

    Pipeline de execução:
        1. Downscaling: reduz resolução espacial pelos fatores especificados
        2. Threshold adaptativo: calcula limite superior por percentil
        3. Thresholding com offset: aplica limites e garante valores positivos
        4. Extração LCC: mantém maior componente (2D por slice ou 3D total)
        5. Remoção de offset: restaura valores originais

    Args:
        image (np.ndarray): Volume 3D de entrada (tipicamente CCTA em HU)
        downscale_factors (tuple): Fatores de downscale (ex: (2, 2, 1))
        min_threshold (int): Limite inferior de intensidade. Default: -300
        max_threshold_percentile (float): Percentil para calcular limite superior.
            Default: 99.5 (remove 0.5% dos voxels mais brilhantes)
        lcc_per_slice (bool): Se True, aplica LCC em cada slice 2D separadamente.
            Se False, aplica LCC no volume 3D completo. Default: True
        order (int): Ordem de interpolação para downscaling com scipy (0-5).
            Default: 3. Ignorado se use_opencv=True
        use_opencv (bool): Se True, usa OpenCV para downscaling (cv2.resize).
            Se False, usa scipy (ndi.zoom). Default: False
        opencv_interpolation (int): Método de interpolação do OpenCV (ex: cv2.INTER_AREA).
            Ignorado se use_opencv=False. Se None, usa cv2.INTER_AREA. Default: None

    Returns:
        tuple: (down_image, thresh_image, lcc_image, thresh_vals) contendo:
            - down_image (np.ndarray): Volume após downscaling
            - thresh_image (np.ndarray): Volume após thresholding (com offset)
            - lcc_image (np.ndarray): Volume final com maior componente (sem offset)
            - thresh_vals (tuple): Valores de threshold usados (min, max)

    Example:
        >>> ccta = load_volume()  # Shape: (512, 512, 300)
        >>> down, thresh, lcc, vals = run_core_preprocessing_pipeline(
        ...     ccta,
        ...     downscale_factors=(2, 2, 1),
        ...     lcc_per_slice=True,
        ...     use_opencv=True
        ... )
        >>> print(f"Shape reduzido: {down.shape}")  # (256, 256, 300)
        >>> print(f"Thresholds: {vals}")  # (-300, 620) aproximadamente

    Note:
        - lcc_per_slice=True é mais rápido e previne conexões espúrias entre slices
        - lcc_per_slice=False garante conectividade 3D completa
        - max_threshold é adaptativo (percentil) para lidar com variações de contraste
        - Offset é removido no final para preservar valores Hounsfield originais
        - thresh_vals pode ser usado para reproduzir o threshold em outras imagens
        - OpenCV (use_opencv=True) pode ser mais rápido para downscaling 2D
    """
    # 1. Downscaling para reduzir custo computacional
    if use_opencv:
        if opencv_interpolation is None:
            opencv_interpolation = cv2.INTER_AREA
        down_image = downscale_image_opencv(
            image, downscale_factors, interpolation=opencv_interpolation
        )
    else:
        down_image = downscale_image_ndi(image, downscale_factors, order=order)

    # 2. Calcular threshold superior adaptativo por percentil
    thresh_vals = (
        min_threshold,
        int(np.percentile(down_image, max_threshold_percentile)),
    )

    # 3. Aplicar thresholding com offset para valores positivos
    thresh_image, thresh_mask, offset = threshold_image_with_offset(
        down_image, *thresh_vals
    )

    # 4. Extrair maior componente conectado (2D por slice ou 3D completo)
    if lcc_per_slice:
        lcc_image = np.zeros_like(thresh_image, dtype=float)
        for z in range(thresh_image.shape[2]):
            slice_mask = thresh_mask[:, :, z]
            slice_image = thresh_image[:, :, z]
            lcc_slice, _ = largest_connected_component(slice_image, slice_mask)
            lcc_image[:, :, z] = lcc_slice
    else:
        lcc_image, lcc_mask = largest_connected_component(thresh_image, thresh_mask)

    # 5. Remover offset para restaurar valores originais
    lcc_image -= offset

    return down_image, thresh_image, lcc_image, thresh_vals


# =============================================================================
# Funções Públicas - Downscaling com GPU
# =============================================================================


def downscale_image(image, factors, order=3, use_opencv=False, opencv_interpolation=None):
    """
    Downscale otimizado com suporte automático a GPU.

    Estratégia automática para máxima performance:
    1. Se use_opencv=True, usa OpenCV cv2.resize
    2. Se GPU disponível, usa CuPy (mais rápido para volumes grandes)
    3. Caso contrário usa scipy.ndimage.zoom (CPU estável)

    Args:
        image (np.ndarray or cp.ndarray): Imagem a redimensionar
        factors (tuple): Fatores de downscale para cada dimensão
        order (int): Ordem de interpolação (0-5). Default: 3 (cúbica)
        use_opencv (bool): Se True, força OpenCV em vez de GPU/scipy
        opencv_interpolation (int): Método de interpolação do OpenCV (ex: cv2.INTER_AREA)

    Returns:
        np.ndarray: Imagem redimensionada

    Note:
        - GPU é ~2-5x mais rápido que CPU para volumes grandes
        - Fallback automático para CPU se GPU falhar
        - OpenCV pode ser mais rápido para volumes pequenos
    """
    # Força OpenCV se solicitado
    if use_opencv:
        if opencv_interpolation is None:
            opencv_interpolation = cv2.INTER_AREA
        return downscale_image_opencv(image, factors, interpolation=opencv_interpolation)

    # Tenta GPU primeiro se disponível
    if GPU_AVAILABLE:
        try:
            img_gpu = to_gpu(image)
            zoom_factors = tuple(1.0 / f for f in factors)
            result_gpu = cu_ndi.zoom(img_gpu, zoom=zoom_factors, order=order)
            return to_cpu(result_gpu)
        except Exception as e:
            # Fallback para CPU se GPU falhar
            warnings.warn(f"GPU downscaling falhou ({type(e).__name__}), usando CPU.", UserWarning)
            return downscale_image_ndi(image, factors, order=order)
    else:
        # CPU version (scipy)
        return downscale_image_ndi(image, factors, order=order)
