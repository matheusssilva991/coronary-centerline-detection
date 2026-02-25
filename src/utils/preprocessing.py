"""
Módulo para pré-processamento de imagens médicas CCTA.

Este módulo implementa operações de pré-processamento essenciais para volumes CCTA,
incluindo normalização, redimensionamento, thresholding e extração de componentes
conectados para isolar estruturas anatômicas de interesse.
"""

import numpy as np
import scipy.ndimage as ndi


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
# Funções Públicas
# =============================================================================


def normalize_image(img):
    """
    Normaliza a imagem para o intervalo [0, 1] usando min-max scaling.

    A normalização é feita pela fórmula: (img - min) / (max - min)

    Args:
        img (np.ndarray): Imagem de entrada de qualquer dimensão

    Returns:
        np.ndarray: Imagem normalizada no intervalo [0, 1]. Se a imagem tiver
            valores constantes (max == min), retorna a imagem original

    Example:
        >>> img = np.array([[-100, 0, 200], [300, 400, 500]])
        >>> norm_img = normalize_image(img)
        >>> print(f"Range: [{norm_img.min()}, {norm_img.max()}]")
        Range: [0.0, 1.0]

    Note:
        - Útil antes de aplicar filtros que esperam valores normalizados
        - Preserva a distribuição relativa dos valores
        - Retorna cópia da imagem original se todos valores forem iguais
    """
    min_val, max_val = np.min(img), np.max(img)
    if max_val - min_val == 0:
        return img
    return (img - min_val) / (max_val - min_val)


def downscale_image(image, factors, order=3):
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


def keep_largest_component(mask):
    """
    Mantém apenas o maior componente conectado da máscara binária.

    Versão simplificada de largest_connected_component() que opera apenas
    na máscara, sem aplicar em uma imagem.

    Args:
        mask (np.ndarray): Máscara binária de entrada

    Returns:
        np.ndarray: Máscara binária (dtype=uint8) contendo apenas o maior
            componente conectado

    Example:
        >>> noisy_mask = segment_structure(volume)
        >>> clean_mask = keep_largest_component(noisy_mask)
        >>> print(f"Redução: {noisy_mask.sum()} → {clean_mask.sum()} voxels")

    Note:
        - Equivalente a largest_connected_component() mas sem imagem associada
        - Mais eficiente quando só precisa da máscara
        - Retorna máscara original se vazia
    """
    labeled, num = ndi.label(mask)
    if num == 0:
        return mask

    sizes = ndi.sum(mask, labeled, range(1, num + 1))
    largest_label = np.argmax(sizes) + 1

    return (labeled == largest_label).astype(np.uint8)


def run_core_preprocessing_pipeline(
    image,
    downscale_factors,
    min_threshold=-300,
    max_threshold_percentile=99.5,
    lcc_per_slice=True,
    order=3,
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
        order (int): Ordem de interpolação para downscaling (0-5). Default: 3

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
        ...     lcc_per_slice=True
        ... )
        >>> print(f"Shape reduzido: {down.shape}")  # (256, 256, 300)
        >>> print(f"Thresholds: {vals}")  # (-300, 620) aproximadamente

    Note:
        - lcc_per_slice=True é mais rápido e previne conexões espúrias entre slices
        - lcc_per_slice=False garante conectividade 3D completa
        - max_threshold é adaptativo (percentil) para lidar com variações de contraste
        - Offset é removido no final para preservar valores Hounsfield originais
        - thresh_vals pode ser usado para reproduzir o threshold em outras imagens
    """
    # 1. Downscaling para reduzir custo computacional
    down_image = downscale_image(image, downscale_factors, order=order)

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
