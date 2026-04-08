"""
Módulo de utilitários gerais para processamento de imagens médicas.

Este módulo fornece funções para carregar, salvar, processar e visualizar
volumes médicos em formato NIfTI, incluindo extração de ROIs, segmentação
por unidades Hounsfield e cálculo de métricas de avaliação.
"""

import os
import json

import matplotlib.pyplot as plt
from matplotlib import patches
import nibabel as nib
import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Funções de Normalização
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


def robust_normalize(img, p_min=0, p_max=99.8):
    """
    Normaliza ignorando outliers extremos (cálcio/stents/artefatos).
    Tudo acima do percentil máximo vira 1.0.

    Funciona com NumPy ou CuPy arrays. Retorna o mesmo tipo que recebeu
    (mantém na GPU se entrada for GPU).

    Args:
        img (np.ndarray or cp.ndarray): Imagem de entrada de qualquer dimensão
        p_min (float): Percentil mínimo para clipping (padrão 0)
        p_max (float): Percentil máximo para clipping (padrão 99.8, ignora outliers muito altos)

    Returns:
        np.ndarray or cp.ndarray: Imagem normalizada no intervalo [0, 1].
            Tipo (NumPy ou CuPy) é o mesmo da entrada.

    Example:
        >>> img = np.array([[0, 100, 200], [300, 5000, 6000]])  # Valores com outliers
        >>> norm = robust_normalize(img, p_min=1, p_max=99)
        >>> print(f"Range: [{norm.min()}, {norm.max()}]")
        Range: [0.0, 1.0]

    Note:
        - Útil para remover efeito de outliers extremos (metal, cálcio em CT)
        - Preserva estruturas importantes ignorando apenas extremos
        - Mantém tipo da entrada (NumPy ou CuPy array)
        - Se img.size == 0, retorna img sem alterações
    """
    # Detecta GPU se necessário
    try:
        import cupy as cp

        is_gpu = isinstance(img, cp.ndarray)
    except ImportError:
        is_gpu = False

    xp = cp if is_gpu else np

    if img.size == 0:
        return img

    val_min = xp.percentile(img, p_min)
    val_max = xp.percentile(img, p_max)

    # Clipar os valores para ficar dentro do intervalo "seguro"
    img_clipped = xp.clip(img, val_min, val_max)

    # Evita divisão por zero
    if val_max - val_min == 0:
        return xp.zeros_like(img, dtype=float)

    return (img_clipped - val_min) / (val_max - val_min)


# =============================================================================
# Funções de I/O (Input/Output)
# =============================================================================


def load_json_file(path: str) -> dict:
    """
    Carrega um arquivo JSON com validações e mensagens de erro claras.

    Args:
        path (str): Caminho do arquivo JSON.

    Returns:
        dict: Conteúdo do arquivo JSON.

    Raises:
        FileNotFoundError: Quando o arquivo não existe.
        IsADirectoryError: Quando o caminho informado é um diretório.
        ValueError: Quando o conteúdo não é um objeto JSON válido.
        OSError: Quando ocorre erro de leitura no sistema de arquivos.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo JSON não encontrado: {path}")
    if os.path.isdir(path):
        raise IsADirectoryError(
            f"Caminho aponta para diretório, não arquivo JSON: {path}"
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON inválido em {path} (linha {exc.lineno}, coluna {exc.colno})"
        ) from exc
    except OSError as exc:
        raise OSError(f"Erro ao ler arquivo JSON em {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Conteúdo JSON deve ser um objeto (dict), mas recebeu {type(data).__name__} em {path}"
        )

    return data


def save_json_file(
    data: dict, path: str, indent: int = 2, ensure_ascii: bool = False
) -> None:
    """
    Salva um dicionário em arquivo JSON com criação de diretório e validações.

    Args:
        data (dict): Conteúdo a ser salvo.
        path (str): Caminho de saída do arquivo JSON.
        indent (int): Quantidade de espaços para indentação. Padrão: 2.
        ensure_ascii (bool): Se True, força escape ASCII. Padrão: False.

    Raises:
        TypeError: Quando os dados não são serializáveis em JSON.
        OSError: Quando ocorre erro de escrita no sistema de arquivos.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            f.write("\n")
    except TypeError as exc:
        raise TypeError(f"Dados não serializáveis para JSON em {path}: {exc}") from exc
    except OSError as exc:
        raise OSError(f"Erro ao salvar JSON em {path}: {exc}") from exc


def load_img_and_label(
    img_path: str, label_path: str = None
) -> tuple[NDArray, NDArray]:
    """
    Carrega uma imagem NIfTI e seu rótulo correspondente como arrays NumPy.

    Args:
        img_path (str): Caminho para o arquivo de imagem NIfTI (.nii ou .nii.gz)
        label_path (str, optional): Caminho para o arquivo de rótulo NIfTI.
            Se None, retorna apenas a imagem. Default: None

    Returns:
        tuple[NDArray, NDArray]: Tupla (imagem, label) onde:
            - imagem (np.ndarray): Volume 3D da imagem, ou None se img_path for None
            - label (np.ndarray): Volume 3D do rótulo, ou None se label_path for None

    Example:
        >>> img, label = load_img_and_label('volume.nii.gz', 'segmentation.nii.gz')
        >>> print(f"Image shape: {img.shape}, Label shape: {label.shape}")
        Image shape: (512, 512, 300), Label shape: (512, 512, 300)

        >>> img, _ = load_img_and_label('volume.nii.gz')  # Sem label
        >>> print(f"Image shape: {img.shape}")
        Image shape: (512, 512, 300)

    Note:
        - Usa nibabel.load().get_fdata() para converter para array NumPy
        - Retorna None para parâmetros não fornecidos
        - Para manter objetos Nibabel originais, use load_raw_img_and_label()
    """
    img, label = None, None

    if img_path:
        img = nib.load(img_path).get_fdata()
    if label_path:
        label = nib.load(label_path).get_fdata()

    return img, label


def load_raw_img_and_label(img_path: str, label_path: str = None) -> tuple:
    """
    Carrega imagem e rótulo NIfTI como objetos Nibabel sem conversão para array.

    Útil quando você precisa acessar metadados (affine, header) além dos dados.

    Args:
        img_path (str): Caminho para o arquivo de imagem NIfTI (.nii ou .nii.gz)
        label_path (str, optional): Caminho para o arquivo de rótulo NIfTI.
            Se None, retorna apenas a imagem. Default: None

    Returns:
        tuple: Tupla (img_nib, label_nib) contendo objetos Nibabel, ou None
            para parâmetros não fornecidos

    Example:
        >>> img_nib, label_nib = load_raw_img_and_label('volume.nii.gz', 'seg.nii.gz')
        >>> print(f"Affine shape: {img_nib.affine.shape}")
        >>> print(f"Pixel dims: {img_nib.header.get_zooms()[:3]}")
        >>> data = img_nib.get_fdata()  # Converter para array quando necessário
        Affine shape: (4, 4)
        Pixel dims: (0.5, 0.5, 0.625)

    Note:
        - Preserva metadados completos (affine, header, voxel spacing, etc.)
        - Use para operações que precisam de transformações espaciais
        - Mais eficiente se você não precisa dos dados imediatamente
    """
    img, label = None, None

    if img_path:
        img = nib.load(img_path)
    if label_path:
        label = nib.load(label_path)

    return img, label


def save_nii_image(image, affine, path_to_save="."):
    """
    Salva um array NumPy como arquivo NIfTI com transformação affine.

    Args:
        image (np.ndarray): Volume 3D a ser salvo
        affine (np.ndarray): Matriz affine 4x4 definindo transformação espacial
        path_to_save (str): Caminho completo do arquivo de saída (incluindo extensão).
            Default: "." (salvará no diretório atual)

    Returns:
        None

    Example:
        >>> volume = segment_aorta(ccta)
        >>> affine = img_nib.affine  # Preservar affine da imagem original
        >>> save_nii_image(volume, affine, 'output/aorta_seg.nii.gz')
        Imagem salva em: output/aorta_seg.nii.gz .

    Note:
        - Affine deve vir da imagem original para preservar orientação espacial
        - Formato NIfTI-1 (compatível com a maioria das ferramentas)
        - Imprime mensagem de sucesso ou erro
        - Para arrays simples sem metadados, use save_npy_array()
    """
    nifti_img = nib.Nifti1Image(image, affine)

    try:
        nib.save(nifti_img, path_to_save)
        print("Imagem salva em:", path_to_save, ".")
    except Exception as e:
        print("Erro ao salvar a imagem:", e)


def save_npy_array(array: NDArray, path: str):
    """
    Salva um array NumPy em formato binário .npy.

    Args:
        array (NDArray): Array NumPy de qualquer dimensão ou dtype
        path (str): Caminho do arquivo de saída (deve terminar em .npy)

    Returns:
        None

    Example:
        >>> circles = detect_aorta_circles(volume)
        >>> save_npy_array(circles, 'output/detected_circles.npy')
        Array salvo em: output/detected_circles.npy

        >>> loaded = np.load('output/detected_circles.npy')

    Note:
        - Formato .npy é específico do NumPy (binário, rápido)
        - Preserva dtype e shape exatos
        - Para dados médicos com metadados espaciais, use save_nii_image()
        - Útil para resultados intermediários e caches
    """
    try:
        np.save(path, array)
        print(f"Array salvo em: {path}")
    except Exception as e:
        print(f"Erro ao salvar o array: {e}")


# =============================================================================
# Funções de Extração de ROI (Region of Interest)
# =============================================================================


def extract_square_region(image, x_min, x_max, y_min, y_max):
    """
    Extrai uma região retangular do volume 3D usando coordenadas de bounding box.

    Args:
        image (np.ndarray): Volume 3D de entrada com shape (H, W, D)
        x_min (int): Coordenada inicial no eixo x (altura)
        x_max (int): Coordenada final no eixo x (altura)
        y_min (int): Coordenada inicial no eixo y (largura)
        y_max (int): Coordenada final no eixo y (largura)

    Returns:
        np.ndarray: Subvolume extraído com shape (x_max-x_min, y_max-y_min, D)

    Raises:
        ValueError: Se coordenadas são inválidas (min >= max)

    Example:
        >>> volume = load_volume()  # Shape: (512, 512, 300)
        >>> roi = extract_square_region(volume, 100, 400, 150, 450)
        >>> print(f"ROI shape: {roi.shape}")  # (300, 300, 300)

    Note:
        - Coordenadas são automaticamente clampadas aos limites da imagem
        - Profundidade (eixo z) é preservada integralmente
        - Útil para focar processamento em região anatômica específica
        - Para ROI circular, use extract_circular_region()
    """
    h, w, d = image.shape

    # Garantir que as coordenadas estejam dentro dos limites da imagem
    x_min = max(0, x_min)
    x_max = min(h, x_max)
    y_min = max(0, y_min)
    y_max = min(w, y_max)

    # Verificar se as coordenadas formam uma região válida
    if x_min >= x_max or y_min >= y_max:
        raise ValueError(
            "Coordenadas inválidas: x_min deve ser menor que x_max e y_min deve ser menor que y_max"
        )

    # Extrair a região
    return image[x_min:x_max, y_min:y_max, :]


def extract_circular_region(image, center=None, radius=None, mask_background=True):
    """
    Extrai uma região circular do volume 3D, aplicando máscara em cada slice.

    Cria ROI retangular baseado no círculo e opcionalmente mascara voxels fora
    do círculo. O círculo é aplicado em cada slice 2D ao longo do eixo z.

    Args:
        image (np.ndarray): Volume 3D de entrada com shape (H, W, D)
        center (tuple, optional): Centro do círculo (x, y) em coordenadas de pixel.
            Se None, usa centro da imagem (H//2, W//2). Default: None
        radius (int, optional): Raio do círculo em pixels.
            Se None, usa min(H, W) // 4. Default: None
        mask_background (bool): Se True, zera voxels fora do círculo.
            Se False, retorna ROI retangular sem máscara. Default: True

    Returns:
        np.ndarray: Subvolume extraído. Se mask_background=True, voxels fora
            do círculo são zerados

    Example:
        >>> volume = load_volume()  # Shape: (512, 512, 300)
        >>> # Extrair região circular central
        >>> roi = extract_circular_region(volume, center=(256, 256), radius=100)
        >>> print(f"ROI shape: {roi.shape}")  # (200, 200, 300)

        >>> # Sem máscara (retângulo)
        >>> roi_rect = extract_circular_region(volume, radius=100, mask_background=False)

    Note:
        - Máscara circular é aplicada identicamente em todos os slices z
        - Útil para focar em região central (ex: aorta no centro do tórax)
        - center é relativo à imagem original, não ao subvolume
        - Para ROI retangular simples, use extract_square_region()
    """
    h, w, d = image.shape

    if center is None:
        center = (h // 2, w // 2)

    if radius is None:
        radius = min(h, w) // 4

    # Definir os limites do corte retangular
    x_min, x_max = max(0, center[0] - radius), min(h, center[0] + radius)
    y_min, y_max = max(0, center[1] - radius), min(w, center[1] + radius)

    # Extrair o subvolume retangular
    sub_volume = image[x_min:x_max, y_min:y_max, :]

    if mask_background:
        # Aplicar máscara circular dentro do subvolume
        sub_h, sub_w = sub_volume.shape[0], sub_volume.shape[1]
        sub_center = (sub_h // 2, sub_w // 2)

        # Criar malha de coordenadas
        y, x = np.ogrid[:sub_h, :sub_w]
        # Calcular a distância ao quadrado de cada ponto até o centro
        dist_from_center = (x - sub_center[1]) ** 2 + (y - sub_center[0]) ** 2
        # Criar máscara circular
        mask = dist_from_center <= radius**2

        # Aplicar a máscara
        masked_volume = np.zeros_like(sub_volume)
        for z in range(sub_volume.shape[2]):
            masked_volume[:, :, z] = sub_volume[:, :, z] * mask

        return masked_volume

    return sub_volume


# =============================================================================
# Funções de Segmentação e Processamento
# =============================================================================


def segment_by_hu(img_3d, include_labels=None):
    """
    Segmenta volume CT em 8 classes de tecidos baseado em unidades Hounsfield.

    Aplica thresholding multi-classe usando intervalos HU característicos de
    diferentes tecidos anatômicos em imagens de tomografia computadorizada.

    Args:
        img_3d (np.ndarray): Volume 3D em unidades Hounsfield (HU)
        include_labels (list, optional): Lista de labels (1-8) a incluir.
            Se None, inclui todas as 8 classes. Default: None

    Returns:
        tuple: (segmented, hu_ranges) contendo:
            - segmented (np.ndarray): Volume segmentado (dtype=uint8) com labels 1-8
            - hu_ranges (dict): Dicionário {label: {name, range, color}} das classes

    Classes de segmentação (label: nome, intervalo HU):
        1: Ar (-1050, -950)
        2: Pulmão (-950, -500)
        3: Gordura (-190, -30)
        4: Água/Fluidos (-30, 30)
        5: Tecidos Moles (30, 100)
        6: Osso Esponjoso (100, 400)
        7: Osso Cortical/Denso (400, 3000)
        8: Metal/Implantes (3000, 4000)

    Example:
        >>> ccta = load_volume()
        >>> seg, ranges = segment_by_hu(ccta)
        >>> print(f"Classes encontradas: {np.unique(seg)}")
        Classes encontradas: [0 1 2 3 5 6 7]

        >>> # Segmentar apenas tecidos moles e osso
        >>> seg_subset, _ = segment_by_hu(ccta, include_labels=[5, 6, 7])

    Note:
        - Label 0 representa voxels fora de todos os intervalos
        - Intervalos HU baseados em literatura médica padrão
        - Útil para visualização anatômica e pré-processamento
        - hu_ranges['color'] pode ser usado para visualização RGB
    """
    # Dicionário de intervalos HU para diferentes tecidos/estruturas
    hu_ranges = {
        1: {"name": "Ar", "range": (-1050, -950), "color": [0, 0, 0]},
        2: {"name": "Pulmão", "range": (-950, -500), "color": [194, 220, 232]},
        3: {"name": "Gordura", "range": (-190, -30), "color": [255, 255, 150]},
        4: {"name": "Água/Fluidos", "range": (-30, 30), "color": [170, 250, 250]},
        5: {"name": "Tecidos Moles", "range": (30, 100), "color": [230, 170, 170]},
        6: {"name": "Osso Esponjoso", "range": (100, 400), "color": [255, 180, 100]},
        7: {
            "name": "Osso Cortical/Denso",
            "range": (400, 3000),
            "color": [255, 255, 255],
        },
        8: {"name": "Metal/Implantes", "range": (3000, 4000), "color": [220, 220, 50]},
    }
    # Filtrar as classes a serem incluídas
    if include_labels:
        hu_ranges = {k: v for k, v in hu_ranges.items() if k in include_labels}

    # Inicializar imagem segmentada com zeros
    segmented = np.zeros_like(img_3d, dtype=np.uint8)

    # Aplicar segmentação para cada intervalo de HU
    for label, props in hu_ranges.items():
        min_hu, max_hu = props["range"]
        mask = (img_3d >= min_hu) & (img_3d <= max_hu)
        segmented[mask] = label

    return segmented, hu_ranges


# =============================================================================
# Funções de Avaliação e Métricas
# =============================================================================


def dice_score(pred, target):
    """
    Calcula o coeficiente de Dice (F1-score para segmentação).

    O coeficiente de Dice mede a sobreposição entre duas máscaras binárias,
    frequentemente usado para avaliar qualidade de segmentações médicas.

    Fórmula: Dice = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        pred (np.ndarray): Máscara de segmentação predita (valores > 0 são positivos)
        target (np.ndarray): Máscara de segmentação ground truth (valores > 0 são positivos)

    Returns:
        float: Dice coefficient no intervalo [0, 1] onde:
            - 1.0 = overlap perfeito
            - 0.0 = sem overlap
            - 1.0 também retornado se ambas máscaras estão vazias

    Example:
        >>> pred = segment_aorta(volume)
        >>> gt = load_ground_truth()
        >>> dice = dice_score(pred, gt)
        >>> print(f"Dice coefficient: {dice:.4f}")
        Dice coefficient: 0.8752

    Note:
        - Binarização automática: valores > 0 são considerados positivos
        - Simétrico: dice(A, B) = dice(B, A)
        - Equivalente a 2*TP / (2*TP + FP + FN)
        - Popular em desafios de segmentação médica (ex: Medical Segmentation Decathlon)
    """
    pred_binary = (pred > 0).astype(bool)
    target_binary = (target > 0).astype(bool)

    intersection = np.sum(pred_binary & target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return 2.0 * intersection / union
