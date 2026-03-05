import numpy as np
from skimage.filters import ridges, gaussian
import os
import pickle
import warnings

# Detecta disponibilidade de GPU (teste simplificado)
GPU_AVAILABLE = False
cp = None
gpu_filters = None

try:
    import cupy as cp
    import cucim.skimage.filters as gpu_filters

    # Teste simples: cria array pequeno na GPU
    _ = cp.array([1, 2, 3])

    GPU_AVAILABLE = True

except Exception as e:
    # Qualquer erro = usa CPU
    cp = None
    gpu_filters = None
    GPU_AVAILABLE = False
    warnings.warn(f"GPU não disponível ({type(e).__name__}). Usando CPU.", UserWarning)


def use_gpu():
    """Retorna True se GPU está disponível e CuPy está instalado."""
    return GPU_AVAILABLE


def to_gpu(arr):
    """Converte array NumPy para CuPy se GPU disponível."""
    if GPU_AVAILABLE and isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    return arr


def to_cpu(arr):
    """Converte array CuPy para NumPy se necessário."""
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


def normalize(img):
    """
    Normaliza para [0, 1] com proteção contra divisão por zero.
    Funciona com NumPy ou CuPy arrays.
    Retorna o mesmo tipo que recebeu (mantém na GPU se entrada for GPU).
    """
    is_gpu = GPU_AVAILABLE and isinstance(img, cp.ndarray)
    xp = cp if is_gpu else np

    min_val, max_val = xp.min(img), xp.max(img)
    if max_val - min_val == 0:
        return xp.zeros_like(img, dtype=float)
    return (img - min_val) / (max_val - min_val)


def robust_normalize(img, p_min=0, p_max=99.8):
    """
    Normaliza ignorando outliers extremos (cálcio/stents).
    Tudo acima do percentil 99.5 vira 1.0.
    Funciona com NumPy ou CuPy arrays.
    Retorna o mesmo tipo que recebeu (mantém na GPU se entrada for GPU).
    """
    if img.size == 0:
        return img

    is_gpu = GPU_AVAILABLE and isinstance(img, cp.ndarray)
    xp = cp if is_gpu else np

    val_min = xp.percentile(img, p_min)
    val_max = xp.percentile(img, p_max)

    # Clipar os valores para ficar dentro do intervalo "seguro"
    img_clipped = xp.clip(img, val_min, val_max)

    # Evita divisão por zero
    if val_max - val_min == 0:
        return xp.zeros_like(img, dtype=float)

    return (img_clipped - val_min) / (val_max - val_min)


def get_gf(image_volume):
    """
    Calcula a medida de grayness (Gf) baseada na Equação (7) do artigo.
    Funciona com NumPy ou CuPy arrays.

    Args:
      image_volume: Um array NumPy/CuPy 3D representando a imagem CCTA.

    Returns:
      Um array NumPy/CuPy 3D com a medida Gf para cada voxel.
    """
    is_gpu = GPU_AVAILABLE and isinstance(image_volume, cp.ndarray)
    xp = cp if is_gpu else np

    # t é o valor médio da intensidade dos pixels da imagem 3D
    t = xp.mean(image_volume)
    # I_max é o valor máximo de escala de cinza da imagem 3D
    i_max = xp.max(image_volume)

    # Evita divisão por zero se i_max for 0
    if i_max == 0:
        return xp.zeros_like(image_volume, dtype=float)

    gf = xp.abs(image_volume - t) / i_max

    return gf


def get_gd(image_volume):
    """
    Calcula a medida de gradiente (Gd).
    Funciona com NumPy ou CuPy arrays.

    Args:
      image_volume: Um array NumPy/CuPy 3D representando a imagem CCTA.

    Returns:
      Um array NumPy/CuPy 3D com a medida Gd para cada voxel.
    """
    is_gpu = GPU_AVAILABLE and isinstance(image_volume, cp.ndarray)
    xp = cp if is_gpu else np

    gz, gy, gx = xp.gradient(image_volume)
    g_mag = xp.sqrt(gx**2 + gy**2 + gz**2)  # Magnitude euclidiana correta
    i_max = xp.max(image_volume)

    if i_max == 0:
        return xp.zeros_like(image_volume)

    gd = (g_mag - image_volume) / i_max

    # Clipar outliers de gradiente (ex: stents metálicos)
    gd = xp.clip(gd, xp.percentile(gd, 1), xp.percentile(gd, 99))

    return gd  # Retorna sem normalizar novamente


def get_vesselness(
    image,
    sigmas=np.arange(1.0, 4.0, 0.5),
    alpha=0.5,
    beta=0.5,
    gamma=None,
    black_ridges=False,
    normalization="none",
    gpu=None,
):
    """
    Calcula o mapa de vesselness usando o filtro de Frangi.
    Usa GPU se disponível, caso contrário usa CPU.

    Args:
        image: Imagem 3D de entrada (NumPy ou CuPy array)
        sigmas: Range de sigmas para multi-escala
        alpha: Sensibilidade a estruturas blob (0.1-1.0, padrão 0.5)
        beta: Sensibilidade ao ruído de fundo (0.1-1.0, padrão 0.5)
        gamma: Sensibilidade ao contraste (padrão None)
        black_ridges: Se True, detecta estruturas escuras
        normalization: Método de normalização ('robust', 'minmax', 'none')
            - 'robust': Ignora outliers usando percentis (padrão)
            - 'minmax': Normalização simples [0, 1]
            - 'none': Sem normalização
        gpu: Se None (padrão), detecta automaticamente. Se True, força GPU. Se False, força CPU.

    Returns:
        vesselness_norm: Mapa de vesselness normalizado (ou não), como NumPy array
    """
    # Determina se deve usar GPU
    use_gpu_flag = gpu if gpu is not None else GPU_AVAILABLE

    if use_gpu_flag and GPU_AVAILABLE:
        # Converte para GPU se necessário
        image_gpu = to_gpu(image)

        # IMPORTANTE: sigmas deve ser lista Python, não array CuPy
        # Evita problemas com driver NVIDIA em algumas versões do cuCIM
        sigmas_list = list(sigmas) if hasattr(sigmas, '__iter__') else [sigmas]

        # Usa cuCIM para Frangi na GPU
        vesselness = gpu_filters.frangi(
            image_gpu,
            sigmas=sigmas_list,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            black_ridges=black_ridges
        )

        # Mantém na GPU durante normalização para eficiência
        if normalization == "robust":
            vesselness = robust_normalize(vesselness)
        elif normalization == "minmax":
            vesselness = normalize(vesselness)
        elif normalization != "none":
            raise ValueError(
                f"Método de normalização '{normalization}' inválido. Use 'robust', 'minmax' ou 'none'."
            )

        # Só converte para CPU no final
        return to_cpu(vesselness)
    else:
        # Usa skimage na CPU
        if isinstance(image, np.ndarray):
            img_cpu = image
        else:
            img_cpu = to_cpu(image)

        vesselness = ridges.frangi(
            img_cpu,
            sigmas=sigmas,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            black_ridges=black_ridges
        )

        # Normalização na CPU
        if normalization == "robust":
            vesselness = robust_normalize(vesselness)
        elif normalization == "minmax":
            vesselness = normalize(vesselness)
        elif normalization != "none":
            raise ValueError(
                f"Método de normalização '{normalization}' inválido. Use 'robust', 'minmax' ou 'none'."
            )

        return vesselness


def get_vesselness_optimized(
    image,
    sigmas=np.arange(1.0, 4.0, 0.5),
    alpha=0.5,
    beta=0.5,
    normalization="none",
    smooth_sigma=1.0,
    gpu=None,
):
    """
    Pipeline otimizado com pré-processamento e medidas auxiliares.
    Usa GPU se disponível, caso contrário usa CPU.

    Args:
        image: Imagem 3D de entrada (NumPy ou CuPy array)
        sigmas: Range de sigmas para multi-escala
        alpha: Sensibilidade a estruturas blob (0.1-1.0, padrão 0.5)
        beta: Sensibilidade ao ruído de fundo (0.1-1.0, padrão 0.5)
        normalization: Método de normalização ('robust', 'minmax', 'none')
        smooth_sigma: Sigma para suavização Gaussiana
        gpu: Se None (padrão), detecta automaticamente. Se True, força GPU. Se False, força CPU.

    Returns:
        modified_vesselness: Vesselness modificado com Gf e Gd (como NumPy array)
    """
    # Determina se deve usar GPU
    use_gpu_flag = gpu if gpu is not None else GPU_AVAILABLE

    if use_gpu_flag and GPU_AVAILABLE:
        # Converte para GPU
        img_gpu = to_gpu(image)

        # IMPORTANTE: sigmas deve ser lista Python, não array CuPy
        sigmas_list = list(sigmas) if hasattr(sigmas, '__iter__') else [sigmas]

        # Suavização leve para remover ruído na GPU
        img_smooth = gpu_filters.gaussian(img_gpu, sigma=smooth_sigma, preserve_range=True)

        # Frangi tunado na GPU
        vesselness = gpu_filters.frangi(
            img_smooth,
            sigmas=sigmas_list,
            alpha=alpha,
            beta=beta,
            black_ridges=False,  # Vasos são brancos no CCTA
        )

        # Medidas auxiliares na GPU (mantém na GPU para eficiência)
        gf = get_gf(img_smooth)
        gd = get_gd(img_smooth)

        # Combinação na GPU
        modified_vesselness = vesselness * (gf / gd)

        # Normalização na GPU se necessário
        if normalization == "robust":
            modified_vesselness = robust_normalize(modified_vesselness)
        elif normalization == "minmax":
            modified_vesselness = normalize(modified_vesselness)

        # Só converte para CPU no final
        return to_cpu(modified_vesselness)
    else:
        # Pipeline na CPU
        if not isinstance(image, np.ndarray):
            image = to_cpu(image)

        # Suavização leve para remover ruído
        img_smooth = gaussian(image, sigma=smooth_sigma, preserve_range=True)

        # Frangi tunado
        vesselness = ridges.frangi(
            img_smooth,
            sigmas=sigmas,
            alpha=alpha,
            beta=beta,
            black_ridges=False,  # Vasos são brancos no CCTA
        )

        # Medidas auxiliares
        gf = get_gf(img_smooth)
        gd = get_gd(img_smooth)

        # Combinação
        modified_vesselness = vesselness * (gf / gd)

        # Normalização na CPU
        if normalization == "robust":
            modified_vesselness = robust_normalize(modified_vesselness)
        elif normalization == "minmax":
            modified_vesselness = normalize(modified_vesselness)

        return modified_vesselness


def save_vesselness_cache(vesselness_i, img_id, cache_dir="../cache"):
    """
    Salva o mapa de vesselness em cache.
    Converte para NumPy se necessário.

    Args:
        vesselness_i: Mapa de vesselness (NumPy ou CuPy array)
        img_id: ID da imagem
        cache_dir: Diretório para salvar o cache
    """
    os.makedirs(cache_dir, exist_ok=True)
    # Converte para NumPy se necessário
    vesselness_i = to_cpu(vesselness_i)
    cache_path = os.path.join(cache_dir, f"vesselness_{img_id}.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(vesselness_i, f)


def load_vesselness_cache(img_id, cache_dir="../cache"):
    """
    Carrega o mapa de vesselness do cache se disponível.

    Args:
        img_id: ID da imagem
        cache_dir: Diretório do cache

    Returns:
        vesselness_i como NumPy array, ou None se não encontrado
    """
    cache_path = os.path.join(cache_dir, f"vesselness_{img_id}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None
