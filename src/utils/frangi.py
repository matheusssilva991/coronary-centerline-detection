import numpy as np
from skimage.filters import ridges, gaussian
import os
import pickle

# Importa utilitários de GPU centralizados
from .gpu_utils import (
    use_gpu,
    to_gpu,
    to_cpu,
    GPU_AVAILABLE,
    cp,
)

# Importa funções de normalização
from .utils import normalize_image, robust_normalize

# Importa cucim filters se GPU disponível
gpu_filters = None
if GPU_AVAILABLE:
    try:
        import cucim.skimage.filters as gpu_filters
    except ImportError:
        gpu_filters = None


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
            vesselness = normalize_image(vesselness)
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
            vesselness = normalize_image(vesselness)
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
            modified_vesselness = normalize_image(modified_vesselness)

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
            modified_vesselness = normalize_image(modified_vesselness)

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
