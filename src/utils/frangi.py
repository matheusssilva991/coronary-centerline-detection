import numpy as np
from skimage.filters import ridges, gaussian
import os
import pickle


def normalize(img):
    """Normaliza para [0, 1] com proteção contra divisão por zero."""
    min_val, max_val = np.min(img), np.max(img)
    if max_val - min_val == 0:
        return np.zeros_like(img, dtype=float)
    return (img - min_val) / (max_val - min_val)


def robust_normalize(img, p_min=0, p_max=99.8):
    """
    Normaliza ignorando outliers extremos (cálcio/stents).
    Tudo acima do percentil 99.5 vira 1.0.
    """
    if img.size == 0:
        return img

    val_min = np.percentile(img, p_min)
    val_max = np.percentile(img, p_max)

    # Clipar os valores para ficar dentro do intervalo "seguro"
    img_clipped = np.clip(img, val_min, val_max)

    # Evita divisão por zero
    if val_max - val_min == 0:
        return np.zeros_like(img, dtype=float)

    return (img_clipped - val_min) / (val_max - val_min)


def get_gf(image_volume):
    """
    Calcula a medida de grayness (Gf) baseada na Equação (7) do artigo.

    Args:
      image_volume: Um array NumPy 3D representando a imagem CCTA.

    Returns:
      Um array NumPy 3D com a medida Gf para cada voxel.
    """
    # t é o valor médio da intensidade dos pixels da imagem 3D
    t = np.mean(image_volume)
    # I_max é o valor máximo de escala de cinza da imagem 3D
    i_max = np.max(image_volume)

    # Evita divisão por zero se i_max for 0
    if i_max == 0:
        return np.zeros_like(image_volume, dtype=float)

    gf = np.abs(image_volume - t) / i_max

    return gf


def get_gd(image_volume):
    """
    Calcula a medida de gradiente (Gd).

    Args:
      image_volume: Um array NumPy 3D representando a imagem CCTA.

    Returns:
      Um array NumPy 3D com a medida Gd para cada voxel.
    """
    gz, gy, gx = np.gradient(image_volume)
    g_mag = np.sqrt(gx**2 + gy**2 + gz**2)  # Magnitude euclidiana correta
    i_max = np.max(image_volume)

    if i_max == 0:
        return np.zeros_like(image_volume)

    gd = (g_mag - image_volume) / i_max

    # Clipar outliers de gradiente (ex: stents metálicos)
    gd = np.clip(gd, np.percentile(gd, 1), np.percentile(gd, 99))

    return gd  # Retorna sem normalizar novamente


def get_vesselness(
    image,
    sigmas=np.arange(1.0, 4.0, 0.5),
    alpha=0.5,
    beta=0.5,
    gamma=None,
    black_ridges=False,
    normalization="none",
):
    """
    Calcula o mapa de vesselness usando o filtro de Frangi.

    Args:
        image: Imagem 3D de entrada
        sigmas: Range de sigmas para multi-escala
        alpha: Sensibilidade a estruturas blob (0.1-1.0, padrão 0.5)
        beta: Sensibilidade ao ruído de fundo (0.1-1.0, padrão 0.5)
        gamma: Sensibilidade ao contraste (padrão None)
        black_ridges: Se True, detecta estruturas escuras
        normalization: Método de normalização ('robust', 'minmax', 'none')
            - 'robust': Ignora outliers usando percentis (padrão)
            - 'minmax': Normalização simples [0, 1]
            - 'none': Sem normalização

    Returns:
        vesselness_norm: Mapa de vesselness normalizado (ou não)
    """
    vesselness = ridges.frangi(
        image, sigmas=sigmas, alpha=alpha, beta=beta, gamma=gamma,
        black_ridges=black_ridges # type: ignore
    )

    if normalization == "robust":
        return robust_normalize(vesselness)
    elif normalization == "minmax":
        return normalize(vesselness)
    elif normalization == "none":
        return vesselness
    else:
        raise ValueError(
            f"Método de normalização '{normalization}' inválido. Use 'robust', 'minmax' ou 'none'."
        )


def get_vesselness_optimized(
    image, sigmas=np.arange(1.0, 4.0, 0.5), alpha=0.5, beta=0.5, normalization="none", smooth_sigma=1.0
):
    """
    Pipeline otimizado com pré-processamento e medidas auxiliares.

    Args:
        image: Imagem 3D de entrada
        sigmas: Range de sigmas para multi-escala
        alpha: Sensibilidade a estruturas blob (0.1-1.0, padrão 0.5)
        beta: Sensibilidade ao ruído de fundo (0.1-1.0, padrão 0.5)

    Returns:
        modified_vesselness: Vesselness modificado com Gf e Gd
    """
    # Suavização leve para remover ruído
    img_smooth = gaussian(image, sigma=smooth_sigma, preserve_range=True)

    # Frangi tunado
    vesselness = ridges.frangi(
        img_smooth,
        sigmas=sigmas, # type: ignore
        alpha=alpha,
        beta=beta,
        black_ridges=False,  # Vasos são brancos no CCTA
    )

    # Medidas auxiliares
    gf = get_gf(img_smooth)
    gd = get_gd(img_smooth)

    # Combinação
    modified_vesselness = vesselness * (gf / gd)

    if normalization == "robust":
        modified_vesselness = robust_normalize(modified_vesselness)
    elif normalization == "minmax":
        modified_vesselness = normalize(modified_vesselness)

    return modified_vesselness


def save_vesselness_cache(vesselness_i, img_id, cache_dir="../cache"):
    """
    Salva o mapa de vesselness em cache.

    Args:
        vesselness_i: Mapa de vesselness
        img_id: ID da imagem
        cache_dir: Diretório para salvar o cache
    """
    os.makedirs(cache_dir, exist_ok=True)
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
        vesselness_i ou None se não encontrado
    """
    cache_path = os.path.join(cache_dir, f"vesselness_{img_id}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None
