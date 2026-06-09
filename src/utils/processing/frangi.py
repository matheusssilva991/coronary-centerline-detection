import numpy as np
from skimage.filters import ridges, gaussian
import os
import pickle
from typing import Any, Optional, Sequence, Tuple
from numpy.typing import NDArray

# Importa utilitários de GPU centralizados
from .gpu_utils import (
    to_gpu,
    to_cpu,
    GPU_AVAILABLE,
    cp,
)

# Importa funções de normalização
from ..utils import normalize_image, robust_normalize

# Importa cucim filters se GPU disponível
gpu_filters = None
if GPU_AVAILABLE:
    try:
        import cucim.skimage.filters as gpu_filters
    except ImportError:
        gpu_filters = None


def get_gf(
    image_volume: NDArray[Any],
    center_percentile: float = 50.0,
    scale_percentile: float = 99.5,
    scale_epsilon: float = 1e-6,
) -> NDArray[Any]:
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

    center = xp.percentile(image_volume, center_percentile)
    abs_deviation = xp.abs(image_volume - center)
    scale = xp.percentile(abs_deviation, scale_percentile)

    if float(scale) <= scale_epsilon:
        return xp.zeros_like(image_volume, dtype=float)

    gf = abs_deviation / scale

    return gf


def get_gd(
    image_volume: NDArray[Any],
    spacing: Optional[Sequence[float]] = None,
    clip_percentiles: Tuple[float, float] = (1.0, 99.0),
) -> NDArray[Any]:
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

    if spacing is None:
        gradients = xp.gradient(image_volume)
    else:
        gradients = xp.gradient(image_volume, *spacing)
    g_mag = xp.sqrt(sum(axis_grad**2 for axis_grad in gradients))

    # Clipar outliers de gradiente (ex: stents metálicos)
    gd = xp.clip(
        g_mag,
        xp.percentile(g_mag, clip_percentiles[0]),
        xp.percentile(g_mag, clip_percentiles[1]),
    )

    return gd  # Retorna sem normalizar novamente


def _as_sigma_sequence(sigmas: Sequence[float]) -> list[float]:
    if np.isscalar(sigmas):
        sigma_values = [float(sigmas)]
    else:
        sigma_values = [float(sigma) for sigma in sigmas]

    if not sigma_values:
        raise ValueError("sigmas deve conter pelo menos um valor.")
    if any(sigma <= 0 for sigma in sigma_values):
        raise ValueError("Todos os valores de sigmas devem ser maiores que zero.")

    return sigma_values


def _validate_normalization_method(normalization: str) -> None:
    if normalization not in {"none", "robust", "minmax"}:
        raise ValueError(
            f"Método de normalização '{normalization}' inválido. "
            "Use 'robust', 'minmax' ou 'none'."
        )


def _apply_normalization(array: Any, normalization: str) -> Any:
    _validate_normalization_method(normalization)
    if normalization == "robust":
        return robust_normalize(array)
    if normalization == "minmax":
        return normalize_image(array)
    return array


def get_vesselness(
    image: Any,
    sigmas: Sequence[float] = np.arange(1.0, 4.0, 0.5),
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: Optional[float] = None,
    black_ridges: bool = False,
    normalization: str = "none",
    smooth_sigma: float = 0.0,
    gpu: Optional[bool] = None,
    return_cpu: bool = True,
) -> NDArray[Any]:
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
        smooth_sigma: Sigma opcional para suavização gaussiana antes do Frangi.
            Use 0 para desativar.
        gpu: Se None (padrão), detecta automaticamente. Se True, força GPU. Se False, força CPU.
        return_cpu: Se True, converte resultado GPU para NumPy antes de retornar.

    Returns:
        vesselness_norm: Mapa de vesselness normalizado (ou não), como NumPy array
    """
    sigma_values = _as_sigma_sequence(sigmas)
    _validate_normalization_method(normalization)
    use_gpu_flag = gpu if gpu is not None else GPU_AVAILABLE

    if use_gpu_flag and GPU_AVAILABLE and gpu_filters is not None:
        image_gpu = to_gpu(image)
        frangi_input = (
            gpu_filters.gaussian(
                image_gpu,
                sigma=smooth_sigma,
                preserve_range=True,
            )
            if smooth_sigma > 0
            else image_gpu
        )
        vesselness = gpu_filters.frangi(
            frangi_input,
            sigmas=sigma_values,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            black_ridges=black_ridges,
        )
        vesselness = _apply_normalization(vesselness, normalization)
        return to_cpu(vesselness) if return_cpu else vesselness

    img_cpu = image if isinstance(image, np.ndarray) else to_cpu(image)
    frangi_input = (
        gaussian(img_cpu, sigma=smooth_sigma, preserve_range=True)
        if smooth_sigma > 0
        else img_cpu
    )
    vesselness = ridges.frangi(
        frangi_input,
        sigmas=sigma_values,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=black_ridges,
    )
    return _apply_normalization(vesselness, normalization)


def get_modified_vesselness(
    image: Any,
    sigmas: Sequence[float] = np.arange(1.0, 4.0, 0.5),
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: Optional[float] = None,
    black_ridges: bool = False,
    normalization: str = "none",
    smooth_sigma: float = 1.0,
    gd_epsilon: float = 1e-6,
    gf_center_percentile: float = 50.0,
    gf_scale_percentile: float = 99.5,
    gd_clip_percentiles: Tuple[float, float] = (1.0, 99.0),
    ratio_clip: Optional[Tuple[float, float]] = (0.0, 10.0),
    spacing: Optional[Sequence[float]] = None,
    gpu: Optional[bool] = None,
) -> NDArray[Any]:
    """
    Vesselness modificada com pré-suavização e ponderação por grayness/gradient.
    Usa GPU se disponível, caso contrário usa CPU.

    Args:
        image: Imagem 3D de entrada (NumPy ou CuPy array)
        sigmas: Range de sigmas para multi-escala
        alpha: Sensibilidade a estruturas blob (0.1-1.0, padrão 0.5)
        beta: Sensibilidade ao ruído de fundo (0.1-1.0, padrão 0.5)
        gamma: Sensibilidade ao contraste (padrão None)
        black_ridges: Se True, detecta estruturas escuras
        normalization: Método de normalização ('robust', 'minmax', 'none')
        smooth_sigma: Sigma para suavização Gaussiana
        gd_epsilon: Piso numérico para evitar divisão por zero em Gf/Gd
        gf_center_percentile: Percentil usado como centro robusto da intensidade
        gf_scale_percentile: Percentil usado como escala robusta da grayness
        gd_clip_percentiles: Percentis usados para limitar outliers do gradiente
        ratio_clip: Intervalo opcional para limitar a razão Gf/Gd
        spacing: Espaçamento físico por eixo do array, na ordem (y, x, z)
        gpu: Se None (padrão), detecta automaticamente. Se True, força GPU. Se False, força CPU.

    Returns:
        modified_vesselness: Vesselness modificado com Gf e Gd (como NumPy array)
    """
    # Determina se deve usar GPU
    use_gpu_flag = gpu if gpu is not None else GPU_AVAILABLE

    if use_gpu_flag and GPU_AVAILABLE and gpu_filters is not None:
        # Converte para GPU
        img_gpu = to_gpu(image)

        # Suavização leve para remover ruído na GPU
        img_smooth = gpu_filters.gaussian(
            img_gpu, sigma=smooth_sigma, preserve_range=True
        )

        vesselness = get_vesselness(
            img_smooth,
            sigmas=sigmas,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            black_ridges=black_ridges,
            normalization="none",
            smooth_sigma=0.0,
            gpu=True,
            return_cpu=False,
        )

        # Medidas auxiliares na GPU (mantém na GPU para eficiência)
        gf = get_gf(
            img_smooth,
            center_percentile=gf_center_percentile,
            scale_percentile=gf_scale_percentile,
            scale_epsilon=gd_epsilon,
        )
        gd = get_gd(
            img_smooth,
            spacing=spacing,
            clip_percentiles=gd_clip_percentiles,
        )

        xp = cp
        gd_safe = xp.maximum(xp.abs(gd), gd_epsilon)
        ratio = gf / gd_safe
        if ratio_clip is not None:
            ratio = xp.clip(ratio, ratio_clip[0], ratio_clip[1])

        # Combinação na GPU
        modified_vesselness = vesselness * ratio
        modified_vesselness = _apply_normalization(modified_vesselness, normalization)

        # Só converte para CPU no final
        return to_cpu(modified_vesselness)
    else:
        # Pipeline na CPU
        if not isinstance(image, np.ndarray):
            image = to_cpu(image)

        # Suavização leve para remover ruído
        img_smooth = gaussian(image, sigma=smooth_sigma, preserve_range=True)

        vesselness = get_vesselness(
            img_smooth,
            sigmas=sigmas,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            black_ridges=black_ridges,
            normalization="none",
            smooth_sigma=0.0,
            gpu=False,
        )

        # Medidas auxiliares
        gf = get_gf(
            img_smooth,
            center_percentile=gf_center_percentile,
            scale_percentile=gf_scale_percentile,
            scale_epsilon=gd_epsilon,
        )
        gd = get_gd(
            img_smooth,
            spacing=spacing,
            clip_percentiles=gd_clip_percentiles,
        )
        gd_safe = np.maximum(np.abs(gd), gd_epsilon)
        ratio = gf / gd_safe
        if ratio_clip is not None:
            ratio = np.clip(ratio, ratio_clip[0], ratio_clip[1])

        # Combinação
        modified_vesselness = vesselness * ratio
        modified_vesselness = _apply_normalization(modified_vesselness, normalization)

        return modified_vesselness


def save_vesselness_cache(
    vesselness_i: Any, img_id: Any, cache_dir: str = "../cache"
) -> None:
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


def load_vesselness_cache(
    img_id: Any, cache_dir: str = "../cache"
) -> Optional[NDArray[Any]]:
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
