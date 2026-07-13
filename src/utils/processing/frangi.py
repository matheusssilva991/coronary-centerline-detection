import numpy as np
from skimage.filters import ridges, gaussian
import os
import pickle
import warnings
from typing import Any, Optional, Sequence
from numpy.typing import NDArray

# Importa utilitários de GPU centralizados
from .gpu_utils import (
    to_gpu,
    to_cpu,
    GPU_AVAILABLE,
)

# Importa funções de normalização
from ..utils import normalize_image, robust_normalize

# Importa cucim filters se GPU disponível
gpu_filters = None
if GPU_AVAILABLE:
    try:
        import cucim.skimage.filters as gpu_filters
    except Exception as e:
        warnings.warn(
            f"cuCIM indisponível para Frangi GPU ({type(e).__name__}: {e}). "
            "Frangi usará CPU.",
            UserWarning,
        )
        gpu_filters = None


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
        try:
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
        except Exception as e:
            warnings.warn(
                f"Frangi GPU falhou ({type(e).__name__}: {e}). Usando CPU.",
                UserWarning,
            )

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


def save_vesselness_cache(
    vesselness_i: Any, img_id: Any, cache_dir: str = "../cache"
) -> None:
    """
    Salva o mapa de vesselness em cache comprimido sem alterar dtype.

    O formato atual usa ``.npz`` com compressão lossless. Isso reduz uso de
    disco sem converter para float16, então evita pequenas diferenças numéricas
    nos resultados.

    Args:
        vesselness_i: Mapa de vesselness (NumPy ou CuPy array)
        img_id: ID da imagem
        cache_dir: Diretório para salvar o cache
    """
    os.makedirs(cache_dir, exist_ok=True)
    # Converte para NumPy se necessário
    vesselness_i = np.asarray(to_cpu(vesselness_i))
    cache_path = os.path.join(cache_dir, f"vesselness_{img_id}.npz")
    np.savez_compressed(cache_path, vesselness=vesselness_i)


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
    compressed_cache_path = os.path.join(cache_dir, f"vesselness_{img_id}.npz")
    if os.path.exists(compressed_cache_path):
        with np.load(compressed_cache_path) as data:
            return data["vesselness"]

    # Compatibilidade com caches antigos gerados em pickle.
    legacy_cache_path = os.path.join(cache_dir, f"vesselness_{img_id}.pkl")
    if os.path.exists(legacy_cache_path):
        with open(legacy_cache_path, "rb") as f:
            return pickle.load(f)
    return None
