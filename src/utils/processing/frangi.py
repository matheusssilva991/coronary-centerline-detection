import numpy as np
from skimage.filters import ridges, gaussian
import warnings
from typing import Any, Optional, Sequence, cast
from numpy.typing import ArrayLike, NDArray

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


def _as_sigma_sequence(sigmas: ArrayLike) -> list[float]:
    sigma_values = np.atleast_1d(np.asarray(sigmas, dtype=float)).tolist()

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
    sigmas: Sequence[float] = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5),
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

    # O caminho GPU mantém os dados no dispositivo até a normalização final.
    if use_gpu_flag and GPU_AVAILABLE and gpu_filters is not None:
        try:
            filters_gpu = cast(Any, gpu_filters)
            image_gpu = to_gpu(image)
            frangi_input = (
                filters_gpu.gaussian(
                    image_gpu,
                    sigma=smooth_sigma,
                    preserve_range=True,
                )
                if smooth_sigma > 0
                else image_gpu
            )
            vesselness = filters_gpu.frangi(
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
            # Falhas do backend não devem interromper a execução do pipeline.
            warnings.warn(
                f"Frangi GPU falhou ({type(e).__name__}: {e}). Usando CPU.",
                UserWarning,
            )

    # O fallback CPU reproduz suavização e parâmetros usados no backend GPU.
    img_cpu = image if isinstance(image, np.ndarray) else to_cpu(image)
    gaussian_filter = cast(Any, gaussian)
    frangi_filter = cast(Any, ridges.frangi)
    frangi_input = (
        gaussian_filter(img_cpu, sigma=smooth_sigma, preserve_range=True)
        if smooth_sigma > 0
        else img_cpu
    )
    vesselness = frangi_filter(
        frangi_input,
        sigmas=sigma_values,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=black_ridges,
    )
    return np.asarray(_apply_normalization(vesselness, normalization))
