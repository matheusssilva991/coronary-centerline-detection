"""Cálculo de vesselness de Frangi em 3D com aceleração opcional em GPU e normalização."""

import warnings
from typing import Any, Optional, Sequence, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from skimage.filters import gaussian, ridges

from ..utils import normalize_image, robust_normalize
from .gpu_utils import (
    GPU_AVAILABLE,
    to_cpu,
    to_gpu,
)

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
    """Converte e valida a sequência de sigmas positivos para o filtro multiescala.

    Args:
        sigmas (ArrayLike): Escalas gaussianas (em voxels) informadas como escalar ou
            sequência.

    Returns:
        list[float]: Lista validada de sigmas positivos em ponto flutuante.
    """
    sigma_values = np.atleast_1d(np.asarray(sigmas, dtype=float)).tolist()

    if not sigma_values:
        raise ValueError("sigmas deve conter pelo menos um valor.")
    if any(sigma <= 0 for sigma in sigma_values):
        raise ValueError("Todos os valores de sigmas devem ser maiores que zero.")

    return sigma_values


def _validate_normalization_method(normalization: str) -> None:
    """Valida o método de normalização suportado pelo filtro de Frangi.

    Args:
        normalization (str): Nome do método de normalização pós-filtro (``none``,
            ``robust`` ou ``minmax``).
    """
    if normalization not in {"none", "robust", "minmax"}:
        raise ValueError(
            f"Método de normalização '{normalization}' inválido. "
            "Use 'robust', 'minmax' ou 'none'."
        )


def _apply_normalization(array: Any, normalization: str) -> Any:
    """Aplica normalização opcional ao mapa de vesselness gerado.

    Args:
        array (Any): Mapa de vesselness calculado em CPU ou GPU.
        normalization (str): Método desejado (``none``, ``robust`` ou ``minmax``).

    Returns:
        Any: Array normalizado segundo o método selecionado, preservando o dispositivo
            quando ``normalization`` é ``none``.
    """
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
    """Calcula o mapa de vesselness 3D pelo filtro de Frangi, usando GPU se disponível.

    Args:
        image (Any): Volume 3D de entrada (CPU ou GPU), tipicamente pré-processado em
            HU.
        sigmas (Sequence[float]): Escalas gaussianas positivas (em voxels) usadas no
            filtro multiescala.
        alpha (float): Sensibilidade da razão de Frangi entre estruturas tubulares e
            laminares.
        beta (float): Sensibilidade da razão de Frangi que distingue estruturas tipo
            bolha.
        gamma (Optional[float]): Escala de contraste de fundo; ``None`` usa estimativa
            automática do filtro.
        black_ridges (bool): Se ``True``, realça estruturas escuras em vez de vasos
            claros.
        normalization (str): Método pós-filtro (``none``, ``robust`` por percentis ou
            ``minmax``).
        smooth_sigma (float): Desvio-padrão gaussiano (em voxels) da suavização prévia;
            ``0`` desativa.
        gpu (Optional[bool]): Preferência de backend; ``None`` detecta automaticamente e
            booleanos forçam GPU ou CPU com fallback.
        return_cpu (bool): Se ``True``, converte o resultado para NumPy antes de
            retornar.

    Returns:
        NDArray[Any]: Mapa 3D de vesselness com o mesmo shape de ``image``.
    """
    # Valida escalas e método de normalização antes de selecionar o backend.
    sigma_values = _as_sigma_sequence(sigmas)
    _validate_normalization_method(normalization)
    use_gpu_flag = gpu if gpu is not None else GPU_AVAILABLE

    # O caminho GPU mantém os dados no dispositivo até a normalização final.
    if use_gpu_flag and GPU_AVAILABLE and gpu_filters is not None:
        try:
            filters_gpu = cast(Any, gpu_filters)
            image_gpu = to_gpu(image)
            # Aplica suavização prévia opcional e executa o filtro multiescala na GPU.
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
