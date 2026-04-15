"""
Módulo com utilitários comuns para aceleração GPU.

Este módulo centraliza a detecção de GPU e funções auxiliares de conversão
entre NumPy e CuPy, usadas em todos os módulos que suportam GPU.
"""

import numpy as np
import warnings

# =============================================================================
# Detecção de GPU e Imports
# =============================================================================

GPU_AVAILABLE = False
cp = None
cu_ndi = None

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cu_ndi

    # Teste simples: cria array pequeno na GPU
    _ = cp.array([1, 2, 3])
    GPU_AVAILABLE = True

except ImportError:
    # CuPy não instalado
    cp = None
    cu_ndi = None
    GPU_AVAILABLE = False

except Exception as e:
    # Outro erro (ex: CUDA não encontrado, GPU driver, etc)
    cp = None
    cu_ndi = None
    GPU_AVAILABLE = False
    warnings.warn(
        f"GPU não disponível ({type(e).__name__}). Operações usarão CPU.",
        UserWarning
    )


# =============================================================================
# Funções Públicas
# =============================================================================


def use_gpu():
    """
    Retorna True se GPU está disponível e CuPy está instalado.

    Returns:
        bool: True se GPU disponível, False caso contrário

    Example:
        >>> if use_gpu():
        ...     print("GPU disponível!")
        ... else:
        ...     print("Usando CPU")
    """
    return GPU_AVAILABLE


def to_gpu(arr):
    """
    Converte array NumPy para CuPy (GPU) se GPU disponível.

    Se GPU não estiver disponível, retorna o array inalterado.

    Args:
        arr (np.ndarray or cp.ndarray): Array a converter

    Returns:
        cp.ndarray or np.ndarray: Array na GPU se disponível, senão retorna entrada

    Example:
        >>> arr_cpu = np.array([1, 2, 3])
        >>> arr_gpu = to_gpu(arr_cpu)  # cp.ndarray se GPU disponível
    """
    if GPU_AVAILABLE and isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    return arr


def to_cpu(arr):
    """
    Converte array CuPy (GPU) para NumPy (CPU) se necessário.

    Se o array já for NumPy, retorna inalterado.

    Args:
        arr (np.ndarray or cp.ndarray): Array a converter

    Returns:
        np.ndarray: Array na CPU (sempre NumPy)

    Example:
        >>> arr_gpu = cp.array([1, 2, 3])  # Na GPU
        >>> arr_cpu = to_cpu(arr_gpu)  # np.ndarray na CPU
    """
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


def get_array_module(arr):
    """
    Retorna o módulo apropriado (numpy ou cupy) baseado no tipo do array.

    Útil para escrever código que funciona tanto com NumPy quanto CuPy.

    Args:
        arr (np.ndarray or cp.ndarray): Array de entrada

    Returns:
        module: numpy ou cupy

    Example:
        >>> arr = cp.array([1, 2, 3])
        >>> xp = get_array_module(arr)
        >>> result = xp.sum(arr)  # Usa cp.sum() automaticamente
    """
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp
    return np


def ensure_cpu(arr):
    """
    Garante que o array está na CPU (NumPy).

    Alias para to_cpu() com nome mais descritivo.

    Args:
        arr: Array de qualquer tipo

    Returns:
        np.ndarray: Array na CPU
    """
    return to_cpu(arr)


def ensure_gpu(arr):
    """
    Garante que o array está na GPU (CuPy) se GPU disponível.

    Alias para to_gpu() com nome mais descritivo.

    Args:
        arr: Array de qualquer tipo

    Returns:
        cp.ndarray or np.ndarray: Array na GPU se disponível
    """
    return to_gpu(arr)
