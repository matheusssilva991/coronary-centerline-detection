"""
Módulo com utilitários comuns para aceleração GPU.

Este módulo centraliza a detecção de GPU e funções auxiliares de conversão
entre NumPy e CuPy, usadas em todos os módulos que suportam GPU.
"""

import warnings
from typing import Any

import numpy as np

GPU_AVAILABLE = False
cp = None
cu_ndi = None

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cu_ndi

    # Valida alocação mínima para confirmar runtime CUDA funcional.
    _ = cp.array([1, 2, 3])
    GPU_AVAILABLE = True

except ImportError:
    cp = None
    cu_ndi = None
    GPU_AVAILABLE = False

except Exception as e:
    cp = None
    cu_ndi = None
    GPU_AVAILABLE = False
    warnings.warn(
        f"GPU não disponível ({type(e).__name__}). Operações usarão CPU.", UserWarning
    )


def use_gpu() -> bool:
    """Informa se o runtime GPU (CuPy) está disponível para aceleração.

    Returns:
        bool: ``True`` quando o runtime CUDA/CuPy foi inicializado com sucesso.
    """
    return GPU_AVAILABLE


def to_gpu(arr: Any) -> Any:
    """Transfere um array NumPy para CuPy se a GPU estiver disponível.

    Mantém objetos que já estão em GPU ou fora do formato ``np.ndarray`` inalterados.

    Args:
        arr (Any): Array NumPy a transferir ou objeto já em GPU/outro formato.

    Returns:
        Any: Array CuPy na GPU quando disponível; caso contrário, o objeto original.
    """
    if GPU_AVAILABLE and cp is not None and isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    return arr


def to_cpu(arr: Any) -> Any:
    """Converte um array CuPy da GPU para NumPy na CPU, mantendo arrays CPU.

    Args:
        arr (Any): Array CuPy na GPU ou objeto já residente na CPU.

    Returns:
        Any: Array NumPy na CPU quando ``arr`` for CuPy; caso contrário, o objeto
            original.
    """
    if GPU_AVAILABLE and cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


def get_array_module(arr: Any) -> Any:
    """Retorna o módulo ``numpy`` ou ``cupy`` compatível com o array informado.

    Args:
        arr (Any): Array de referência usado para escolher o módulo numérico.

    Returns:
        Any: Módulo ``cupy`` se ``arr`` for um array CuPy ativo; caso contrário,
            ``numpy``.
    """
    if GPU_AVAILABLE and cp is not None and isinstance(arr, cp.ndarray):
        return cp
    return np
