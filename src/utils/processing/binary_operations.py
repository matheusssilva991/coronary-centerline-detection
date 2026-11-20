"""
Módulo para operações morfológicas em imagens médicas com suporte GPU.

Este módulo implementa operações morfológicas binárias (closing, dilation, erosion)
e labeling de componentes conectados, com aceleração automática por GPU quando
disponível.
"""

from typing import Any, Optional, Tuple, cast

import numpy as np
import scipy.ndimage as ndi
from numpy.typing import NDArray

from .gpu_utils import (
    GPU_AVAILABLE,
    cu_ndi,
    to_cpu,
    to_gpu,
)


def _use_gpu_ndi(gpu: Optional[bool]) -> bool:
    """Resolve se a operação morfológica deve usar o backend CuPy.

    Args:
        gpu (Optional[bool]): Preferência explícita de execução em GPU; ``None`` usa
            a detecção automática de disponibilidade.

    Returns:
        bool: ``True`` se a operação deve usar ``cupyx.scipy.ndimage`` na GPU;
            ``False`` para executar na CPU.
    """
    requested = gpu if gpu is not None else GPU_AVAILABLE
    return bool(requested and GPU_AVAILABLE and cu_ndi is not None)


def binary_closing(
    mask: NDArray[Any],
    structure: Optional[NDArray[Any]] = None,
    gpu: Optional[bool] = None,
) -> NDArray[np.uint8]:
    """Aplica fechamento binário para preencher pequenas lacunas.

    Executa dilatação seguida de erosão em GPU (CuPy) ou CPU (SciPy).

    Args:
        mask (NDArray[Any]): Máscara binária 2D ou 3D de entrada.
        structure (Optional[NDArray[Any]]): Elemento estruturante; ``None`` usa a
            conectividade padrão da biblioteca.
        gpu (Optional[bool]): Preferência de backend; ``None`` detecta GPU
            automaticamente e booleanos forçam CuPy ou SciPy na CPU.

    Returns:
        NDArray[np.uint8]: Máscara fechada na CPU com valores ``uint8``.
    """
    if _use_gpu_ndi(gpu):
        assert cu_ndi is not None
        mask_gpu = to_gpu(mask)
        struct_gpu = to_gpu(structure) if structure is not None else None

        # Fechamento em GPU: dilatação seguida de erosão antes de retornar à CPU.
        result = cu_ndi.binary_dilation(mask_gpu, structure=struct_gpu)
        result = cu_ndi.binary_erosion(result, structure=struct_gpu)

        return to_cpu(result).astype(np.uint8)
    else:
        return ndi.binary_closing(mask, structure=structure).astype(np.uint8)


def binary_dilation(
    mask: NDArray[Any],
    structure: Optional[NDArray[Any]] = None,
    gpu: Optional[bool] = None,
) -> NDArray[np.uint8]:
    """Aplica dilatação binária para expandir regiões ou unir segmentos próximos.

    Args:
        mask (NDArray[Any]): Máscara binária 2D ou 3D de entrada.
        structure (Optional[NDArray[Any]]): Elemento estruturante; ``None`` usa a
            conectividade padrão da biblioteca.
        gpu (Optional[bool]): Preferência de backend; ``None`` detecta GPU
            automaticamente e booleanos forçam CuPy ou SciPy na CPU.

    Returns:
        NDArray[np.uint8]: Máscara dilatada na CPU com valores ``uint8``.
    """
    if _use_gpu_ndi(gpu):
        assert cu_ndi is not None
        mask_gpu = to_gpu(mask)
        struct_gpu = to_gpu(structure) if structure is not None else None

        result = cu_ndi.binary_dilation(mask_gpu, structure=struct_gpu)
        return to_cpu(result).astype(np.uint8)
    else:
        return ndi.binary_dilation(mask, structure=structure).astype(np.uint8)


def binary_erosion(
    mask: NDArray[Any],
    structure: Optional[NDArray[Any]] = None,
    gpu: Optional[bool] = None,
) -> NDArray[np.uint8]:
    """Aplica erosão binária para remover bordas finas e desconectar artefatos.

    Args:
        mask (NDArray[Any]): Máscara binária 2D ou 3D de entrada.
        structure (Optional[NDArray[Any]]): Elemento estruturante; ``None`` usa a
            conectividade padrão da biblioteca.
        gpu (Optional[bool]): Preferência de backend; ``None`` detecta GPU
            automaticamente e booleanos forçam CuPy ou SciPy na CPU.

    Returns:
        NDArray[np.uint8]: Máscara erodida na CPU com valores ``uint8``.
    """
    if _use_gpu_ndi(gpu):
        assert cu_ndi is not None
        mask_gpu = to_gpu(mask)
        struct_gpu = to_gpu(structure) if structure is not None else None

        result = cu_ndi.binary_erosion(mask_gpu, structure=struct_gpu)
        return to_cpu(result).astype(np.uint8)
    else:
        return ndi.binary_erosion(mask, structure=structure).astype(np.uint8)


def label(mask: NDArray[Any], gpu: Optional[bool] = None) -> Tuple[NDArray[Any], int]:
    """Rotula componentes conectados na máscara binária de entrada.

    Args:
        mask (NDArray[Any]): Máscara binária 2D ou 3D de entrada.
        gpu (Optional[bool]): Preferência de backend; ``None`` detecta GPU
            automaticamente e booleanos forçam CuPy ou SciPy na CPU.

    Returns:
        Tuple[NDArray[Any], int]: Tupla ``(labeled_array, num_features)`` com rótulos
            inteiros na CPU e total de componentes conectados.
    """
    if _use_gpu_ndi(gpu):
        assert cu_ndi is not None
        mask_gpu = to_gpu(mask)
        labeled_gpu, num_features = cast(
            tuple[Any, int],
            cu_ndi.label(mask_gpu),
        )
        return to_cpu(labeled_gpu), int(num_features)
    else:
        labeled, count = cast(tuple[NDArray[Any], int], ndi.label(mask))
        return labeled, int(count)


def keep_largest_component(
    mask: NDArray[Any], gpu: Optional[bool] = None
) -> NDArray[np.uint8]:
    """Mantém apenas o maior componente conectado da máscara binária em ``uint8``.

    Args:
        mask (NDArray[Any]): Máscara binária 2D ou 3D de entrada.
        gpu (Optional[bool]): Preferência de backend; ``None`` detecta GPU
            automaticamente e booleanos forçam CuPy ou SciPy na CPU.

    Returns:
        NDArray[np.uint8]: Máscara ``uint8`` com apenas o maior componente conectado
            (ou zerada se vazia).
    """
    # Rotula os componentes conectados na GPU ou CPU e retorna cedo se vazia.
    if _use_gpu_ndi(gpu):
        assert cu_ndi is not None
        mask_gpu = to_gpu(mask)
        labeled_gpu, num = cast(tuple[Any, int], cu_ndi.label(mask_gpu))

        if num == 0:
            return to_cpu(mask_gpu).astype(np.uint8)

        labeled = to_cpu(labeled_gpu)
    else:
        labeled, num = cast(tuple[NDArray[Any], int], ndi.label(mask))

        if num == 0:
            return mask.astype(np.uint8) if isinstance(mask, np.ndarray) else mask

    # Ignora o rótulo de fundo (0) para selecionar o componente com mais voxels.
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest_label = int(np.argmax(counts))

    return (labeled == largest_label).astype(np.uint8)


def binary_opening(
    input: NDArray[Any],
    structure: Optional[NDArray[Any]] = None,
    gpu: Optional[bool] = None,
) -> NDArray[np.uint8]:
    """Aplica abertura binária para remover ruídos isolados em ``uint8``.

    Executa erosão seguida de dilatação usando elemento estruturante padrão ``3x3x3``
    quando nenhum kernel é fornecido, com fallback automático para CPU em falhas GPU.

    Args:
        input (NDArray[Any]): Máscara binária 2D ou 3D de entrada.
        structure (Optional[NDArray[Any]]): Elemento estruturante; ``None`` utiliza um
            cubo booleano ``3x3x3``.
        gpu (Optional[bool]): Preferência de backend; ``None`` detecta GPU
            automaticamente e booleanos forçam CuPy ou SciPy na CPU (com fallback).

    Returns:
        NDArray[np.uint8]: Máscara aberta na CPU com valores ``uint8``.
    """
    # Garante elemento estruturante 3D padrão quando nenhum kernel é informado.
    if structure is None:
        structure = np.ones((3, 3, 3), dtype=bool)

    if _use_gpu_ndi(gpu):
        assert cu_ndi is not None
        try:
            input_gpu = to_gpu(input)
            struct_gpu = to_gpu(structure)

            # Abertura em GPU: erosão seguida de dilatação com fallback para SciPy em falhas de dispositivo.
            opened_gpu = cu_ndi.binary_erosion(input_gpu, structure=struct_gpu)
            opened_gpu = cu_ndi.binary_dilation(opened_gpu, structure=struct_gpu)

            return to_cpu(opened_gpu).astype(np.uint8)
        except Exception:
            return ndi.binary_opening(input, structure=structure).astype(np.uint8)
    else:
        return ndi.binary_opening(input, structure=structure).astype(np.uint8)
