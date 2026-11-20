"""Utilitários centrais de pré-processamento para volumes CCTA.

Este módulo centraliza downscaling, thresholding, filtragem por maior
componente conectado e o pipeline principal de pré-processamento usado
pelas etapas de segmentação.
"""

import warnings
from typing import Any, Optional, Sequence, Tuple, cast

import cv2
import numpy as np
import scipy.ndimage as ndi
from numpy.typing import NDArray

from ..project.config import IMAGE_GRID_LABEL_PROTOCOL, LEGACY_LABEL_PROTOCOL
from .gpu_utils import GPU_AVAILABLE, cu_ndi, to_cpu, to_gpu


def _find_largest_component_label(labeled_array: NDArray[Any]) -> Optional[int]:
    """Retorna o rótulo do maior componente conectado, excluindo o fundo.

    Args:
        labeled_array (NDArray[Any]): Array inteiro de rótulos retornado por
            ``scipy.ndimage.label``.

    Returns:
        Optional[int]: Rótulo do componente com mais voxels ou ``None`` se só houver
            fundo.
    """
    comp_sizes = np.bincount(labeled_array.ravel())
    comp_sizes[0] = 0

    if len(comp_sizes) <= 1:
        return None

    return int(np.argmax(comp_sizes))


def downscale_image_ndi(
    image: NDArray[Any], factors: Sequence[float], order: int = 3
) -> NDArray[Any]:
    """Reduz a resolução da imagem usando ``scipy.ndimage.zoom``.

    Args:
        image (NDArray[Any]): Imagem ou volume NumPy a redimensionar.
        factors (Sequence[float]): Fatores divisores de redução por eixo alinhados ao
            shape de ``image``.
        order (int): Ordem da interpolação spline de ``scipy.ndimage.zoom``
            (``0`` equivale a vizinho mais próximo).

    Returns:
        NDArray[Any]: Array NumPy redimensionado conforme os fatores informados.
    """
    zoom_factors = tuple(1.0 / f for f in factors)
    return np.asarray(ndi.zoom(image, zoom=zoom_factors, order=order))


def downscale_image_opencv(
    image: NDArray[Any], factors: Sequence[float], interpolation: int = cv2.INTER_LINEAR
) -> NDArray[Any]:
    """Reduz a resolução de imagem 2D/3D usando ``cv2.resize``.

    Args:
        image (NDArray[Any]): Imagem 2D ou volume 3D a redimensionar.
        factors (Sequence[float]): Fatores divisores de redução por eixo alinhados ao
            shape de ``image``.
        interpolation (int): Código de interpolação do OpenCV (ex.:
            ``cv2.INTER_LINEAR``).

    Returns:
        NDArray[Any]: Array redimensionado preservando o dtype de ``image``.
    """
    # Caso 2D: devolve cópia para fator unitário ou redimensiona em uma passagem.
    if image.ndim == 2:
        if tuple(float(f) for f in factors[:2]) == (1.0, 1.0):
            return image.copy()
        new_shape = (
            int(image.shape[1] / factors[1]),
            int(image.shape[0] / factors[0]),
        )
        return cv2.resize(image, new_shape, interpolation=interpolation)

    # Caso 3D: calcula as dimensões alvo e separa a redução entre XY e profundidade z.
    if image.ndim == 3:
        if tuple(float(f) for f in factors) == (1.0, 1.0, 1.0):
            return image.copy()
        factor_x, factor_y, factor_z = factors
        new_shape_xy = (
            int(image.shape[1] / factor_y),
            int(image.shape[0] / factor_x),
        )
        new_shape_z = int(image.shape[2] / factor_z)

        # Redimensiona primeiro cada plano axial para preservar a organização 3D.
        volume_resized_xy = np.zeros(
            (new_shape_xy[1], new_shape_xy[0], image.shape[2]), dtype=image.dtype
        )

        for z in range(image.shape[2]):
            volume_resized_xy[:, :, z] = cv2.resize(
                image[:, :, z], new_shape_xy, interpolation=interpolation
            )

        if factor_z != 1:
            # A segunda passagem ajusta a profundidade sem misturar os eixos XY.
            volume_final = np.zeros(
                (new_shape_xy[1], new_shape_xy[0], new_shape_z), dtype=image.dtype
            )

            for y in range(new_shape_xy[1]):
                slice_xz = volume_resized_xy[y, :, :]
                resized_xz = cv2.resize(
                    slice_xz,
                    (new_shape_z, new_shape_xy[0]),
                    interpolation=interpolation,
                )
                volume_final[y, :, :] = resized_xz

            return volume_final

        return volume_resized_xy

    raise ValueError(f"Imagem deve ser 2D ou 3D, recebido shape: {image.shape}")


def downscale_image(
    image: NDArray[Any],
    factors: Sequence[float],
    order: int = 3,
    use_opencv: bool = False,
    opencv_interpolation: Optional[int] = None,
) -> NDArray[Any]:
    """Reduz a resolução de imagem/volume com suporte opcional a OpenCV e GPU.

    Args:
        image (NDArray[Any]): Imagem 2D ou volume 3D de entrada.
        factors (Sequence[float]): Fatores divisores de redução por eixo alinhados ao
            shape de ``image``.
        order (int): Ordem da interpolação spline usada no caminho SciPy/CuPy.
        use_opencv (bool): Se ``True``, usa OpenCV por planos em vez de SciPy/CuPy
            ``zoom``.
        opencv_interpolation (Optional[int]): Código ``cv2.INTER_*``; ``None`` seleciona
            ``cv2.INTER_LINEAR``.

    Returns:
        NDArray[Any]: Array redimensionado segundo o backend selecionado.
    """
    # Prioriza o caminho OpenCV por fatias quando configurado.
    if use_opencv:
        if opencv_interpolation is None:
            opencv_interpolation = cv2.INTER_LINEAR
        return downscale_image_opencv(
            image, factors, interpolation=opencv_interpolation
        )

    # Tenta GPU apenas no caminho SciPy; OpenCV já foi tratado acima.
    if GPU_AVAILABLE and cu_ndi is not None:
        try:
            img_gpu = to_gpu(image)
            zoom_factors = tuple(1.0 / f for f in factors)
            result_gpu = cu_ndi.zoom(img_gpu, zoom=zoom_factors, order=order)
            return to_cpu(result_gpu)
        except Exception as e:
            # Mantém o pipeline funcional quando a operação não é suportada na GPU.
            warnings.warn(
                f"GPU downscaling falhou ({type(e).__name__}), usando CPU.", UserWarning
            )
            return downscale_image_ndi(image, factors, order=order)

    return downscale_image_ndi(image, factors, order=order)


def downscale_label(
    label: NDArray[Any],
    config: dict[str, Any],
    *,
    allow_gpu_zoom: bool = False,
) -> NDArray[Any]:
    """Amostra classes discretas na grade da imagem ou reproduz o zoom histórico.

    Args:
        label (NDArray[Any]): Volume 3D de rótulos inteiros alinhado à imagem original.
        config (dict[str, Any]): Configuração efetiva contendo ``DOWNSCALE_FACTORS``,
            ``DOWNSCALE_METHOD`` e ``LABEL_RESAMPLING``.
        allow_gpu_zoom (bool): Se ``True``, permite usar CuPy ``zoom`` ordem ``0`` no
            caminho não-OpenCV.

    Returns:
        NDArray[Any]: Volume de rótulos reamostrado preservando os valores inteiros das
            classes.
    """
    # Valida o protocolo de reamostragem e evita trabalho extra para fator unitário.
    factors = config["DOWNSCALE_FACTORS"]
    protocol = config.get("LABEL_RESAMPLING", IMAGE_GRID_LABEL_PROTOCOL)
    if protocol not in (IMAGE_GRID_LABEL_PROTOCOL, LEGACY_LABEL_PROTOCOL):
        raise ValueError(f"Protocolo de redução dos rótulos desconhecido: {protocol}")
    if tuple(float(f) for f in factors) == (1.0, 1.0, 1.0):
        return np.asarray(label).copy()
    # Seleciona interpolação discreta via OpenCV ou zoom ordem 0 para manter classes.
    if protocol == IMAGE_GRID_LABEL_PROTOCOL and config["DOWNSCALE_METHOD"] == "opencv":
        interpolation = (
            cv2.INTER_NEAREST
            if config.get("OPENCV_INTERPOLATION") == "nearest"
            else cv2.INTER_NEAREST_EXACT
        )
        return downscale_image_opencv(label, factors, interpolation)
    if allow_gpu_zoom:
        return downscale_image(label, factors, order=0)
    return downscale_image_ndi(label, factors, order=0)


def threshold_image_with_offset(
    image: NDArray[Any], min_val: float = -300, max_val: float = 675
) -> Tuple[NDArray[Any], NDArray[Any], float]:
    """Aplica o limiar e desloca os valores válidos para uma faixa positiva.

    Args:
        image (NDArray[Any]): Imagem ou volume NumPy (em HU quando CCTA).
        min_val (float): Limite inferior inclusivo de intensidade, em HU.
        max_val (float): Limite superior inclusivo de intensidade, em HU.

    Returns:
        Tuple[NDArray[Any], NDArray[Any], float]: Tupla
            ``(thresh_img, thresh_mask, offset)`` com fundo zerado antes do LCC.
    """
    offset = np.abs(min_val)

    # Cria a máscara e preserva zero fora da faixa selecionada.
    thresh_mask = (image >= min_val) & (image <= max_val)
    thresh_img = np.zeros_like(image)
    thresh_img[thresh_mask] = image[thresh_mask] + offset

    return thresh_img, thresh_mask, offset


def largest_connected_component(
    image: NDArray[Any], mask: NDArray[Any]
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """Mantém apenas o maior componente conectado e o aplica à imagem.

    Args:
        image (NDArray[Any]): Imagem ou volume NumPy previamente limiarizado com fundo
            zerado.
        mask (NDArray[Any]): Máscara binária 2D ou 3D correspondente aos candidatos
            ativos.

    Returns:
        Tuple[NDArray[Any], NDArray[Any]]: Tupla ``(lcc_img, largest_comp_mask)`` (ou
            a entrada inalterada se a máscara estiver vazia).
    """
    # Rotula os componentes para selecionar aquele com maior número de voxels.
    labeled_array, num_features = cast(
        tuple[NDArray[Any], int],
        ndi.label(mask),
    )
    if num_features == 0:
        return image, mask

    # Seleciona o rótulo predominante e aplica sua máscara sobre a imagem de entrada.
    largest_comp_label = _find_largest_component_label(labeled_array)
    if largest_comp_label is None:
        return image, mask

    largest_comp_mask = labeled_array == largest_comp_label
    lcc_img = image * largest_comp_mask

    return lcc_img, largest_comp_mask


def build_lcc_image_from_mask(
    volume: NDArray[Any],
    mask: NDArray[Any],
    offset: float,
    per_slice: bool = True,
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """Aplica uma máscara binária e extrai o maior componente conectado (LCC).

    Args:
        volume (NDArray[Any]): Volume 3D original (em HU) sobre o qual a máscara é
            aplicada.
        mask (NDArray[Any]): Máscara binária 3D dos voxels candidatos.
        offset (float): Deslocamento somado temporariamente antes do LCC para separar
            valores negativos do fundo.
        per_slice (bool): Se ``True``, extrai o LCC em cada fatia axial; se ``False``,
            em 3D global.

    Returns:
        Tuple[NDArray[Any], NDArray[Any]]: Tupla ``(lcc_image, lcc_mask)`` em
            ``float32`` e booleano com o offset já subtraído.
    """
    # Recria a imagem limiarizada a partir da máscara candidata.
    thresholded = np.zeros_like(volume, dtype=np.float32)
    thresholded[mask] = volume[mask] + offset

    if per_slice:
        # Mantém o comportamento histórico do pipeline: LCC por fatia axial.
        lcc_image = np.zeros_like(thresholded, dtype=np.float32)
        lcc_mask = np.zeros_like(mask, dtype=bool)
        for z_idx in range(thresholded.shape[2]):
            lcc_slice, lcc_mask_slice = largest_connected_component(
                thresholded[:, :, z_idx], mask[:, :, z_idx]
            )
            lcc_image[:, :, z_idx] = lcc_slice
            lcc_mask[:, :, z_idx] = lcc_mask_slice
    else:
        # Alternativa global em 3D para comparar uma seleção mais restritiva.
        lcc_image, lcc_mask = largest_connected_component(thresholded, mask)

    # Remove o offset temporário para devolver as intensidades na escala original.
    return (lcc_image - offset).astype(np.float32), lcc_mask.astype(bool)


def run_core_preprocessing_pipeline(
    image: NDArray[Any],
    downscale_factors: Sequence[float],
    min_threshold: int = -300,
    max_threshold_percentile: float = 99.5,
    lcc_per_slice: bool = True,
    order: int = 3,
    use_opencv: bool = False,
    opencv_interpolation: Optional[int] = None,
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], Tuple[int, int]]:
    """Executa pipeline de downscale, threshold adaptativo e maior componente.

    Args:
        image (NDArray[Any]): Volume 3D bruto de entrada (tipicamente em HU).
        downscale_factors (Sequence[float]): Fatores divisores de redução por eixo
            alinhados ao shape de ``image``.
        min_threshold (int): Limiar inferior inclusivo em HU.
        max_threshold_percentile (float): Percentil superior usado para definir o teto
            de intensidade.
        lcc_per_slice (bool): Se ``True``, aplica LCC por fatia axial; se ``False``, em
            3D global.
        order (int): Ordem da interpolação spline quando ``use_opencv=False``.
        use_opencv (bool): Se ``True``, usa OpenCV no downscaling em vez de SciPy.
        opencv_interpolation (Optional[int]): Código ``cv2.INTER_*``; ``None`` usa
            ``cv2.INTER_LINEAR``.

    Returns:
        Tuple[NDArray[Any], NDArray[Any], NDArray[Any], Tuple[int, int]]: Tupla
            ``(down_image, thresh_image, lcc_image, thresh_vals)`` com o volume
            reduzido, limiarizado, filtrado por LCC e os limiares aplicados.
    """
    # Reduz a resolução espacial do volume pelo backend selecionado.
    if use_opencv:
        if opencv_interpolation is None:
            opencv_interpolation = cv2.INTER_LINEAR
        down_image = downscale_image_opencv(
            image, downscale_factors, interpolation=opencv_interpolation
        )
    else:
        down_image = downscale_image_ndi(image, downscale_factors, order=order)

    # O limite superior acompanha a distribuição de intensidades de cada exame.
    thresh_vals = (
        min_threshold,
        int(np.percentile(down_image, max_threshold_percentile)),
    )

    # Limiariza o volume e desloca valores válidos para faixa positiva antes do LCC.
    thresh_image, thresh_mask, offset = threshold_image_with_offset(
        down_image, *thresh_vals
    )

    # Permite comparar o comportamento histórico por fatia com uma LCC 3D global.
    if lcc_per_slice:
        lcc_image = np.zeros_like(thresh_image, dtype=thresh_image.dtype)
        for z in range(thresh_image.shape[2]):
            slice_mask = thresh_mask[:, :, z]
            slice_image = thresh_image[:, :, z]
            lcc_slice, _ = largest_connected_component(slice_image, slice_mask)
            lcc_image[:, :, z] = lcc_slice
    else:
        lcc_image, _ = largest_connected_component(thresh_image, thresh_mask)

    # Subtrai o offset para restaurar as intensidades relativas originais.
    lcc_image -= offset

    return down_image, thresh_image, lcc_image, thresh_vals


__all__ = [
    "downscale_image",
    "downscale_image_ndi",
    "downscale_image_opencv",
    "build_lcc_image_from_mask",
    "largest_connected_component",
    "run_core_preprocessing_pipeline",
    "threshold_image_with_offset",
]
