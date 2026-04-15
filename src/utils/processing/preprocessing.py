"""Utilitários centrais de pré-processamento para volumes CCTA.

Este módulo centraliza downscaling, thresholding, filtragem por maior
componente conectado e o pipeline principal de pré-processamento usado
pelas etapas de segmentação.
"""

import warnings

import cv2
import numpy as np
import scipy.ndimage as ndi

from .gpu_utils import GPU_AVAILABLE, cu_ndi, to_cpu, to_gpu


def _find_largest_component_label(labeled_array):
    """Retorna o rótulo do maior componente conectado, excluindo o fundo."""
    comp_sizes = np.bincount(labeled_array.ravel())
    comp_sizes[0] = 0

    if len(comp_sizes) <= 1:
        return None

    return np.argmax(comp_sizes)


def downscale_image_ndi(image, factors, order=3):
    """Reduz a resolução da imagem usando scipy.ndimage.zoom."""
    zoom_factors = tuple(1.0 / f for f in factors)
    return ndi.zoom(image, zoom=zoom_factors, order=order)


def downscale_image_opencv(image, factors, interpolation=cv2.INTER_LINEAR):
    """Reduz a resolução de imagem 2D/3D usando OpenCV resize."""
    if image.ndim == 2:
        new_shape = (
            int(image.shape[1] / factors[1]),
            int(image.shape[0] / factors[0]),
        )
        return cv2.resize(image, new_shape, interpolation=interpolation)

    if image.ndim == 3:
        factor_x, factor_y, factor_z = factors
        new_shape_xy = (
            int(image.shape[1] / factor_y),
            int(image.shape[0] / factor_x),
        )
        new_shape_z = int(image.shape[2] / factor_z)

        volume_resized_xy = np.zeros(
            (new_shape_xy[1], new_shape_xy[0], image.shape[2]), dtype=image.dtype
        )

        for z in range(image.shape[2]):
            volume_resized_xy[:, :, z] = cv2.resize(
                image[:, :, z], new_shape_xy, interpolation=interpolation
            )

        if factor_z != 1:
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
    image, factors, order=3, use_opencv=False, opencv_interpolation=None
):
    """Downscale otimizado com suporte opcional a OpenCV e GPU."""
    if use_opencv:
        if opencv_interpolation is None:
            opencv_interpolation = cv2.INTER_AREA
        return downscale_image_opencv(
            image, factors, interpolation=opencv_interpolation
        )

    if GPU_AVAILABLE:
        try:
            img_gpu = to_gpu(image)
            zoom_factors = tuple(1.0 / f for f in factors)
            result_gpu = cu_ndi.zoom(img_gpu, zoom=zoom_factors, order=order)
            return to_cpu(result_gpu)
        except Exception as e:
            warnings.warn(
                f"GPU downscaling falhou ({type(e).__name__}), usando CPU.", UserWarning
            )
            return downscale_image_ndi(image, factors, order=order)

    return downscale_image_ndi(image, factors, order=order)


def threshold_image(image, min_val=-300, max_val=675):
    """Aplica máscara de threshold por faixa inclusiva de HU/intensidade."""
    thresh_mask = (image >= min_val) & (image <= max_val)
    thresh_img = thresh_mask * image
    return thresh_img, thresh_mask


def threshold_image_with_offset(image, min_val=-300, max_val=675):
    """Aplica threshold e desloca os valores para evitar números negativos."""
    offset = np.abs(min_val)
    image_offset = image + offset

    thresh_mask = (image >= min_val) & (image <= max_val)
    thresh_img = thresh_mask * image_offset

    return thresh_img, thresh_mask, offset


def largest_connected_component(image, mask):
    """Mantém apenas o maior componente conectado e o aplica à imagem."""
    labeled_array, num_features = ndi.label(mask)
    if num_features == 0:
        return image, mask

    largest_comp_label = _find_largest_component_label(labeled_array)
    if largest_comp_label is None:
        return image, mask

    largest_comp_mask = labeled_array == largest_comp_label
    lcc_img = image * largest_comp_mask

    return lcc_img, largest_comp_mask


def run_core_preprocessing_pipeline(
    image,
    downscale_factors,
    min_threshold=-300,
    max_threshold_percentile=99.5,
    lcc_per_slice=True,
    order=3,
    use_opencv=False,
    opencv_interpolation=None,
):
    """Executa pipeline com downscale, threshold adaptativo e maior componente conectado."""
    if use_opencv:
        if opencv_interpolation is None:
            opencv_interpolation = cv2.INTER_AREA
        down_image = downscale_image_opencv(
            image, downscale_factors, interpolation=opencv_interpolation
        )
    else:
        down_image = downscale_image_ndi(image, downscale_factors, order=order)

    thresh_vals = (
        min_threshold,
        int(np.percentile(down_image, max_threshold_percentile)),
    )

    thresh_image, thresh_mask, offset = threshold_image_with_offset(
        down_image, *thresh_vals
    )

    if lcc_per_slice:
        lcc_image = np.zeros_like(thresh_image, dtype=float)
        for z in range(thresh_image.shape[2]):
            slice_mask = thresh_mask[:, :, z]
            slice_image = thresh_image[:, :, z]
            lcc_slice, _ = largest_connected_component(slice_image, slice_mask)
            lcc_image[:, :, z] = lcc_slice
    else:
        lcc_image, _ = largest_connected_component(thresh_image, thresh_mask)

    lcc_image -= offset

    return down_image, thresh_image, lcc_image, thresh_vals


__all__ = [
    "downscale_image",
    "downscale_image_ndi",
    "downscale_image_opencv",
    "largest_connected_component",
    "run_core_preprocessing_pipeline",
    "threshold_image",
    "threshold_image_with_offset",
]
