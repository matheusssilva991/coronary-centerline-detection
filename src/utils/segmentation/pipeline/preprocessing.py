"""Carregamento, pré-processamento e vesselness reutilizáveis no pipeline.

Este módulo concentra as etapas que transformam um caso ImageCAS bruto em
volumes prontos para detecção de aorta/óstios e segmentação arterial.
"""

from typing import Any, Dict, Sequence

import cv2
import numpy as np

from utils.processing.frangi import get_vesselness
from utils.processing.preprocessing import (
    build_lcc_image_from_mask,
    downscale_image_ndi,
    downscale_image_opencv,
    downscale_label,
)
from utils.segmentation.thresholding import (
    fuzzy_threshold_from_config,
    get_thresholding_config,
    normalize_threshold_mode,
    resolve_lower_threshold,
)
from utils.utils.nifti_io import load_raw_img_and_label


def load_and_preprocess_image(
    img_id: str,
    base_path: str,
    config: Dict[str, Any],
    *,
    include_intermediates: bool = False,
) -> Dict[str, Any]:
    """Carrega exame/label do ImageCAS e executa o pré-processamento padrão até a LCC.

    Args:
        img_id (str): Identificador do exame no ImageCAS (sem sufixo ``.img.nii.gz``).
        base_path (str): Diretório base contendo os volumes e rótulos do ImageCAS.
        config (Dict[str, Any]): Configuração já normalizada e escalada para a resolução
            de trabalho.
        include_intermediates (bool): Se ``True``, inclui volumes intermediários
            (evitar em lote para poupar RAM).

    Returns:
        Dict[str, Any]: Dicionário produzido por ``preprocess_ccta_volume`` com
            ``lcc_image``, ``label``, ``spacing``, ``scaled_spacing``,
            ``downscale_factors`` e detalhes.
    """
    # Carrega o volume CCTA e a máscara de referência do ImageCAS.
    nii_img, nii_label = load_raw_img_and_label(
        f"{base_path}/{img_id}.img.nii.gz", f"{base_path}/{img_id}.label.nii.gz"
    )
    if nii_label is None:
        raise ValueError(f"Label não carregado para o exame {img_id}.")
    spacing = nii_img.header.get_zooms()
    img = np.asarray(nii_img.get_fdata(dtype=np.float32), dtype=np.float32)
    label = np.asarray(nii_label.get_fdata(dtype=np.float32), dtype=np.uint8)

    # Delega downscaling, limiarização e extração da LCC ao fluxo comum de volumes.
    return preprocess_ccta_volume(
        img,
        spacing,
        config,
        label=label,
        include_intermediates=include_intermediates,
    )


def preprocess_ccta_volume(
    image: Any,
    spacing: Sequence[float],
    config: Dict[str, Any],
    *,
    label: Any | None = None,
    include_intermediates: bool = False,
) -> Dict[str, Any]:
    """Pré-processa um volume CCTA, com label de referência opcional.

    Entrada comum para ImageCAS e volumes externos (OrCaScore/MM-WHS), assumindo
    layout ``(x, y, z)`` e espaçamento positivo nessa mesma ordem.

    Args:
        image (Any): Volume CCTA 3D bruto em HU no layout ``(x, y, z)``.
        spacing (Sequence[float]): Espaçamento físico positivo em ``mm`` na ordem
            ``(dx, dy, dz)``.
        config (Dict[str, Any]): Configuração efetiva já normalizada e escalada para a
            resolução desejada.
        label (Any | None): Rótulo 3D opcional no mesmo shape de ``image`` (preservado
            por vizinho mais próximo).
        include_intermediates (bool): Se ``True``, inclui ``image``, ``down_image``,
            ``threshold_mask`` e ``lcc_mask``.

    Returns:
        Dict[str, Any]: Dicionário com ``lcc_image`` (``float32``), ``label``,
            ``spacing``, ``scaled_spacing``, ``downscale_factors`` e
            ``preprocessing_details``.
    """
    # Valida dimensionalidade do volume, espaçamento físico e compatibilidade do label.
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 3:
        raise ValueError("O volume CCTA deve ser tridimensional.")

    spacing = tuple(float(value) for value in spacing[:3])
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError("spacing deve conter três valores positivos.")

    label_array = None
    if label is not None:
        label_array = np.asarray(label, dtype=np.uint8)
        if label_array.shape != img.shape:
            raise ValueError("Imagem e label devem possuir o mesmo shape.")

    # Define o método de downsampling usado para imagem e label.
    downscale_factors = config["DOWNSCALE_FACTORS"]
    use_opencv = config["DOWNSCALE_METHOD"] == "opencv"
    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos4": cv2.INTER_LANCZOS4,
    }
    opencv_interpolation = interpolation_map.get(
        config["OPENCV_INTERPOLATION"], cv2.INTER_LINEAR
    )

    thresholding_config = get_thresholding_config(config)
    threshold_mode = normalize_threshold_mode(thresholding_config.get("method"))

    preprocessing_details: Dict[str, Any] = {
        "threshold_mode": threshold_mode,
    }

    # A redução é idêntica para todos os modos de pré-processamento.
    if use_opencv:
        down_image = downscale_image_opencv(
            img,
            downscale_factors,
            interpolation=opencv_interpolation,
        )
    else:
        down_image = downscale_image_ndi(img, downscale_factors, order=3)

    # Aplica o modo de threshold escolhido (none, normal ou fuzzy) e extrai a LCC.
    if threshold_mode == "none":
        # Sem corte de HU, todos os voxels finitos continuam candidatos.
        # A LCC da máscara completa não filtraria as fatias, então a omitimos.
        candidate_mask = np.isfinite(down_image)
        lcc_image = (
            down_image
            if bool(candidate_mask.all())
            else np.where(candidate_mask, down_image, 0).astype(np.float32)
        )
        lcc_mask = candidate_mask
        selected_mask = candidate_mask
        preprocessing_details.update(
            {
                "lower_threshold_method": None,
                "min_threshold": None,
                "max_threshold": None,
                "effective_upper_threshold_hu": None,
                "threshold_voxels": None,
                "lcc_voxels": None,
            }
        )
    elif threshold_mode == "normal":
        # Caminho histórico: threshold HU superior por percentil + LCC.
        min_threshold, lower_threshold_details = resolve_lower_threshold(
            down_image,
            config,
        )
        thresh_vals = (
            float(min_threshold),
            float(np.percentile(down_image, config["MAX_THRESHOLD_PERCENTILE"])),
        )
        thresh_mask = (down_image >= thresh_vals[0]) & (down_image <= thresh_vals[1])
        selected_mask = thresh_mask
        lcc_image, lcc_mask = build_lcc_image_from_mask(
            down_image,
            thresh_mask,
            offset=abs(float(min_threshold)),
            per_slice=True,
        )
        preprocessing_details.update(
            {
                **lower_threshold_details,
                "min_threshold": float(thresh_vals[0]),
                "max_threshold": float(thresh_vals[1]),
                "effective_upper_threshold_hu": float(thresh_vals[1]),
                "threshold_voxels": int(np.sum(thresh_mask)),
                "lcc_voxels": int(np.sum(lcc_mask)),
            }
        )
    else:
        # Fuzzy threshold: mantém apenas voxels cuja classe dominante é objeto.
        fuzzy_config = dict(config)
        fuzzy_threshold_config = dict(thresholding_config.get("fuzzy", {}))
        fuzzy_lower_config = dict(config.get("LOWER_THRESHOLD", {}))
        fuzzy_lower_config["percentile"] = float(
            fuzzy_threshold_config.get(
                "lower_percentile",
                fuzzy_lower_config.get("percentile", 10.5),
            )
        )
        fuzzy_config["LOWER_THRESHOLD"] = fuzzy_lower_config
        min_threshold, lower_threshold_details = resolve_lower_threshold(
            down_image,
            fuzzy_config,
        )
        fuzzy_config["MIN_THRESHOLD"] = float(min_threshold)
        fuzzy_mask, fuzzy_details = fuzzy_threshold_from_config(
            down_image,
            fuzzy_config,
        )
        selected_mask = fuzzy_mask
        lcc_image, lcc_mask = build_lcc_image_from_mask(
            down_image,
            fuzzy_mask,
            offset=abs(float(min_threshold)),
            per_slice=True,
        )
        preprocessing_details.update(
            {
                **lower_threshold_details,
                "min_threshold": float(min_threshold),
                "max_threshold": None,
                "effective_upper_threshold_hu": None,
                "threshold_voxels": int(np.sum(fuzzy_mask)),
                "lcc_voxels": int(np.sum(lcc_mask)),
                **fuzzy_details,
            }
        )
    # O label usa interpolação de vizinho mais próximo para preservar classes.
    down_label = (
        downscale_label(label_array, config) if label_array is not None else None
    )

    # Recalcula o espaçamento após downscaling e monta o dicionário de retorno.
    dx, dy, dz = (
        spacing[0] * downscale_factors[0],
        spacing[1] * downscale_factors[1],
        spacing[2] * downscale_factors[2],
    )

    result = {
        "lcc_image": lcc_image,
        "label": down_label,
        "preprocessing_details": preprocessing_details,
        "spacing": spacing,
        "scaled_spacing": (dx, dy, dz),
        "downscale_factors": downscale_factors,
    }
    if include_intermediates:
        result.update(
            {
                "image": img,
                "down_image": down_image,
                "threshold_mask": selected_mask.astype(bool),
                "lcc_mask": lcc_mask.astype(bool),
            }
        )
    return result


def compute_vesselness(
    image: Any,
    vesselness_config: Dict[str, Any],
    use_gpu: bool = False,
) -> Any:
    """Calcula o mapa de vasos (Frangi) para um volume 3D.

    Args:
        image (Any): Volume 3D pré-processado (tipicamente ``lcc_image``).
        vesselness_config (Dict[str, Any]): Dicionário de configuração
            (``VESSELNESS_AORTA`` ou ``VESSELNESS_ARTERY``).
        use_gpu (bool): Se ``True``, habilita execução GPU no filtro de Frangi quando
            disponível.

    Returns:
        Any: Mapa 3D de vesselness com o mesmo shape de ``image``.
    """
    # Monta os parâmetros do Frangi validado para aorta e artérias.
    vesselness_kwargs = {
        "sigmas": vesselness_config["sigmas"],
        "black_ridges": vesselness_config.get("black_ridges", False),
        "alpha": vesselness_config["alpha"],
        "beta": vesselness_config["beta"],
        "gamma": vesselness_config["gamma"],
        "normalization": vesselness_config.get("normalization", "none"),
        "smooth_sigma": vesselness_config.get("smooth_sigma", 0.0),
        "gpu": bool(use_gpu),
    }
    # Calcula o mapa de vasos usado na detecção dos óstios ou segmentação arterial.
    return get_vesselness(image, **vesselness_kwargs)
