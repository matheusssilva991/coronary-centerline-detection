"""Etapa de segmentação das artérias coronárias a partir dos óstios.

O módulo recebe os óstios já detectados, calcula o vesselness arterial,
segmenta as artérias por crescimento de região e aplica o pós-processamento
morfológico final antes de calcular métricas.
"""

from typing import Any, Dict, Optional, Sequence

import numpy as np
from skimage.morphology import ball

from .artery_segmentation import normal_region_growing_from_ostia
from .pipeline_preprocessing import get_or_compute_vesselness
from ..processing.binary_operations import binary_closing, binary_dilation
from ..utils.metrics import dice_score


def postprocess_artery_mask(
    mask: Any,
    config: Dict[str, Any],
    *,
    closing_radius: Optional[int] = None,
    dilation_radius: Optional[int] = None,
) -> np.ndarray:
    """Aplica fechamento + dilatação usados nas máscaras arteriais."""
    post_config = config["POSTPROCESSING"]
    close_radius = post_config["closing_radius"] if closing_radius is None else int(closing_radius)
    dilate_radius = post_config["dilation_radius"] if dilation_radius is None else int(dilation_radius)

    closed_mask = binary_closing(
        np.asarray(mask) > 0,
        structure=ball(close_radius),
        # Mantém a morfologia final na CPU para reduzir diferenças discretas
        # entre execuções CPU/GPU.
        gpu=False,
    )
    dilated_mask = binary_dilation(
        closed_mask,
        structure=ball(dilate_radius),
        gpu=False,
    )
    return dilated_mask.astype(np.uint8)


def segment_arteries_from_ostia(
    img_id: str,
    lcc_image: Any,
    label_artery: Any,
    ostia_left: Optional[Sequence[int]],
    ostia_right: Optional[Sequence[int]],
    config: Dict[str, Any],
    base_save_path: str,
    scaled_spacing: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Calcula vesselness arterial, executa region growing e avalia Dice."""
    vesselness_spacing = (
        (scaled_spacing[1], scaled_spacing[0], scaled_spacing[2])
        if scaled_spacing is not None
        else None
    )
    # Calcula o mapa de vasos usado pela segmentação arterial.
    vesselness_artery = get_or_compute_vesselness(
        img_id,
        lcc_image,
        cache_dir=(
            f"{base_save_path}/vesselness_artery_cache_"
            f"{'gpu' if config.get('USE_GPU', False) else 'cpu'}"
        ),
        vesselness_config=config["VESSELNESS_ARTERY"],
        load_cache=config["LOAD_CACHE"],
        save_cache=config["SAVE_CACHE"],
        use_gpu=config.get("USE_GPU", False),
        spacing=vesselness_spacing,
    )

    # Segmenta as artérias a partir dos óstios esquerdo e direito.
    artery_mask = normal_region_growing_from_ostia(
        vesselness_artery,
        ostia_left,
        ostia_right,
        config,
    )
    # Fecha pequenas falhas e dilata a máscara final conforme o pipeline.
    artery_mask = postprocess_artery_mask(artery_mask, config)

    return {
        "artery_mask": artery_mask,
        "artery_voxels": int(np.sum(artery_mask)),
        "dice_artery": float(dice_score(artery_mask, label_artery)),
    }
