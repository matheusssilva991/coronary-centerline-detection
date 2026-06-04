"""Etapa de segmentação das artérias coronárias a partir dos óstios."""

from typing import Any, Dict, Optional, Sequence

import numpy as np
from skimage.morphology import ball

from .artery_segmentation import region_growing_segmentation
from .pipeline_preprocessing import get_or_compute_vesselness
from ..processing.binary_operations import binary_closing, binary_dilation
from ..utils.metrics import dice_score


def segment_arteries_from_ostia(
    img_id: str,
    lcc_image: Any,
    label_artery: Any,
    ostia_left: Optional[Sequence[int]],
    ostia_right: Optional[Sequence[int]],
    config: Dict[str, Any],
    base_save_path: str,
) -> Dict[str, Any]:
    """Calcula vesselness arterial, executa region growing e avalia Dice."""
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
    )

    rg_config = config["REGION_GROWING"]
    region_growing_params = {
        "threshold": (vesselness_artery.max() - vesselness_artery.min())
        / rg_config["threshold_divisor"],
        "max_volume": rg_config["max_volume"],
        "min_vesselness": vesselness_artery.max()
        * rg_config["min_vesselness_fraction"],
        "relaxed_floor_factor": rg_config["relaxed_floor_factor"],
        "switch_at_voxels": rg_config["switch_at_voxels"],
        "comparison_window": rg_config["comparison_window"],
        "smooth_relaxation": rg_config["smooth_relaxation"],
        "verbose": False,
    }

    left_mask = (
        region_growing_segmentation(
            vesselness_artery, seed_point=ostia_left, **region_growing_params
        )
        if ostia_left is not None
        else np.zeros_like(vesselness_artery, dtype=np.uint8)
    )

    right_mask = (
        region_growing_segmentation(
            vesselness_artery, seed_point=ostia_right, **region_growing_params
        )
        if ostia_right is not None
        else np.zeros_like(vesselness_artery, dtype=np.uint8)
    )

    artery_mask = ((left_mask > 0) | (right_mask > 0)).astype(np.uint8)
    post_config = config["POSTPROCESSING"]
    closed_mask = binary_closing(
        artery_mask > 0,
        structure=ball(post_config["closing_radius"]),
        gpu=config.get("USE_GPU", False),
    )
    dilated_mask = binary_dilation(
        closed_mask,
        structure=ball(post_config["dilation_radius"]),
        gpu=config.get("USE_GPU", False),
    )
    artery_mask = dilated_mask

    return {
        "artery_mask": artery_mask,
        "artery_voxels": int(np.sum(artery_mask)),
        "dice_artery": float(dice_score(artery_mask, label_artery)),
    }
