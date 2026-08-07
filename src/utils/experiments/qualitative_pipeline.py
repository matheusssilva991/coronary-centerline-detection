"""Execução detalhada do pipeline para análises qualitativas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.segmentation.pipeline_arteries import segment_arteries_from_ostia
from utils.segmentation.pipeline_detection import (
    detect_and_evaluate_ostia,
    locate_aorta_circles,
    segment_aorta,
)
from utils.segmentation.pipeline_preprocessing import (
    compute_vesselness,
    load_and_preprocess_image,
)


def run_qualitative_pipeline_case(
    img_id: int | str,
    config: dict[str, Any],
    imagecas_path: str | Path,
) -> dict[str, Any]:
    """Executa um caso e preserva as máscaras usadas na visualização.

    Diferentemente da execução em lote, esta função mantém em memória a máscara
    da aorta, os óstios, a artéria segmentada e o ground truth. Ela é destinada
    somente a notebooks e diagnósticos qualitativos.
    """
    image_id = int(img_id)

    image_data = load_and_preprocess_image(
        str(image_id),
        str(imagecas_path),
        config,
        include_intermediates=True,
    )
    lcc_image = image_data["lcc_image"]
    label = image_data["label"]
    scaled_spacing = image_data["scaled_spacing"]
    downscale_factors = image_data["downscale_factors"]

    vesselness_ostia = compute_vesselness(
        lcc_image,
        vesselness_config=config["VESSELNESS_AORTA"],
        use_gpu=config.get("USE_GPU", False),
    )
    detected_circles = locate_aorta_circles(
        lcc_image,
        downscale_factors,
        scaled_spacing,
        config["CIRCLE_DETECTION"],
    )
    aorta_mask = segment_aorta(
        lcc_image,
        detected_circles,
        config["LEVEL_SET"],
        use_gpu=config.get("USE_GPU", False),
    )
    ostia_results = detect_and_evaluate_ostia(
        aorta_mask,
        vesselness_ostia,
        label,
        scaled_spacing,
        config,
        detected_circles=detected_circles,
    )
    artery_results = segment_arteries_from_ostia(
        lcc_image,
        ostia_results["label_artery"],
        ostia_results["ostia_left"],
        ostia_results["ostia_right"],
        config,
    )

    return {
        "img_id": image_id,
        "image": image_data.get("image"),
        "down_image": image_data.get("down_image"),
        "threshold_mask": image_data.get("threshold_mask"),
        "lcc_mask": image_data.get("lcc_mask"),
        "lcc_image": lcc_image,
        "label": label,
        "label_artery": ostia_results["label_artery"],
        "detected_circles": detected_circles,
        "aorta_mask": aorta_mask,
        "vesselness_ostia": vesselness_ostia,
        "ostia_left": ostia_results["ostia_left"],
        "ostia_right": ostia_results["ostia_right"],
        "ostia_results": ostia_results,
        "artery_mask": artery_results["artery_mask"],
        "raw_artery_mask": artery_results.get("raw_artery_mask"),
        "artery_results": artery_results,
        "scaled_spacing": scaled_spacing,
        "downscale_factors": downscale_factors,
    }


__all__ = ["run_qualitative_pipeline_case"]
