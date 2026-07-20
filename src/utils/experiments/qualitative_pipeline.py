"""Execução detalhada do pipeline para análises qualitativas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.segmentation.pipeline_arteries import segment_arteries_from_ostia
from utils.segmentation.pipeline_detection import (
    detect_and_evaluate_ostia,
    get_or_detect_aorta_circles,
    get_or_segment_aorta,
)
from utils.segmentation.pipeline_preprocessing import (
    get_or_compute_vesselness,
    load_and_preprocess_image,
)


def run_qualitative_pipeline_case(
    img_id: int | str,
    config: dict[str, Any],
    imagecas_path: str | Path,
    cache_dir: str | Path,
    *,
    load_cache: bool = True,
    save_cache: bool = False,
) -> dict[str, Any]:
    """Executa um caso e preserva as máscaras usadas na visualização.

    Diferentemente da execução em lote, esta função mantém em memória a máscara
    da aorta, os óstios, a artéria segmentada e o ground truth. Ela é destinada
    somente a notebooks e diagnósticos qualitativos.
    """
    image_id = int(img_id)
    cache_path = Path(cache_dir)
    backend = "gpu" if config.get("USE_GPU", False) else "cpu"

    image_data = load_and_preprocess_image(image_id, str(imagecas_path), config)
    lcc_image = image_data["lcc_image"]
    label = image_data["label"]
    scaled_spacing = image_data["scaled_spacing"]
    downscale_factors = image_data["downscale_factors"]
    vesselness_spacing = (
        scaled_spacing[1],
        scaled_spacing[0],
        scaled_spacing[2],
    )

    vesselness_ostia = get_or_compute_vesselness(
        image_id,
        lcc_image,
        cache_dir=str(cache_path / f"vesselness_ostia_{backend}"),
        vesselness_config=config["VESSELNESS_AORTA"],
        load_cache=load_cache,
        save_cache=save_cache,
        use_gpu=config.get("USE_GPU", False),
        spacing=vesselness_spacing,
    )
    detected_circles = get_or_detect_aorta_circles(
        image_id,
        lcc_image,
        downscale_factors,
        scaled_spacing,
        config["CIRCLE_DETECTION"],
        str(cache_path),
        load_cache=load_cache,
        save_cache=save_cache,
    )
    aorta_mask = get_or_segment_aorta(
        image_id,
        lcc_image,
        detected_circles,
        config["LEVEL_SET"],
        str(cache_path),
        load_cache=load_cache,
        save_cache=save_cache,
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
        image_id,
        lcc_image,
        ostia_results["label_artery"],
        ostia_results["ostia_left"],
        ostia_results["ostia_right"],
        config,
        str(cache_path),
        scaled_spacing=scaled_spacing,
    )

    return {
        "img_id": image_id,
        "lcc_image": lcc_image,
        "label": label,
        "label_artery": ostia_results["label_artery"],
        "detected_circles": detected_circles,
        "aorta_mask": aorta_mask,
        "ostia_left": ostia_results["ostia_left"],
        "ostia_right": ostia_results["ostia_right"],
        "ostia_results": ostia_results,
        "artery_mask": artery_results["artery_mask"],
        "artery_results": artery_results,
        "scaled_spacing": scaled_spacing,
        "downscale_factors": downscale_factors,
    }


__all__ = ["run_qualitative_pipeline_case"]
