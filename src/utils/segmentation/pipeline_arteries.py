"""Etapa de segmentação das artérias coronárias a partir dos óstios.

O módulo recebe os óstios já detectados, calcula o vesselness arterial,
seleciona o método de segmentação das artérias e aplica o pós-processamento
morfológico final antes de calcular métricas.
"""

from typing import Any, Dict, Optional, Sequence

import numpy as np
from skimage.morphology import ball

from .artery_segmentation import normal_region_growing_from_ostia
from .pipeline_preprocessing import get_or_compute_vesselness
from ..processing.binary_operations import binary_closing, binary_dilation
from ..utils.metrics import dice_score


def _normalize_artery_segmentation_method(method: Any) -> str:
    """Normaliza o nome do método de segmentação arterial."""
    normalized = str(method or "region_growing").strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    aliases = {
        "rg": "region_growing",
        "region": "region_growing",
        "region_growing": "region_growing",
        "fuzzy": "fuzzy_connectedness",
        "fc": "fuzzy_connectedness",
        "fuzzy_connectedness": "fuzzy_connectedness",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(
            f"Método de segmentação arterial inválido: {method!r}. Use: {valid}."
        )
    return aliases[normalized]


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


def _segment_with_region_growing(
    vesselness_artery: Any,
    ostia_left: Optional[Sequence[int]],
    ostia_right: Optional[Sequence[int]],
    config: Dict[str, Any],
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Executa o region growing padrão e retorna máscara + metadados."""
    # Segmenta as artérias a partir dos óstios esquerdo e direito.
    raw_mask = normal_region_growing_from_ostia(
        vesselness_artery,
        ostia_left,
        ostia_right,
        config,
    )
    # Fecha pequenas falhas e dilata a máscara final conforme o pipeline.
    artery_mask = postprocess_artery_mask(raw_mask, config)
    return artery_mask, {"raw_artery_voxels": int(np.sum(raw_mask))}


def _segment_with_fuzzy_connectedness(
    lcc_image: Any,
    vesselness_artery: Any,
    ostia_left: Optional[Sequence[int]],
    ostia_right: Optional[Sequence[int]],
    config: Dict[str, Any],
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Executa fuzzy connectedness arterial e retorna máscara + metadados."""
    from .fuzzy_connectedness import segment_artery_fuzzy_connectedness

    fc_config = dict(config.get("FUZZY_CONNECTEDNESS", {}))
    max_candidate_voxels = fc_config.pop("max_candidate_voxels", 500_000)
    max_processed_voxels = fc_config.pop("max_processed_voxels", 500_000)

    # A LCC preserva intensidades acima do threshold mínimo e zera o restante
    # para o offset HU. Essa máscara restringe a FC à anatomia candidata.
    min_threshold = float(config.get("MIN_THRESHOLD", -300))
    lcc_mask = np.asarray(lcc_image) > min_threshold

    fc_result = segment_artery_fuzzy_connectedness(
        np.asarray(lcc_image),
        np.asarray(vesselness_artery),
        [ostia_left, ostia_right],
        lcc_mask,
        config,
        params=fc_config,
        max_candidate_voxels=max_candidate_voxels,
        max_processed_voxels=max_processed_voxels,
    )
    return fc_result["artery_mask"], fc_result.get("details", {})


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
    """Calcula vesselness arterial, segmenta artérias e avalia Dice."""
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

    method = _normalize_artery_segmentation_method(
        config.get("ARTERY_SEGMENTATION", {}).get("method", "region_growing")
    )
    if method == "fuzzy_connectedness":
        artery_mask, details = _segment_with_fuzzy_connectedness(
            lcc_image,
            vesselness_artery,
            ostia_left,
            ostia_right,
            config,
        )
    else:
        artery_mask, details = _segment_with_region_growing(
            vesselness_artery,
            ostia_left,
            ostia_right,
            config,
        )

    return {
        "artery_mask": artery_mask,
        "artery_voxels": int(np.sum(artery_mask)),
        "dice_artery": float(dice_score(artery_mask, label_artery)),
        "artery_segmentation_method": method,
        "fc_processed_voxels": details.get("processed_voxels"),
        "fc_effective_alpha": details.get("effective_alpha"),
        "fc_object_seed_count": details.get("object_seed_count"),
        "fc_candidate_voxels_final": details.get("candidate_voxels_final"),
    }
