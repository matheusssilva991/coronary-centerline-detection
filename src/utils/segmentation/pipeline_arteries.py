"""Etapa de segmentação das artérias coronárias a partir dos óstios.

O módulo recebe os óstios já detectados, calcula o vesselness arterial,
seleciona o método de segmentação das artérias e aplica o pós-processamento
morfológico final antes de calcular métricas.
"""

from typing import Any, Dict, Optional, Sequence

import numpy as np
from skimage.morphology import ball

from ..processing.binary_operations import binary_closing, binary_dilation
from ..utils.metrics import dice_score
from .artery_segmentation import normal_region_growing_from_ostia
from .pipeline_preprocessing import compute_vesselness


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


def get_artery_postprocessing_stages(
    mask: Any,
    config: Dict[str, Any],
    *,
    closing_radius: Optional[int] = None,
    dilation_radius: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Retorna as máscaras bruta, fechada e dilatada do pós-processamento."""
    post_config = config["POSTPROCESSING"]
    close_radius = (
        post_config["closing_radius"] if closing_radius is None else int(closing_radius)
    )
    dilate_radius = (
        post_config["dilation_radius"]
        if dilation_radius is None
        else int(dilation_radius)
    )

    raw_mask = (np.asarray(mask) > 0).astype(np.uint8)
    closing_structure = np.asarray(ball(close_radius))
    dilation_structure = np.asarray(ball(dilate_radius))
    closed_mask = binary_closing(
        raw_mask > 0,
        structure=closing_structure,
        # Mantém a morfologia final na CPU para reduzir diferenças discretas
        # entre execuções CPU/GPU.
        gpu=False,
    )
    final_mask = binary_dilation(
        closed_mask,
        structure=dilation_structure,
        gpu=False,
    )
    return {
        "raw_mask": raw_mask,
        "closed_mask": np.asarray(closed_mask, dtype=np.uint8),
        "final_mask": np.asarray(final_mask, dtype=np.uint8),
    }


def postprocess_artery_mask(
    mask: Any,
    config: Dict[str, Any],
    *,
    closing_radius: Optional[int] = None,
    dilation_radius: Optional[int] = None,
) -> np.ndarray:
    """Aplica fechamento + dilatação usados nas máscaras arteriais."""
    stages = get_artery_postprocessing_stages(
        mask,
        config,
        closing_radius=closing_radius,
        dilation_radius=dilation_radius,
    )
    return stages["final_mask"]


def _segment_with_region_growing(
    vesselness_artery: Any,
    ostia_left: Optional[Sequence[int]],
    ostia_right: Optional[Sequence[int]],
    config: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
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
    return artery_mask, raw_mask, {"raw_artery_voxels": int(np.sum(raw_mask))}


def _segment_with_fuzzy_connectedness(
    lcc_image: Any,
    vesselness_artery: Any,
    ostia_left: Optional[Sequence[int]],
    ostia_right: Optional[Sequence[int]],
    config: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
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
    return (
        fc_result["artery_mask"],
        fc_result["raw_mask"],
        fc_result.get("details", {}),
    )


def segment_arteries_from_ostia(
    lcc_image: Any,
    label_artery: Any,
    ostia_left: Optional[Sequence[int]],
    ostia_right: Optional[Sequence[int]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Calcula vesselness arterial, segmenta artérias e avalia Dice."""
    # Calcula o mapa de vasos usado pela segmentação arterial.
    vesselness_artery = compute_vesselness(
        lcc_image,
        vesselness_config=config["VESSELNESS_ARTERY"],
        use_gpu=config.get("USE_GPU", False),
    )

    return segment_arteries_from_vesselness(
        lcc_image,
        label_artery,
        vesselness_artery,
        ostia_left,
        ostia_right,
        config,
    )


def segment_arteries_from_vesselness(
    lcc_image: Any,
    label_artery: Any,
    vesselness_artery: Any,
    ostia_left: Optional[Sequence[int]],
    ostia_right: Optional[Sequence[int]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Segmenta artérias reutilizando um mapa de vesselness já calculado.

    Esta entrada é útil em experimentos pareados que alteram apenas o Region
    Growing ou a morfologia. O fluxo principal continua usando
    :func:`segment_arteries_from_ostia`, que calcula o mapa e delega para esta
    função.
    """

    # Despacha RG ou FC mantendo o mesmo contrato de máscaras e métricas.
    method = _normalize_artery_segmentation_method(
        config.get("ARTERY_SEGMENTATION", {}).get("method", "region_growing")
    )
    if method == "fuzzy_connectedness":
        artery_mask, raw_artery_mask, details = _segment_with_fuzzy_connectedness(
            lcc_image,
            vesselness_artery,
            ostia_left,
            ostia_right,
            config,
        )
    else:
        artery_mask, raw_artery_mask, details = _segment_with_region_growing(
            vesselness_artery,
            ostia_left,
            ostia_right,
            config,
        )

    # Mede separadamente o método de crescimento e o efeito da morfologia final.
    dice_before = float(dice_score(raw_artery_mask, label_artery))
    dice_after = float(dice_score(artery_mask, label_artery))
    raw_artery_voxels = int(np.sum(raw_artery_mask))
    artery_voxels = int(np.sum(artery_mask))

    return {
        # As máscaras são removidas pela orquestração antes da persistência.
        "artery_mask": artery_mask,
        "raw_artery_mask": raw_artery_mask,
        "artery_voxels_before_morphology": raw_artery_voxels,
        "artery_voxels_after_morphology": artery_voxels,
        "artery_voxels": artery_voxels,
        "dice_artery_before_morphology": dice_before,
        "dice_artery_after_morphology": dice_after,
        "dice_artery_morphology_delta": dice_after - dice_before,
        # Alias mantido para compatibilidade com relatórios antigos.
        "dice_artery": dice_after,
        "artery_segmentation_method": method,
        "fc_processed_voxels": details.get("processed_voxels"),
        "fc_effective_alpha": details.get("effective_alpha"),
        "fc_object_seed_count": details.get("object_seed_count"),
        "fc_candidate_voxels_final": details.get("candidate_voxels_final"),
    }
