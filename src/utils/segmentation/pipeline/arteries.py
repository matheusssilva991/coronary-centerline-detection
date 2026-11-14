"""Etapa de segmentação das artérias coronárias a partir dos óstios.

O módulo recebe os óstios já detectados, calcula o vesselness arterial,
seleciona o método de segmentação das artérias e aplica o pós-processamento
morfológico final antes de calcular métricas.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
from skimage.morphology import ball

from utils.processing.binary_operations import binary_closing, binary_dilation
from utils.project.results.columns import ARTERY_BRANCH_COLUMNS
from utils.project.evaluation.artery_distances import evaluate_artery_distances
from utils.utils.metrics import dice_score
from utils.segmentation.artery_segmentation import normal_region_growing_from_ostia
from utils.segmentation.pipeline.preprocessing import compute_vesselness


@dataclass(frozen=True)
class ArterySegmentationResult:
    """Máscaras e diagnósticos produzidos pela segmentação arterial."""

    raw_mask: np.ndarray
    closed_mask: np.ndarray
    final_mask: np.ndarray
    method: str
    details: Dict[str, Any]


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


def evaluate_artery_postprocessing_stages(
    raw_mask: Any,
    closed_mask: Any,
    final_mask: Any,
    *,
    label_artery: Any = None,
    spacing: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Mede os estágios já calculados, sem reaplicar a morfologia.

    O Dice exige uma referência compatível; volumes em mL exigem o espaçamento
    efetivo da resolução de trabalho, em milímetros.
    """
    masks = (raw_mask, closed_mask, final_mask)
    counts = [int(np.sum(mask)) for mask in masks]
    stages = ("before_morphology", "after_closing", "after_morphology")
    metrics: Dict[str, Any] = {
        f"artery_voxels_{stage}": count
        for stage, count in zip(stages, counts, strict=True)
    }
    metrics["artery_voxels"] = counts[-1]
    if spacing is not None:
        spacing_array = np.asarray(spacing, dtype=float)
        if (
            spacing_array.shape != (3,)
            or not np.isfinite(spacing_array).all()
            or np.any(spacing_array <= 0)
        ):
            raise ValueError("O espaçamento deve conter três valores positivos em mm.")
        voxel_volume_mm3 = float(np.prod(spacing_array))
        metrics.update(
            {
                f"artery_volume_{stage}_ml": count * voxel_volume_mm3 / 1000.0
                for stage, count in zip(stages, counts, strict=True)
            }
        )
    if label_artery is not None:
        before, closed, after = (
            float(dice_score(mask, label_artery)) for mask in masks
        )
        metrics.update(
            {
                "dice_artery_before_morphology": before,
                "dice_artery_after_closing": closed,
                "dice_artery_after_morphology": after,
                "dice_artery_closing_delta": closed - before,
                "dice_artery_dilation_delta": after - closed,
                "dice_artery_morphology_delta": after - before,
                "dice_artery": after,
            }
        )
    return metrics


def _segment_with_region_growing(
    vesselness_artery: Any,
    ostia_left: Optional[Sequence[int]],
    ostia_right: Optional[Sequence[int]],
    config: Dict[str, Any],
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Executa o region growing padrão e retorna máscara + metadados."""
    # Segmenta as artérias a partir dos óstios esquerdo e direito.
    details: Dict[str, Any] = {}
    raw_mask = normal_region_growing_from_ostia(
        vesselness_artery,
        ostia_left,
        ostia_right,
        config,
        branch_diagnostics=details,
    )
    return raw_mask, {
        **details,
        "raw_artery_voxels": int(np.sum(raw_mask)),
    }


def _segment_with_fuzzy_connectedness(
    lcc_image: Any,
    vesselness_artery: Any,
    ostia_left: Optional[Sequence[int]],
    ostia_right: Optional[Sequence[int]],
    config: Dict[str, Any],
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Executa fuzzy connectedness arterial e retorna máscara + metadados."""
    from utils.segmentation.fuzzy.connectedness import (
        segment_artery_fuzzy_connectedness,
    )

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
        apply_postprocessing=False,
    )
    return fc_result["raw_mask"], fc_result.get("details", {})


def segment_artery_masks_from_vesselness(
    lcc_image: Any,
    vesselness_artery: Any,
    ostia_left: Optional[Sequence[int]],
    ostia_right: Optional[Sequence[int]],
    config: Dict[str, Any],
    *,
    method: Any = None,
) -> ArterySegmentationResult:
    """Segmenta artérias sem exigir uma referência coronariana."""
    configured_method = config.get("ARTERY_SEGMENTATION", {}).get(
        "method", "region_growing"
    )
    normalized_method = _normalize_artery_segmentation_method(
        configured_method if method is None else method
    )
    if normalized_method == "fuzzy_connectedness":
        raw_mask, details = _segment_with_fuzzy_connectedness(
            lcc_image,
            vesselness_artery,
            ostia_left,
            ostia_right,
            config,
        )
    else:
        raw_mask, details = _segment_with_region_growing(
            vesselness_artery,
            ostia_left,
            ostia_right,
            config,
        )

    stages = get_artery_postprocessing_stages(raw_mask, config)
    return ArterySegmentationResult(
        raw_mask=stages["raw_mask"],
        closed_mask=stages["closed_mask"],
        final_mask=stages["final_mask"],
        method=normalized_method,
        details=details,
    )


def segment_arteries_from_ostia(
    lcc_image: Any,
    label_artery: Any,
    ostia_left: Optional[Sequence[int]],
    ostia_right: Optional[Sequence[int]],
    config: Dict[str, Any],
    *,
    spacing: Optional[Sequence[float]] = None,
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
        spacing=spacing,
    )


def segment_arteries_from_vesselness(
    lcc_image: Any,
    label_artery: Any,
    vesselness_artery: Any,
    ostia_left: Optional[Sequence[int]],
    ostia_right: Optional[Sequence[int]],
    config: Dict[str, Any],
    *,
    spacing: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Segmenta artérias reutilizando um mapa de vesselness já calculado.

    Esta entrada é útil em experimentos pareados que alteram apenas o Region
    Growing ou a morfologia. O fluxo principal continua usando
    :func:`segment_arteries_from_ostia`, que calcula o mapa e delega para esta
    função.
    """

    segmentation = segment_artery_masks_from_vesselness(
        lcc_image,
        vesselness_artery,
        ostia_left,
        ostia_right,
        config,
    )
    artery_mask = segmentation.final_mask
    raw_artery_mask = segmentation.raw_mask
    details = segmentation.details

    metrics = evaluate_artery_postprocessing_stages(
        raw_artery_mask,
        segmentation.closed_mask,
        artery_mask,
        label_artery=label_artery,
    )

    return {
        # As máscaras são removidas pela orquestração antes da persistência.
        **{column: details.get(column) for column in ARTERY_BRANCH_COLUMNS},
        "artery_mask": artery_mask,
        "raw_artery_mask": raw_artery_mask,
        **metrics,
        **evaluate_artery_distances(artery_mask, label_artery, spacing, config),
        "artery_segmentation_method": segmentation.method,
        "fc_processed_voxels": details.get("processed_voxels"),
        "fc_effective_alpha": details.get("effective_alpha"),
        "fc_object_seed_count": details.get("object_seed_count"),
        "fc_candidate_voxels_final": details.get("candidate_voxels_final"),
    }
