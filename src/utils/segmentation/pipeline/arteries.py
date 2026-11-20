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
from utils.project.evaluation.artery_distances import evaluate_artery_distances
from utils.project.results.columns import ARTERY_BRANCH_COLUMNS
from utils.segmentation.artery_segmentation import normal_region_growing_from_ostia
from utils.segmentation.pipeline.preprocessing import compute_vesselness
from utils.utils.metrics import dice_score
from utils.utils.roi import mask_voxel_bbox_slices


@dataclass(frozen=True)
class ArterySegmentationResult:
    """Máscaras e diagnósticos produzidos pela segmentação arterial."""

    raw_mask: np.ndarray
    closed_mask: np.ndarray
    final_mask: np.ndarray
    method: str
    details: Dict[str, Any]


def _normalize_artery_segmentation_method(method: Any) -> str:
    """Normaliza o nome do método de segmentação arterial.

    Args:
        method (Any): Nome ou alias do método (``rg``/``region_growing`` ou
            ``fc``/``fuzzy_connectedness``).

    Returns:
        str: Identificador canônico (``"region_growing"`` ou
            ``"fuzzy_connectedness"``).
    """
    # Normaliza separadores e resolve aliases conhecidos para o identificador canônico.
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
    """Retorna as máscaras bruta, fechada e dilatada do pós-processamento na CPU.

    Args:
        mask (Any): Máscara binária 3D bruta gerada pela segmentação arterial.
        config (Dict[str, Any]): Configuração efetiva contendo ``POSTPROCESSING``.
        closing_radius (int | None): Raio esférico em voxels do fechamento; ``None``
            usa ``POSTPROCESSING.closing_radius``.
        dilation_radius (int | None): Raio esférico em voxels da dilatação; ``None``
            usa ``POSTPROCESSING.dilation_radius``.

    Returns:
        Dict[str, np.ndarray]: Dicionário com ``raw_mask``, ``closed_mask`` e
            ``final_mask`` em ``uint8``.
    """
    # Resolve raios efetivos e recorta pela bounding box para acelerar a morfologia.
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
    pad = max(close_radius, 0) + max(dilate_radius, 0) + 1
    bbox = mask_voxel_bbox_slices(raw_mask, pad)
    if bbox is None:
        return {
            "raw_mask": raw_mask,
            "closed_mask": np.zeros_like(raw_mask),
            "final_mask": np.zeros_like(raw_mask),
        }

    # Aplica fechamento seguido de dilatação esférica e restaura o shape completo.
    closing_structure = np.asarray(ball(close_radius))
    dilation_structure = np.asarray(ball(dilate_radius))
    closed_sub = binary_closing(
        raw_mask[bbox] > 0,
        structure=closing_structure,
        # Mantém a morfologia final na CPU para reduzir diferenças discretas
        # entre execuções CPU/GPU.
        gpu=False,
    )
    final_sub = binary_dilation(
        closed_sub,
        structure=dilation_structure,
        gpu=False,
    )
    closed_mask = np.zeros_like(raw_mask)
    closed_mask[bbox] = np.asarray(closed_sub, dtype=np.uint8)
    final_mask = np.zeros_like(raw_mask)
    final_mask[bbox] = np.asarray(final_sub, dtype=np.uint8)
    return {
        "raw_mask": raw_mask,
        "closed_mask": closed_mask,
        "final_mask": final_mask,
    }


def postprocess_artery_mask(
    mask: Any,
    config: Dict[str, Any],
    *,
    closing_radius: Optional[int] = None,
    dilation_radius: Optional[int] = None,
) -> np.ndarray:
    """Aplica fechamento e dilatação esférica usados nas máscaras arteriais.

    Args:
        mask (Any): Máscara binária 3D bruta gerada pela segmentação arterial.
        config (Dict[str, Any]): Configuração efetiva contendo ``POSTPROCESSING``.
        closing_radius (int | None): Raio em voxels do fechamento; ``None`` usa
            ``POSTPROCESSING.closing_radius``.
        dilation_radius (int | None): Raio em voxels da dilatação; ``None`` usa
            ``POSTPROCESSING.dilation_radius``.

    Returns:
        np.ndarray: Máscara binária ``uint8`` 3D após fechamento e dilatação.
    """
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
    """Mede voxels, volumes e Dice dos estágios já calculados, sem reaplicar morfologia.

    Args:
        raw_mask (Any): Máscara arterial bruta antes da morfologia.
        closed_mask (Any): Máscara após o fechamento esférico.
        final_mask (Any): Máscara final após fechamento e dilatação.
        label_artery (Any): Máscara binária de referência opcional para cálculo de
            Dice e deltas.
        spacing (Sequence[float] | None): Espaçamento físico em ``mm`` na resolução de
            trabalho para converter voxels em ``mL``.

    Returns:
        Dict[str, Any]: Dicionário com contagens de voxels e, quando disponíveis,
            volumes em ``mL`` e Dice por estágio.
    """
    # Etapa 1: contabiliza voxels por estágio morfológico (bruto, fechado e dilatado).
    masks = (raw_mask, closed_mask, final_mask)
    counts = [int(np.sum(mask)) for mask in masks]
    stages = ("before_morphology", "after_closing", "after_morphology")
    metrics: Dict[str, Any] = {
        f"artery_voxels_{stage}": count
        for stage, count in zip(stages, counts, strict=True)
    }
    metrics["artery_voxels"] = counts[-1]
    # Etapa 2: converte contagens em mL apenas quando o espaçamento físico da resolução é informado.
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
    # Etapa 3: calcula o ganho/perda de Dice em cada operação morfológica quando há ground truth.
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
    """Executa o region growing padrão e retorna máscara e metadados.

    Args:
        vesselness_artery (Any): Mapa 3D de vesselness arterial pré-calculado.
        ostia_left (Sequence[int] | None): Coordenada ``(y, x, z)`` do óstio esquerdo
            ou ``None``.
        ostia_right (Sequence[int] | None): Coordenada ``(y, x, z)`` do óstio direito
            ou ``None``.
        config (Dict[str, Any]): Configuração efetiva contendo ``REGION_GROWING``.

    Returns:
        tuple[np.ndarray, Dict[str, Any]]: Tupla ``(raw_mask, metadados)`` com a
            máscara bruta e contagens diagnósticas por ramo.
    """
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
    """Executa fuzzy connectedness arterial e retorna máscara e metadados.

    Args:
        lcc_image (Any): Volume 3D pré-processado em HU (pós-LCC).
        vesselness_artery (Any): Mapa 3D de vesselness arterial pré-calculado.
        ostia_left (Sequence[int] | None): Coordenada ``(y, x, z)`` do óstio esquerdo
            ou ``None``.
        ostia_right (Sequence[int] | None): Coordenada ``(y, x, z)`` do óstio direito
            ou ``None``.
        config (Dict[str, Any]): Configuração efetiva contendo ``FUZZY_CONNECTEDNESS``
            e ``MIN_THRESHOLD``.

    Returns:
        tuple[np.ndarray, Dict[str, Any]]: Tupla ``(raw_mask, metadados)`` com a
            máscara binária pré-morfologia e estatísticas da propagação.
    """
    from utils.segmentation.fuzzy.connectedness import (
        segment_artery_fuzzy_connectedness,
    )

    # Extrai hiperparâmetros da FC preservando o dicionário original da configuração.
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
    """Segmenta artérias coronárias e pós-processa sem exigir referência anotada.

    Args:
        lcc_image (Any): Volume 3D pré-processado (usado pela variante fuzzy
            connectedness).
        vesselness_artery (Any): Mapa 3D de vesselness arterial pré-calculado.
        ostia_left (Sequence[int] | None): Coordenada ``(y, x, z)`` do óstio esquerdo
            ou ``None``.
        ostia_right (Sequence[int] | None): Coordenada ``(y, x, z)`` do óstio direito
            ou ``None``.
        config (Dict[str, Any]): Configuração efetiva contendo método, parâmetros de
            crescimento/FC e ``POSTPROCESSING``.
        method (Any): Sobrescrita opcional do método (``rg``/``region_growing`` ou
            ``fc``/``fuzzy_connectedness``).

    Returns:
        ArterySegmentationResult: Estrutura com ``raw_mask``, ``closed_mask``,
            ``final_mask``, ``method`` e ``details``.
    """
    # Etapa 1: resolve o método arterial (region_growing ou fuzzy_connectedness) e gera a máscara bruta.
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

    # Etapa 2: aplica fechamento e dilatação configurados para obter os três estágios da máscara.
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
    """Calcula vesselness arterial, segmenta artérias e avalia métricas.

    Args:
        lcc_image (Any): Volume 3D pré-processado (pós-LCC).
        label_artery (Any): Máscara binária 3D de referência arterial.
        ostia_left (Sequence[int] | None): Coordenada ``(y, x, z)`` do óstio esquerdo
            ou ``None``.
        ostia_right (Sequence[int] | None): Coordenada ``(y, x, z)`` do óstio direito
            ou ``None``.
        config (Dict[str, Any]): Configuração efetiva contendo ``VESSELNESS_ARTERY``,
            método arterial, morfologia e avaliação.
        spacing (Sequence[float] | None): Espaçamento físico em ``mm`` na ordem
            ``(dy, dx, dz)`` (ou ``None`` para omitir distâncias físicas).

    Returns:
        Dict[str, Any]: Dicionário com ``artery_mask``, ``raw_artery_mask``, métricas
            de Dice, distâncias e diagnósticos.
    """
    # Calcula o mapa de vasos usado pela segmentação arterial.
    vesselness_artery = compute_vesselness(
        lcc_image,
        vesselness_config=config["VESSELNESS_ARTERY"],
        use_gpu=config.get("USE_GPU", False),
    )

    # Delega segmentação e avaliação reutilizando o mapa de vesselness recém-calculado.
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

    Entrada útil em experimentos pareados que variam apenas RG/FC ou morfologia;
    o fluxo padrão entra por :func:`segment_arteries_from_ostia`.

    Args:
        lcc_image (Any): Volume 3D pré-processado (usado pela variante fuzzy
            connectedness).
        label_artery (Any): Máscara binária 3D de referência arterial.
        vesselness_artery (Any): Mapa 3D de vesselness arterial pré-calculado.
        ostia_left (Sequence[int] | None): Coordenada ``(y, x, z)`` do óstio esquerdo
            ou ``None``.
        ostia_right (Sequence[int] | None): Coordenada ``(y, x, z)`` do óstio direito
            ou ``None``.
        config (Dict[str, Any]): Configuração efetiva com parâmetros arteriais,
            pós-processamento e avaliação.
        spacing (Sequence[float] | None): Espaçamento físico em ``mm`` para cálculo
            opcional das distâncias arteriais.

    Returns:
        Dict[str, Any]: Dicionário com ``artery_mask``, ``raw_artery_mask``, métricas
            de Dice por estágio, distâncias físicas e metadados do método arterial.
    """

    # Etapa 1: segmenta artérias brutas e pós-processadas usando o método escolhido na configuração.
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

    # Etapa 2: mede Dice por estágio morfológico e distâncias físicas finais (mm) quando habilitadas.
    metrics = evaluate_artery_postprocessing_stages(
        raw_artery_mask,
        segmentation.closed_mask,
        artery_mask,
        label_artery=label_artery,
    )

    # Consolida diagnósticos por ramo, métricas morfológicas, distâncias e FC.
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
