"""Avaliação exploratória da localização da aorta com mapas de Frangi."""

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import isclose, isfinite
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from utils.segmentation.pipeline.detection import (
    locate_and_filter_aorta_circles,
    segment_aorta_with_diagnostics,
)
from utils.segmentation.pipeline.preprocessing import compute_vesselness
from utils.utils.metrics import dice_score


@dataclass(frozen=True)
class AortaVesselnessResult:
    """Reúne a máscara, as medidas e os tempos de uma variante por exame."""

    status: str
    mask: NDArray[np.uint8] | None = None
    circles_before_filter: int | None = None
    circles_after_filter: int | None = None
    coverage_percent: float | None = None
    volume_ml: float | None = None
    dice: float | None = None
    frangi_seconds: float | None = None
    localization_seconds: float | None = None
    segmentation_seconds: float | None = None
    vesselness: NDArray[Any] | None = None

    @property
    def total_seconds(self) -> float:
        """Soma os tempos de Frangi, localização e segmentação medidos."""
        return sum(
            duration
            for duration in (
                self.frangi_seconds,
                self.localization_seconds,
                self.segmentation_seconds,
            )
            if duration is not None
        )


def normalize_aorta_sigmas(sigmas: Sequence[float]) -> tuple[float, ...]:
    """Valida sigmas positivos em voxels e preserva sua ordem na lista."""
    scales = np.asarray(sigmas, dtype=float)
    if (
        scales.ndim != 1
        or not scales.size
        or not np.all(np.isfinite(scales) & (scales > 0))
    ):
        raise ValueError(
            "Os sigmas devem formar uma sequência de valores finitos e positivos."
        )
    return tuple(scales.tolist())


def evaluate_aorta_localization(
    image: NDArray[Any],
    downscale_factors: Sequence[int],
    scaled_spacing: Sequence[float],
    config: dict[str, Any],
    *,
    reference: NDArray[Any] | None,
    frangi_config: dict[str, Any] | None = None,
    use_gpu: bool = False,
    include_vesselness: bool = False,
) -> AortaVesselnessResult:
    """Localiza e segmenta a aorta na grade processada de um exame.

    Usa a tomografia pós-LCC para o level set. Quando recebe frangi_config,
    calcula um mapa nessa mesma grade para a Hough. scaled_spacing contém
    (dx, dy, dz) em mm. A referência, quando disponível, é uma máscara da
    aorta na mesma grade e deve conter voxels positivos. include_vesselness
    inclui o mapa Frangi válido no resultado para inspeção.
    """
    localization_image = image
    frangi_seconds = None
    if frangi_config is not None:
        started = perf_counter()
        localization_image = compute_vesselness(image, frangi_config, use_gpu=use_gpu)
        frangi_seconds = perf_counter() - started
        if (
            localization_image.shape != image.shape
            or not np.isfinite(localization_image).all()
        ):
            return AortaVesselnessResult(
                "Mapa Frangi inválido", frangi_seconds=frangi_seconds
            )
        if localization_image.min() == localization_image.max():
            return AortaVesselnessResult(
                "Mapa Frangi constante", frangi_seconds=frangi_seconds
            )

    started = perf_counter()
    tracking = locate_and_filter_aorta_circles(
        localization_image,
        downscale_factors,
        scaled_spacing,
        config["CIRCLE_DETECTION"],
    )
    localization_seconds = perf_counter() - started
    vesselness = (
        localization_image if include_vesselness and frangi_config is not None else None
    )
    del localization_image

    started = perf_counter()
    mask = segment_aorta_with_diagnostics(
        image, tracking.filtered_circles, config["LEVEL_SET"]
    ).mask
    segmentation_seconds = perf_counter() - started

    before_filter = len(tracking.original_circles)
    after_filter = len(tracking.filtered_circles)
    voxel_count = int(np.count_nonzero(mask))
    if not before_filter:
        status = "Nenhum círculo detectado"
    elif not after_filter:
        status = "Nenhum círculo após o filtro"
    elif not voxel_count:
        status = "Máscara vazia"
    else:
        status = "Máscara produzida"

    return AortaVesselnessResult(
        status=status,
        mask=mask,
        circles_before_filter=before_filter,
        circles_after_filter=after_filter,
        coverage_percent=100 * (after_filter / image.shape[2]),
        volume_ml=voxel_count * (float(np.prod(scaled_spacing)) / 1000),
        dice=dice_score(mask, reference) if reference is not None else None,
        frangi_seconds=frangi_seconds,
        localization_seconds=localization_seconds,
        segmentation_seconds=segmentation_seconds,
        vesselness=vesselness,
    )


def recommend_aorta_sigmas(
    results: Mapping[tuple[float, ...], AortaVesselnessResult],
    reference_sigmas: tuple[float, ...],
    max_dice_loss: float,
    *,
    require_faster: bool = False,
) -> tuple[float, ...] | None:
    """Seleciona a lista menor mais rápida dentro da perda máxima de Dice.

    A perda é expressa em Dice na escala de 0 a 1. Uma referência sem máscara
    ou Dice válido impede a seleção. Com require_faster, a lista precisa ter
    tempo estritamente menor que o da referência. Empates de tempo favorecem
    menos escalas e maior Dice, nessa ordem.
    """
    if not 0 <= max_dice_loss <= 1:
        raise ValueError("A perda máxima deve estar entre 0 e 1.")

    reference = results[reference_sigmas]
    if (
        reference.dice is None
        or not isfinite(reference.dice)
        or not reference.volume_ml
    ):
        return None
    minimum_dice = reference.dice - max_dice_loss
    eligible = [
        (result.total_seconds, len(sigmas), -result.dice, sigmas)
        for sigmas, result in results.items()
        if len(sigmas) < len(reference_sigmas)
        and result.volume_ml
        and result.dice is not None
        and isfinite(result.dice)
        and (not require_faster or result.total_seconds < reference.total_seconds)
        and (
            result.dice >= minimum_dice
            or isclose(result.dice, minimum_dice, rel_tol=0, abs_tol=1e-12)
        )
    ]
    return min(eligible)[-1] if eligible else None


def search_reduced_aorta_sigmas(
    reference_sigmas: tuple[float, ...],
    results: MutableMapping[tuple[float, ...], AortaVesselnessResult],
    evaluate: Callable[[tuple[float, ...]], AortaVesselnessResult],
    *,
    max_dice_loss: float,
) -> tuple[dict[tuple[float, ...], AortaVesselnessResult], tuple[float, ...] | None]:
    """Testa subconjuntos de sigmas em ordem crescente de quantidade de escalas.

    Reaproveita results do mesmo exame e grava nele cada avaliação concluída.
    Retorna a referência e os resultados da busca atual, incluindo a lista
    escolhida quando houver. Sem Dice de referência válido, avalia apenas os
    sigmas isolados para inspeção.
    """
    if not 0 <= max_dice_loss <= 1:
        raise ValueError("A perda máxima deve estar entre 0 e 1.")
    if reference_sigmas not in results:
        results[reference_sigmas] = evaluate(reference_sigmas)
    reference = results[reference_sigmas]
    tested = {reference_sigmas: reference}
    has_reference_dice = (
        reference.dice is not None
        and isfinite(reference.dice)
        and bool(reference.volume_ml)
    )

    for size in range(1, len(reference_sigmas)):
        stage = {reference_sigmas: reference}
        for sigmas in combinations(reference_sigmas, size):
            if sigmas not in results:
                results[sigmas] = evaluate(sigmas)
            stage[sigmas] = results[sigmas]
        tested.update(stage)
        if not has_reference_dice:
            break
        recommendation = recommend_aorta_sigmas(
            stage, reference_sigmas, max_dice_loss, require_faster=True
        )
        if recommendation is not None:
            return tested, recommendation

    return tested, None
