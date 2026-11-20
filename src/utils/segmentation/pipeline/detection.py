"""Etapas de detecção/segmentação da aorta e avaliação dos óstios."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from utils.processing.binary_operations import keep_largest_component
from utils.segmentation.aorta.diagnostics import (
    summarize_aorta_circles,
    summarize_aorta_volume,
)
from utils.segmentation.aorta.localization import (
    detect_aorta_circles,
    filter_aorta_circle_trajectory,
)
from utils.segmentation.aorta.segmentation import (
    calculate_circle_mask_metrics,
    calculate_slice_area_jump_p95,
    classify_aorta_segmentation_feedback,
    initialize_level_set_mask,
    prepare_level_set_evolution,
    remove_leaks_morphology,
    restrict_mask_to_circle_trajectory,
)
from utils.segmentation.ostia_detection import check_ostium_intersection, find_ostia


@dataclass(frozen=True)
class AortaStageResult:
    """Resultado unificado da localização, segmentação e diagnóstico da aorta."""

    circle_tracking: "AortaCircleTrackingResult"
    aorta_mask: np.ndarray
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class AortaSegmentationResult:
    """Máscara final da aorta acompanhada dos diagnósticos do level set."""

    mask: np.ndarray
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class AortaCircleTrackingResult:
    """Trajetórias original e filtrada dos círculos da aorta."""

    original_circles: List[Dict[str, Any]]
    filtered_circles: List[Dict[str, Any]]
    filter_diagnostics: Dict[str, Any]


def locate_aorta_circles(
    lcc_image: Any,
    downscale_factors: Sequence[int],
    scaled_spacing: Sequence[float],
    circle_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Detecta círculos da aorta sempre na CPU via transformada de Hough.

    Args:
        lcc_image (Any): Volume 3D pré-processado (pós-LCC).
        downscale_factors (Sequence[int]): Fatores de redução da resolução (mantido
            por compatibilidade de interface).
        scaled_spacing (Sequence[float]): Espaçamento físico em ``mm`` na ordem
            ``(dx, dy, dz)`` após downscaling.
        circle_config (Dict[str, Any]): Bloco ``CIRCLE_DETECTION`` já escalado para a
            resolução de trabalho.

    Returns:
        List[Dict[str, Any]]: Lista ordenada de círculos detectados (``slice_index``,
            ``center_x``, ``center_y``, ``radius`` e ``accum``).
    """
    del downscale_factors  # Os raios já chegam escalados na configuração efetiva.
    # Define faixa de raios em pixels e espaçamento médio no plano axial.
    dx, dy, _ = scaled_spacing
    hough_radii = np.arange(
        circle_config["radii_start_px"],
        circle_config["radii_end_px"],
        circle_config.get("radius_step_px", 1),
    )
    pixel_spacing = (dx + dy) / 2.0

    # Localiza a aorta fatia a fatia por candidatos circulares.
    return detect_aorta_circles(
        lcc_image,
        hough_radii,
        pixel_spacing,
        tol_radius_mm=circle_config["tol_radius_mm"],
        tol_distance_mm=circle_config["tol_distance_mm"],
        quadrant_offset=tuple(circle_config["quadrant_offset"]),
        max_slice_miss_threshold=circle_config["max_slice_miss_threshold"],
        neighbor_distance_threshold=circle_config["neighbor_distance_threshold"],
        total_num_peaks_initial=circle_config["total_num_peaks_initial"],
        total_num_peaks=circle_config["total_num_peaks"],
        canny_sigma=circle_config["canny_sigma"],
        use_local_roi=circle_config.get("use_local_roi", True),
        local_roi_padding=circle_config.get("local_roi_padding", 20),
        interpolate_missed_circles=circle_config.get(
            "interpolate_missed_circles", True
        ),
    )


def filter_located_aorta_circles(
    detected_circles: Sequence[Dict[str, Any]],
    scaled_spacing: Sequence[float],
    image_slice_count: int,
    circle_config: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Aplica o filtro robusto opcional à trajetória localizada pela Hough.

    Args:
        detected_circles (Sequence[Dict[str, Any]]): Círculos detectados pela Hough ao
            longo das fatias axiais.
        scaled_spacing (Sequence[float]): Espaçamento físico em ``mm`` na ordem
            ``(dx, dy, dz)`` após downscaling.
        image_slice_count (int): Total de fatias axiais do volume (denominador da
            cobertura).
        circle_config (Dict[str, Any]): Bloco ``CIRCLE_DETECTION`` contendo
            ``trajectory_filter`` opcional.

    Returns:
        tuple[List[Dict[str, Any]], Dict[str, Any]]: Tupla ``(filtered_circles,
            filter_diagnostics)`` com círculos mantidos/extrapolados e metadados.
    """
    dx, dy, _ = scaled_spacing
    pixel_spacing = (float(dx) + float(dy)) / 2.0
    return filter_aorta_circle_trajectory(
        detected_circles,
        pixel_spacing,
        image_slice_count,
        circle_config.get("trajectory_filter", {}),
    )


def locate_and_filter_aorta_circles(
    lcc_image: Any,
    downscale_factors: Sequence[int],
    scaled_spacing: Sequence[float],
    circle_config: Dict[str, Any],
) -> AortaCircleTrackingResult:
    """Localiza os círculos da aorta por Hough e filtra a trajetória para segmentação.

    Args:
        lcc_image (Any): Volume 3D pré-processado (pós-LCC).
        downscale_factors (Sequence[int]): Fatores de redução aplicados no
            pré-processamento.
        scaled_spacing (Sequence[float]): Espaçamento físico em ``mm`` na ordem
            ``(dx, dy, dz)``.
        circle_config (Dict[str, Any]): Bloco ``CIRCLE_DETECTION`` já escalado para a
            resolução de trabalho.

    Returns:
        AortaCircleTrackingResult: Objeto com ``original_circles``,
            ``filtered_circles`` e ``filter_diagnostics``.
    """
    # Etapa 1: detecta círculos candidatos em cada fatia pela transformada de Hough.
    original_circles = locate_aorta_circles(
        lcc_image,
        downscale_factors,
        scaled_spacing,
        circle_config,
    )
    # Etapa 2: aplica filtro opcional de trajetória para remover caudas incompatíveis.
    filtered_circles, filter_diagnostics = filter_located_aorta_circles(
        original_circles,
        scaled_spacing,
        int(np.asarray(lcc_image).shape[2]),
        circle_config,
    )
    return AortaCircleTrackingResult(
        original_circles=original_circles,
        filtered_circles=filtered_circles,
        filter_diagnostics=filter_diagnostics,
    )


def _postprocess_aorta_mask(
    mask_refined: Any,
    detected_circles: Sequence[Dict[str, Any]],
    level_set_config: Dict[str, Any],
) -> np.ndarray:
    """Aplica abertura, envelope opcional e maior componente conectada.

    Args:
        mask_refined (Any): Máscara bruta 3D produzida pela evolução do level set.
        detected_circles (Sequence[Dict[str, Any]]): Círculos filtrados da aorta
            usados para restringir a trajetória.
        level_set_config (Dict[str, Any]): Bloco ``LEVEL_SET`` com raios e margens de
            pós-processamento.

    Returns:
        np.ndarray: Máscara binária ``uint8`` 3D pós-processada na CPU.
    """
    # Remove pequenos vazamentos por abertura morfológica antes do envelope.
    aorta_mask = remove_leaks_morphology(
        mask_refined,
        radius=level_set_config["leak_removal_radius"],
        use_gpu=False,
    )
    trajectory_radius_factor = level_set_config.get("trajectory_radius_factor")
    if trajectory_radius_factor is not None:
        # Limita vazamentos ao tubo acompanhado pelos círculos da aorta.
        aorta_mask = restrict_mask_to_circle_trajectory(
            aorta_mask,
            detected_circles,
            radius_factor=float(trajectory_radius_factor),
            axial_margin_slices=int(
                level_set_config.get("trajectory_axial_margin_slices", 0)
            ),
        )
    # Preserva apenas a maior componente conectada para eliminar ilhas isoladas.
    aorta_mask = keep_largest_component(aorta_mask, gpu=False)
    return np.asarray(aorta_mask, dtype=np.uint8)


def _fixed_level_set_result(
    lcc_image: Any,
    detected_circles: Sequence[Dict[str, Any]],
    level_set_config: Dict[str, Any],
) -> AortaSegmentationResult:
    """Executa o level set com o número configurado de iterações.

    Args:
        lcc_image (Any): Volume 3D pré-processado no formato ``(y, x, z)``.
        detected_circles (Sequence[Dict[str, Any]]): Trajetória circular usada para
            inicializar e opcionalmente restringir o contorno.
        level_set_config (Dict[str, Any]): Bloco ``LEVEL_SET`` com parâmetros do
            MorphGAC e morfologia.

    Returns:
        AortaSegmentationResult: Máscara final ``uint8`` acompanhada dos diagnósticos
            quantitativos do level set.
    """
    # Etapa 1: evolui o MorphGAC na ROI quando há círculos rastreados; caso contrário, retorna máscara vazia.
    num_iter = int(level_set_config["num_iter"])
    if detected_circles:
        context = prepare_level_set_evolution(
            lcc_image,
            detected_circles,
            radius_reduction_factor=level_set_config["radius_reduction_factor"],
            roi_margin=level_set_config.get("roi_margin", 10),
            use_roi=level_set_config.get("use_roi", True),
            alpha=level_set_config.get("alpha", 1000),
            sigma=level_set_config.get("sigma", 2),
            use_gpu=False,
        )
        initial_voxel_count = int(np.count_nonzero(context.current_mask))
        mask_refined = context.evolve(
            num_iter,
            balloon=level_set_config["balloon"],
            smoothing=level_set_config["smoothing"],
            threshold=level_set_config.get("threshold", "auto"),
        )
    else:
        initial_voxel_count = 0
        mask_refined = np.zeros_like(lcc_image, dtype=np.uint8)

    # Etapa 2: mede o volume bruto do contorno antes da remoção morfológica de vazamentos.
    raw_voxel_count = int(np.count_nonzero(mask_refined))
    aorta_mask = _postprocess_aorta_mask(
        mask_refined,
        detected_circles,
        level_set_config,
    )
    # Etapa 3: extrai métricas geométricas para diagnosticar sub/supersegmentação sem alterar a máscara.
    circle_metrics = calculate_circle_mask_metrics(aorta_mask, detected_circles)
    slice_area_jump_p95 = calculate_slice_area_jump_p95(aorta_mask)
    image_voxel_count = int(mask_refined.size)
    return AortaSegmentationResult(
        mask=aorta_mask,
        diagnostics={
            "aorta_level_set_initial_voxel_count": initial_voxel_count,
            "aorta_level_set_raw_voxel_count": raw_voxel_count,
            "aorta_level_set_initial_volume_fraction": (
                initial_voxel_count / image_voxel_count
            ),
            "aorta_level_set_raw_volume_fraction": (
                raw_voxel_count / image_voxel_count
            ),
            "aorta_level_set_iterations_used": num_iter,
            "aorta_level_set_circle_fill_q25": circle_metrics["circle_fill_q25"],
            "aorta_level_set_circle_area_ratio_p90": circle_metrics[
                "circle_area_ratio_p90"
            ],
            "aorta_slice_area_jump_p95": slice_area_jump_p95,
        },
    )


def segment_aorta_with_diagnostics(
    lcc_image: Any,
    detected_circles: List[Dict[str, Any]],
    level_set_config: Dict[str, Any],
    use_gpu: bool = False,
) -> AortaSegmentationResult:
    """Segmenta a aorta e retorna máscara pós-processada e métricas do level set.

    Args:
        lcc_image (Any): Volume 3D pré-processado no formato ``(y, x, z)``.
        detected_circles (List[Dict[str, Any]]): Trajetória circular da aorta usada na
            inicialização e envelope.
        level_set_config (Dict[str, Any]): Bloco ``LEVEL_SET`` da configuração efetiva.
        use_gpu (bool): Mantido por compatibilidade de assinatura; MorphGAC e
            morfologia rodam na CPU.

    Returns:
        AortaSegmentationResult: Estrutura com ``mask`` (``uint8``) e ``diagnostics``.
    """
    del use_gpu  # MorphGAC e pós-processamento permanecem na CPU.
    return _fixed_level_set_result(lcc_image, detected_circles, level_set_config)


def locate_and_segment_aorta(
    lcc_image: Any,
    downscale_factors: Sequence[int],
    scaled_spacing: Sequence[float],
    config: Dict[str, Any],
    *,
    circle_summary_source: str = "original",
    require_non_empty: bool = False,
    on_circles_detected: Callable[[List[Dict[str, Any]]], None] | None = None,
) -> AortaStageResult:
    """Localiza a aorta, filtra a trajetória, segmenta e gera diagnósticos.

    Args:
        lcc_image (Any): Volume 3D pré-processado no formato ``(y, x, z)``.
        downscale_factors (Sequence[int]): Fatores de redução aplicados ao volume.
        scaled_spacing (Sequence[float]): Espaçamento físico em ``mm`` na ordem
            ``(dx, dy, dz)`` após downscaling.
        config (Dict[str, Any]): Configuração efetiva contendo ``CIRCLE_DETECTION``,
            ``LEVEL_SET`` e ``USE_GPU``.
        circle_summary_source (str): Origem dos círculos resumidos nas métricas
            (``"original"`` ou ``"filtered"``).
        require_non_empty (bool): Se ``True``, lança erro caso falte círculo ou a
            máscara resulte vazia.
        on_circles_detected (Callable[[List[Dict[str, Any]]], None] | None): Callback
            opcional chamado com os círculos filtrados antes do level set.

    Returns:
        AortaStageResult: Estrutura com ``circle_tracking``, ``aorta_mask`` e
            ``diagnostics``.
    """
    image_array = np.asarray(lcc_image)
    # Etapa 1: detecta círculos por Hough e aplica filtro opcional de trajetória.
    circle_tracking = locate_and_filter_aorta_circles(
        lcc_image,
        downscale_factors,
        scaled_spacing,
        config["CIRCLE_DETECTION"],
    )
    if require_non_empty:
        if not circle_tracking.original_circles:
            raise RuntimeError("Nenhum círculo da aorta foi detectado.")
        if not circle_tracking.filtered_circles:
            raise RuntimeError("O filtro removeu todos os círculos da aorta.")

    # Etapa 2: resume os círculos (originais no ImageCAS; filtrados no batch externo) mantendo precedência histórica.
    if circle_summary_source == "original":
        summary_circles = circle_tracking.original_circles
    elif circle_summary_source == "filtered":
        summary_circles = circle_tracking.filtered_circles
    else:
        raise ValueError(f"circle_summary_source inválido: {circle_summary_source!r}")

    circle_summary = summarize_aorta_circles(
        summary_circles,
        int(image_array.shape[2]),
        scaled_spacing,
        config["CIRCLE_DETECTION"],
    )
    diagnostics: Dict[str, Any] = {}
    if circle_summary_source == "original":
        diagnostics.update(circle_summary)
        diagnostics.update(circle_tracking.filter_diagnostics)
    else:
        diagnostics.update(circle_tracking.filter_diagnostics)
        diagnostics.update(circle_summary)

    if on_circles_detected is not None:
        on_circles_detected(circle_tracking.filtered_circles)

    # Etapa 3: segmenta a aorta com level set a partir da trajetória filtrada e avalia qualidade da máscara.
    aorta_segmentation = segment_aorta_with_diagnostics(
        lcc_image,
        circle_tracking.filtered_circles,
        config["LEVEL_SET"],
        use_gpu=config.get("USE_GPU", False),
    )
    aorta_mask = aorta_segmentation.mask
    if require_non_empty and not np.any(aorta_mask):
        raise RuntimeError("A segmentação da aorta produziu uma máscara vazia.")

    # Consolida métricas volumétricas e classifica o feedback qualitativo da máscara.
    diagnostics.update(aorta_segmentation.diagnostics)
    diagnostics.update(summarize_aorta_volume(aorta_mask, int(image_array.size)))
    diagnostics["aorta_segmentation_feedback"] = classify_aorta_segmentation_feedback(
        diagnostics.get("aorta_level_set_circle_fill_q25"),
        diagnostics.get("aorta_level_set_circle_area_ratio_p90"),
        diagnostics.get("aorta_volume_fraction"),
        config.get("LEVEL_SET", {}).get("quality_feedback"),
    )
    return AortaStageResult(
        circle_tracking=circle_tracking,
        aorta_mask=aorta_mask,
        diagnostics=diagnostics,
    )


def segment_aorta_without_level_set(
    lcc_image: Any,
    detected_circles: List[Dict[str, Any]],
    level_set_config: Dict[str, Any],
    *,
    radius_factor: float | None = None,
) -> np.ndarray:
    """Usa discos dos círculos sem MorphGAC, preservando o pós-processamento da aorta.

    Args:
        lcc_image (Any): Volume 3D pré-processado (define o shape de saída).
        detected_circles (List[Dict[str, Any]]): Trajetória circular detectada da
            aorta.
        level_set_config (Dict[str, Any]): Bloco ``LEVEL_SET`` usado para parâmetros de
            ROI, abertura e envelope.
        radius_factor (float | None): Fator de raio dos discos em ``(0, 1]``; ``None``
            usa ``level_set_config["radius_reduction_factor"]``.

    Returns:
        np.ndarray: Máscara binária ``uint8`` 3D após o pós-processamento padrão da
            aorta.
    """
    # Valida o fator de raio e gera discos iniciais sem evolução MorphGAC.
    factor = (
        level_set_config["radius_reduction_factor"]
        if radius_factor is None
        else float(radius_factor)
    )
    if not np.isfinite(factor) or not 0 < factor <= 1:
        raise ValueError("O fator do raio dos círculos deve estar em (0, 1].")
    initial = initialize_level_set_mask(
        np.asarray(lcc_image).shape,
        detected_circles,
        radius_reduction_factor=factor,
        roi_margin=level_set_config.get("roi_margin", 10),
        use_roi=level_set_config.get("use_roi", True),
    )
    # Aplica o mesmo pós-processamento morfológico da segmentação padrão.
    return _postprocess_aorta_mask(initial, detected_circles, level_set_config)


def segment_aorta(
    lcc_image: Any,
    detected_circles: List[Dict[str, Any]],
    level_set_config: Dict[str, Any],
    use_gpu: bool = False,
) -> Any:
    """Segmenta a aorta preservando a interface simples que retorna apenas a máscara.

    Args:
        lcc_image (Any): Volume 3D pré-processado no formato ``(y, x, z)``.
        detected_circles (List[Dict[str, Any]]): Trajetória circular detectada para
            inicializar o level set.
        level_set_config (Dict[str, Any]): Bloco ``LEVEL_SET`` da configuração efetiva.
        use_gpu (bool): Mantido por compatibilidade de assinatura; MorphGAC roda na CPU.

    Returns:
        Any: Máscara binária ``uint8`` 3D final da aorta.
    """
    return segment_aorta_with_diagnostics(
        lcc_image,
        detected_circles,
        level_set_config,
        use_gpu=use_gpu,
    ).mask


def detect_and_evaluate_ostia(
    aorta_mask: Any,
    vesselness_ostios: Any,
    label: Any,
    scaled_spacing: Sequence[float],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Detecta os dois óstios coronarianos e avalia proximidade física contra o label.

    Args:
        aorta_mask (Any): Máscara binária 3D da aorta em ordem ``(y, x, z)``.
        vesselness_ostios (Any): Mapa 3D de vesselness calculado para óstios.
        label (Any): Volume 3D de rótulos de referência (a artéria coronária
            corresponde a ``label == 1``).
        scaled_spacing (Sequence[float]): Espaçamento físico em ``mm`` na ordem
            ``(dx, dy, dz)``, reordenado para ``(dy, dx, dz)`` internamente.
        config (Dict[str, Any]): Configuração efetiva contendo ``OSTIA_DETECTION`` e
            ``OSTIA_VALIDATION``.

    Returns:
        Dict[str, Any]: Dicionário com ``ostia_left``, ``ostia_right``,
            ``label_artery``, ``left_info``, ``right_info``, ``both_correct`` e
            ``both_tolerable``.
    """
    # Etapa 1: localiza o par de óstios sobre a superfície erodida da aorta.
    ostia_left, ostia_right = detect_ostia(
        aorta_mask,
        vesselness_ostios,
        scaled_spacing,
        config,
    )

    # Etapa 2: mede interseção direta e distância física em mm contra o rótulo arterial (label == 1).
    dx, dy, dz = scaled_spacing
    label_artery = (label == 1).astype(np.uint8)
    left_coords = tuple(int(value) for value in ostia_left)
    right_coords = (
        tuple(int(value) for value in ostia_right) if ostia_right is not None else None
    )
    left_info = check_ostium_intersection(
        left_coords,
        label_artery,
        spacing=(dy, dx, dz),
        ostium_name="Óstio esquerdo",
    )
    right_info = check_ostium_intersection(
        right_coords,
        label_artery,
        spacing=(dy, dx, dz),
        ostium_name="Óstio direito",
    )

    # Etapa 3: classifica desfecho em ambos corretos vs. ambos toleráveis dentro do limiar em mm.
    tolerable = config["OSTIA_VALIDATION"]["distance_threshold_mm"]
    both_correct = left_info["intersects"] and right_info["intersects"]
    both_tolerable_inclusive = (
        left_info["intersects"] or left_info["physical_dist"] <= tolerable
    ) and (right_info["intersects"] or right_info["physical_dist"] <= tolerable)

    return {
        "ostia_left": ostia_left,
        "ostia_right": ostia_right,
        "label_artery": label_artery,
        "left_info": left_info,
        "right_info": right_info,
        "both_correct": both_correct,
        "both_tolerable": both_tolerable_inclusive and (not both_correct),
    }


def detect_ostia(
    aorta_mask: Any,
    vesselness_ostia: Any,
    scaled_spacing: Sequence[float],
    config: Dict[str, Any],
) -> tuple[Any, Any]:
    """Detecta os dois óstios coronarianos sem depender de máscara de referência.

    Args:
        aorta_mask (Any): Máscara binária 3D da aorta em ordem ``(y, x, z)``.
        vesselness_ostia (Any): Mapa 3D de vesselness usado na seleção de candidatos
            (``VESSELNESS_AORTA``).
        scaled_spacing (Sequence[float]): Espaçamento físico em ``mm`` na ordem
            ``(dx, dy, dz)``, convertido para ``(dy, dx, dz)`` ao chamar ``find_ostia``.
        config (Dict[str, Any]): Configuração efetiva contendo ``OSTIA_DETECTION``.

    Returns:
        tuple[Any, Any]: Tupla ``(ostia_left, ostia_right)`` em coordenadas
            ``(y, x, z)`` (``ostia_right`` pode ser ``None``).
    """
    # Reordena o spacing para (dy, dx, dz) seguindo a convenção espacial de find_ostia.
    dx, dy, dz = scaled_spacing
    ostia_config = config["OSTIA_DETECTION"]
    return find_ostia(
        aorta_mask,
        vesselness_ostia,
        spacing=(dy, dx, dz),
        top_n=ostia_config["top_n"],
        max_z_diff_mm=ostia_config["max_z_diff_mm"],
        lower_fraction=ostia_config["lower_fraction"],
        min_center_distance_factor=ostia_config["min_center_distance_factor"],
        min_lateral_factor=ostia_config["min_lateral_factor"],
        erosion_radius=ostia_config["erosion_radius"],
        surface_padding_radius=ostia_config.get("surface_padding_radius", 0),
        pair_distance_mode=ostia_config.get("pair_distance_mode", "voxel_xyz"),
    )
