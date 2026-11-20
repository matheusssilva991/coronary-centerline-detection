"""Localização de aorta baseada em círculos para volumes 3D de CCTA.

Este módulo detecta e rastreia candidatos circulares da aorta entre fatias
usando Canny + transformada de Hough com restrições de continuidade geométrica.
"""

from typing import Any, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from skimage import feature
from skimage.transform import hough_circle, hough_circle_peaks

OUT_OF_TOLERANCE = "out_of_tolerance"


def _circle_geometry_departure(
    circle: dict[str, Any],
    reference: dict[str, float],
    pixel_spacing: float,
) -> tuple[float, float]:
    """Mede o afastamento de raio e centro em relação a uma referência.

    Args:
        circle (dict[str, Any]): Círculo avaliado contendo ``radius``, ``center_x`` e
            ``center_y``.
        reference (dict[str, float]): Geometria de referência com ``radius``,
            ``center_x`` e ``center_y`` em pixels.
        pixel_spacing (float): Espaçamento médio no plano XY, em ``mm/px``.

    Returns:
        tuple[float, float]: Diferença absoluta de raio e distância euclidiana entre
            centros, em ``mm``.
    """
    radius_mm = abs(float(circle["radius"]) - reference["radius"]) * pixel_spacing
    center_mm = float(
        np.hypot(
            float(circle["center_x"]) - reference["center_x"],
            float(circle["center_y"]) - reference["center_y"],
        )
        * pixel_spacing
    )
    return radius_mm, center_mm


def _circle_step_geometry(
    previous: dict[str, Any],
    current: dict[str, Any],
    pixel_spacing: float,
) -> tuple[float, float]:
    """Calcula mudanças físicas de raio e centro normalizadas por fatia.

    Args:
        previous (dict[str, Any]): Círculo da fatia anterior na trajetória.
        current (dict[str, Any]): Círculo da fatia atual na trajetória.
        pixel_spacing (float): Espaçamento médio no plano XY, em ``mm/px``.

    Returns:
        tuple[float, float]: Variação absoluta de raio e deslocamento de centro por
            fatia, em ``mm/fatia``.
    """
    # Normaliza o afastamento geométrico pelo salto axial entre as duas fatias.
    slice_distance = max(
        abs(int(current["slice_index"]) - int(previous["slice_index"])),
        1,
    )
    reference = {
        "radius": float(previous["radius"]),
        "center_x": float(previous["center_x"]),
        "center_y": float(previous["center_y"]),
    }
    radius_mm, center_mm = _circle_geometry_departure(
        current,
        reference,
        pixel_spacing,
    )
    return radius_mm / slice_distance, center_mm / slice_distance


def _median_circle_reference(circles: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Resume uma janela estável da trajetória por medianas geométricas.

    Args:
        circles (Sequence[dict[str, Any]]): Janela de círculos usada para estimar a
            posição e o raio estáveis.

    Returns:
        dict[str, float]: Dicionário com medianas de ``radius``, ``center_x`` e
            ``center_y`` em pixels.
    """
    return {
        "radius": float(np.median([float(circle["radius"]) for circle in circles])),
        "center_x": float(np.median([float(circle["center_x"]) for circle in circles])),
        "center_y": float(np.median([float(circle["center_y"]) for circle in circles])),
    }


def _find_incompatible_tail_start(
    circles: Sequence[dict[str, Any]],
    pixel_spacing: float,
    config: dict[str, Any],
) -> int | None:
    """Localiza uma cauda que se afasta persistentemente da trajetória anterior.

    Args:
        circles (Sequence[dict[str, Any]]): Trajetória ordenada de círculos da aorta ao
            longo do eixo axial.
        pixel_spacing (float): Espaçamento médio no plano XY, em ``mm/px``.
        config (dict[str, Any]): Parâmetros de janela, persistência, tolerâncias físicas
            e confiança mínima da Hough.

    Returns:
        int | None: Índice inicial da cauda incompatível em ``circles``, ou ``None`` se
            nenhum desvio persistente for encontrado.
    """
    # Configura janelas de referência/persistência e limites axiais de busca da cauda.
    reference_window = max(3, int(config.get("reference_window", 5)))
    persistence_window = max(3, int(config.get("persistence_window", 5)))
    persistence_required = min(
        persistence_window,
        max(2, int(config.get("persistence_required", 4))),
    )
    min_tail_circles = max(
        persistence_window,
        int(config.get("min_tail_circles", 8)),
    )
    min_remaining = max(reference_window, int(config.get("min_remaining_circles", 30)))
    search_start = max(
        min_remaining,
        int(
            round(len(circles) * float(config.get("tail_search_start_fraction", 0.35)))
        ),
    )
    search_stop = len(circles) - min_tail_circles + 1

    max_radius_step = float(config.get("max_radius_step_mm", 4.8))
    max_center_step = float(config.get("max_center_step_mm", 8.0))
    min_accumulator = float(config.get("min_hough_accumulator", 0.408))

    # Procura uma transição abrupta seguida de divergência persistente na janela.
    for index in range(search_start, search_stop):
        previous = circles[index - 1]
        current = circles[index]
        radius_step, center_step = _circle_step_geometry(
            previous,
            current,
            pixel_spacing,
        )
        transition_is_abrupt = (
            radius_step > max_radius_step or center_step > max_center_step
        )
        if not transition_is_abrupt:
            continue

        # Compara a janela posterior com a mediana anterior por geometria e acumulador.
        reference = _median_circle_reference(circles[index - reference_window : index])
        incompatible = 0
        geometry_incompatible = 0
        for circle in circles[index : index + persistence_window]:
            radius_departure, center_departure = _circle_geometry_departure(
                circle,
                reference,
                pixel_spacing,
            )
            geometry_signal = (
                radius_departure > max_radius_step or center_departure > max_center_step
            )
            confidence_signal = (
                circle.get("accum") is not None
                and float(circle["accum"]) < min_accumulator
            )
            geometry_incompatible += int(geometry_signal)
            incompatible += int(geometry_signal or confidence_signal)

        # Confiança baixa isolada não basta: a cauda também deve divergir
        # geometricamente da janela anterior.
        if incompatible >= persistence_required and geometry_incompatible >= max(
            2, persistence_required - 1
        ):
            return index
    return None


def extrapolate_stable_circle_tail(
    stable_circles: Sequence[dict[str, Any]],
    *,
    synthetic_slices: int,
    pixel_spacing: float,
    reference_window: int,
    max_radius_step_mm: float,
    max_center_step_mm: float,
) -> list[dict[str, Any]]:
    """Prolonga uma trajetória estável sem reutilizar a cauda rejeitada.

    Estima a tendência por fatia pela mediana das diferenças recentes e limita o
    deslocamento acumulado às tolerâncias físicas informadas.

    Args:
        stable_circles (Sequence[dict[str, Any]]): Sequência ordenada de círculos
            estáveis antes do corte.
        synthetic_slices (int): Quantidade máxima de fatias sintéticas geradas abaixo
            do último círculo.
        pixel_spacing (float): Espaçamento médio no plano XY em ``mm/px``.
        reference_window (int): Número de círculos finais usados para estimar a
            inclinação mediana.
        max_radius_step_mm (float): Limite físico por fatia para variação do raio, em
            ``mm``.
        max_center_step_mm (float): Limite físico por fatia para deslocamento do centro
            no plano XY, em ``mm``.

    Returns:
        list[dict[str, Any]]: Lista de círculos sintéticos com ``interpolated=True`` e
            metadados do filtro.
    """
    if synthetic_slices <= 0 or not stable_circles:
        return []

    # Seleciona a janela final estável e define os índices das fatias sintéticas.
    reference = list(stable_circles[-max(2, reference_window) :])
    last = reference[-1]
    last_slice = int(last["slice_index"])
    target_slices = list(
        range(last_slice - 1, max(-1, last_slice - synthetic_slices - 1), -1)
    )
    if not target_slices:
        return []

    # Estima a tendência mediana por fatia para centro, raio e acumulador.
    slopes: dict[str, list[float]] = {
        "center_x": [],
        "center_y": [],
        "radius": [],
    }
    for previous, current in zip(reference, reference[1:]):
        delta_slice = int(current["slice_index"]) - int(previous["slice_index"])
        if delta_slice == 0:
            continue
        for field in slopes:
            slopes[field].append(
                (float(current[field]) - float(previous[field])) / delta_slice
            )
    median_slopes = {
        field: float(np.median(values)) if values else 0.0
        for field, values in slopes.items()
    }
    median_accumulator = float(
        np.median(
            [
                float(circle["accum"])
                for circle in reference
                if circle.get("accum") is not None
            ]
            or [0.0]
        )
    )
    radius_step_limit_px = max_radius_step_mm / pixel_spacing
    center_step_limit_px = max_center_step_mm / pixel_spacing
    synthetic: list[dict[str, Any]] = []

    # Projeta cada fatia sintética mantendo o desvio acumulado dentro do limite físico.
    for target_slice in target_slices:
        delta_slice = target_slice - last_slice
        center_x = float(last["center_x"]) + median_slopes["center_x"] * delta_slice
        center_y = float(last["center_y"]) + median_slopes["center_y"] * delta_slice
        radius = float(last["radius"]) + median_slopes["radius"] * delta_slice

        # Limita o afastamento acumulado para impedir que a extrapolação
        # replique uma mudança geométrica tão forte quanto a cauda removida.
        max_steps = abs(delta_slice)
        center_dx = center_x - float(last["center_x"])
        center_dy = center_y - float(last["center_y"])
        center_distance = float(np.hypot(center_dx, center_dy))
        max_center_distance = center_step_limit_px * max_steps
        if center_distance > max_center_distance > 0:
            scale = max_center_distance / center_distance
            center_x = float(last["center_x"]) + center_dx * scale
            center_y = float(last["center_y"]) + center_dy * scale
        radius_delta = np.clip(
            radius - float(last["radius"]),
            -radius_step_limit_px * max_steps,
            radius_step_limit_px * max_steps,
        )

        synthetic.append(
            {
                "slice_index": target_slice,
                "center_x": center_x,
                "center_y": center_y,
                "radius": max(1.0, float(last["radius"]) + float(radius_delta)),
                "accum": median_accumulator,
                "interpolated": True,
                "trajectory_filtered": True,
                "trajectory_filter_action": "extrapolated_stable_tail",
            }
        )
    return synthetic


def filter_aorta_circle_trajectory(
    detected_circles: Sequence[dict[str, Any]],
    pixel_spacing: float,
    image_slice_count: int,
    filter_config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filtra caudas incompatíveis da trajetória da aorta.

    O detector percorre o volume das fatias finais para as iniciais; uma cauda
    removida corresponde ao trecho final após perda de consistência da Hough.

    Args:
        detected_circles (Sequence[dict[str, Any]]): Sequência de círculos detectados
            pela Hough ao longo das fatias.
        pixel_spacing (float): Espaçamento médio no plano XY em ``mm/px``.
        image_slice_count (int): Número total de fatias axiais do volume (usado na
            cobertura).
        filter_config (dict[str, Any] | None): Subdicionário do filtro de trajetória
            (``method``, tolerâncias e extrapolação); ``None`` mantém inalterado.

    Returns:
        tuple[list[dict[str, Any]], dict[str, Any]]: Tupla
            ``(filtered_circles, diagnostics)`` com a trajetória filtrada e diagnósticos
            do corte.
    """
    # Ordena a trajetória de cima para baixo e inicializa os diagnósticos padrão.
    config = filter_config or {}
    method = str(config.get("method", "none"))
    original = sorted(
        (dict(circle) for circle in detected_circles),
        key=lambda circle: int(circle["slice_index"]),
        reverse=True,
    )
    diagnostics = {
        "aorta_circle_filter_method": method,
        "aorta_circle_filter_applied": False,
        "aorta_circle_original_count": len(original),
        "aorta_circle_used_count": len(original),
        "aorta_circle_filter_synthetic_tail_count": 0,
        "aorta_circle_filter_trimmed_tail_count": 0,
        "aorta_circle_filter_trim_start_slice": None,
        "aorta_circle_filter_detected_tail_start_slice": None,
        "aorta_circle_filter_original_coverage": (
            len(original) / image_slice_count if image_slice_count else None
        ),
        "aorta_circle_filter_used_coverage": (
            len(original) / image_slice_count if image_slice_count else None
        ),
        "aorta_circle_filter_reason": "disabled" if method == "none" else "unchanged",
    }
    # Valida método e parâmetros físicos antes de avaliar cortes na trajetória.
    if method == "none" or len(original) < 3:
        return original, diagnostics
    if method != "robust":
        raise ValueError(f"Método de filtro de círculos desconhecido: {method}")
    if pixel_spacing <= 0:
        raise ValueError("pixel_spacing deve ser maior que zero")

    # Elimina uma cauda persistentemente desviada e preserva o trecho estável.
    original_coverage = len(original) / image_slice_count if image_slice_count else 0.0
    min_tail_coverage = float(config.get("min_tail_coverage", 0.8))
    max_tail_trim_fraction = float(config.get("max_tail_trim_fraction", 1.0))
    if not 0.0 < max_tail_trim_fraction <= 1.0:
        raise ValueError("max_tail_trim_fraction deve estar no intervalo (0, 1]")
    detected_tail_start = None
    if original_coverage >= min_tail_coverage:
        detected_tail_start = _find_incompatible_tail_start(
            original,
            pixel_spacing,
            config,
        )
    tail_start = detected_tail_start
    trim_rejected = False
    synthetic_tail: list[dict[str, Any]] = []
    if tail_start is None:
        trimmed = original
        trimmed_count = 0
        trim_start_slice = None
    else:
        candidate_trimmed_count = len(original) - tail_start
        candidate_trim_fraction = candidate_trimmed_count / len(original)
        if candidate_trim_fraction > max_tail_trim_fraction:
            # Um corte axial muito longo pode remover a região onde os óstios
            # serão procurados. Nesse caso, conserva a trajetória original.
            trimmed = original
            trimmed_count = 0
            trim_start_slice = None
            tail_start = None
            trim_rejected = True
        # Aplica o corte validado e extrapola a extremidade estável quando configurado.
        if not trim_rejected:
            if tail_start is None:
                raise RuntimeError("O início da cauda não foi definido para o corte.")
            trimmed = original[:tail_start]
            trimmed_count = candidate_trimmed_count
            trim_start_slice = int(original[tail_start]["slice_index"])
            trim_rejected = False
            synthetic_tail = extrapolate_stable_circle_tail(
                trimmed,
                synthetic_slices=max(0, int(config.get("synthetic_tail_slices", 0))),
                pixel_spacing=pixel_spacing,
                reference_window=max(2, int(config.get("reference_window", 5))),
                max_radius_step_mm=float(config.get("max_radius_step_mm", 4.8)),
                max_center_step_mm=float(config.get("max_center_step_mm", 8.0)),
            )

    # Monta a trajetória resultante e atualiza o dicionário de diagnóstico do filtro.
    filtered = [dict(circle) for circle in trimmed]
    filtered.extend(synthetic_tail)
    applied = bool(trimmed_count)
    reasons = []
    if trimmed_count:
        reasons.append("persistent_tail_trimmed")
    if synthetic_tail:
        reasons.append("stable_tail_extrapolated")

    diagnostics.update(
        {
            "aorta_circle_filter_applied": applied,
            "aorta_circle_used_count": len(filtered),
            "aorta_circle_filter_synthetic_tail_count": len(synthetic_tail),
            "aorta_circle_filter_trimmed_tail_count": trimmed_count,
            "aorta_circle_filter_trim_start_slice": trim_start_slice,
            "aorta_circle_filter_detected_tail_start_slice": (
                int(original[detected_tail_start]["slice_index"])
                if detected_tail_start is not None
                else None
            ),
            "aorta_circle_filter_used_coverage": (
                len(filtered) / image_slice_count if image_slice_count else None
            ),
            "aorta_circle_filter_reason": (
                "+".join(reasons)
                if reasons
                else "tail_trim_fraction_exceeded"
                if trim_rejected
                else "coverage_below_tail_threshold"
                if original_coverage < min_tail_coverage
                else "unchanged"
            ),
        }
    )
    return filtered, diagnostics


def _interpolate_missing_circles(
    previous_circle: dict,
    next_circle: dict,
    missing_slice_indices: Sequence[int],
) -> list[dict]:
    """Interpola círculos para fatias puladas entre duas detecções válidas.

    Args:
        previous_circle (dict[str, Any]): Último círculo válido antes da lacuna axial.
        next_circle (dict[str, Any]): Primeiro círculo válido após a lacuna axial.
        missing_slice_indices (Sequence[int]): Índices das fatias intermediárias sem
            detecção.

    Returns:
        list[dict[str, Any]]: Lista de círculos interpolados linearmente com
            ``interpolated=True``.
    """
    # Evita divisão por zero caso as duas referências apontem para a mesma fatia.
    previous_slice = int(previous_circle["slice_index"])
    next_slice = int(next_circle["slice_index"])
    denominator = next_slice - previous_slice
    if denominator == 0:
        return []

    # Distribui centro, raio e acumulador linearmente entre duas detecções válidas.
    interpolated = []
    for slice_index in missing_slice_indices:
        fraction = (int(slice_index) - previous_slice) / denominator
        circle = {
            "slice_index": int(slice_index),
            "center_x": float(
                previous_circle["center_x"]
                + fraction * (next_circle["center_x"] - previous_circle["center_x"])
            ),
            "center_y": float(
                previous_circle["center_y"]
                + fraction * (next_circle["center_y"] - previous_circle["center_y"])
            ),
            "radius": float(
                previous_circle["radius"]
                + fraction * (next_circle["radius"] - previous_circle["radius"])
            ),
            "accum": float(
                previous_circle.get("accum", 0.0)
                + fraction
                * (next_circle.get("accum", 0.0) - previous_circle.get("accum", 0.0))
            ),
            "interpolated": True,
        }
        interpolated.append(circle)
    return interpolated


def _calculate_distances_vectorized(
    cx: ArrayLike, cy: ArrayLike, ref_x: float, ref_y: float
) -> NDArray[Any]:
    """Calcula distâncias euclidianas de forma vetorizada usando NumPy broadcasting.

    Args:
        cx (ArrayLike): Coordenadas ``x`` dos centros candidatos (em pixels).
        cy (ArrayLike): Coordenadas ``y`` dos centros candidatos (em pixels).
        ref_x (float): Coordenada ``x`` de referência (em pixels).
        ref_y (float): Coordenada ``y`` de referência (em pixels).

    Returns:
        NDArray[Any]: Array de distâncias euclidianas (em pixels) até a referência.
    """
    cx_arr = np.asarray(cx)
    cy_arr = np.asarray(cy)
    return np.sqrt((cx_arr - ref_x) ** 2 + (cy_arr - ref_y) ** 2)


def _detect_circles_in_slice(
    img_slice: NDArray[Any],
    hough_radii: Sequence[float],
    total_num_peaks: int,
    canny_sigma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detecta círculos em uma fatia usando Canny e Hough na CPU.

    Args:
        img_slice (NDArray[Any]): Fatia axial 2D pré-processada no formato ``(y, x)``.
        hough_radii (Sequence[float]): Raios candidatos (em pixels) testados pela
            transformada de Hough.
        total_num_peaks (int): Número máximo de picos retornados na fatia.
        canny_sigma (float): Desvio-padrão gaussiano (em pixels) do detector de bordas
            Canny.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tupla
            ``(accums, cx, cy, radii)`` com os melhores picos da Hough na fatia.
    """
    edges = feature.canny(img_slice.astype(float), sigma=canny_sigma)

    # A Hough recebe o mapa binário de bordas e devolve os melhores círculos.
    hough_res = hough_circle(edges, hough_radii)
    return hough_circle_peaks(hough_res, hough_radii, total_num_peaks=total_num_peaks)


def _select_initial_circle_candidate(
    cx: ArrayLike,
    cy: ArrayLike,
    img_shape: Sequence[int],
    quadrant_offset: Sequence[int],
) -> int | None:
    """Seleciona o primeiro pico da Hough no quadrante anatômico esperado.

    Args:
        cx (ArrayLike): Coordenadas ``x`` dos picos da Hough na fatia inicial.
        cy (ArrayLike): Coordenadas ``y`` dos picos da Hough na fatia inicial.
        img_shape (Sequence[int]): Dimensões ``(altura, largura)`` da fatia axial.
        quadrant_offset (Sequence[int]): Deslocamento ``(dx, dy)`` em pixels que ajusta
            o quadrante anatômico da busca inicial.

    Returns:
        int | None: Índice do primeiro pico dentro do quadrante esperado ou ``None`` se
            nenhum candidato atender ao critério.
    """
    height, width = img_shape
    center_x = (width // 2) - quadrant_offset[0]
    center_y = (height // 2) + quadrant_offset[1]

    # A aorta ascendente esperada fica no quadrante anatômico configurado.
    cx_arr = np.asarray(cx, dtype=float)
    cy_arr = np.asarray(cy, dtype=float)
    mask = (cx_arr > center_x) & (cy_arr < center_y)
    candidate_indices = np.where(mask)[0]
    if len(candidate_indices) == 0:
        return None
    return int(candidate_indices[0])


def _select_circle_candidate(
    cx: ArrayLike,
    cy: ArrayLike,
    radii: ArrayLike,
    ref_x: float,
    ref_y: float,
    ref_radius: float,
    radius_tolerance: float,
    distance_tolerance: float,
) -> Tuple[int, float]:
    """Seleciona o candidato válido mais próximo do círculo anterior.

    Args:
        cx (ArrayLike): Coordenadas ``x`` dos círculos candidatos na fatia (em pixels).
        cy (ArrayLike): Coordenadas ``y`` dos círculos candidatos na fatia (em pixels).
        radii (ArrayLike): Raios dos círculos candidatos (em pixels).
        ref_x (float): Coordenada ``x`` do círculo de referência (em pixels).
        ref_y (float): Coordenada ``y`` do círculo de referência (em pixels).
        ref_radius (float): Raio do círculo de referência (em pixels).
        radius_tolerance (float): Variação máxima permitida para o raio (em pixels).
        distance_tolerance (float): Deslocamento máximo permitido para o centro (em
            pixels).

    Returns:
        Tuple[int, float]: Tupla ``(best_idx, distance)`` com o índice escolhido e a
            distância euclidiana em pixels até a referência.
    """
    distances = _calculate_distances_vectorized(cx, cy, ref_x, ref_y)
    radii_arr = np.asarray(radii, dtype=float)
    radius_diff = np.abs(radii_arr - ref_radius)
    tolerance_mask = (distances <= distance_tolerance) & (
        radius_diff <= radius_tolerance
    )

    # Filtra por continuidade geométrica antes de escolher o centro mais próximo.
    candidate_indices = np.where(tolerance_mask)[0]
    if len(candidate_indices) == 0:
        candidate_indices = np.arange(len(radii_arr))

    best_idx = int(candidate_indices[np.argmin(distances[candidate_indices])])
    return best_idx, float(distances[best_idx])


def _is_circle_within_tolerance(
    circle_radius: float,
    circle_distance: float,
    ref_radius: float,
    radius_tolerance: float,
    distance_tolerance: float,
) -> bool:
    """Valida se raio e distância estão dentro das tolerâncias definidas.

    Args:
        circle_radius (float): Raio do círculo candidato (em pixels).
        circle_distance (float): Distância euclidiana até o centro de referência (em
            pixels).
        ref_radius (float): Raio do círculo de referência (em pixels).
        radius_tolerance (float): Variação máxima aceita para o raio (em pixels).
        distance_tolerance (float): Deslocamento máximo aceito para o centro (em
            pixels).

    Returns:
        bool: ``True`` se distância e diferença de raio estiverem dentro dos limites.
    """
    radius_diff = abs(circle_radius - ref_radius)
    return circle_distance <= distance_tolerance and radius_diff <= radius_tolerance


def _compute_local_roi_bounds(
    img_shape: Sequence[int],
    ref_x: float,
    ref_y: float,
    ref_radius: float,
    distance_tolerance: float,
    radius_tolerance: float,
    local_roi_padding: int,
) -> Tuple[int, int, int, int]:
    """Calcula os limites de ROI local para busca de círculos na fatia.

    Args:
        img_shape (Sequence[int]): Dimensões ``(altura, largura)`` da fatia axial.
        ref_x (float): Coordenada ``x`` do círculo de referência (em pixels).
        ref_y (float): Coordenada ``y`` do círculo de referência (em pixels).
        ref_radius (float): Raio do círculo de referência (em pixels).
        distance_tolerance (float): Tolerância de deslocamento do centro (em pixels).
        radius_tolerance (float): Tolerância de variação do raio (em pixels).
        local_roi_padding (int): Margem extra (em pixels) adicionada ao raio de busca.

    Returns:
        Tuple[int, int, int, int]: Limites recortados ``(x_min, x_max, y_min, y_max)``
            dentro da fatia.
    """
    height, width = img_shape

    # A ROI cobre o círculo anterior, a tolerância de movimento e uma margem extra.
    search_radius = (
        ref_radius + distance_tolerance + radius_tolerance + local_roi_padding
    )
    half_size = int(np.ceil(max(8.0, search_radius)))

    cx = int(round(ref_x))
    cy = int(round(ref_y))

    # Garante que os limites da ROI permaneçam dentro das dimensões da fatia.
    x_min = max(0, cx - half_size)
    x_max = min(width, cx + half_size)
    y_min = max(0, cy - half_size)
    y_max = min(height, cy + half_size)

    return x_min, x_max, y_min, y_max


def _process_slice(
    img_slice: NDArray[Any],
    hough_radii: Sequence[float],
    reference_circle: dict,
    radius_tolerance: float,
    distance_tolerance: float,
    neighbor_distance_threshold: float,
    total_num_peaks: int,
    canny_sigma: float,
    use_local_roi: bool = True,
    local_roi_padding: int = 20,
    verbose: bool = True,
) -> dict[str, Any] | str | None:
    """Processa uma fatia e retorna o melhor círculo rastreado.

    Args:
        img_slice (NDArray[Any]): Fatia axial 2D pré-processada no formato ``(y, x)``.
        hough_radii (Sequence[float]): Raios candidatos (em pixels) testados pela
            transformada de Hough.
        reference_circle (dict[str, Any]): Círculo da última fatia válida com
            ``center_x``, ``center_y`` e ``radius``.
        radius_tolerance (float): Variação máxima permitida para o raio (em pixels).
        distance_tolerance (float): Deslocamento máximo permitido para o centro (em
            pixels).
        neighbor_distance_threshold (float): Distância máxima (em pixels) para média
            local de candidatos vizinhos.
        total_num_peaks (int): Número máximo de picos avaliados pela Hough na fatia.
        canny_sigma (float): Desvio-padrão gaussiano (em pixels) do detector de bordas
            Canny.
        use_local_roi (bool): Se ``True``, restringe a busca à ROI local ao redor de
            ``reference_circle``.
        local_roi_padding (int): Margem adicional (em pixels) para o recorte da ROI
            local.
        verbose (bool): Se ``True``, imprime aviso quando o melhor candidato excede as
            tolerâncias.

    Returns:
        dict[str, Any] | str | None: Dicionário do círculo refinado,
            ``OUT_OF_TOLERANCE`` se divergir ou ``None`` se nenhum pico for detectado.
    """
    ref_x = reference_circle["center_x"]
    ref_y = reference_circle["center_y"]
    ref_radius = reference_circle["radius"]

    accums, cx, cy, radii = None, None, None, None

    if use_local_roi:
        # Primeiro tenta uma busca local ao redor do círculo da fatia anterior.
        x_min, x_max, y_min, y_max = _compute_local_roi_bounds(
            img_slice.shape,
            ref_x,
            ref_y,
            ref_radius,
            distance_tolerance,
            radius_tolerance,
            local_roi_padding,
        )

        roi_slice = img_slice[y_min:y_max, x_min:x_max]
        accums, cx, cy, radii = _detect_circles_in_slice(
            roi_slice, hough_radii, total_num_peaks, canny_sigma
        )

        if len(radii) > 0:
            cx = cx + x_min
            cy = cy + y_min
        else:
            # Se a ROI inicial falha, expande a busca sem voltar ao volume inteiro.
            expanded_padding = min(
                local_roi_padding * 2,
                int(np.sqrt(img_slice.shape[0] ** 2 + img_slice.shape[1] ** 2) / 2),
            )
            x_min, x_max, y_min, y_max = _compute_local_roi_bounds(
                img_slice.shape,
                ref_x,
                ref_y,
                ref_radius,
                distance_tolerance,
                radius_tolerance,
                expanded_padding,
            )
            roi_slice = img_slice[y_min:y_max, x_min:x_max]
            accums, cx, cy, radii = _detect_circles_in_slice(
                roi_slice, hough_radii, total_num_peaks, canny_sigma
            )
            if len(radii) > 0:
                cx = cx + x_min
                cy = cy + y_min
    else:
        accums, cx, cy, radii = _detect_circles_in_slice(
            img_slice, hough_radii, total_num_peaks, canny_sigma
        )

    if len(radii) == 0:
        return None

    # Seleciona um candidato consistente com o círculo anterior.
    min_idx, min_dist = _select_circle_candidate(
        cx,
        cy,
        radii,
        ref_x,
        ref_y,
        ref_radius,
        radius_tolerance,
        distance_tolerance,
    )

    if not _is_circle_within_tolerance(
        radii[min_idx], min_dist, ref_radius, radius_tolerance, distance_tolerance
    ):
        # Candidato fora da tolerância encerra o rastreamento.
        slice_idx = int(reference_circle.get("slice_index", -1))
        if verbose:
            print(
                f"Parada na fatia {slice_idx - 1}: Δr={abs(radii[min_idx] - ref_radius):.2f} ou dist={min_dist:.2f}"
            )
        return OUT_OF_TOLERANCE

    # Usa círculos vizinhos para reduzir jitter no centro e no raio escolhidos.
    cx_mean, cy_mean, radius_mean = refine_circle_with_neighbors(
        cx,
        cy,
        radii,
        float(cx[min_idx]),
        float(cy[min_idx]),
        neighbor_distance_threshold,
    )

    if radius_mean is None:
        radius_mean = float(radii[min_idx])

    return {
        "center_x": cx_mean,
        "center_y": cy_mean,
        "radius": radius_mean,
        "accum": float(accums[min_idx]),
    }


def detect_initial_circle(
    img_slice: NDArray[Any],
    hough_radii: Sequence[float],
    quadrant_offset: Sequence[int] = (30, 30),
    total_num_peaks: int = 10,
    canny_sigma: float = 3,
) -> Optional[dict]:
    """Detecta o círculo inicial da aorta em uma fatia de referência.

    Args:
        img_slice (NDArray[Any]): Fatia axial 2D pré-processada no formato ``(y, x)``.
        hough_radii (Sequence[float]): Raios candidatos (em pixels) testados pela
            transformada de Hough.
        quadrant_offset (Sequence[int]): Deslocamento ``(dx, dy)`` em pixels que define
            o quadrante anatômico esperado da busca inicial.
        total_num_peaks (int): Quantidade máxima de picos retornados pela Hough na
            fatia.
        canny_sigma (float): Desvio-padrão gaussiano (em pixels) do detector de bordas
            Canny.

    Returns:
        Optional[dict[str, float]]: Dicionário
            ``{"center_x", "center_y", "radius", "accum"}`` ou ``None`` se nada for
            encontrado.
    """
    # Extrai candidatos circulares na fatia de referência via Canny e Hough.
    accums, cx, cy, radii = _detect_circles_in_slice(
        img_slice, hough_radii, total_num_peaks, canny_sigma
    )

    if len(accums) == 0:
        return None

    # Seleciona o primeiro pico posicionado no quadrante anatômico esperado.
    idx = _select_initial_circle_candidate(
        cx,
        cy,
        img_slice.shape,
        quadrant_offset,
    )
    if idx is None:
        return None

    return {
        "center_x": float(cx[idx]),
        "center_y": float(cy[idx]),
        "radius": float(radii[idx]),
        "accum": float(accums[idx]),
    }


def get_initial_circle_diagnostics(
    img_slice: NDArray[Any],
    hough_radii: Sequence[float],
    quadrant_offset: Sequence[int] = (30, 30),
    total_num_peaks_initial: int = 10,
    canny_sigma: float = 3,
    neighbor_distance_threshold: float = 5,
) -> dict:
    """Retorna o círculo inicial, os candidatos da fatia e o círculo refinado.

    Args:
        img_slice (NDArray[Any]): Fatia axial 2D pré-processada no formato ``(y, x)``.
        hough_radii (Sequence[float]): Raios candidatos (em pixels) testados pela
            transformada de Hough.
        quadrant_offset (Sequence[int]): Deslocamento ``(dx, dy)`` em pixels do
            quadrante inicial da busca.
        total_num_peaks_initial (int): Quantidade máxima de picos retornados pela Hough.
        canny_sigma (float): Desvio-padrão gaussiano (em pixels) do detector de bordas
            Canny.
        neighbor_distance_threshold (float): Distância máxima (em pixels) para incluir
            vizinhos no refinamento médio.

    Returns:
        dict[str, Any]: Dicionário com ``initial_circle``, ``refined_circle``,
            ``candidates`` e ``refinement_candidates``.
    """
    # Executa a Hough uma única vez para inspecionar todos os picos iniciais da fatia.
    accums, cx, cy, radii = _detect_circles_in_slice(
        img_slice, hough_radii, total_num_peaks_initial, canny_sigma
    )

    if len(accums) == 0:
        return {
            "initial_circle": None,
            "refined_circle": None,
            "candidates": [],
            "refinement_candidates": [],
        }

    # Encontra o círculo inicial no quadrante sem repetir a Hough.
    idx = _select_initial_circle_candidate(cx, cy, img_slice.shape, quadrant_offset)
    if idx is None:
        return {
            "initial_circle": None,
            "refined_circle": None,
            "candidates": [],
            "refinement_candidates": [],
        }

    idx = int(idx)
    initial_circle = {
        "center_x": float(cx[idx]),
        "center_y": float(cy[idx]),
        "radius": float(radii[idx]),
        "accum": float(accums[idx]),
    }

    # Separa candidatos usados para explicar/refazer o refinamento inicial.
    distances = _calculate_distances_vectorized(
        cx, cy, initial_circle["center_x"], initial_circle["center_y"]
    )
    refinement_candidates = [
        {
            "center_x": float(cx[idx]),
            "center_y": float(cy[idx]),
            "radius": float(radii[idx]),
            "accum": float(accums[idx]),
        }
        for idx in range(len(cx))
        if distances[idx] <= neighbor_distance_threshold
    ]

    # Calcula o círculo refinado pela média dos vizinhos e serializa os candidatos.
    refined_x, refined_y, refined_radius = refine_circle_with_neighbors(
        cx,
        cy,
        radii,
        initial_circle["center_x"],
        initial_circle["center_y"],
        distance_threshold=neighbor_distance_threshold,
    )

    refined_circle = {
        "center_x": float(refined_x),
        "center_y": float(refined_y),
        "radius": float(refined_radius)
        if refined_radius is not None
        else float(initial_circle["radius"]),
        "accum": float(initial_circle["accum"]),
    }

    candidates = [
        {
            "center_x": float(cx[idx]),
            "center_y": float(cy[idx]),
            "radius": float(radii[idx]),
            "accum": float(accums[idx]),
        }
        for idx in range(len(cx))
    ]

    return {
        "initial_circle": initial_circle,
        "refined_circle": refined_circle,
        "candidates": candidates,
        "refinement_candidates": refinement_candidates,
    }


def refine_circle_with_neighbors(
    cx: ArrayLike,
    cy: ArrayLike,
    radii: ArrayLike,
    ref_x: float,
    ref_y: float,
    distance_threshold: float = 5,
) -> Tuple[float, float, Optional[float]]:
    """Refina centro e raio pela média dos círculos vizinhos próximos.

    Args:
        cx (ArrayLike): Coordenadas ``x`` dos círculos candidatos na fatia (em pixels).
        cy (ArrayLike): Coordenadas ``y`` dos círculos candidatos na fatia (em pixels).
        radii (ArrayLike): Raios dos círculos candidatos (em pixels).
        ref_x (float): Coordenada ``x`` do círculo de referência (em pixels).
        ref_y (float): Coordenada ``y`` do círculo de referência (em pixels).
        distance_threshold (float): Raio máximo de distância euclidiana (em pixels) para
            incluir vizinhos.

    Returns:
        Tuple[float, float, Optional[float]]: Tupla ``(x_mean, y_mean, radius_mean)``,
            com ``radius_mean=None`` se nenhum vizinho for encontrado.
    """
    distances = _calculate_distances_vectorized(cx, cy, ref_x, ref_y)
    mask = distances <= distance_threshold

    if not np.any(mask):
        return ref_x, ref_y, None

    # Média local dos candidatos próximos suaviza pequenas variações da Hough.
    radii_arr = np.asarray(radii)
    cx_arr = np.asarray(cx)
    cy_arr = np.asarray(cy)

    radius_mean = float(np.mean(radii_arr[mask]))
    x_mean = float(np.mean(cx_arr[mask]))
    y_mean = float(np.mean(cy_arr[mask]))

    return x_mean, y_mean, radius_mean


def detect_aorta_circles(
    img_volume: NDArray[Any],
    hough_radii: Sequence[float],
    pixel_spacing: float,
    tol_radius_mm: float = 9.0,
    tol_distance_mm: float = 18.0,
    max_slice_miss_threshold: int = 5,
    neighbor_distance_threshold: float = 5,
    quadrant_offset: Sequence[int] = (30, 30),
    total_num_peaks_initial: int = 10,
    total_num_peaks: int = 8,
    canny_sigma: float = 3,
    use_local_roi: bool = True,
    local_roi_padding: int = 20,
    interpolate_missed_circles: bool = True,
    verbose: bool = True,
) -> list:
    """Detecta e rastreia círculos da aorta fatia a fatia por transformada de Hough.

    Args:
        img_volume (NDArray[Any]): Volume 3D pré-processado no formato ``(y, x, z)``.
        hough_radii (Sequence[float]): Raios candidatos (em pixels) testados em cada
            fatia pela Hough.
        pixel_spacing (float): Espaçamento médio no plano XY (``mm/px``) para converter
            tolerâncias físicas em pixels.
        tol_radius_mm (float): Variação máxima de raio entre fatias consecutivas, em
            ``mm``.
        tol_distance_mm (float): Deslocamento máximo do centro entre fatias
            consecutivas, em ``mm``.
        max_slice_miss_threshold (int): Máximo de fatias consecutivas sem detecção antes
            de interromper o rastreamento.
        neighbor_distance_threshold (float): Distância máxima (em pixels) para média
            local de candidatos vizinhos.
        quadrant_offset (Sequence[int]): Deslocamento ``(dx, dy)`` em pixels do
            quadrante esperado para o círculo inicial.
        total_num_peaks_initial (int): Máximo de picos da Hough avaliados na primeira
            fatia.
        total_num_peaks (int): Máximo de picos da Hough avaliados nas demais fatias.
        canny_sigma (float): Desvio-padrão gaussiano (em pixels) do detector de bordas
            Canny.
        use_local_roi (bool): Se ``True``, restringe a busca ao redor do círculo
            anterior.
        local_roi_padding (int): Margem (em pixels) adicionada ao redor do raio de
            referência na ROI local.
        interpolate_missed_circles (bool): Se ``True``, preenche por interpolação linear
            lacunas curtas quando a aorta reaparece antes do limite.
        verbose (bool): Se ``True``, imprime mensagens de diagnóstico no console.

    Returns:
        list[dict[str, Any]]: Lista de dicionários por fatia com ``slice_index``,
            ``center_x``, ``center_y``, ``radius`` e ``accum``.
    """
    # Valida dimensionalidade do volume e parâmetros obrigatórios do rastreamento.
    if img_volume.ndim != 3:
        raise ValueError(f"img_volume deve ser 3D, recebido shape={img_volume.shape}")
    if len(hough_radii) == 0:
        raise ValueError("hough_radii não pode ser vazio")
    if pixel_spacing <= 0:
        raise ValueError("pixel_spacing deve ser maior que 0")

    num_slices = img_volume.shape[2]
    first_slice_idx = num_slices - 1

    # Converte tolerâncias físicas (mm) para pixels da resolução atual.
    radius_tolerance = tol_radius_mm / pixel_spacing
    distance_tolerance = tol_distance_mm / pixel_spacing

    # Etapa 1: ancora a aorta na última fatia axial (região superior do exame ImageCAS).
    initial_slice = img_volume[:, :, first_slice_idx]
    accums, cx, cy, radii = _detect_circles_in_slice(
        initial_slice,
        hough_radii,
        total_num_peaks_initial,
        canny_sigma,
    )
    initial_idx = (
        _select_initial_circle_candidate(
            cx,
            cy,
            initial_slice.shape,
            quadrant_offset,
        )
        if len(accums) > 0
        else None
    )

    if initial_idx is None:
        if verbose:
            print("Nenhum círculo inicial detectado.")
        return []

    # Etapa 2: refina posição e raio iniciais combinando candidatos próximos.
    ref_x, ref_y, ref_radius = refine_circle_with_neighbors(
        cx,
        cy,
        radii,
        float(cx[initial_idx]),
        float(cy[initial_idx]),
        neighbor_distance_threshold,
    )
    refined_initial = {
        "center_x": ref_x,
        "center_y": ref_y,
        "radius": float(radii[initial_idx]) if ref_radius is None else ref_radius,
        "accum": float(accums[initial_idx]),
    }

    detected_circles = [{"slice_index": first_slice_idx, **refined_initial}]
    miss_counter = 0
    pending_missed_slices: list[int] = []

    # Etapa 3: rastreia para baixo fatia a fatia usando sempre o último círculo válido.
    for slice_idx in range(first_slice_idx - 1, -1, -1):
        reference_circle = detected_circles[-1]
        result = _process_slice(
            img_volume[:, :, slice_idx],
            hough_radii,
            reference_circle,
            radius_tolerance,
            distance_tolerance,
            neighbor_distance_threshold,
            total_num_peaks,
            canny_sigma,
            use_local_roi,
            local_roi_padding,
            verbose=verbose,
        )

        if result is None:
            # Sem candidato: guarda a fatia para possível interpolação futura.
            miss_counter += 1
            pending_missed_slices.append(slice_idx)
            if miss_counter >= max_slice_miss_threshold:
                if verbose:
                    print(
                        f"Parada: {max_slice_miss_threshold} fatias consecutivas sem detecção."
                    )
                break
            continue

        if result == OUT_OF_TOLERANCE or not isinstance(result, dict):
            break

        next_circle = {"slice_index": slice_idx, **result, "interpolated": False}
        if pending_missed_slices and interpolate_missed_circles:
            # Preenche lacunas curtas quando o rastreamento volta a encontrar a aorta.
            detected_circles.extend(
                _interpolate_missing_circles(
                    detected_circles[-1],
                    next_circle,
                    pending_missed_slices,
                )
            )
        detected_circles.append(next_circle)
        miss_counter = 0
        pending_missed_slices = []

    return detected_circles


__all__ = [
    "detect_aorta_circles",
    "detect_initial_circle",
    "extrapolate_stable_circle_tail",
    "filter_aorta_circle_trajectory",
    "get_initial_circle_diagnostics",
    "refine_circle_with_neighbors",
]
