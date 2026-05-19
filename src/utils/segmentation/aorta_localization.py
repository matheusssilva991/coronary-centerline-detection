"""Localização de aorta baseada em círculos para volumes 3D de CCTA.

Este módulo detecta e rastreia candidatos circulares da aorta entre fatias
usando Canny + transformada de Hough com restrições de continuidade geométrica.
"""

import numpy as np
from skimage import feature
from skimage.transform import hough_circle, hough_circle_peaks
from typing import Any, Optional, Sequence, Tuple
from numpy.typing import NDArray

# GPU utilities (consistent style with binary_operations)
from ..processing.gpu_utils import GPU_AVAILABLE, to_gpu, to_cpu, cu_ndi, cp


def _calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calcula a distância euclidiana entre dois pontos 2D."""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _calculate_distances_vectorized(
    cx: NDArray[Any], cy: NDArray[Any], ref_x: float, ref_y: float
) -> NDArray[Any]:
    """Calcula distâncias euclidianas de forma vetorizada usando NumPy broadcasting."""
    cx_arr = np.asarray(cx)
    cy_arr = np.asarray(cy)
    return np.sqrt((cx_arr - ref_x) ** 2 + (cy_arr - ref_y) ** 2)


def _detect_circles_in_slice(
    img_slice: NDArray[Any],
    hough_radii: Sequence[float],
    total_num_peaks: int,
    canny_sigma: float,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detecta círculos em uma fatia usando Canny (ou GPU-preproc) + Hough.

    If `use_gpu` and CuPy available, do blur+Sobel magnitude on GPU and
    transfer a boolean edge map to CPU for the existing Hough implementation.
    """
    if use_gpu and GPU_AVAILABLE:
        img_gpu = to_gpu(img_slice.astype(np.float32))
        blurred = cu_ndi.gaussian_filter(img_gpu, sigma=canny_sigma)
        gx = cu_ndi.sobel(blurred, axis=1)
        gy = cu_ndi.sobel(blurred, axis=0)
        gmag = cp.sqrt(gx ** 2 + gy ** 2)
        try:
            thr = float(cp.percentile(gmag, 75))
        except Exception:
            thr = float(gmag.mean())
        edges = to_cpu(gmag > thr)
    else:
        edges = feature.canny(img_slice.astype(float), sigma=canny_sigma)

    hough_res = hough_circle(edges, hough_radii)
    return hough_circle_peaks(hough_res, hough_radii, total_num_peaks=total_num_peaks)


def _find_closest_circle(
    cx: Sequence[float],
    cy: Sequence[float],
    radii: Sequence[float],
    ref_x: float,
    ref_y: float,
) -> Tuple[int, float]:
    """Encontra o círculo detectado mais próximo de um ponto de referência (vetorizado)."""
    distances = _calculate_distances_vectorized(cx, cy, ref_x, ref_y)
    min_idx = int(np.argmin(distances))
    return min_idx, float(distances[min_idx])


def _is_circle_within_tolerance(
    circle_radius: float,
    circle_distance: float,
    ref_radius: float,
    radius_tolerance: float,
    distance_tolerance: float,
) -> bool:
    """Valida se raio e distância estão dentro das tolerâncias definidas."""
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
    """Calcula os limites de ROI local para busca de círculos na fatia."""
    height, width = img_shape

    search_radius = (
        ref_radius + distance_tolerance + radius_tolerance + local_roi_padding
    )
    half_size = int(np.ceil(max(8.0, search_radius)))

    cx = int(round(ref_x))
    cy = int(round(ref_y))

    x_min = max(0, cx - half_size)
    x_max = min(width, cx + half_size)
    y_min = max(0, cy - half_size)
    y_max = min(height, cy + half_size)

    return x_min, x_max, y_min, y_max


def _process_initial_circle(
    img_slice: NDArray[Any],
    hough_radii: Sequence[float],
    initial_circle: dict,
    neighbor_distance_threshold: float,
    total_num_peaks: int,
    canny_sigma: float,
    use_gpu: bool = False,
) -> dict:
    """Refina o círculo inicial com base em vizinhos próximos."""
    _, cx, cy, radii = _detect_circles_in_slice(
        img_slice, hough_radii, total_num_peaks, canny_sigma, use_gpu=use_gpu
    )

    ref_x, ref_y, ref_radius = refine_circle_with_neighbors(
        cx,
        cy,
        radii,
        initial_circle["center_x"],
        initial_circle["center_y"],
        neighbor_distance_threshold,
    )

    if ref_radius is None:
        ref_radius = initial_circle["radius"]

    return {
        "center_x": ref_x,
        "center_y": ref_y,
        "radius": ref_radius,
        "accum": initial_circle["accum"],
    }


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
    use_gpu: bool = False,
) -> Optional[dict]:
    """Processa uma fatia e retorna o melhor círculo rastreado (evita detecção duplicada)."""
    ref_x = reference_circle["center_x"]
    ref_y = reference_circle["center_y"]
    ref_radius = reference_circle["radius"]

    accums, cx, cy, radii = None, None, None, None

    if use_local_roi:
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
            roi_slice, hough_radii, total_num_peaks, canny_sigma, use_gpu=use_gpu
        )

        if len(radii) > 0:
            cx = cx + x_min
            cy = cy + y_min
        else:
            # Se não encontrou na ROI local, tentar com ROI expandida ao invés de fallback para volume inteiro
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
                roi_slice, hough_radii, total_num_peaks, canny_sigma, use_gpu=use_gpu
            )
            if len(radii) > 0:
                cx = cx + x_min
                cy = cy + y_min
    else:
        accums, cx, cy, radii = _detect_circles_in_slice(
            img_slice, hough_radii, total_num_peaks, canny_sigma, use_gpu=use_gpu
        )

    if len(radii) == 0:
        return None

    min_idx, min_dist = _find_closest_circle(cx, cy, radii, ref_x, ref_y)

    if not _is_circle_within_tolerance(
        radii[min_idx], min_dist, ref_radius, radius_tolerance, distance_tolerance
    ):
        slice_idx = reference_circle.get("slice_index", "N/A")
        print(
            f"Parada na fatia {slice_idx - 1}: Δr={abs(radii[min_idx] - ref_radius):.2f} ou dist={min_dist:.2f}"
        )
        return "out_of_tolerance"

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
    use_gpu: bool = False,
) -> Optional[dict]:
    """Detecta o círculo inicial da aorta em uma fatia de referência."""
    accums, cx, cy, radii = _detect_circles_in_slice(
        img_slice, hough_radii, total_num_peaks, canny_sigma, use_gpu=use_gpu
    )

    if len(accums) == 0:
        return None

    height, width = img_slice.shape
    center_x = (width // 2) - quadrant_offset[0]
    center_y = (height // 2) + quadrant_offset[1]

    cx_arr = np.asarray(cx)
    cy_arr = np.asarray(cy)
    mask = (cx_arr > center_x) & (cy_arr < center_y)
    first_quad_indices = np.where(mask)[0]

    if len(first_quad_indices) == 0:
        return None

    idx = int(first_quad_indices[0])
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
    use_gpu: bool = False,
) -> dict:
    """Retorna o círculo inicial, os candidatos da fatia e o círculo refinado."""
    accums, cx, cy, radii = _detect_circles_in_slice(
        img_slice, hough_radii, total_num_peaks_initial, canny_sigma, use_gpu=use_gpu
    )

    if len(accums) == 0:
        return {
            "initial_circle": None,
            "refined_circle": None,
            "candidates": [],
            "refinement_candidates": [],
        }

    # Encontrar círculo inicial no quadrante (sem chamar detect_initial_circle novamente)
    height, width = img_slice.shape
    center_x = (width // 2) - quadrant_offset[0]
    center_y = (height // 2) + quadrant_offset[1]

    cx_arr = np.asarray(cx)
    cy_arr = np.asarray(cy)
    mask = (cx_arr > center_x) & (cy_arr < center_y)
    first_quad_indices = np.where(mask)[0]

    if len(first_quad_indices) == 0:
        return {
            "initial_circle": None,
            "refined_circle": None,
            "candidates": [],
            "refinement_candidates": [],
        }

    idx = int(first_quad_indices[0])
    initial_circle = {
        "center_x": float(cx[idx]),
        "center_y": float(cy[idx]),
        "radius": float(radii[idx]),
        "accum": float(accums[idx]),
    }

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
    cx: Sequence[float],
    cy: Sequence[float],
    radii: Sequence[float],
    ref_x: float,
    ref_y: float,
    distance_threshold: float = 5,
) -> Tuple[float, float, Optional[float]]:
    """Refina centro e raio pela média dos círculos vizinhos próximos (vetorizado)."""
    distances = _calculate_distances_vectorized(cx, cy, ref_x, ref_y)
    mask = distances <= distance_threshold

    if not np.any(mask):
        return ref_x, ref_y, None

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
    use_gpu: bool = False,
) -> list:
    """Detecta círculos da aorta ao longo do volume 3D fatia a fatia."""
    num_slices = img_volume.shape[2]
    first_slice_idx = num_slices - 1

    radius_tolerance = tol_radius_mm / pixel_spacing
    distance_tolerance = tol_distance_mm / pixel_spacing

    initial_circle = detect_initial_circle(
        img_volume[:, :, first_slice_idx],
        hough_radii,
        quadrant_offset,
        total_num_peaks_initial,
        canny_sigma,
        use_gpu=use_gpu,
    )

    if initial_circle is None:
        print("Nenhum círculo inicial detectado.")
        return []

    refined_initial = _process_initial_circle(
        img_volume[:, :, first_slice_idx],
        hough_radii,
        initial_circle,
        neighbor_distance_threshold,
        total_num_peaks_initial,
        canny_sigma,
        use_gpu=use_gpu,
    )

    detected_circles = [{"slice_index": first_slice_idx, **refined_initial}]
    miss_counter = 0

    for slice_idx in range(first_slice_idx - 1, -1, -1):
        result = _process_slice(
            img_volume[:, :, slice_idx],
            hough_radii,
            detected_circles[-1],
            radius_tolerance,
            distance_tolerance,
            neighbor_distance_threshold,
            total_num_peaks,
            canny_sigma,
            use_local_roi,
            local_roi_padding,
            use_gpu=use_gpu,
        )

        if result is None:
            miss_counter += 1
            if miss_counter >= max_slice_miss_threshold:
                print(
                    f"Parada: {max_slice_miss_threshold} fatias consecutivas sem detecção."
                )
                break
            continue

        if result == "out_of_tolerance":
            break

        detected_circles.append({"slice_index": slice_idx, **result})
        miss_counter = 0

    return detected_circles


__all__ = [
    "detect_aorta_circles",
    "detect_initial_circle",
    "get_initial_circle_diagnostics",
    "refine_circle_with_neighbors",
]
