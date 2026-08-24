"""Segmentação da aorta em volumes 3D usando Level Set.

Este módulo implementa segmentação baseada em contornos ativos geodésicos
morfológicos (Morphological Geodesic Active Contour - MGAC) usando círculos
detectados como inicialização.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Dict, Sequence, cast

import numpy as np
from numpy.typing import NDArray
from skimage.draw import disk
from skimage.segmentation import (
    inverse_gaussian_gradient,
    morphological_geodesic_active_contour,
    morphsnakes,
)
from skimage.morphology import ball

# Operação morfológica reutilizada do pacote de processamento.
from ..processing.binary_operations import (
    binary_dilation,
    binary_erosion,
    binary_opening,
    label,
)

# Utilitários de GPU usados em trechos pontuais do level set/morfologia.
from ..processing.gpu_utils import GPU_AVAILABLE, to_gpu, to_cpu, cu_ndi, cp


# =============================================================================
# Funções Auxiliares Privadas
# =============================================================================


def _calculate_roi_bounds(
    detected_circles: Sequence[Dict[str, Any]],
    volume_shape: Sequence[int],
    roi_margin: int,
) -> Dict[str, int]:
    """
    Calcula os limites da região de interesse (ROI) baseado nos círculos detectados.

    Args:
        detected_circles (list): Lista de dicionários com círculos detectados
        volume_shape (tuple): Shape do volume 3D (altura, largura, profundidade)
        roi_margin (int): Margem extra em voxels ao redor da ROI

    Returns:
        dict: Dicionário com os limites da ROI contendo:
            - 'x_min', 'x_max': Limites no eixo x
            - 'y_min', 'y_max': Limites no eixo y
            - 'z_min', 'z_max': Limites no eixo z
    """
    # A ROI acompanha as fatias onde os círculos foram detectados.
    slice_indices = [int(c["slice_index"]) for c in detected_circles]
    z_min = max(0, min(slice_indices) - roi_margin)
    z_max = min(volume_shape[2], max(slice_indices) + roi_margin + 1)

    # Expande x/y pelo maior raio detectado e pela margem configurada.
    x_coords = [c["center_x"] for c in detected_circles]
    y_coords = [c["center_y"] for c in detected_circles]
    radii = [c["radius"] for c in detected_circles]
    max_radius = max(radii) if radii else 50

    x_min = max(0, int(min(x_coords) - max_radius - roi_margin))
    x_max = min(volume_shape[1], int(max(x_coords) + max_radius + roi_margin))
    y_min = max(0, int(min(y_coords) - max_radius - roi_margin))
    y_max = min(volume_shape[0], int(max(y_coords) + max_radius + roi_margin))

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "z_min": z_min,
        "z_max": z_max,
    }


def _adjust_circles_to_roi(
    detected_circles: Sequence[Dict[str, Any]], roi_bounds: Dict[str, int]
) -> Sequence[Dict[str, Any]]:
    """
    Ajusta as coordenadas dos círculos para o sistema de coordenadas da ROI.

    Args:
        detected_circles (list): Lista de círculos em coordenadas globais
        roi_bounds (dict): Limites da ROI com 'x_min', 'y_min', 'z_min', etc.

    Returns:
        list: Lista de círculos com coordenadas ajustadas para a ROI
    """
    roi_circles = []
    z_min = roi_bounds["z_min"]
    z_max = roi_bounds["z_max"]
    x_min = roi_bounds["x_min"]
    y_min = roi_bounds["y_min"]

    for c in detected_circles:
        if z_min <= c["slice_index"] < z_max:
            # Desloca centro/fatia para o sistema local da ROI.
            roi_c = {
                "slice_index": int(c["slice_index"]) - z_min,
                "center_x": c["center_x"] - x_min,
                "center_y": c["center_y"] - y_min,
                "radius": c["radius"],
            }
            roi_circles.append(roi_c)

    return roi_circles


def _initialize_level_set_from_circles(
    volume_shape: Sequence[int],
    circles: Sequence[Dict[str, Any]],
    radius_reduction_factor: float = 0.8,
) -> NDArray[Any]:
    """
    Inicializa a máscara do level set usando círculos detectados como sementes.

    Cria uma máscara binária onde cada círculo é desenhado como um disco
    preenchido na fatia correspondente.

    Args:
        volume_shape (tuple): Shape do volume (altura, largura, profundidade)
        circles (list): Lista de dicionários com círculos contendo
            'slice_index', 'center_x', 'center_y', 'radius'
        radius_reduction_factor (float): Fator para reduzir o raio inicial.
            Valores < 1.0 criam sementes menores que os círculos detectados.
            Default: 0.8

    Returns:
        np.ndarray: Máscara binária 3D (dtype=int8) com as sementes inicializadas
    """
    init_level_set = np.zeros(volume_shape, dtype=np.int8)
    height, width = volume_shape[:2]

    for circle in circles:
        # Cada círculo vira uma semente circular na fatia correspondente.
        slice_idx = int(circle["slice_index"])
        cx = circle["center_x"]
        cy = circle["center_y"]
        r = max(1, circle["radius"] * radius_reduction_factor)

        # Desenhar disco preenchido na fatia
        rr, cc = disk((cy, cx), r, shape=(height, width))
        init_level_set[rr, cc, slice_idx] = 1

    return init_level_set


def _prepare_level_set_inputs(
    volume_ccta: NDArray[Any],
    detected_circles: Sequence[Dict[str, Any]],
    radius_reduction_factor: float,
    roi_margin: int,
    use_roi: bool,
    alpha: float,
    sigma: float,
    use_gpu: bool,
) -> tuple[NDArray[Any], NDArray[Any], Dict[str, int] | None]:
    """Prepara gradiente, sementes e ROI compartilhados por todas as iterações."""
    if use_roi:
        roi_bounds = _calculate_roi_bounds(
            detected_circles,
            volume_ccta.shape,
            roi_margin,
        )
        work_volume = volume_ccta[
            roi_bounds["y_min"] : roi_bounds["y_max"],
            roi_bounds["x_min"] : roi_bounds["x_max"],
            roi_bounds["z_min"] : roi_bounds["z_max"],
        ]
        work_circles = _adjust_circles_to_roi(detected_circles, roi_bounds)
    else:
        roi_bounds = None
        work_volume = volume_ccta
        work_circles = detected_circles

    init_level_set = _initialize_level_set_from_circles(
        work_volume.shape,
        work_circles,
        radius_reduction_factor,
    )

    # O mapa de bordas é calculado uma única vez e reutilizado nos checkpoints.
    if use_gpu and GPU_AVAILABLE and cu_ndi is not None and cp is not None:
        vol_gpu = to_gpu(work_volume.astype(np.float32))
        blurred = cu_ndi.gaussian_filter(vol_gpu, sigma=sigma)
        gx = cast(Any, cu_ndi.sobel(blurred, axis=1))
        gy = cast(Any, cu_ndi.sobel(blurred, axis=0))
        gmag = cp.sqrt(gx**2 + gy**2)
        gimage = to_cpu(1.0 / (1.0 + alpha * (gmag**2)))
    else:
        gimage = inverse_gaussian_gradient(work_volume, alpha=alpha, sigma=sigma)

    return np.asarray(gimage), init_level_set, roi_bounds


def _restore_level_set_mask(
    work_mask: NDArray[Any],
    volume_shape: Sequence[int],
    roi_bounds: Dict[str, int] | None,
) -> NDArray[np.uint8]:
    """Reinsere uma máscara local da ROI nas coordenadas do volume completo."""
    if roi_bounds is None:
        return np.asarray(work_mask, dtype=np.uint8)

    full_mask = np.zeros(volume_shape, dtype=np.uint8)
    full_mask[
        roi_bounds["y_min"] : roi_bounds["y_max"],
        roi_bounds["x_min"] : roi_bounds["x_max"],
        roi_bounds["z_min"] : roi_bounds["z_max"],
    ] = work_mask
    return full_mask


def _crop_level_set_mask(
    full_mask: NDArray[Any],
    roi_bounds: Dict[str, int] | None,
) -> NDArray[np.uint8]:
    """Converte uma máscara completa para o sistema local usado pelo MorphGAC."""
    mask = np.asarray(full_mask, dtype=np.uint8)
    if roi_bounds is None:
        return mask.copy()
    return mask[
        roi_bounds["y_min"] : roi_bounds["y_max"],
        roi_bounds["x_min"] : roi_bounds["x_max"],
        roi_bounds["z_min"] : roi_bounds["z_max"],
    ].copy()


def _evolve_level_set(
    gimage: NDArray[Any],
    init_level_set: NDArray[Any],
    num_iter: int,
    smoothing: int,
    balloon: float,
    threshold: Any,
) -> NDArray[np.uint8]:
    """Executa um bloco do MorphGAC a partir da máscara recebida."""
    active_contour = cast(Any, morphological_geodesic_active_contour)
    return np.asarray(
        active_contour(
            gimage,
            num_iter=num_iter,
            init_level_set=init_level_set,
            smoothing=smoothing,
            balloon=balloon,
            threshold=threshold,
        ),
        dtype=np.uint8,
    )


def reset_morphgac_curvature_cycle() -> None:
    """Reinicia a alternância global dos operadores de curvatura do MorphGAC.

    O scikit-image mantém essa alternância em um iterador de módulo. O reset é
    usado apenas pelo modo adaptativo para impedir que uma contração de uma
    imagem altere a fase inicial da imagem seguinte.
    """
    cycle_factory = getattr(morphsnakes, "_fcycle")
    setattr(
        morphsnakes,
        "_curvop",
        cycle_factory(
            [
                lambda mask: morphsnakes.sup_inf(morphsnakes.inf_sup(mask)),
                lambda mask: morphsnakes.inf_sup(morphsnakes.sup_inf(mask)),
            ]
        ),
    )


@dataclass
class LevelSetEvolutionContext:
    """Estado reutilizável do MorphGAC para expansão e refinamento.

    O mapa de gradiente e a ROI são preparados uma única vez. A máscara de
    trabalho pode evoluir em blocos consecutivos ou ser substituída por uma
    máscara completa, como ocorre antes da segunda passagem contrativa.
    """

    gimage: NDArray[Any]
    current_mask: NDArray[np.uint8]
    volume_shape: tuple[int, int, int]
    roi_bounds: Dict[str, int] | None
    completed_iterations: int = 0

    def evolve(
        self,
        num_iter: int,
        *,
        smoothing: int,
        balloon: float,
        threshold: Any,
    ) -> NDArray[np.uint8]:
        """Evolui o estado atual e retorna a máscara em coordenadas globais."""
        if num_iter < 0:
            raise ValueError("num_iter não pode ser negativo")
        if num_iter:
            self.current_mask = _evolve_level_set(
                self.gimage,
                self.current_mask,
                num_iter,
                smoothing,
                balloon,
                threshold,
            )
            self.completed_iterations += num_iter
        return self.full_mask()

    def reset_from_full_mask(self, full_mask: NDArray[Any]) -> None:
        """Define uma nova inicialização sem recalcular gradiente ou ROI."""
        if tuple(full_mask.shape) != self.volume_shape:
            raise ValueError("A máscara de reinicialização deve ter o shape do volume")
        self.current_mask = _crop_level_set_mask(full_mask, self.roi_bounds)
        self.completed_iterations = 0

    def iter_checkpoints(
        self,
        checkpoint_iterations: Sequence[int],
        *,
        smoothing: int,
        balloon: float,
        threshold: Any,
    ) -> Iterator[tuple[int, NDArray[np.uint8]]]:
        """Produz snapshots acumulados a partir do estado atual."""
        checkpoints = sorted({int(value) for value in checkpoint_iterations})
        if not checkpoints or checkpoints[0] <= 0:
            raise ValueError("checkpoint_iterations deve conter inteiros positivos")

        completed = 0
        for checkpoint in checkpoints:
            yield checkpoint, self.evolve(
                checkpoint - completed,
                smoothing=smoothing,
                balloon=balloon,
                threshold=threshold,
            )
            completed = checkpoint

    def full_mask(self) -> NDArray[np.uint8]:
        """Restaura a máscara de trabalho no volume completo."""
        return _restore_level_set_mask(
            self.current_mask,
            self.volume_shape,
            self.roi_bounds,
        )


def prepare_level_set_evolution(
    volume_ccta: NDArray[Any],
    detected_circles: Sequence[Dict[str, Any]],
    *,
    radius_reduction_factor: float = 0.8,
    roi_margin: int = 10,
    use_roi: bool = True,
    alpha: float = 1000,
    sigma: float = 2,
    use_gpu: bool = False,
    reset_curvature_cycle: bool = False,
) -> LevelSetEvolutionContext:
    """Prepara uma evolução reutilizável do level set para um volume 3D."""
    if volume_ccta.ndim != 3:
        raise ValueError(f"volume_ccta deve ser 3D, recebido shape={volume_ccta.shape}")
    if not detected_circles:
        raise ValueError("detected_circles não pode ser vazio")
    if reset_curvature_cycle:
        reset_morphgac_curvature_cycle()

    gimage, init_level_set, roi_bounds = _prepare_level_set_inputs(
        volume_ccta,
        detected_circles,
        radius_reduction_factor,
        roi_margin,
        use_roi,
        alpha,
        sigma,
        use_gpu,
    )
    return LevelSetEvolutionContext(
        gimage=gimage,
        current_mask=np.asarray(init_level_set, dtype=np.uint8),
        volume_shape=(
            int(volume_ccta.shape[0]),
            int(volume_ccta.shape[1]),
            int(volume_ccta.shape[2]),
        ),
        roi_bounds=roi_bounds,
    )


def build_circle_trajectory_envelope(
    volume_shape: Sequence[int],
    detected_circles: Sequence[Dict[str, Any]],
    radius_factor: float = 1.5,
) -> NDArray[np.uint8]:
    """Cria um envelope 3D interpolado ao redor da trajetória dos círculos.

    O envelope limita a máscara da aorta à região anatomicamente acompanhada
    pelo rastreamento circular. Centros e raios são interpolados nas fatias sem
    círculo explícito entre a primeira e a última detecção.
    """
    if radius_factor <= 0:
        raise ValueError("radius_factor deve ser maior que zero")
    if not detected_circles:
        return np.zeros(volume_shape, dtype=np.uint8)

    circles_by_slice: dict[int, Dict[str, Any]] = {}
    for circle in detected_circles:
        slice_index = int(circle["slice_index"])
        if 0 <= slice_index < volume_shape[2]:
            circles_by_slice[slice_index] = circle
    if not circles_by_slice:
        return np.zeros(volume_shape, dtype=np.uint8)

    source_slices = np.array(sorted(circles_by_slice), dtype=float)
    centers_x = np.array(
        [circles_by_slice[int(z)]["center_x"] for z in source_slices], dtype=float
    )
    centers_y = np.array(
        [circles_by_slice[int(z)]["center_y"] for z in source_slices], dtype=float
    )
    radii = np.array(
        [circles_by_slice[int(z)]["radius"] for z in source_slices], dtype=float
    )
    target_slices = np.arange(int(source_slices[0]), int(source_slices[-1]) + 1)
    interp_x = np.interp(target_slices, source_slices, centers_x)
    interp_y = np.interp(target_slices, source_slices, centers_y)
    interp_radii = np.interp(target_slices, source_slices, radii)

    envelope = np.zeros(volume_shape, dtype=np.uint8)
    height, width = volume_shape[:2]
    for z, center_x, center_y, radius in zip(
        target_slices, interp_x, interp_y, interp_radii
    ):
        rr, cc = disk(
            (center_y, center_x),
            max(1.0, radius * radius_factor),
            shape=(height, width),
        )
        envelope[rr, cc, int(z)] = 1
    return envelope


def restrict_mask_to_circle_trajectory(
    aorta_mask: NDArray[Any],
    detected_circles: Sequence[Dict[str, Any]],
    radius_factor: float,
) -> NDArray[np.uint8]:
    """Remove da máscara voxels fora do envelope da trajetória circular."""
    envelope = build_circle_trajectory_envelope(
        aorta_mask.shape,
        detected_circles,
        radius_factor=radius_factor,
    )
    return (aorta_mask.astype(bool) & envelope.astype(bool)).astype(np.uint8)


def correct_anomalous_aorta_slices(
    aorta_mask: NDArray[Any],
    detected_circles: Sequence[Dict[str, Any]],
    area_ratio_threshold: float,
    radius_factor: float = 1.75,
    transition_slices: int = 0,
    transition_radius_step: float = 0.35,
) -> NDArray[np.uint8]:
    """Restringe fatias anômalas com uma transição opcional no eixo axial.

    Ao contrário do envelope global, fatias compatíveis com a trajetória são
    preservadas integralmente. Quando ``transition_slices`` é positivo, as
    fatias vizinhas usam discos progressivamente maiores para evitar planos de
    corte abruptos entre regiões corrigidas e não corrigidas.
    """
    if area_ratio_threshold <= 0:
        raise ValueError("area_ratio_threshold deve ser maior que zero")
    if radius_factor <= 0:
        raise ValueError("radius_factor deve ser maior que zero")
    if transition_slices < 0:
        raise ValueError("transition_slices não pode ser negativo")
    if transition_radius_step < 0:
        raise ValueError("transition_radius_step não pode ser negativo")

    corrected = np.asarray(aorta_mask, dtype=np.uint8).copy()
    height, width, depth = corrected.shape
    circles_by_slice = {
        int(circle["slice_index"]): circle
        for circle in detected_circles
        if 0 <= int(circle["slice_index"]) < depth
    }
    if not circles_by_slice:
        return corrected

    # Primeiro identifica somente as fatias realmente incompatíveis.
    anomalous_slices: list[int] = []
    for z, circle in circles_by_slice.items():
        radius = max(float(circle["radius"]), 1.0)
        expected_area = np.pi * radius**2
        segmented_area = float(corrected[:, :, z].sum())
        if segmented_area > area_ratio_threshold * expected_area:
            anomalous_slices.append(z)
    if not anomalous_slices:
        return corrected

    # Interpola a trajetória para suavizar a correção entre círculos detectados.
    source_slices = np.array(sorted(circles_by_slice), dtype=float)
    centers_x = np.array(
        [circles_by_slice[int(z)]["center_x"] for z in source_slices],
        dtype=float,
    )
    centers_y = np.array(
        [circles_by_slice[int(z)]["center_y"] for z in source_slices],
        dtype=float,
    )
    radii = np.array(
        [circles_by_slice[int(z)]["radius"] for z in source_slices],
        dtype=float,
    )
    first_slice = int(source_slices[0])
    last_slice = int(source_slices[-1])
    target_slices = range(first_slice, last_slice + 1)

    for z in target_slices:
        distance_to_anomaly = min(abs(z - anomaly) for anomaly in anomalous_slices)
        if distance_to_anomaly > transition_slices:
            continue

        center_x = float(np.interp(z, source_slices, centers_x))
        center_y = float(np.interp(z, source_slices, centers_y))
        radius = max(float(np.interp(z, source_slices, radii)), 1.0)
        effective_radius_factor = (
            radius_factor + transition_radius_step * distance_to_anomaly
        )

        # O disco cresce nas bordas da região corrigida, suavizando a transição.
        allowed = np.zeros((height, width), dtype=bool)
        rr, cc = disk(
            (center_y, center_x),
            radius * effective_radius_factor,
            shape=(height, width),
        )
        allowed[rr, cc] = True
        corrected[:, :, z] = (corrected[:, :, z].astype(bool) & allowed).astype(
            np.uint8
        )
    return corrected


# =============================================================================
# Funções Públicas
# =============================================================================


def level_set_segmentation(
    volume_ccta: NDArray[Any],
    detected_circles: Sequence[Dict[str, Any]],
    num_iter: int = 50,
    smoothing: int = 1,
    balloon: int = 1,
    threshold: Any = "auto",
    radius_reduction_factor: float = 0.8,
    roi_margin: int = 10,
    use_roi: bool = True,
    alpha: float = 1000,
    sigma: float = 2,
    use_gpu: bool = False,
) -> NDArray[Any]:
    """
    Segmenta a aorta usando Level Set 3D inicializado com círculos detectados.

    Esta função implementa segmentação por contorno ativo geodésico morfológico,
    usando os círculos detectados pela transformada de Hough como inicialização.
    Opcionalmente pode processar apenas uma região de interesse (ROI) para
    maior eficiência computacional.

    O algoritmo segue estas etapas:
    1. Calcula ROI baseada nos círculos detectados (se use_roi=True)
    2. Inicializa level set desenhando discos nos círculos detectados
    3. Calcula gradiente inverso para guiar a evolução do contorno
    4. Aplica contorno ativo geodésico morfológico
    5. Retorna máscara no volume completo (se usou ROI)

    Args:
        volume_ccta (np.ndarray): Volume 3D original (altura, largura, profundidade),
            já pré-processado/normalizado
        detected_circles (list): Lista de dicionários, cada um contendo:
            - 'slice_index' (int): Índice da fatia
            - 'center_x' (float): Coordenada x do centro
            - 'center_y' (float): Coordenada y do centro
            - 'radius' (float): Raio do círculo
        num_iter (int): Número de iterações do algoritmo Level Set. Default: 50
        smoothing (int): Número de iterações de suavização do contorno a cada
            passo. Valores maiores = contornos mais suaves. Default: 1
        balloon (int): Força de expansão (+) ou contração (-) do contorno.
            Default: 1 (leve expansão)
        threshold (str or float): Critério de parada. 'auto' usa critério
            automático baseado no gradiente. Default: 'auto'
        radius_reduction_factor (float): Fator multiplicativo para reduzir o
            raio inicial dos círculos (0.0-1.0). Sementes menores permitem que
            o contorno expanda até as bordas. Default: 0.8
        roi_margin (int): Margem extra em voxels ao redor da ROI para incluir
            contexto adicional. Default: 10
        use_roi (bool): Se True, processa apenas ROI ao redor dos círculos.
            Se False, processa volume completo. Default: True
        alpha (float): Sensibilidade às bordas no cálculo do gradiente.
            Valores maiores = bordas mais fracas também influenciam o contorno.
            Default: 1000
        sigma (float): Desvio padrão da suavização Gaussiana antes do gradiente.
            Valores maiores = mais suavização, menos sensibilidade a ruído.
            Default: 2

    Returns:
        np.ndarray: Máscara binária 3D (dtype=int8) com a segmentação da aorta,
            com o mesmo shape do volume_ccta de entrada. Valores: 0 (fundo) e 1 (aorta)

    Example:
        >>> volume = load_ccta_volume()  # shape: (512, 512, 200)
        >>> circles = detect_aorta_circles(volume, ...)
        >>> mask = level_set_segmentation(
        ...     volume, circles, num_iter=100, balloon=2
        ... )
        >>> print(f"Volume aorta: {mask.sum()} voxels")

    Note:
        - Para volumes grandes, use_roi=True é altamente recomendado
        - O parâmetro balloon controla se o contorno expande ou contrai
        - Ajuste alpha e sigma se houver muito ruído ou bordas fracas
    """
    if volume_ccta.ndim != 3:
        raise ValueError(f"volume_ccta deve ser 3D, recebido shape={volume_ccta.shape}")
    if not detected_circles:
        return np.zeros_like(volume_ccta, dtype=np.int8)

    context = prepare_level_set_evolution(
        volume_ccta,
        detected_circles,
        radius_reduction_factor=radius_reduction_factor,
        roi_margin=roi_margin,
        use_roi=use_roi,
        alpha=alpha,
        sigma=sigma,
        use_gpu=use_gpu,
    )
    return context.evolve(
        num_iter,
        smoothing=smoothing,
        balloon=balloon,
        threshold=threshold,
    )


def iter_level_set_checkpoints(
    volume_ccta: NDArray[Any],
    detected_circles: Sequence[Dict[str, Any]],
    checkpoint_iterations: Sequence[int],
    smoothing: int = 1,
    balloon: float = 1,
    threshold: Any = "auto",
    radius_reduction_factor: float = 0.8,
    roi_margin: int = 10,
    use_roi: bool = True,
    alpha: float = 1000,
    sigma: float = 2,
    use_gpu: bool = False,
    context: LevelSetEvolutionContext | None = None,
) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    """Produz máscaras em iterações acumuladas sem recalcular o gradiente.

    Cada bloco começa na máscara produzida pelo bloco anterior. Como o MorphGAC
    não mantém outro estado entre iterações, o último checkpoint é equivalente
    a uma execução contínua com o mesmo total de iterações.
    """
    if context is None:
        if volume_ccta.ndim != 3:
            raise ValueError(
                f"volume_ccta deve ser 3D, recebido shape={volume_ccta.shape}"
            )
        if not detected_circles:
            return

    checkpoints = sorted({int(value) for value in checkpoint_iterations})
    if not checkpoints or checkpoints[0] <= 0:
        raise ValueError("checkpoint_iterations deve conter inteiros positivos")

    evolution = context or prepare_level_set_evolution(
        volume_ccta,
        detected_circles,
        radius_reduction_factor=radius_reduction_factor,
        roi_margin=roi_margin,
        use_roi=use_roi,
        alpha=alpha,
        sigma=sigma,
        use_gpu=use_gpu,
    )
    yield from evolution.iter_checkpoints(
        checkpoints,
        smoothing=smoothing,
        balloon=balloon,
        threshold=threshold,
    )


def calculate_mask_change_fraction(
    previous_mask: NDArray[Any],
    current_mask: NDArray[Any],
) -> float:
    """Calcula a fração alterada entre máscaras usando a união como referência."""
    previous = np.asarray(previous_mask, dtype=bool)
    current = np.asarray(current_mask, dtype=bool)
    if previous.shape != current.shape:
        raise ValueError("As máscaras comparadas devem possuir o mesmo shape")

    union_count = int(np.logical_or(previous, current).sum())
    if union_count == 0:
        return 0.0
    changed_count = int(np.logical_xor(previous, current).sum())
    return changed_count / union_count


def calculate_circle_mask_metrics(
    aorta_mask: NDArray[Any],
    detected_circles: Sequence[Dict[str, Any]],
) -> Dict[str, float | None]:
    """Mede preenchimento dos círculos e excesso de área nas respectivas fatias."""
    mask = np.asarray(aorta_mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("aorta_mask deve ser uma máscara 3D")

    fill_ratios: list[float] = []
    area_ratios: list[float] = []
    height, width, depth = mask.shape
    segmented_area_by_slice = mask.sum(axis=(0, 1), dtype=np.int64)
    for circle in detected_circles:
        slice_index = int(circle["slice_index"])
        if not 0 <= slice_index < depth:
            continue

        radius = max(float(circle["radius"]), 1.0)
        rr, cc = disk(
            (float(circle["center_y"]), float(circle["center_x"])),
            radius,
            shape=(height, width),
        )
        disk_area = len(rr)
        if disk_area == 0:
            continue

        fill_ratios.append(float(mask[rr, cc, slice_index].sum()) / disk_area)
        segmented_area = float(segmented_area_by_slice[slice_index])
        area_ratios.append(segmented_area / (np.pi * radius**2))

    if not fill_ratios:
        return {
            "circle_fill_q25": None,
            "circle_area_ratio_p90": None,
        }
    return {
        "circle_fill_q25": float(np.quantile(fill_ratios, 0.25)),
        "circle_area_ratio_p90": float(np.quantile(area_ratios, 0.90)),
    }


def find_anomalous_circle_slices(
    aorta_mask: NDArray[Any],
    detected_circles: Sequence[Dict[str, Any]],
    area_ratio_threshold: float,
) -> list[int]:
    """Retorna fatias cuja área segmentada excede a área circular esperada."""
    if area_ratio_threshold <= 0:
        raise ValueError("area_ratio_threshold deve ser maior que zero")

    mask = np.asarray(aorta_mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("aorta_mask deve ser uma máscara 3D")

    slice_areas = mask.sum(axis=(0, 1), dtype=np.int64)
    anomalous: list[int] = []
    for circle in detected_circles:
        slice_index = int(circle["slice_index"])
        if not 0 <= slice_index < mask.shape[2]:
            continue
        radius = max(float(circle["radius"]), 1.0)
        expected_area = np.pi * radius**2
        if float(slice_areas[slice_index]) > area_ratio_threshold * expected_area:
            anomalous.append(slice_index)
    return sorted(set(anomalous))


def build_axial_refinement_region(
    depth: int,
    anomalous_slices: Sequence[int],
    margin_slices: int,
) -> NDArray[np.bool_]:
    """Expande as fatias anômalas por uma margem axial configurável."""
    if depth <= 0:
        raise ValueError("depth deve ser maior que zero")
    if margin_slices < 0:
        raise ValueError("margin_slices não pode ser negativo")

    selected = np.zeros(depth, dtype=bool)
    for slice_index in anomalous_slices:
        start = max(0, int(slice_index) - margin_slices)
        stop = min(depth, int(slice_index) + margin_slices + 1)
        selected[start:stop] = True
    return selected


def build_axial_refinement_schedule(
    depth: int,
    anomalous_slices: Sequence[int],
    margin_slices: int,
    max_iterations: int,
    mode: str = "gradual",
) -> NDArray[np.int16]:
    """Define quantas iterações contrativas serão usadas em cada fatia.

    No modo ``gradual``, a fatia anômala usa a contração máxima, a metade
    interna da margem usa aproximadamente metade das iterações e a metade
    externa usa uma iteração. Sobreposições sempre preservam o maior nível.
    """
    if max_iterations <= 0:
        raise ValueError("max_iterations deve ser maior que zero")
    if mode not in {"uniform", "gradual"}:
        raise ValueError("mode deve ser 'uniform' ou 'gradual'")

    region = build_axial_refinement_region(depth, anomalous_slices, margin_slices)
    schedule = np.zeros(depth, dtype=np.int16)
    if mode == "uniform":
        schedule[region] = max_iterations
        return schedule

    middle_iterations = max(1, int(round(max_iterations / 2)))
    inner_margin = max(1, int(np.ceil(margin_slices / 2)))
    for anomaly in anomalous_slices:
        for z in range(
            max(0, int(anomaly) - margin_slices),
            min(depth, int(anomaly) + margin_slices + 1),
        ):
            distance = abs(z - int(anomaly))
            if distance == 0:
                iterations = max_iterations
            elif distance <= inner_margin:
                iterations = middle_iterations
            else:
                iterations = 1
            schedule[z] = max(schedule[z], iterations)
    return schedule


def calculate_slice_area_jump_p95(aorta_mask: NDArray[Any]) -> float:
    """Calcula o percentil 95 dos saltos relativos de área entre fatias."""
    mask = np.asarray(aorta_mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("aorta_mask deve ser uma máscara 3D")

    areas = mask.sum(axis=(0, 1), dtype=np.int64).astype(float)
    occupied = np.flatnonzero(areas > 0)
    if occupied.size < 2:
        return 0.0

    # Considera o intervalo segmentado completo para capturar cortes internos.
    areas = areas[occupied[0] : occupied[-1] + 1]
    denominator = np.maximum.reduce(
        [areas[1:], areas[:-1], np.ones(areas.size - 1)]
    )
    jumps = np.abs(np.diff(areas)) / denominator
    return float(np.quantile(jumps, 0.95)) if jumps.size else 0.0


@dataclass(frozen=True)
class CircleSeededNeckPruningResult:
    """Resultado da remoção experimental de vazamentos por colo estreito."""

    mask: NDArray[np.uint8]
    attempted: bool
    accepted: bool
    anomalous_slice_count: int
    removed_voxels: int
    volume_loss_fraction: float
    area_ratio_before: float | None
    area_ratio_after: float | None
    fill_q25_before: float | None
    fill_q25_after: float | None
    slice_area_jump_p95_before: float
    slice_area_jump_p95_after: float
    rejection_reason: str


@dataclass(frozen=True)
class CircleAreaJumpPruningResult:
    """Resultado da poda guiada por uma expansão abrupta da área axial."""

    mask: NDArray[np.uint8]
    attempted: bool
    accepted: bool
    trigger_slice: int | None
    neck_slice: int | None
    removed_voxels: int
    volume_loss_fraction: float
    area_ratio_p90_before: float | None
    area_ratio_p90_after: float | None
    fill_q25_before: float | None
    fill_q25_after: float | None
    voxels_per_slice_before: float
    voxels_per_slice_after: float
    rejection_reason: str


def _neck_pruning_result(
    original_mask: NDArray[np.uint8],
    *,
    attempted: bool,
    accepted: bool = False,
    anomalous_slice_count: int = 0,
    candidate_mask: NDArray[np.uint8] | None = None,
    rejection_reason: str,
    detected_circles: Sequence[Dict[str, Any]],
) -> CircleSeededNeckPruningResult:
    """Monta diagnósticos consistentes para aceitações e rejeições."""
    candidate = original_mask if candidate_mask is None else candidate_mask
    before = calculate_circle_mask_metrics(original_mask, detected_circles)
    after = calculate_circle_mask_metrics(candidate, detected_circles)
    original_voxels = int(original_mask.sum())
    removed_voxels = max(0, original_voxels - int(candidate.sum()))
    return CircleSeededNeckPruningResult(
        mask=candidate if accepted else original_mask.copy(),
        attempted=attempted,
        accepted=accepted,
        anomalous_slice_count=anomalous_slice_count,
        removed_voxels=removed_voxels if accepted else 0,
        volume_loss_fraction=(removed_voxels / max(original_voxels, 1)) if accepted else 0.0,
        area_ratio_before=before["circle_area_ratio_p90"],
        area_ratio_after=after["circle_area_ratio_p90"],
        fill_q25_before=before["circle_fill_q25"],
        fill_q25_after=after["circle_fill_q25"],
        slice_area_jump_p95_before=calculate_slice_area_jump_p95(original_mask),
        slice_area_jump_p95_after=calculate_slice_area_jump_p95(candidate),
        rejection_reason=rejection_reason,
    )


def prune_circle_seeded_narrow_necks(
    aorta_mask: NDArray[Any],
    detected_circles: Sequence[Dict[str, Any]],
    *,
    erosion_radius: int = 2,
    area_ratio_threshold: float = 3.0,
    core_radius_factor: float = 0.85,
    inferior_fraction: float = 0.50,
    anomaly_margin_slices: int = 5,
    max_fill_loss: float = 0.01,
    max_volume_loss_fraction: float = 0.30,
    max_axial_jump_increase_fraction: float = 0.10,
) -> CircleSeededNeckPruningResult:
    """Remove vazamentos inferiores unidos à aorta por conexões estreitas.

    A máscara é erodida para romper colos finos. Entre as componentes erodidas,
    somente aquelas que ainda intersectam o núcleo interpolado dos círculos são
    reconstruídas por dilatação. A substituição fica restrita às fatias
    inferiores anômalas e sempre é uma submáscara da segmentação recebida.

    A correção é conservadora: qualquer perda do núcleo, de uma fatia inteira,
    de preenchimento circular ou de continuidade axial faz a máscara original
    ser devolvida exatamente como foi recebida.
    """
    original = np.asarray(aorta_mask, dtype=np.uint8)
    if original.ndim != 3:
        raise ValueError("aorta_mask deve ser uma máscara 3D")
    if erosion_radius <= 0:
        raise ValueError("erosion_radius deve ser maior que zero")
    if area_ratio_threshold <= 0:
        raise ValueError("area_ratio_threshold deve ser maior que zero")
    if not 0 < core_radius_factor <= 1:
        raise ValueError("core_radius_factor deve estar no intervalo (0, 1]")
    if not 0 < inferior_fraction <= 1:
        raise ValueError("inferior_fraction deve estar no intervalo (0, 1]")
    if anomaly_margin_slices < 0:
        raise ValueError("anomaly_margin_slices não pode ser negativo")
    if not 0 <= max_fill_loss <= 1:
        raise ValueError("max_fill_loss deve estar no intervalo [0, 1]")
    if not 0 <= max_volume_loss_fraction <= 1:
        raise ValueError("max_volume_loss_fraction deve estar no intervalo [0, 1]")
    if max_axial_jump_increase_fraction < 0:
        raise ValueError("max_axial_jump_increase_fraction não pode ser negativo")
    if not detected_circles or not original.any():
        return _neck_pruning_result(
            original,
            attempted=False,
            rejection_reason="missing_mask_or_circles",
            detected_circles=detected_circles,
        )

    valid_slices = sorted(
        {
            int(circle["slice_index"])
            for circle in detected_circles
            if 0 <= int(circle["slice_index"]) < original.shape[2]
        }
    )
    if not valid_slices:
        return _neck_pruning_result(
            original,
            attempted=False,
            rejection_reason="missing_valid_circle_slices",
            detected_circles=detected_circles,
        )

    # Os óstios são buscados a partir de z_min; portanto, a correção atua na
    # mesma porção inferior da trajetória e não modifica o arco superior.
    z_min, z_max = valid_slices[0], valid_slices[-1]
    inferior_stop = min(
        z_max,
        z_min + max(1, int(np.ceil((z_max - z_min + 1) * inferior_fraction))) - 1,
    )
    anomalous = [
        z
        for z in find_anomalous_circle_slices(
            original,
            detected_circles,
            area_ratio_threshold=area_ratio_threshold,
        )
        if z_min <= z <= inferior_stop
    ]
    if not anomalous:
        return _neck_pruning_result(
            original,
            attempted=False,
            rejection_reason="no_inferior_anomalous_slices",
            detected_circles=detected_circles,
        )

    axial_region = build_axial_refinement_region(
        original.shape[2], anomalous, anomaly_margin_slices
    )
    axial_region[:z_min] = False
    axial_region[inferior_stop + 1 :] = False

    # A erosão rompe conexões finas; o núcleo dos círculos identifica qual
    # componente pertence à aorta e quais componentes representam vazamento.
    structure = ball(erosion_radius)
    eroded = binary_erosion(original, structure=structure, gpu=False)
    labeled, component_count = label(eroded, gpu=False)
    core = build_circle_trajectory_envelope(
        original.shape,
        detected_circles,
        radius_factor=core_radius_factor,
    ).astype(bool)
    trusted_labels = np.unique(labeled[core & eroded.astype(bool)])
    trusted_labels = trusted_labels[trusted_labels > 0]
    if component_count == 0 or trusted_labels.size == 0:
        return _neck_pruning_result(
            original,
            attempted=True,
            anomalous_slice_count=len(anomalous),
            rejection_reason="trusted_component_not_found",
            detected_circles=detected_circles,
        )

    trusted_eroded = np.isin(labeled, trusted_labels)
    reconstructed = binary_dilation(
        trusted_eroded,
        structure=structure,
        gpu=False,
    ).astype(bool)
    reconstructed &= original.astype(bool)

    candidate = original.astype(bool).copy()
    candidate[:, :, axial_region] = reconstructed[:, :, axial_region]
    protected_core = core & original.astype(bool)
    candidate[:, :, axial_region] |= protected_core[:, :, axial_region]
    candidate = candidate.astype(np.uint8)

    before = calculate_circle_mask_metrics(original, detected_circles)
    after = calculate_circle_mask_metrics(candidate, detected_circles)
    before_area = before["circle_area_ratio_p90"]
    after_area = after["circle_area_ratio_p90"]
    before_fill = before["circle_fill_q25"]
    after_fill = after["circle_fill_q25"]
    original_voxels = int(original.sum())
    candidate_voxels = int(candidate.sum())
    removed_voxels = original_voxels - candidate_voxels
    volume_loss = removed_voxels / max(original_voxels, 1)
    original_slices = int(np.count_nonzero(original.sum(axis=(0, 1))))
    candidate_slices = int(np.count_nonzero(candidate.sum(axis=(0, 1))))
    jump_before = calculate_slice_area_jump_p95(original)
    jump_after = calculate_slice_area_jump_p95(candidate)
    _, candidate_components = label(candidate, gpu=False)

    rejection_reason = "accepted"
    if removed_voxels <= 0:
        rejection_reason = "no_voxels_removed"
    elif before_area is None or after_area is None or after_area >= before_area:
        rejection_reason = "area_ratio_not_reduced"
    elif (
        before_fill is None
        or after_fill is None
        or after_fill < before_fill - max_fill_loss
    ):
        rejection_reason = "circle_fill_loss"
    elif volume_loss > max_volume_loss_fraction:
        rejection_reason = "excessive_volume_loss"
    elif candidate_slices < original_slices:
        rejection_reason = "segmented_slice_loss"
    elif candidate_components != 1:
        rejection_reason = "disconnected_candidate"
    elif not np.all(candidate[protected_core]):
        rejection_reason = "protected_core_loss"
    elif jump_after > jump_before * (1 + max_axial_jump_increase_fraction) + 1e-12:
        rejection_reason = "axial_jump_increased"

    accepted = rejection_reason == "accepted"
    return _neck_pruning_result(
        original,
        attempted=True,
        accepted=accepted,
        anomalous_slice_count=len(anomalous),
        candidate_mask=candidate,
        rejection_reason=rejection_reason,
        detected_circles=detected_circles,
    )


def _circle_area_profile(
    aorta_mask: NDArray[Any],
    detected_circles: Sequence[Dict[str, Any]],
) -> list[Dict[str, float]]:
    """Interpola a trajetória e mede área absoluta e relativa por fatia."""
    mask = np.asarray(aorta_mask, dtype=bool)
    circles_by_slice = {
        int(circle["slice_index"]): circle
        for circle in detected_circles
        if 0 <= int(circle["slice_index"]) < mask.shape[2]
    }
    if not circles_by_slice:
        return []

    source_slices = np.asarray(sorted(circles_by_slice), dtype=float)
    source_x = np.asarray(
        [circles_by_slice[int(z)]["center_x"] for z in source_slices],
        dtype=float,
    )
    source_y = np.asarray(
        [circles_by_slice[int(z)]["center_y"] for z in source_slices],
        dtype=float,
    )
    source_r = np.asarray(
        [circles_by_slice[int(z)]["radius"] for z in source_slices],
        dtype=float,
    )
    target_slices = np.arange(
        int(source_slices[0]), int(source_slices[-1]) + 1, dtype=int
    )
    centers_x = np.interp(target_slices, source_slices, source_x)
    centers_y = np.interp(target_slices, source_slices, source_y)
    radii = np.interp(target_slices, source_slices, source_r)
    slice_areas = mask.sum(axis=(0, 1), dtype=np.int64)

    return [
        {
            "slice_index": float(z),
            "center_x": float(center_x),
            "center_y": float(center_y),
            "radius": max(float(radius), 1.0),
            "area": float(slice_areas[z]),
            "area_ratio": float(slice_areas[z])
            / (np.pi * max(float(radius), 1.0) ** 2),
        }
        for z, center_x, center_y, radius in zip(
            target_slices, centers_x, centers_y, radii
        )
    ]


def _area_jump_result(
    original: NDArray[np.uint8],
    detected_circles: Sequence[Dict[str, Any]],
    *,
    attempted: bool,
    accepted: bool = False,
    candidate: NDArray[np.uint8] | None = None,
    trigger_slice: int | None = None,
    neck_slice: int | None = None,
    rejection_reason: str,
) -> CircleAreaJumpPruningResult:
    """Consolida métricas da candidata mantendo a máscara original na rejeição."""
    evaluated = original if candidate is None else candidate
    before = calculate_circle_mask_metrics(original, detected_circles)
    after = calculate_circle_mask_metrics(evaluated, detected_circles)
    original_voxels = int(original.sum())
    removed = max(0, original_voxels - int(evaluated.sum()))

    def mean_slice_voxels(mask: NDArray[np.uint8]) -> float:
        occupied = int(np.count_nonzero(mask.sum(axis=(0, 1))))
        return float(mask.sum()) / occupied if occupied else 0.0

    return CircleAreaJumpPruningResult(
        mask=evaluated if accepted else original.copy(),
        attempted=attempted,
        accepted=accepted,
        trigger_slice=trigger_slice,
        neck_slice=neck_slice,
        removed_voxels=removed if accepted else 0,
        volume_loss_fraction=(removed / max(original_voxels, 1)) if accepted else 0.0,
        area_ratio_p90_before=before["circle_area_ratio_p90"],
        area_ratio_p90_after=after["circle_area_ratio_p90"],
        fill_q25_before=before["circle_fill_q25"],
        fill_q25_after=after["circle_fill_q25"],
        voxels_per_slice_before=mean_slice_voxels(original),
        voxels_per_slice_after=mean_slice_voxels(evaluated),
        rejection_reason=rejection_reason,
    )


def prune_aorta_at_circle_area_jump(
    aorta_mask: NDArray[Any],
    detected_circles: Sequence[Dict[str, Any]],
    *,
    inferior_fraction: float = 0.50,
    area_ratio_threshold: float = 2.8,
    min_ratio_jump: float = 0.50,
    min_relative_area_growth: float = 0.25,
    baseline_window_slices: int = 6,
    neck_search_slices: int = 8,
    cut_half_width_slices: int = 2,
    cut_radius_factor: float = 1.35,
    cut_transition_step: float = 0.20,
    core_radius_factor: float = 0.85,
    min_removed_fraction: float = 0.005,
    max_volume_loss_fraction: float = 0.15,
    max_fill_loss: float = 0.01,
    max_axial_jump_increase_fraction: float = 0.25,
) -> CircleAreaJumpPruningResult:
    """Separa uma expansão distal usando o menor plano anterior ao salto.

    A curva ``área/(pi*r²)`` é comparada a uma mediana local. Quando surge uma
    expansão relevante, procura-se a menor seção nas fatias anteriores. Uma
    faixa curta nesse ponto é restringida ao envelope circular, o que pode
    desconectar a região vazada. Somente a componente com maior sobreposição ao
    núcleo dos círculos é preservada.
    """
    original = np.asarray(aorta_mask, dtype=np.uint8)
    if original.ndim != 3:
        raise ValueError("aorta_mask deve ser uma máscara 3D")
    if not 0 < inferior_fraction <= 1:
        raise ValueError("inferior_fraction deve estar no intervalo (0, 1]")
    if baseline_window_slices <= 0 or neck_search_slices <= 0:
        raise ValueError("As janelas axiais devem ser positivas")
    if cut_half_width_slices < 0:
        raise ValueError("cut_half_width_slices não pode ser negativo")
    if cut_radius_factor <= 0 or not 0 < core_radius_factor <= 1:
        raise ValueError("Os fatores de raio devem ser positivos e válidos")
    if not detected_circles or not original.any():
        return _area_jump_result(
            original,
            detected_circles,
            attempted=False,
            rejection_reason="missing_mask_or_circles",
        )

    profile = _circle_area_profile(original, detected_circles)
    lower_count = max(2, int(np.ceil(len(profile) * inferior_fraction)))
    lower_profile = profile[:lower_count]
    if len(lower_profile) < 3:
        return _area_jump_result(
            original,
            detected_circles,
            attempted=False,
            rejection_reason="insufficient_circle_profile",
        )

    # Compara cada fatia com uma mediana anterior para evitar reagir a ruído.
    candidates: list[tuple[float, int]] = []
    for index in range(1, len(lower_profile)):
        start = max(0, index - baseline_window_slices)
        previous = lower_profile[start:index]
        baseline_ratio = float(np.median([item["area_ratio"] for item in previous]))
        baseline_area = float(np.median([item["area"] for item in previous]))
        current = lower_profile[index]
        ratio_jump = current["area_ratio"] - baseline_ratio
        relative_growth = (current["area"] - baseline_area) / max(baseline_area, 1.0)
        if current["area_ratio"] >= area_ratio_threshold and (
            ratio_jump >= min_ratio_jump
            or relative_growth >= min_relative_area_growth
        ):
            candidates.append((max(ratio_jump, relative_growth), index))

    if not candidates:
        return _area_jump_result(
            original,
            detected_circles,
            attempted=False,
            rejection_reason="no_abrupt_inferior_expansion",
        )

    _, trigger_index = max(candidates)
    neck_start = max(0, trigger_index - neck_search_slices)
    neck_candidates = lower_profile[neck_start:trigger_index]
    if not neck_candidates:
        return _area_jump_result(
            original,
            detected_circles,
            attempted=False,
            trigger_slice=int(lower_profile[trigger_index]["slice_index"]),
            rejection_reason="neck_slice_not_found",
        )
    minimum_ratio = min(item["area_ratio"] for item in neck_candidates)
    # Em empates, usa a menor seção mais próxima do salto para limitar o corte.
    neck = max(
        (
            item
            for item in neck_candidates
            if np.isclose(item["area_ratio"], minimum_ratio)
        ),
        key=lambda item: item["slice_index"],
    )
    trigger_slice = int(lower_profile[trigger_index]["slice_index"])
    neck_slice = int(neck["slice_index"])

    profile_by_slice = {int(item["slice_index"]): item for item in profile}
    cut_mask = original.astype(bool).copy()
    for z in range(
        max(0, neck_slice - cut_half_width_slices),
        min(original.shape[2], neck_slice + cut_half_width_slices + 1),
    ):
        circle = profile_by_slice.get(z)
        if circle is None:
            continue
        distance = abs(z - neck_slice)
        radius_factor = cut_radius_factor + cut_transition_step * distance
        allowed = np.zeros(original.shape[:2], dtype=bool)
        rr, cc = disk(
            (circle["center_y"], circle["center_x"]),
            circle["radius"] * radius_factor,
            shape=original.shape[:2],
        )
        allowed[rr, cc] = True
        cut_mask[:, :, z] &= allowed

    # O corte temporário separa ramos; o núcleo decide qual componente é aorta.
    labeled, component_count = label(cut_mask, gpu=False)
    core = build_circle_trajectory_envelope(
        original.shape,
        detected_circles,
        radius_factor=core_radius_factor,
    ).astype(bool)
    core_labels = labeled[core & cut_mask]
    overlap = np.bincount(core_labels.ravel(), minlength=component_count + 1)
    if overlap.size <= 1 or int(overlap[1:].max(initial=0)) == 0:
        return _area_jump_result(
            original,
            detected_circles,
            attempted=True,
            trigger_slice=trigger_slice,
            neck_slice=neck_slice,
            rejection_reason="trusted_component_not_found",
        )
    trusted_label = int(np.argmax(overlap[1:]) + 1)
    candidate = (labeled == trusted_label).astype(np.uint8)

    before = calculate_circle_mask_metrics(original, detected_circles)
    after = calculate_circle_mask_metrics(candidate, detected_circles)
    original_voxels = int(original.sum())
    removed_fraction = (original_voxels - int(candidate.sum())) / max(
        original_voxels, 1
    )
    before_jump = calculate_slice_area_jump_p95(original)
    after_jump = calculate_slice_area_jump_p95(candidate)
    original_slices = int(np.count_nonzero(original.sum(axis=(0, 1))))
    candidate_slices = int(np.count_nonzero(candidate.sum(axis=(0, 1))))

    reason = "accepted"
    if removed_fraction < min_removed_fraction:
        reason = "insufficient_volume_reduction"
    elif removed_fraction > max_volume_loss_fraction:
        reason = "excessive_volume_loss"
    elif (
        before["circle_area_ratio_p90"] is None
        or after["circle_area_ratio_p90"] is None
        or after["circle_area_ratio_p90"] >= before["circle_area_ratio_p90"]
    ):
        reason = "area_ratio_not_reduced"
    elif (
        before["circle_fill_q25"] is None
        or after["circle_fill_q25"] is None
        or after["circle_fill_q25"] < before["circle_fill_q25"] - max_fill_loss
    ):
        reason = "circle_fill_loss"
    elif candidate_slices < original_slices:
        reason = "segmented_slice_loss"
    elif not np.all(candidate[core & original.astype(bool)]):
        reason = "protected_core_loss"
    elif after_jump > before_jump * (1 + max_axial_jump_increase_fraction) + 1e-12:
        reason = "axial_jump_increased"

    accepted = reason == "accepted"
    return _area_jump_result(
        original,
        detected_circles,
        attempted=True,
        accepted=accepted,
        candidate=candidate,
        trigger_slice=trigger_slice,
        neck_slice=neck_slice,
        rejection_reason=reason,
    )


def remove_leaks_morphology(
    mask_3d: NDArray[Any],
    radius: int = 3,
    use_gpu: bool = False,
) -> NDArray[Any]:
    """
    Remove vazamentos e ruído da máscara 3D usando abertura morfológica.

    A abertura morfológica (erosão seguida de dilatação) é eficaz para:
    - Remover pequenas conexões espúrias (vazamentos)
    - Eliminar ruído e pequenos componentes isolados
    - Suavizar o contorno da máscara
    - Separar objetos que estão fracamente conectados

    O tamanho do elemento estruturante (raio) controla a escala dos artefatos
    removidos. Vazamentos maiores que o raio serão preservados.

    Args:
        mask_3d (np.ndarray): Máscara binária 3D a ser limpa (dtype=bool ou int)
        radius (int): Raio do elemento estruturante esférico (ball) em voxels.
            Valores maiores removem estruturas maiores mas podem alterar
            significativamente a geometria. Default: 3

    Returns:
        np.ndarray: Máscara limpa com o mesmo shape e dtype da entrada

    Example:
        >>> noisy_mask = segment_aorta(volume)
        >>> clean_mask = remove_leaks_morphology(noisy_mask, radius=2)
        >>> print(f"Removed {noisy_mask.sum() - clean_mask.sum()} voxels")

    Note:
        - A operação preserva o tipo de dado da entrada
        - Para máscaras muito ruidosas, considere aplicar múltiplas vezes
          com raios diferentes ou usar outras técnicas de pós-processamento
        - O custo computacional cresce com o raio (O(r³))

    See Also:
        skimage.morphology.opening: Documentação da operação morfológica
        skimage.morphology.ball: Elemento estruturante esférico 3D
    """
    radius = int(radius)
    if radius <= 0:
        return mask_3d.copy()

    kernel = np.asarray(ball(radius))

    # Remove conexões finas/vazamentos com abertura morfológica.
    mask_cleaned = binary_opening(mask_3d, structure=kernel, gpu=bool(use_gpu))

    return mask_cleaned
