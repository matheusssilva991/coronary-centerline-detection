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
from ..processing.binary_operations import binary_opening

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
    axial_margin_slices: int = 0,
) -> NDArray[np.uint8]:
    """Cria um envelope 3D interpolado ao redor da trajetória dos círculos.

    O envelope limita a máscara da aorta à região anatomicamente acompanhada
    pelo rastreamento circular. Centros e raios são interpolados nas fatias sem
    círculo explícito. A margem axial prolonga os círculos extremos para evitar
    cortes imediatamente antes da primeira ou depois da última detecção.
    """
    if radius_factor <= 0:
        raise ValueError("radius_factor deve ser maior que zero")
    if axial_margin_slices < 0:
        raise ValueError("axial_margin_slices deve ser zero ou maior")
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
    first_target_slice = max(0, int(source_slices[0]) - axial_margin_slices)
    last_target_slice = min(
        int(volume_shape[2]) - 1,
        int(source_slices[-1]) + axial_margin_slices,
    )
    target_slices = np.arange(first_target_slice, last_target_slice + 1)
    # np.interp mantém os valores extremos fora do intervalo das detecções.
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
    axial_margin_slices: int = 0,
) -> NDArray[np.uint8]:
    """Remove da máscara voxels fora do envelope da trajetória circular."""
    envelope = build_circle_trajectory_envelope(
        aorta_mask.shape,
        detected_circles,
        radius_factor=radius_factor,
        axial_margin_slices=axial_margin_slices,
    )
    return (aorta_mask.astype(bool) & envelope.astype(bool)).astype(np.uint8)


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


def calculate_circle_mask_profile(
    aorta_mask: NDArray[Any],
    detected_circles: Sequence[Dict[str, Any]],
) -> list[Dict[str, float | int]]:
    """Calcula preenchimento e razão área/círculo para cada fatia rastreada."""
    mask = np.asarray(aorta_mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("aorta_mask deve ser uma máscara 3D")

    profile: list[Dict[str, float | int]] = []
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

        fill_ratio = float(mask[rr, cc, slice_index].sum()) / disk_area
        segmented_area = float(segmented_area_by_slice[slice_index])
        profile.append(
            {
                "slice_index": slice_index,
                "circle_fill_ratio": fill_ratio,
                "circle_area_ratio": segmented_area / (np.pi * radius**2),
            }
        )
    return profile


def calculate_circle_mask_metrics(
    aorta_mask: NDArray[Any],
    detected_circles: Sequence[Dict[str, Any]],
) -> Dict[str, float | None]:
    """Resume o preenchimento dos círculos e o excesso de área da máscara."""
    profile = calculate_circle_mask_profile(aorta_mask, detected_circles)

    if not profile:
        return {
            "circle_fill_q25": None,
            "circle_area_ratio_p90": None,
        }
    fill_ratios = [float(item["circle_fill_ratio"]) for item in profile]
    area_ratios = [float(item["circle_area_ratio"]) for item in profile]
    return {
        "circle_fill_q25": float(np.quantile(fill_ratios, 0.25)),
        "circle_area_ratio_p90": float(np.quantile(area_ratios, 0.90)),
    }


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
