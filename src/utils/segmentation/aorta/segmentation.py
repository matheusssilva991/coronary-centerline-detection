"""Segmentação da aorta em volumes 3D usando Level Set.

Este módulo implementa segmentação baseada em contornos ativos geodésicos
morfológicos (Morphological Geodesic Active Contour - MGAC) usando círculos
detectados como inicialização.
"""

from dataclasses import dataclass
from typing import Any, Dict, Sequence, cast

import numpy as np
from numpy.typing import NDArray
from skimage.draw import disk
from skimage.morphology import ball
from skimage.segmentation import (
    inverse_gaussian_gradient,
    morphological_geodesic_active_contour,
)

# Operação morfológica reutilizada do pacote de processamento.
from utils.processing.binary_operations import binary_opening

# Utilitários de GPU usados em trechos pontuais do level set/morfologia.
from utils.processing.gpu_utils import GPU_AVAILABLE, cp, cu_ndi, to_cpu, to_gpu
from utils.utils.roi import mask_voxel_bbox_slices

AORTA_FEEDBACK_ADEQUATE = "adequate"
AORTA_FEEDBACK_UNDERSEGMENTED = "suspected_undersegmentation"
AORTA_FEEDBACK_OVERSEGMENTED = "suspected_oversegmentation"
AORTA_FEEDBACK_INSUFFICIENT_DATA = "insufficient_data"


def _calculate_roi_bounds(
    detected_circles: Sequence[Dict[str, Any]],
    volume_shape: Sequence[int],
    roi_margin: int,
) -> Dict[str, int]:
    """Calcula os limites da ROI ao redor dos círculos detectados.

    Args:
        detected_circles (Sequence[Dict[str, Any]]): Sequência de círculos detectados
            contendo ``slice_index``, ``center_x``, ``center_y`` e ``radius``.
        volume_shape (Sequence[int]): Dimensões ``(y, x, z)`` do volume completo.
        roi_margin (int): Margem adicional em voxels aplicada aos limites da ROI.

    Returns:
        Dict[str, int]: Dicionário com ``x_min``, ``x_max``, ``y_min``, ``y_max``,
            ``z_min`` e ``z_max`` dentro do volume.
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
    """Converte as coordenadas dos círculos globais para o sistema local da ROI.

    Args:
        detected_circles (Sequence[Dict[str, Any]]): Círculos detectados no sistema de
            coordenadas global.
        roi_bounds (Dict[str, int]): Limites da ROI retornados por
            ``_calculate_roi_bounds``.

    Returns:
        Sequence[Dict[str, Any]]: Lista de círculos filtrados e deslocados para o
            sistema local da ROI.
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
    """Gera máscara binária ``int8`` desenhando discos iniciais reduzidos por fatia.

    Args:
        volume_shape (Sequence[int]): Dimensões ``(y, x, z)`` do volume ou ROI onde as
            sementes serão desenhadas.
        circles (Sequence[Dict[str, Any]]): Círculos alinhados ao sistema de coordenadas
            de ``volume_shape``.
        radius_reduction_factor (float): Fração aplicada ao raio de cada círculo para
            criar sementes internas à parede aórtica.

    Returns:
        NDArray[Any]: Máscara binária ``int8`` preenchida com os discos iniciais.
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


def initialize_level_set_mask(
    volume_shape: Sequence[int],
    detected_circles: Sequence[Dict[str, Any]],
    *,
    radius_reduction_factor: float,
    roi_margin: int = 10,
    use_roi: bool = True,
) -> NDArray[np.uint8]:
    """Reconstrói a máscara binária de sementes do level set no volume completo.

    Args:
        volume_shape (Sequence[int]): Dimensões ``(y, x, z)`` do volume de trabalho.
        detected_circles (Sequence[Dict[str, Any]]): Círculos detectados com
            ``slice_index``, ``center_x``, ``center_y`` e ``radius``.
        radius_reduction_factor (float): Fração do raio aplicada ao desenhar os discos
            iniciais.
        roi_margin (int): Margem em voxels ao redor dos círculos quando
            ``use_roi=True``.
        use_roi (bool): Se ``True``, constrói os discos no recorte da ROI antes de
            reinserir no volume completo.

    Returns:
        NDArray[np.uint8]: Máscara binária ``uint8`` com os discos iniciais no shape
            ``volume_shape``.
    """
    if not detected_circles:
        return np.zeros(volume_shape, dtype=np.uint8)
    # Calcula a ROI opcional e converte os círculos para coordenadas locais.
    bounds = (
        _calculate_roi_bounds(detected_circles, volume_shape, roi_margin)
        if use_roi
        else None
    )
    if bounds is None:
        shape = volume_shape
        circles = detected_circles
    else:
        shape = (
            bounds["y_max"] - bounds["y_min"],
            bounds["x_max"] - bounds["x_min"],
            bounds["z_max"] - bounds["z_min"],
        )
        circles = _adjust_circles_to_roi(detected_circles, bounds)
    # Desenha as sementes reduzidas e reinsere a máscara no sistema global.
    seeds = _initialize_level_set_from_circles(shape, circles, radius_reduction_factor)
    return _restore_level_set_mask(seeds, volume_shape, bounds)


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
    """Prepara gradiente, sementes e ROI compartilhados por todas as iterações.

    Args:
        volume_ccta (NDArray[Any]): Volume 3D pré-processado no formato ``(y, x, z)``.
        detected_circles (Sequence[Dict[str, Any]]): Círculos detectados usados para
            delimitar ROI e sementes iniciais.
        radius_reduction_factor (float): Fração do raio usada nos discos iniciais do
            contorno.
        roi_margin (int): Margem em voxels ao redor dos círculos quando
            ``use_roi=True``.
        use_roi (bool): Se ``True``, recorta volume e gradiente à ROI local.
        alpha (float): Inclinação do potencial inverso do gradiente
            (``inverse_gaussian_gradient``).
        sigma (float): Desvio-padrão gaussiano (em voxels) aplicado antes do gradiente.
        use_gpu (bool): Se ``True``, calcula o potencial de borda na GPU quando
            disponível.

    Returns:
        tuple[NDArray[Any], NDArray[Any], Dict[str, int] | None]: Tupla
            ``(gimage, init_level_set, roi_bounds)`` pronta para evolução do MorphGAC.
    """
    # Etapa 1: restringe o processamento à ROI ao redor dos círculos para economizar memória/tempo.
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

    # Etapa 2: cria o contorno inicial (discos menores para o level set expandir até a parede).
    init_level_set = _initialize_level_set_from_circles(
        work_volume.shape,
        work_circles,
        radius_reduction_factor,
    )

    # Etapa 3: calcula o mapa de bordas inverso (calculado uma única vez para a evolução).
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
    """Reinsere uma máscara local da ROI nas coordenadas do volume completo.

    Args:
        work_mask (NDArray[Any]): Máscara binária gerada na ROI local ou já no volume
            completo.
        volume_shape (Sequence[int]): Dimensões ``(y, x, z)`` do volume completo.
        roi_bounds (Dict[str, int] | None): Limites da ROI no volume completo; ``None``
            indica máscara já global.

    Returns:
        NDArray[np.uint8]: Máscara binária ``uint8`` alinhada às dimensões globais
            ``volume_shape``.
    """
    if roi_bounds is None:
        return np.asarray(work_mask, dtype=np.uint8)

    full_mask = np.zeros(volume_shape, dtype=np.uint8)
    full_mask[
        roi_bounds["y_min"] : roi_bounds["y_max"],
        roi_bounds["x_min"] : roi_bounds["x_max"],
        roi_bounds["z_min"] : roi_bounds["z_max"],
    ] = work_mask
    return full_mask


def _evolve_level_set(
    gimage: NDArray[Any],
    init_level_set: NDArray[Any],
    num_iter: int,
    smoothing: int,
    balloon: float,
    threshold: Any,
) -> NDArray[np.uint8]:
    """Executa um bloco do MorphGAC a partir da máscara recebida.

    Args:
        gimage (NDArray[Any]): Mapa de potencial inverso de bordas usado pelo MorphGAC.
        init_level_set (NDArray[Any]): Máscara binária inicial ou intermediária do
            contorno ativo.
        num_iter (int): Número de iterações de evolução do contorno.
        smoothing (int): Passos de suavização morfológica por iteração.
        balloon (float): Força de expansão (``> 0``) ou contração (``< 0``) aplicada ao
            contorno.
        threshold (Any): Limiar de parada sobre o potencial de borda (``"auto"`` ou
            valor numérico).

    Returns:
        NDArray[np.uint8]: Máscara binária ``uint8`` após as iterações solicitadas.
    """
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


@dataclass
class LevelSetEvolutionContext:
    """Mapa de gradiente, ROI e máscara usados pela evolução do MorphGAC."""

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
        """Evolui o estado atual do MorphGAC e retorna a máscara em coordenadas globais.

        Args:
            num_iter (int): Número não negativo de iterações adicionais a evoluir.
            smoothing (int): Passos de suavização morfológica aplicados a cada iteração.
            balloon (float): Força de expansão (``> 0``) ou contração (``< 0``) do
                contorno.
            threshold (Any): Limiar de parada no mapa de potencial de borda (``"auto"``
                ou float).

        Returns:
            NDArray[np.uint8]: Máscara binária ``uint8`` 3D reinserida no volume
                completo.
        """
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

    def full_mask(self) -> NDArray[np.uint8]:
        """Restaura a máscara atual de trabalho no volume completo.

        Returns:
            NDArray[np.uint8]: Máscara binária ``uint8`` com shape
                ``self.volume_shape``.
        """
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
) -> LevelSetEvolutionContext:
    """Prepara contexto reutilizável de evolução do level set para um volume 3D.

    Args:
        volume_ccta (NDArray[Any]): Volume 3D pré-processado no formato ``(y, x, z)``.
        detected_circles (Sequence[Dict[str, Any]]): Círculos da aorta contendo
            ``slice_index``, ``center_x``, ``center_y`` e ``radius``.
        radius_reduction_factor (float): Fração do raio usada nos discos iniciais do
            contorno.
        roi_margin (int): Margem em voxels adicionada à ROI ao redor dos círculos.
        use_roi (bool): Se ``True``, recorta gradiente e evolução à ROI ao redor dos
            círculos.
        alpha (float): Inclinação do potencial inverso do gradiente
            (``inverse_gaussian_gradient``).
        sigma (float): Desvio-padrão gaussiano (em voxels) no cálculo do potencial de
            borda.
        use_gpu (bool): Se ``True``, calcula o gradiente na GPU quando disponível.

    Returns:
        LevelSetEvolutionContext: Instância de ``LevelSetEvolutionContext`` pronta para
            chamadas de ``evolve()``.
    """
    # Garante volume 3D e lista não vazia de círculos antes de montar a ROI.
    if volume_ccta.ndim != 3:
        raise ValueError(f"volume_ccta deve ser 3D, recebido shape={volume_ccta.shape}")
    if not detected_circles:
        raise ValueError("detected_circles não pode ser vazio")
    # Pré-computa potencial de bordas, sementes e limites da ROI compartilhados.
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

    Limita a máscara da aorta à região acompanhada pelo rastreamento circular,
    interpolando centros e raios nas fatias intermediárias e prolongando os extremos.

    Args:
        volume_shape (Sequence[int]): Dimensões ``(y, x, z)`` do volume de trabalho.
        detected_circles (Sequence[Dict[str, Any]]): Círculos rastreados com
            ``slice_index``, ``center_x``, ``center_y`` e ``radius``.
        radius_factor (float): Fator multiplicativo ``k`` aplicado ao raio de cada
            círculo no tubo ``k*r``.
        axial_margin_slices (int): Número de fatias extras mantidas antes do primeiro e
            após o último círculo.

    Returns:
        NDArray[np.uint8]: Máscara binária ``uint8`` 3D do envelope no shape
            ``volume_shape``.
    """
    if radius_factor <= 0:
        raise ValueError("radius_factor deve ser maior que zero")
    if axial_margin_slices < 0:
        raise ValueError("axial_margin_slices deve ser zero ou maior")
    if not detected_circles:
        return np.zeros(volume_shape, dtype=np.uint8)

    # Indexa o melhor círculo por fatia dentro dos limites axiais do volume.
    circles_by_slice: dict[int, Dict[str, Any]] = {}
    for circle in detected_circles:
        slice_index = int(circle["slice_index"])
        if 0 <= slice_index < volume_shape[2]:
            circles_by_slice[slice_index] = circle
    if not circles_by_slice:
        return np.zeros(volume_shape, dtype=np.uint8)

    # Interpola centro e raio ao longo das fatias intermediárias estendendo pelas margens axiais.
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

    # Rasteriza os discos escalados por radius_factor para formar o tubo de restrição.
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
    """Remove da máscara voxels fora do envelope da trajetória circular.

    Args:
        aorta_mask (NDArray[Any]): Máscara binária 3D da aorta a restringir.
        detected_circles (Sequence[Dict[str, Any]]): Círculos rastreados que definem o
            eixo e o raio do tubo.
        radius_factor (float): Fator multiplicativo ``k`` aplicado ao raio dos círculos
            no envelope ``k*r``.
        axial_margin_slices (int): Fatias extras preservadas antes do primeiro e após o
            último círculo.

    Returns:
        NDArray[np.uint8]: Máscara binária ``uint8`` intersectada com o envelope da
            trajetória.
    """
    envelope = build_circle_trajectory_envelope(
        aorta_mask.shape,
        detected_circles,
        radius_factor=radius_factor,
        axial_margin_slices=axial_margin_slices,
    )
    return (aorta_mask.astype(bool) & envelope.astype(bool)).astype(np.uint8)


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
    """Segmenta a aorta em 3D por MorphGAC inicializado nos círculos detectados.

    Args:
        volume_ccta (NDArray[Any]): Volume 3D pré-processado no formato ``(y, x, z)``.
        detected_circles (Sequence[Dict[str, Any]]): Círculos da aorta com
            ``slice_index``, ``center_x``, ``center_y`` e ``radius``.
        num_iter (int): Número de iterações de evolução do contorno ativo.
        smoothing (int): Passos de suavização morfológica aplicados por iteração.
        balloon (int): Força de expansão (``> 0``) ou contração (``< 0``) do contorno.
        threshold (Any): Limiar de parada sobre o potencial de borda (``"auto"`` ou
            float).
        radius_reduction_factor (float): Fração do raio usada nos discos iniciais do
            contorno.
        roi_margin (int): Margem em voxels adicionada à ROI ao redor dos círculos.
        use_roi (bool): Se ``True``, evolui apenas na ROI e reinsere o resultado no
            volume completo.
        alpha (float): Inclinação do potencial inverso do gradiente
            (``inverse_gaussian_gradient``).
        sigma (float): Desvio-padrão gaussiano (em voxels) no cálculo do potencial de
            borda.
        use_gpu (bool): Se ``True``, usa aceleração GPU no cálculo do potencial quando
            disponível.

    Returns:
        NDArray[Any]: Máscara binária 3D da aorta no mesmo shape de ``volume_ccta``.
    """
    # Valida o volume 3D e retorna máscara vazia caso não haja círculos-semente.
    if volume_ccta.ndim != 3:
        raise ValueError(f"volume_ccta deve ser 3D, recebido shape={volume_ccta.shape}")
    if not detected_circles:
        return np.zeros_like(volume_ccta, dtype=np.int8)

    # Prepara potencial de bordas e sementes na ROI e executa o MorphGAC.
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


def calculate_circle_mask_profile(
    aorta_mask: NDArray[Any],
    detected_circles: Sequence[Dict[str, Any]],
) -> list[Dict[str, float | int]]:
    """Calcula preenchimento e razão área/círculo para cada fatia rastreada.

    Args:
        aorta_mask (NDArray[Any]): Máscara binária 3D da aorta no formato ``(y, x, z)``.
        detected_circles (Sequence[Dict[str, Any]]): Círculos rastreados com
            ``slice_index``, ``center_x``, ``center_y`` e ``radius``.

    Returns:
        list[Dict[str, float | int]]: Lista de dicionários por fatia com
            ``slice_index``, ``circle_fill_ratio`` e ``circle_area_ratio``.
    """
    mask = np.asarray(aorta_mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("aorta_mask deve ser uma máscara 3D")

    # Pré-calcula a área segmentada por fatia para comparar contra os discos.
    profile: list[Dict[str, float | int]] = []
    height, width, depth = mask.shape
    segmented_area_by_slice = mask.sum(axis=(0, 1), dtype=np.int64)
    for circle in detected_circles:
        slice_index = int(circle["slice_index"])
        if not 0 <= slice_index < depth:
            continue

        # Compara o preenchimento interno ao disco com a área segmentada total na fatia.
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
    """Resume o preenchimento dos círculos e o excesso de área da máscara.

    Args:
        aorta_mask (NDArray[Any]): Máscara binária 3D da aorta no formato ``(y, x, z)``.
        detected_circles (Sequence[Dict[str, Any]]): Círculos rastreados usados como
            referência axial por fatia.

    Returns:
        Dict[str, float | None]: Dicionário com ``circle_fill_q25`` e
            ``circle_area_ratio_p90`` (ou ``None`` sem fatias válidas).
    """
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


def classify_aorta_segmentation_feedback(
    circle_fill_q25: float | None,
    circle_area_ratio_p90: float | None,
    volume_fraction: float | None,
    feedback_config: Dict[str, Any] | None = None,
) -> str:
    """Classifica a qualidade provável da máscara sem alterar a segmentação.

    Combina o perfil espacial nos círculos com a fração de volume global como
    alerta automático (não substitui inspeção visual ou máscara de referência).

    Args:
        circle_fill_q25 (float | None): Quartil inferior do preenchimento interno dos
            discos.
        circle_area_ratio_p90 (float | None): Percentil 90 da razão entre área
            segmentada e área circular.
        volume_fraction (float | None): Fração do volume total ocupada pela máscara da
            aorta.
        feedback_config (Dict[str, Any] | None): Limiares opcionais em
            ``LEVEL_SET.quality_feedback``; ``None`` usa os valores padrão.

    Returns:
        str: Código de status (``adequate``, ``suspected_oversegmentation``,
            ``suspected_undersegmentation`` ou ``insufficient_data``).
    """
    # Exige métricas finitas antes de avaliar critérios de super ou subsegmentação.
    config = feedback_config or {}
    if (
        circle_fill_q25 is None
        or circle_area_ratio_p90 is None
        or volume_fraction is None
        or not np.isfinite(circle_fill_q25)
        or not np.isfinite(circle_area_ratio_p90)
        or not np.isfinite(volume_fraction)
    ):
        return AORTA_FEEDBACK_INSUFFICIENT_DATA

    fill = float(circle_fill_q25)
    area_ratio = float(circle_area_ratio_p90)
    volume = float(volume_fraction)

    # Uma razão área/círculo muito alta sinaliza vazamento localizado. Volume
    # alto reforça o alerta quando a razão está próxima do limite principal.
    over_area_ratio = float(config.get("oversegmentation_area_ratio_p90", 3.0))
    over_volume = float(config.get("oversegmentation_volume_fraction", 0.019))
    over_paired_ratio = float(config.get("oversegmentation_paired_area_ratio_p90", 2.8))
    if area_ratio > over_area_ratio or (
        volume > over_volume and area_ratio > over_paired_ratio
    ):
        return AORTA_FEEDBACK_OVERSEGMENTED

    # A subsegmentação exige concordância entre pelo menos dois sinais para
    # reduzir falsos alertas em aortas anatomicamente curtas ou pequenas.
    under_signals = (
        fill < float(config.get("undersegmentation_circle_fill_q25", 0.80)),
        area_ratio < float(config.get("undersegmentation_area_ratio_p90", 1.30)),
        volume < float(config.get("undersegmentation_volume_fraction", 0.008)),
    )
    required_signals = int(config.get("undersegmentation_required_signals", 2))
    if sum(under_signals) >= required_signals:
        return AORTA_FEEDBACK_UNDERSEGMENTED

    return AORTA_FEEDBACK_ADEQUATE


def calculate_slice_area_jump_p95(aorta_mask: NDArray[Any]) -> float:
    """Calcula o percentil 95 dos saltos relativos de área entre fatias consecutivas.

    Args:
        aorta_mask (NDArray[Any]): Máscara binária 3D da aorta no formato ``(y, x, z)``.

    Returns:
        float: Percentil 95 da variação relativa de área axial (``0.0`` se houver menos
            de 2 fatias ocupadas).
    """
    mask = np.asarray(aorta_mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("aorta_mask deve ser uma máscara 3D")

    areas = mask.sum(axis=(0, 1), dtype=np.int64).astype(float)
    occupied = np.flatnonzero(areas > 0)
    if occupied.size < 2:
        return 0.0

    # Considera o intervalo segmentado completo para capturar cortes internos.
    areas = areas[occupied[0] : occupied[-1] + 1]
    denominator = np.maximum.reduce([areas[1:], areas[:-1], np.ones(areas.size - 1)])
    jumps = np.abs(np.diff(areas)) / denominator
    return float(np.quantile(jumps, 0.95)) if jumps.size else 0.0


def remove_leaks_morphology(
    mask_3d: NDArray[Any],
    radius: int = 3,
    use_gpu: bool = False,
) -> NDArray[Any]:
    """Remove vazamentos finos e ruídos isolados por abertura esférica.

    Na CPU, restringe a operação à bounding box da máscara (folga ``radius + 1``)
    para reduzir memória e tempo; ``radius <= 0`` retorna cópia inalterada.

    Args:
        mask_3d (NDArray[Any]): Máscara binária 3D da aorta após o level set.
        radius (int): Raio esférico (em voxels) do elemento estruturante
            ``ball(radius)``; valores ``<= 0`` desativam a filtragem.
        use_gpu (bool): Se ``True``, aplica a abertura em GPU quando disponível.

    Returns:
        NDArray[Any]: Máscara binária ``uint8`` (ou cópia com dtype original se
            ``radius <= 0``).
    """
    radius = int(radius)
    if radius <= 0:
        return mask_3d.copy()

    kernel = np.asarray(ball(radius))

    # Remove conexões finas/vazamentos com abertura morfológica na ROI da máscara.
    if not bool(use_gpu):
        bbox = mask_voxel_bbox_slices(mask_3d, radius + 1)
        if bbox is None:
            return np.zeros(mask_3d.shape, dtype=np.uint8)
        opened_sub = binary_opening(mask_3d[bbox], structure=kernel, gpu=False)
        mask_cleaned = np.zeros(mask_3d.shape, dtype=np.uint8)
        mask_cleaned[bbox] = opened_sub
        return mask_cleaned

    return binary_opening(mask_3d, structure=kernel, gpu=True)
