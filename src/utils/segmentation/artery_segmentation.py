"""Crescimento de região para segmentação de artérias coronárias.

Entrada pública: ``normal_region_growing_from_ostia``, segmentação usada pelo
pipeline.

No crescimento padrão, ``comparison_window`` define a referência de vesselness:
``1`` usa o voxel atual, ``ALL``/``-1`` usa a média de toda a região aceita e um
inteiro maior que um usa a média dos últimos N voxels aceitos.

A conectividade pode considerar 6, 18 ou 26 vizinhos. A vizinhança 26 contém
todos os voxels do cubo 3 x 3 x 3 ao redor do voxel atual, exceto o próprio
centro: 6 compartilham uma face, 12 compartilham uma aresta e 8 compartilham
somente um vértice.

As demais funções são helpers privados usados para validar e selecionar
sementes e controlar os critérios de aceitação do crescimento.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

NEIGHBORS_26 = tuple(
    (dy, dx, dz)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dy, dx, dz) != (0, 0, 0)
)


def _neighbor_offsets(neighborhood: int) -> tuple[tuple[int, int, int], ...]:
    """Retorna os deslocamentos da vizinhança 3D solicitada.

    Args:
        neighborhood (int): Conectividade 3D desejada (``6``, ``18`` ou ``26``).

    Returns:
        tuple[tuple[int, int, int], ...]: Deslocamentos inteiros ``(dy, dx, dz)``
            correspondentes à conectividade escolhida.
    """
    if neighborhood not in {6, 18, 26}:
        raise ValueError("neighborhood deve ser 6, 18 ou 26")

    offsets = []
    for offset in NEIGHBORS_26:
        manhattan = sum(abs(value) for value in offset)
        if neighborhood == 6 and manhattan != 1:
            continue
        if neighborhood == 18 and manhattan > 2:
            continue
        offsets.append(offset)
    return tuple(offsets)


def _validate_seed(
    seed_point: Sequence[int] | None,
    volume_shape: Sequence[int],
) -> tuple[int, int, int] | None:
    """Retorna a semente como tupla inteira se estiver dentro do volume.

    Args:
        seed_point (Sequence[int] | None): Coordenada ``(y, x, z)`` da semente ou
            ``None``.
        volume_shape (Sequence[int]): Dimensões ``(height, width, depth)`` do volume 3D.

    Returns:
        tuple[int, int, int] | None: Coordenada inteira ``(y, x, z)`` válida dentro dos
            limites do volume, ou ``None`` se ausente ou fora do domínio.
    """
    if seed_point is None:
        return None

    sy, sx, sz = map(int, seed_point)
    height, width, depth = volume_shape
    if 0 <= sy < height and 0 <= sx < width and 0 <= sz < depth:
        return (sy, sx, sz)
    return None


def _adaptive_floor(
    count: int,
    min_start: float,
    min_end: float,
    switch_at_voxels: int,
    smooth_relaxation: bool,
) -> float:
    """Calcula o piso mínimo de vesselness no estágio atual do crescimento.

    Args:
        count (int): Quantidade de voxels já aceitos na região.
        min_start (float): Piso inicial de vesselness exigido no início do crescimento.
        min_end (float): Piso final relaxado após atingir ``switch_at_voxels``.
        switch_at_voxels (int): Limiar de voxels para transição ao piso relaxado.
        smooth_relaxation (bool): Se ``True``, interpola linearmente entre ``min_start``
            e ``min_end`` antes da transição.

    Returns:
        float: Piso mínimo de vesselness aplicável ao passo atual.
    """
    if switch_at_voxels <= 0 or count >= switch_at_voxels:
        return min_end
    if not smooth_relaxation:
        return min_start

    progress = count / switch_at_voxels
    return min_start - (min_start - min_end) * progress


def _comparison_mean(
    comparison_window: int,
    use_running_mean: bool,
    current_value: float,
    running_sum: float,
    running_count: int,
    window_sum: float,
    value_history: deque[float] | None,
) -> float:
    """Define a referência usada para comparar o próximo voxel candidato.

    Args:
        comparison_window (int): Tamanho da janela de comparação (``1`` para voxel
            atual, ``-1`` para média acumulada, ou ``> 1`` para janela deslizante).
        use_running_mean (bool): Indica se deve usar a média acumulada de toda a região.
        current_value (float): Valor de vesselness do voxel atual na fronteira.
        running_sum (float): Soma acumulada dos valores de vesselness já aceitos.
        running_count (int): Quantidade total de voxels aceitos até o momento.
        window_sum (float): Soma dos últimos valores mantidos em ``value_history``.
        value_history (deque[float] | None): Histórico recente de valores na janela.

    Returns:
        float: Valor de referência de vesselness para o teste de similaridade.
    """
    if comparison_window == 1:
        return current_value
    if use_running_mean:
        return running_sum / max(running_count, 1)
    if not value_history:
        return current_value
    return window_sum / len(value_history)


def _neighbor_is_acceptable(
    neighbor_val: float,
    comparison_value: float,
    current_floor: float,
    threshold: float,
) -> bool:
    """Verifica piso de vesselness e similaridade com a região atual.

    Args:
        neighbor_val (float): Valor de vesselness do voxel vizinho avaliado.
        comparison_value (float): Intensidade de referência da região atual.
        current_floor (float): Piso mínimo de vesselness permitido no passo atual.
        threshold (float): Tolerância máxima de diferença absoluta até
            ``comparison_value``.

    Returns:
        bool: ``True`` se o vizinho atende ao piso e à tolerância de similaridade.
    """
    if neighbor_val < current_floor:
        return False
    return abs(neighbor_val - comparison_value) <= threshold


def _collect_seed_candidates_by_score(
    seed: Sequence[int] | None,
    score_map: NDArray[Any],
    search_radius: int,
    min_seed_score: float,
    max_candidates: int,
) -> list[tuple[int, int, int]]:
    """Seleciona os melhores voxels locais para iniciar o crescimento.

    Args:
        seed (Sequence[int] | None): Coordenada ``(y, x, z)`` inicial do óstio ou
            ``None``.
        score_map (NDArray[Any]): Mapa 3D de pontuação (vesselness) em ``(y, x, z)``.
        search_radius (int): Raio cúbico em voxels ao redor da semente para busca local.
        min_seed_score (float): Pontuação mínima exigida para aceitar um candidato.
        max_candidates (int): Número máximo de sementes retornadas em ordem decrescente.

    Returns:
        list[tuple[int, int, int]]: Lista de coordenadas ``(y, x, z)`` selecionadas.
    """
    validated_seed = _validate_seed(seed, score_map.shape)
    if validated_seed is None:
        return []

    y, x, z = validated_seed
    radius = max(int(search_radius), 0)
    # Recorta uma vizinhança limitada pelo volume ao redor do óstio detectado.
    y0, y1 = max(0, y - radius), min(score_map.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(score_map.shape[1], x + radius + 1)
    z0, z1 = max(0, z - radius), min(score_map.shape[2], z + radius + 1)

    # Filtra candidatos locais acima do piso e faz fallback na própria semente.
    local_scores = score_map[y0:y1, x0:x1, z0:z1]
    candidate_mask = local_scores >= float(min_seed_score)
    if not np.any(candidate_mask):
        seed_score = float(score_map[validated_seed])
        return [validated_seed] if seed_score >= float(min_seed_score) else []

    # Ordena por vesselness para iniciar o crescimento pelos candidatos mais fortes.
    candidate_indices = np.argwhere(candidate_mask)
    candidate_scores = local_scores[candidate_mask]
    order = np.argsort(candidate_scores)[::-1]

    selected = []
    for idx in order:
        ly, lx, lz = candidate_indices[idx]
        selected.append((int(y0 + ly), int(x0 + lx), int(z0 + lz)))
        if len(selected) >= max(1, int(max_candidates)):
            break
    return selected


def _vesselness_reference(
    vesselness_map: NDArray[Any],
    ostia: Iterable[Sequence[int] | None],
    *,
    scope: str,
    radius: int,
    percentile: float,
) -> float:
    """Calcula a referência global ou local usada pelos limiares do region growing.

    Args:
        vesselness_map (NDArray[Any]): Mapa 3D de vesselness arterial ``(y, x, z)``.
        ostia (Iterable[Sequence[int] | None]): Coordenadas ``(y, x, z)`` dos óstios
            usados na amostragem local (elementos ``None`` são ignorados).
        scope (str): Escopo da referência (``"global"`` ou ``"ostia_local"``).
        radius (int): Raio cúbico em voxels ao redor de cada óstio no modo local.
        percentile (float): Percentil em ``[0, 100]`` aplicado aos valores positivos.

    Returns:
        float: Intensidade de referência para escalar os parâmetros do crescimento.
    """
    # Valida o escopo e o percentil antes de amostrar os óstios.
    global_max = float(np.max(vesselness_map))
    if scope == "global":
        return global_max
    if scope != "ostia_local":
        raise ValueError("reference_scope deve ser 'global' ou 'ostia_local'")
    if not 0 <= percentile <= 100:
        raise ValueError("reference_percentile deve estar em [0, 100]")

    # Reúne respostas em torno de cada óstio para evitar referência global excessiva.
    local_values = []
    local_radius = max(int(radius), 0)
    for ostium in ostia:
        coord = _validate_seed(ostium, vesselness_map.shape)
        if coord is None:
            continue
        y, x, z = coord
        local_values.append(
            vesselness_map[
                max(0, y - local_radius) : min(
                    vesselness_map.shape[0], y + local_radius + 1
                ),
                max(0, x - local_radius) : min(
                    vesselness_map.shape[1], x + local_radius + 1
                ),
                max(0, z - local_radius) : min(
                    vesselness_map.shape[2], z + local_radius + 1
                ),
            ].ravel()
        )

    if not local_values:
        return global_max
    # Usa percentil positivo local para reduzir sensibilidade a ruído ou stents fora do óstio.
    values = np.concatenate(local_values)
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return global_max
    return float(np.percentile(positive, percentile))


def _region_growing_parameters(
    rg_config: dict[str, Any],
    reference: float,
    map_minimum: float,
) -> dict[str, Any]:
    """Converte a referência de vesselness nos parâmetros absolutos do region growing.

    Args:
        rg_config (dict[str, Any]): Bloco ``REGION_GROWING`` da configuração efetiva.
        reference (float): Valor de referência de vesselness do exame ou ramo.
        map_minimum (float): Valor mínimo observado no mapa de vesselness do volume.

    Returns:
        dict[str, Any]: Parâmetros prontos para ``_region_growing_from_seeds``.
    """
    return {
        "threshold": (reference - map_minimum) / float(rg_config["threshold_divisor"]),
        "max_volume": int(rg_config["max_volume"]),
        "min_vesselness": reference * float(rg_config["min_vesselness_fraction"]),
        "relaxed_floor_factor": float(rg_config["relaxed_floor_factor"]),
        "switch_at_voxels": int(rg_config["switch_at_voxels"]),
        "comparison_window": rg_config["comparison_window"],
        "smooth_relaxation": bool(rg_config["smooth_relaxation"]),
        "neighborhood": int(rg_config.get("neighborhood", 26)),
        "verbose": bool(rg_config.get("verbose", False)),
    }


def _region_growing_from_seeds(
    vesselness_map: NDArray[Any],
    seeds: Iterable[Sequence[int]],
    *,
    threshold: float,
    min_vesselness: float,
    max_volume: int,
    relaxed_floor_factor: float,
    switch_at_voxels: int,
    comparison_window: int | str,
    smooth_relaxation: bool,
    neighborhood: int = 26,
    verbose: bool = False,
) -> NDArray[np.uint8]:
    """Cresce uma região usando uma ou mais sementes iniciais.

    Args:
        vesselness_map (NDArray[Any]): Mapa 3D de vesselness arterial ``(y, x, z)``.
        seeds (Iterable[Sequence[int]]): Coordenadas iniciais ``(y, x, z)`` da região.
        threshold (float): Tolerância máxima de diferença até a referência atual.
        min_vesselness (float): Piso inicial mínimo de vesselness para aceitar voxels.
        max_volume (int): Teto de voxels aceitos antes de encerrar a expansão.
        relaxed_floor_factor (float): Fator multiplicativo aplicado a ``min_vesselness``
            após atingir ``switch_at_voxels``.
        switch_at_voxels (int): Limiar de voxels aceitos para relaxar o piso mínimo.
        comparison_window (int | str): Modo de referência (``1``, ``-1``, ``"ALL"`` ou
            tamanho inteiro positivo da janela deslizante).
        smooth_relaxation (bool): Se ``True``, interpola o piso até
            ``switch_at_voxels``.
        neighborhood (int): Conectividade 3D entre vizinhos (``6``, ``18`` ou ``26``).
        verbose (bool): Se ``True``, imprime mensagens de diagnóstico no console.

    Returns:
        NDArray[np.uint8]: Máscara binária ``uint8`` 3D dos voxels segmentados.
    """
    if vesselness_map.ndim != 3:
        raise ValueError(
            f"vesselness_map deve ser 3D, recebido shape={vesselness_map.shape}"
        )

    # Configura a janela de comparação (voxel atual, janela móvel ou média global).
    use_running_mean = False
    value_history: deque[float] | None = None
    if isinstance(comparison_window, str):
        if comparison_window.strip().upper() != "ALL":
            raise ValueError(
                "comparison_window deve ser ALL, -1 ou um inteiro positivo"
            )
        comparison_window = -1

    if comparison_window == -1:
        comparison_window = -1
        use_running_mean = True
    elif comparison_window > 1:
        value_history = deque(maxlen=comparison_window)
    else:
        comparison_window = 1

    # O piso pode relaxar gradualmente depois que a região ganha suporte suficiente.
    min_start = float(min_vesselness)
    min_end = min_start * float(relaxed_floor_factor)
    max_vol = int(max_volume)
    switch_at = int(switch_at_voxels)
    smooth_relax = bool(smooth_relaxation)
    thresh = float(threshold)
    mask = np.zeros_like(vesselness_map, dtype=np.uint8)
    queue: deque[tuple[int, int, int]] = deque()
    running_sum = 0.0
    running_count = 0
    window_sum = 0.0
    neighbor_offsets = _neighbor_offsets(int(neighborhood))

    # Inicializa a fila com as sementes válidas e acima do piso mínimo.
    for seed in seeds:
        coord = _validate_seed(seed, vesselness_map.shape)
        if coord is None or mask[coord]:
            continue

        seed_val = float(vesselness_map[coord])
        if seed_val < min_start:
            continue

        mask[coord] = 1
        queue.append(coord)
        running_sum += seed_val
        running_count += 1
        if value_history is not None:
            if len(value_history) == comparison_window:
                window_sum -= value_history[0]
            value_history.append(seed_val)
            window_sum += seed_val
    if running_count == 0:
        if verbose:
            print("Nenhuma semente válida acima do piso de vesselness.")
        return mask

    # Loop BFS: expande a partir da fronteira até esvaziar a fila ou atingir max_volume.
    count = running_count
    dim0, dim1, dim2 = vesselness_map.shape
    while queue:
        if count >= max_vol:
            if verbose:
                print(f"Volume máximo atingido: {max_vol}")
            break

        cy, cx, cz = queue.popleft()
        current_value = float(vesselness_map[cy, cx, cz])
        # Atualiza a referência e o piso usados para aceitar a próxima fronteira.
        reference = _comparison_mean(
            comparison_window,
            use_running_mean,
            current_value,
            running_sum,
            running_count,
            window_sum,
            value_history,
        )
        current_floor = _adaptive_floor(
            count,
            min_start,
            min_end,
            switch_at,
            smooth_relax,
        )

        # Testa a vizinhança configurada para seguir ramos arteriais em 3D.
        for dy, dx, dz in neighbor_offsets:
            ny, nx, nz = cy + dy, cx + dx, cz + dz
            if not (0 <= ny < dim0 and 0 <= nx < dim1 and 0 <= nz < dim2):
                continue
            if mask[ny, nx, nz]:
                continue

            neighbor_value = float(vesselness_map[ny, nx, nz])
            if not _neighbor_is_acceptable(
                neighbor_value,
                reference,
                current_floor,
                thresh,
            ):
                continue
            mask[ny, nx, nz] = 1
            queue.append((ny, nx, nz))
            count += 1

            # Atualiza os acumuladores de referência conforme o modo de janela ativo.
            if use_running_mean:
                running_sum += neighbor_value
                running_count += 1
            elif value_history is not None:
                if len(value_history) == comparison_window:
                    window_sum -= value_history[0]
                value_history.append(neighbor_value)
                window_sum += neighbor_value
    if verbose:
        print(f"Voxels segmentados: {count}")
    return mask


def normal_region_growing_from_ostia(
    vesselness_artery: NDArray[Any],
    ostia_left: Sequence[int] | None,
    ostia_right: Sequence[int] | None,
    config: dict[str, Any],
    *,
    branch_diagnostics: dict[str, Any] | None = None,
) -> NDArray[np.uint8]:
    """Segmenta artérias coronárias a partir dos óstios por crescimento de região.

    Args:
        vesselness_artery (NDArray[Any]): Mapa 3D de vesselness arterial no formato
            ``(y, x, z)``.
        ostia_left (Sequence[int] | None): Coordenada ``(y, x, z)`` do óstio esquerdo,
            ou ``None`` se ausente.
        ostia_right (Sequence[int] | None): Coordenada ``(y, x, z)`` do óstio direito,
            ou ``None`` se ausente.
        config (dict[str, Any]): Configuração efetiva com o bloco ``REGION_GROWING``.
        branch_diagnostics (dict[str, Any] | None): Dicionário opcional preenchido
            in-place com contagens brutas por ramo.

    Returns:
        NDArray[np.uint8]: Máscara binária ``uint8`` 3D resultante da união das árvores
            coronárias.
    """
    if vesselness_artery.ndim != 3:
        raise ValueError(
            f"vesselness_artery deve ser 3D, recebido shape={vesselness_artery.shape}"
        )

    # Etapa 1: determina a intensidade de referência e converte parâmetros configurados para a escala do exame.
    rg_config = config["REGION_GROWING"]
    ostia = [ostia_left, ostia_right]
    reference_scope = str(rg_config.get("reference_scope", "global"))
    shared_reference_scope = (
        "ostia_local"
        if reference_scope == "ostium_local_per_branch"
        else reference_scope
    )
    max_vesselness = _vesselness_reference(
        vesselness_artery,
        ostia,
        scope=shared_reference_scope,
        radius=int(rg_config.get("reference_radius", 3)),
        percentile=float(rg_config.get("reference_percentile", 95.0)),
    )
    min_vesselness_map = float(np.min(vesselness_artery))
    if max_vesselness <= 0:
        return np.zeros_like(vesselness_artery, dtype=np.uint8)

    params = _region_growing_parameters(rg_config, max_vesselness, min_vesselness_map)

    use_seed_refinement = bool(rg_config.get("use_seed_refinement", False))
    use_multi_seed = bool(rg_config.get("use_multi_seed", False))
    grow_each_ostium_separately = bool(
        rg_config.get("grow_each_ostium_separately", True)
    )
    seed_search_radius = int(rg_config.get("seed_search_radius", 2))
    seed_candidate_radius = int(rg_config.get("seed_candidate_radius", 1))
    max_seed_candidates = int(rg_config.get("max_seed_candidates", 1))

    # Etapa 2: seleciona e opcionalmente refina/multiplica as sementes ao redor de cada óstio.
    def seeds_for_ostium(
        ostium: Sequence[int] | None,
        branch_reference: float,
    ) -> list[tuple[int, int, int]]:
        """Seleciona sementes iniciais para um óstio conforme o modo configurado."""
        if ostium is None:
            return []
        if not (use_seed_refinement or use_multi_seed):
            seed = _validate_seed(ostium, vesselness_artery.shape)
            return [] if seed is None else [seed]

        radius = seed_candidate_radius if use_multi_seed else seed_search_radius
        candidate_count = max_seed_candidates if use_multi_seed else 1
        return _collect_seed_candidates_by_score(
            ostium,
            vesselness_artery,
            radius,
            branch_reference
            * float(rg_config.get("seed_min_vesselness_fraction", 0.02)),
            candidate_count,
        )

    branch_references = [max_vesselness, max_vesselness]
    if reference_scope == "ostium_local_per_branch":
        branch_references = [
            _vesselness_reference(
                vesselness_artery,
                [ostium],
                scope="ostia_local",
                radius=int(rg_config.get("reference_radius", 3)),
                percentile=float(rg_config.get("reference_percentile", 95.0)),
            )
            for ostium in ostia
        ]

    # Mantém as sementes de cada coronária separadas até escolher a estratégia de RG.
    seed_groups = [
        seeds_for_ostium(ostia_left, branch_references[0]),
        seeds_for_ostium(ostia_right, branch_references[1]),
    ]
    if not any(seed_groups):
        return np.zeros_like(vesselness_artery, dtype=np.uint8)

    # Etapa 3: executa o region growing separado por ramo (padrão) ou conjunto.
    if grow_each_ostium_separately:
        # Evita que um ramo mais forte domine a expansão iniciada no outro óstio.
        combined = np.zeros_like(vesselness_artery, dtype=np.uint8)
        for side, seed_group, branch_reference in zip(
            ("left", "right"), seed_groups, branch_references
        ):
            branch_params = _region_growing_parameters(
                rg_config, branch_reference, min_vesselness_map
            )
            branch = _region_growing_from_seeds(
                vesselness_artery,
                seed_group,
                **branch_params,
            )
            if branch_diagnostics is not None:
                branch_diagnostics[f"artery_{side}_raw_voxels"] = int(branch.sum())

            combined |= branch
        return combined.astype(np.uint8)

    # Caso contrário, cresce todos os óstios juntos em uma única fila compartilhada.
    all_seeds = [seed for seed_group in seed_groups for seed in seed_group]
    return _region_growing_from_seeds(
        vesselness_artery,
        all_seeds,
        **params,
    ).astype(np.uint8)


__all__ = [
    "NEIGHBORS_26",
    "normal_region_growing_from_ostia",
]
