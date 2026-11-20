"""Carrega, salva e normaliza configurações JSON."""

import copy
import json
from pathlib import Path

import numpy as np

from ..utils.json_io import save_json_atomic

IMAGE_GRID_LABEL_PROTOCOL = "nearest_on_image_grid_v1"
LEGACY_LABEL_PROTOCOL = "legacy_zoom_order0"


RESOLUTION_SCALING_GROUPS = frozenset(
    {
        "circle_geometry",
        "circle_tracking",
        "level_set_iterations",
        "level_set_morphology",
        "ostia_surface",
        "ostia_candidates",
        "region_growing_heuristics",
        "region_growing_geometry",
        "fuzzy_connectedness_geometry",
        "postprocessing_morphology",
    }
)


def deep_update_dict(base, updates):
    """Atualiza um dicionário recursivamente (merge profundo).

    Args:
        base (dict[str, Any]): Dicionário base atualizado recursivamente no local.
        updates (dict[str, Any]): Dicionário com chaves e subestruturas a mesclar.

    Returns:
        dict[str, Any]: O próprio dicionário base após a atualização.
    """
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_update_dict(base[key], value)
        else:
            base[key] = value
    return base


def normalize_runtime_config(config):
    """Converte tipos carregados de JSON para tipos esperados em runtime.

    Args:
        config (dict[str, Any]): Configuração bruta desserializada de JSON.

    Returns:
        dict[str, Any]: Cópia profunda com tuplas e arrays NumPy restaurados.
    """
    # Cria cópia isolada para não mutar a configuração original recebida.
    cfg = copy.deepcopy(config)

    # Restaura tuplas geométricas convertidas em listas pela serialização JSON.
    if "DOWNSCALE_FACTORS" in cfg and isinstance(cfg["DOWNSCALE_FACTORS"], list):
        cfg["DOWNSCALE_FACTORS"] = tuple(cfg["DOWNSCALE_FACTORS"])

    circle_cfg = cfg.get("CIRCLE_DETECTION", {})
    if "quadrant_offset" in circle_cfg and isinstance(
        circle_cfg["quadrant_offset"], list
    ):
        circle_cfg["quadrant_offset"] = tuple(circle_cfg["quadrant_offset"])

    # Converte listas de sigmas dos filtros vesselness para arrays NumPy.
    if "VESSELNESS_AORTA" in cfg and "sigmas" in cfg["VESSELNESS_AORTA"]:
        cfg["VESSELNESS_AORTA"]["sigmas"] = np.array(cfg["VESSELNESS_AORTA"]["sigmas"])

    if "VESSELNESS_ARTERY" in cfg and "sigmas" in cfg["VESSELNESS_ARTERY"]:
        cfg["VESSELNESS_ARTERY"]["sigmas"] = np.array(
            cfg["VESSELNESS_ARTERY"]["sigmas"]
        )

    return cfg


def serialize_config_for_json(config):
    """Converte config runtime em estrutura serializável para JSON.

    Args:
        config (dict[str, Any]): Dicionário de configuração em formato de execução.

    Returns:
        dict[str, Any]: Cópia profunda com tipos nativos compatíveis com JSON.
    """
    # Cria cópia isolada para preservar tuplas e arrays da config em execução.
    cfg = copy.deepcopy(config)

    # Converte tuplas de redução e deslocamento em listas JSON.
    if "DOWNSCALE_FACTORS" in cfg and isinstance(cfg["DOWNSCALE_FACTORS"], tuple):
        cfg["DOWNSCALE_FACTORS"] = list(cfg["DOWNSCALE_FACTORS"])

    circle_cfg = cfg.get("CIRCLE_DETECTION", {})
    if "quadrant_offset" in circle_cfg and isinstance(
        circle_cfg["quadrant_offset"], tuple
    ):
        circle_cfg["quadrant_offset"] = list(circle_cfg["quadrant_offset"])

    # Converte arrays NumPy de sigmas em listas nativas antes da gravação.
    if "VESSELNESS_AORTA" in cfg and "sigmas" in cfg["VESSELNESS_AORTA"]:
        if hasattr(cfg["VESSELNESS_AORTA"]["sigmas"], "tolist"):
            cfg["VESSELNESS_AORTA"]["sigmas"] = cfg["VESSELNESS_AORTA"][
                "sigmas"
            ].tolist()

    if "VESSELNESS_ARTERY" in cfg and "sigmas" in cfg["VESSELNESS_ARTERY"]:
        if hasattr(cfg["VESSELNESS_ARTERY"]["sigmas"], "tolist"):
            cfg["VESSELNESS_ARTERY"]["sigmas"] = cfg["VESSELNESS_ARTERY"][
                "sigmas"
            ].tolist()

    return cfg


def load_config_json(path, base_config):
    """Carrega configuração JSON e aplica sobre ``base_config``.

    Args:
        path (str | Path): Caminho do arquivo JSON de configuração.
        base_config (dict[str, Any]): Configuração padrão usada como base do merge.

    Returns:
        dict[str, Any]: Configuração mesclada e normalizada para runtime.
    """
    with open(path, "r", encoding="utf-8") as f:
        file_cfg = json.load(f)

    # Snapshots antigos sem LABEL_RESAMPLING usam o protocolo legado de reamostragem.
    if Path(path).name == "effective_pipeline_config.json":
        file_cfg.setdefault("LABEL_RESAMPLING", LEGACY_LABEL_PROTOCOL)
    merged = copy.deepcopy(base_config)
    deep_update_dict(merged, file_cfg)
    # Configs modernas com DOWNSCALE_METHOD assumem alinhamento na grade da imagem.
    if "DOWNSCALE_METHOD" in merged:
        merged.setdefault("LABEL_RESAMPLING", IMAGE_GRID_LABEL_PROTOCOL)
    return normalize_runtime_config(merged)


def save_config_json(config, path):
    """Salva configuração normalizada em arquivo JSON de forma atômica.

    Args:
        config (dict[str, Any]): Dicionário de configuração em formato de execução.
        path (str | Path): Caminho do arquivo JSON de destino.
    """
    save_json_atomic(serialize_config_for_json(config), path)


def scale_config_to_resolution(
    config,
    reference_downscale_xy=2,
    enabled_groups=None,
):
    """Escala raios, margens, sigmas e limites volumétricos em px para ``DOWNSCALE_FACTORS``.

    Os valores-base em ``pipeline_config.json`` são calibrados para resolução média
    (fator XY = 2) e não devem ser reescalados mais de uma vez; o eixo Z não muda.
    ``enabled_groups=None`` aplica todos os grupos registrados.

    Args:
        config (dict[str, Any]): Configuração base calibrada na resolução de referência.
        reference_downscale_xy (int | float): Fator XY de referência (padrão ``2``).
        enabled_groups (Iterable[str] | None): Grupos de escala ativos ou ``None`` para
            aplicar todos.

    Returns:
        dict[str, Any]: Cópia profunda da configuração ajustada à resolução alvo.
    """
    # Valida grupos solicitados e calcula fatores de escala linear e de área XY.
    cfg = copy.deepcopy(config)
    cfg.setdefault("LABEL_RESAMPLING", IMAGE_GRID_LABEL_PROTOCOL)
    groups = (
        set(RESOLUTION_SCALING_GROUPS)
        if enabled_groups is None
        else set(enabled_groups)
    )
    unknown_groups = groups.difference(RESOLUTION_SCALING_GROUPS)
    if unknown_groups:
        raise ValueError(
            f"Grupos de escala desconhecidos: {', '.join(sorted(unknown_groups))}"
        )

    factor_xy = cfg["DOWNSCALE_FACTORS"][0]
    scale = reference_downscale_xy / factor_xy
    scale_area = scale**2

    # Ajustes discretos específicos para resolução alta (fator XY = 1).
    if "region_growing_heuristics" in groups and "REGION_GROWING" in cfg:
        if factor_xy == 1:
            cfg["REGION_GROWING"]["threshold_divisor"] = 12
            cfg["REGION_GROWING"]["min_vesselness_fraction"] = 0.05

    if "level_set_iterations" in groups and "LEVEL_SET" in cfg:
        if factor_xy == 1:
            cfg["LEVEL_SET"]["num_iter"] = 70

    if scale == 1.0:
        return cfg

    # Raios de Hough, passo e deslocamento de quadrante escalam linearmente no plano XY.
    if "circle_geometry" in groups:
        cfg["CIRCLE_DETECTION"]["radii_start_px"] *= scale
        cfg["CIRCLE_DETECTION"]["radii_end_px"] *= scale
        cfg["CIRCLE_DETECTION"]["radius_step_px"] = (
            cfg["CIRCLE_DETECTION"].get("radius_step_px", 1) * scale
        )

        qx, qy = cfg["CIRCLE_DETECTION"]["quadrant_offset"]
        cfg["CIRCLE_DETECTION"]["quadrant_offset"] = (
            int(round(qx * scale)),
            int(round(qy * scale)),
        )

    # Parâmetros de rastreamento, level set e óstios escalam linearmente no plano XY.
    if "circle_tracking" in groups:
        cfg["CIRCLE_DETECTION"]["canny_sigma"] *= scale
        cfg["CIRCLE_DETECTION"]["neighbor_distance_threshold"] *= scale
        cfg["CIRCLE_DETECTION"]["local_roi_padding"] = round(
            cfg["CIRCLE_DETECTION"]["local_roi_padding"] * scale
        )

    if "level_set_morphology" in groups:
        cfg["LEVEL_SET"]["leak_removal_radius"] = max(
            1, round(cfg["LEVEL_SET"]["leak_removal_radius"] * scale)
        )

    if "ostia_surface" in groups:
        cfg["OSTIA_DETECTION"]["erosion_radius"] = max(
            1, round(cfg["OSTIA_DETECTION"]["erosion_radius"] * scale)
        )

    # Contagens por fatia e volumes escalam com a área XY porque o eixo Z permanece fixo.
    if "ostia_candidates" in groups:
        cfg["OSTIA_DETECTION"]["top_n"] = round(
            cfg["OSTIA_DETECTION"]["top_n"] * scale_area
        )

    # Limites volumétricos escalam por área XY e raios de busca linearmente.
    if "region_growing_geometry" in groups:
        cfg["REGION_GROWING"]["max_volume"] = round(
            cfg["REGION_GROWING"]["max_volume"] * scale_area
        )
        cfg["REGION_GROWING"]["switch_at_voxels"] = round(
            cfg["REGION_GROWING"]["switch_at_voxels"] * scale_area
        )
        for key in ("seed_search_radius", "seed_candidate_radius"):
            if key in cfg["REGION_GROWING"]:
                cfg["REGION_GROWING"][key] = max(
                    1, round(cfg["REGION_GROWING"][key] * scale)
                )

    if "fuzzy_connectedness_geometry" in groups and "FUZZY_CONNECTEDNESS" in cfg:
        fc_config = cfg["FUZZY_CONNECTEDNESS"]
        if "seed_search_radius" in fc_config:
            fc_config["seed_search_radius"] = max(
                1, round(fc_config["seed_search_radius"] * scale)
            )
        for key in ("max_candidate_voxels", "max_processed_voxels"):
            if fc_config.get(key) is not None:
                fc_config[key] = round(fc_config[key] * scale_area)

    if "postprocessing_morphology" in groups:
        cfg["POSTPROCESSING"]["closing_radius"] = max(
            1, round(cfg["POSTPROCESSING"]["closing_radius"] * scale)
        )
        cfg["POSTPROCESSING"]["dilation_radius"] = max(
            1, round(cfg["POSTPROCESSING"]["dilation_radius"] * scale)
        )

    return cfg
