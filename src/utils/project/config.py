"""Utilities para carregar/salvar e normalizar configurações JSON."""

import copy
import json
import os

import numpy as np


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
    """Atualiza um dicionário recursivamente (merge profundo)."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_update_dict(base[key], value)
        else:
            base[key] = value
    return base


def normalize_runtime_config(config):
    """Converte tipos carregados de JSON para tipos esperados em runtime."""
    cfg = copy.deepcopy(config)

    if "DOWNSCALE_FACTORS" in cfg and isinstance(cfg["DOWNSCALE_FACTORS"], list):
        cfg["DOWNSCALE_FACTORS"] = tuple(cfg["DOWNSCALE_FACTORS"])

    circle_cfg = cfg.get("CIRCLE_DETECTION", {})
    if "quadrant_offset" in circle_cfg and isinstance(
        circle_cfg["quadrant_offset"], list
    ):
        circle_cfg["quadrant_offset"] = tuple(circle_cfg["quadrant_offset"])

    if "VESSELNESS_AORTA" in cfg and "sigmas" in cfg["VESSELNESS_AORTA"]:
        cfg["VESSELNESS_AORTA"]["sigmas"] = np.array(cfg["VESSELNESS_AORTA"]["sigmas"])

    if "VESSELNESS_ARTERY" in cfg and "sigmas" in cfg["VESSELNESS_ARTERY"]:
        cfg["VESSELNESS_ARTERY"]["sigmas"] = np.array(
            cfg["VESSELNESS_ARTERY"]["sigmas"]
        )

    return cfg


def serialize_config_for_json(config):
    """Converte config runtime em estrutura serializável para JSON."""
    cfg = copy.deepcopy(config)

    if "DOWNSCALE_FACTORS" in cfg and isinstance(cfg["DOWNSCALE_FACTORS"], tuple):
        cfg["DOWNSCALE_FACTORS"] = list(cfg["DOWNSCALE_FACTORS"])

    circle_cfg = cfg.get("CIRCLE_DETECTION", {})
    if "quadrant_offset" in circle_cfg and isinstance(
        circle_cfg["quadrant_offset"], tuple
    ):
        circle_cfg["quadrant_offset"] = list(circle_cfg["quadrant_offset"])

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
    """Carrega configuração JSON e aplica sobre base_config."""
    with open(path, "r", encoding="utf-8") as f:
        file_cfg = json.load(f)

    merged = copy.deepcopy(base_config)
    deep_update_dict(merged, file_cfg)
    return normalize_runtime_config(merged)


def save_config_json(config, path):
    """Salva configuração (normalizada) em arquivo JSON."""
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialize_config_for_json(config), f, indent=2, ensure_ascii=False)


def scale_config_to_resolution(
    config,
    reference_downscale_xy=2,
    enabled_groups=None,
):
    """
    Escala parâmetros em pixels para a resolução definida em DOWNSCALE_FACTORS.

    Os parâmetros padrão (em pipeline_config.json) são definidos para MID RESOLUTION
    (factor_xy=2). Esta função escala os parâmetros para outras resoluções:
    - MID RESOLUTION (factor=2): usa valores padrão (18-31 pixels)
    - HIGH RESOLUTION (factor=1): multiplica por 2 (36-62 pixels)

    Args:
        config: Dicionário de configuração original
        reference_downscale_xy: Fator de downscale de referência (padrão: 2).
        enabled_groups: Grupos de escala que serão aplicados. ``None`` mantém
            o comportamento histórico e ativa todos os grupos.

    Returns:
        Configuração escalada com todos os parâmetros espaciais ajustados
    """
    cfg = copy.deepcopy(config)
    groups = (
        set(RESOLUTION_SCALING_GROUPS)
        if enabled_groups is None
        else set(enabled_groups)
    )
    unknown_groups = groups.difference(RESOLUTION_SCALING_GROUPS)
    if unknown_groups:
        raise ValueError(
            "Grupos de escala desconhecidos: "
            f"{', '.join(sorted(unknown_groups))}"
        )

    factor_xy = cfg["DOWNSCALE_FACTORS"][0]
    scale = reference_downscale_xy / factor_xy
    scale_area = scale**2

    # =========================================================================
    # AJUSTES ESPECIAIS POR RESOLUÇÃO (HIGH = factor 1, MID = factor 2)
    # =========================================================================
    if "region_growing_heuristics" in groups and "REGION_GROWING" in cfg:
        if factor_xy == 1:  # HIGH RESOLUTION (sem downscale)
            # Mais iterações e critérios mais rigorosos para alta resolução
            cfg["REGION_GROWING"]["threshold_divisor"] = 12
            cfg["REGION_GROWING"]["min_vesselness_fraction"] = 0.05

    if "level_set_iterations" in groups and "LEVEL_SET" in cfg:
        if factor_xy == 1:  # HIGH RESOLUTION (sem downscale)
            original_num_iter = int(cfg["LEVEL_SET"]["num_iter"])
            cfg["LEVEL_SET"]["num_iter"] = 70

            # Mantém as proporções temporais do controle adaptativo quando o
            # checkpoint nominal é ampliado para alta resolução.
            adaptive = cfg["LEVEL_SET"].get("adaptive")
            if adaptive and original_num_iter > 0:
                iteration_scale = cfg["LEVEL_SET"]["num_iter"] / original_num_iter
                for key in (
                    "min_iter",
                    "check_interval",
                    "early_stop_iteration",
                    "permissive_start_iteration",
                ):
                    if key in adaptive:
                        adaptive[key] = max(
                            1,
                            int(round(adaptive[key] * iteration_scale)),
                        )
                permissive = adaptive.get("permissive")
                if permissive and "target_iterations" in permissive:
                    permissive["target_iterations"] = max(
                        1,
                        int(round(permissive["target_iterations"] * iteration_scale)),
                    )
                if "oversegmented_voxels_per_slice" in adaptive:
                    adaptive["oversegmented_voxels_per_slice"] *= scale_area

    # Se não há scaling necessário, retorna configuração sem mudanças
    if scale == 1.0:
        return cfg

    # =========================================================================
    # PARÂMETROS DE DETECÇÃO DE CÍRCULOS (Transformada de Hough)
    # =========================================================================
    # Raios de busca: escalados proporcionalmente ao fator de resolução
    # Para HIGH RESOLUTION (factor=1): multiplica por 2 (18→36, 31→62)
    if "circle_geometry" in groups:
        cfg["CIRCLE_DETECTION"]["radii_start_px"] *= scale
        cfg["CIRCLE_DETECTION"]["radii_end_px"] *= scale
        cfg["CIRCLE_DETECTION"]["radius_step_px"] = (
            cfg["CIRCLE_DETECTION"].get("radius_step_px", 1) * scale
        )

        # O offset também representa uma distância no plano axial.
        qx, qy = cfg["CIRCLE_DETECTION"]["quadrant_offset"]
        cfg["CIRCLE_DETECTION"]["quadrant_offset"] = (
            int(round(qx * scale)),
            int(round(qy * scale)),
        )

    if "circle_tracking" in groups:
        # Parâmetros auxiliares do rastreamento ainda precisam ser validados em high.
        cfg["CIRCLE_DETECTION"]["canny_sigma"] *= scale
        cfg["CIRCLE_DETECTION"]["neighbor_distance_threshold"] *= scale
        cfg["CIRCLE_DETECTION"]["local_roi_padding"] = round(
            cfg["CIRCLE_DETECTION"]["local_roi_padding"] * scale
        )

    # =========================================================================
    # PARÂMETROS DE SEGMENTAÇÃO DA AORTA (Level Set)
    # =========================================================================
    if "level_set_morphology" in groups:
        cfg["LEVEL_SET"]["leak_removal_radius"] = max(
            1, round(cfg["LEVEL_SET"]["leak_removal_radius"] * scale)
        )

    # =========================================================================
    # PARÂMETROS DE DETECÇÃO DE ÓSTIOS
    # =========================================================================
    # Raio de erosão para extração de superfície
    if "ostia_surface" in groups:
        cfg["OSTIA_DETECTION"]["erosion_radius"] = max(
            1, round(cfg["OSTIA_DETECTION"]["erosion_radius"] * scale)
        )

    # Número de candidatos (escala com área)
    if "ostia_candidates" in groups:
        cfg["OSTIA_DETECTION"]["top_n"] = round(
            cfg["OSTIA_DETECTION"]["top_n"] * scale_area
        )

    # =========================================================================
    # PARÂMETROS DE CRESCIMENTO DE REGIÃO (Region Growing)
    # =========================================================================
    # Volume máximo (escala com área)
    if "region_growing_geometry" in groups:
        cfg["REGION_GROWING"]["max_volume"] = round(
            cfg["REGION_GROWING"]["max_volume"] * scale_area
        )

        # O eixo z não sofre downsampling; por isso volumes escalam com a área XY.
        cfg["REGION_GROWING"]["switch_at_voxels"] = round(
            cfg["REGION_GROWING"]["switch_at_voxels"] * scale_area
        )
        for key in ("seed_search_radius", "seed_candidate_radius"):
            if key in cfg["REGION_GROWING"]:
                cfg["REGION_GROWING"][key] = max(
                    1, round(cfg["REGION_GROWING"][key] * scale)
                )

    # =========================================================================
    # PARÂMETROS DE FUZZY CONNECTEDNESS
    # =========================================================================
    if "fuzzy_connectedness_geometry" in groups and "FUZZY_CONNECTEDNESS" in cfg:
        fc_config = cfg["FUZZY_CONNECTEDNESS"]
        if "seed_search_radius" in fc_config:
            fc_config["seed_search_radius"] = max(
                1, round(fc_config["seed_search_radius"] * scale)
            )
        for key in ("max_candidate_voxels", "max_processed_voxels"):
            if fc_config.get(key) is not None:
                fc_config[key] = round(fc_config[key] * scale_area)

    # =========================================================================
    # PARÂMETROS DE PÓS-PROCESSAMENTO
    # =========================================================================
    # Operações morfológicas de fechamento e dilatação
    if "postprocessing_morphology" in groups:
        cfg["POSTPROCESSING"]["closing_radius"] = max(
            1, round(cfg["POSTPROCESSING"]["closing_radius"] * scale)
        )
        cfg["POSTPROCESSING"]["dilation_radius"] = max(
            1, round(cfg["POSTPROCESSING"]["dilation_radius"] * scale)
        )

    return cfg
