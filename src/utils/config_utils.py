"""Utilities para carregar/salvar e normalizar configurações JSON."""

import copy
import json
import os

import numpy as np


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
    if "quadrant_offset" in circle_cfg and isinstance(circle_cfg["quadrant_offset"], list):
        circle_cfg["quadrant_offset"] = tuple(circle_cfg["quadrant_offset"])

    if "VESSELNESS_AORTA" in cfg and "sigmas" in cfg["VESSELNESS_AORTA"]:
        cfg["VESSELNESS_AORTA"]["sigmas"] = np.array(cfg["VESSELNESS_AORTA"]["sigmas"])

    if "VESSELNESS_ARTERY" in cfg and "sigmas" in cfg["VESSELNESS_ARTERY"]:
        cfg["VESSELNESS_ARTERY"]["sigmas"] = np.array(cfg["VESSELNESS_ARTERY"]["sigmas"])

    return cfg


def serialize_config_for_json(config):
    """Converte config runtime em estrutura serializável para JSON."""
    cfg = copy.deepcopy(config)

    if "DOWNSCALE_FACTORS" in cfg and isinstance(cfg["DOWNSCALE_FACTORS"], tuple):
        cfg["DOWNSCALE_FACTORS"] = list(cfg["DOWNSCALE_FACTORS"])

    circle_cfg = cfg.get("CIRCLE_DETECTION", {})
    if "quadrant_offset" in circle_cfg and isinstance(circle_cfg["quadrant_offset"], tuple):
        circle_cfg["quadrant_offset"] = list(circle_cfg["quadrant_offset"])

    if "VESSELNESS_AORTA" in cfg and "sigmas" in cfg["VESSELNESS_AORTA"]:
        if hasattr(cfg["VESSELNESS_AORTA"]["sigmas"], "tolist"):
            cfg["VESSELNESS_AORTA"]["sigmas"] = cfg["VESSELNESS_AORTA"]["sigmas"].tolist()

    if "VESSELNESS_ARTERY" in cfg and "sigmas" in cfg["VESSELNESS_ARTERY"]:
        if hasattr(cfg["VESSELNESS_ARTERY"]["sigmas"], "tolist"):
            cfg["VESSELNESS_ARTERY"]["sigmas"] = cfg["VESSELNESS_ARTERY"]["sigmas"].tolist()

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


def scale_config_to_resolution(config, reference_downscale_xy=2):
    """Escala parâmetros em voxels para a resolução definida em DOWNSCALE_FACTORS."""
    cfg = copy.deepcopy(config)
    factor_xy = cfg["DOWNSCALE_FACTORS"][0]
    scale = reference_downscale_xy / factor_xy

    if scale == 1.0:
        return cfg

    scale_area = scale ** 2

    #cfg["VESSELNESS_AORTA"]["sigmas"] = cfg["VESSELNESS_AORTA"]["sigmas"] * scale
    #cfg["VESSELNESS_ARTERY"]["sigmas"] = cfg["VESSELNESS_ARTERY"]["sigmas"] * scale
    cfg['CIRCLE_DETECTION']['tol_distance_mm'] /= scale
    #cfg['CIRCLE_DETECTION']['tol_radius_mm'] /= scale

    #cfg["CIRCLE_DETECTION"]["canny_sigma"] *= scale
    #cfg["CIRCLE_DETECTION"]["neighbor_distance_threshold"] *= scale
    qx, qy = cfg["CIRCLE_DETECTION"]["quadrant_offset"]
    cfg["CIRCLE_DETECTION"]["quadrant_offset"] = (
        int(round(qx * scale)),
        int(round(qy * scale)),
    )

    cfg["LEVEL_SET"]["leak_removal_radius"] = max(
        1, round(cfg["LEVEL_SET"]["leak_removal_radius"] * scale)
    )
    cfg["OSTIA_DETECTION"]["erosion_radius"] = max(
        1, round(cfg["OSTIA_DETECTION"]["erosion_radius"] * scale)
    )

    cfg["REGION_GROWING"]["max_volume"] = round(
        cfg["REGION_GROWING"]["max_volume"] * scale_area
    )
    cfg["REGION_GROWING"]["switch_at_voxels"] = round(
        cfg["REGION_GROWING"]["switch_at_voxels"] * scale_area
    )

    cfg["POSTPROCESSING"]["closing_radius"] = max(
        1, round(cfg["POSTPROCESSING"]["closing_radius"] * scale)
    )
    cfg["POSTPROCESSING"]["dilation_radius"] = max(
        1, round(cfg["POSTPROCESSING"]["dilation_radius"] * scale)
    )

    return cfg
