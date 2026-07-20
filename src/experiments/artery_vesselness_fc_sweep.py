"""Compare vesselness arterial e parâmetros de fuzzy connectedness.

O experimento mantém threshold e aorta/óstios fixos, processa cada imagem uma
única vez e reutiliza cada mapa de vesselness entre RG e FC. O estágio
``refinement`` também reutiliza cada máscara bruta para comparar diferentes
pós-processamentos morfológicos sem repetir RG ou FC.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.experiments.sweep_common import (  # noqa: E402
    csv_safe,
    sanitize_name,
    select_ids,
    write_json,
)
from utils.project.config import (  # noqa: E402
    apply_aorta_ostia_method,
    load_config_json,
    scale_config_to_resolution,
)
from utils.project.notebook_env import resolve_imagecas_base_path  # noqa: E402
from utils.segmentation.artery_segmentation import (  # noqa: E402
    normal_region_growing_from_ostia,
)
from utils.segmentation.fuzzy_connectedness import (  # noqa: E402
    segment_artery_fuzzy_connectedness,
)
from utils.segmentation.pipeline_arteries import (  # noqa: E402
    postprocess_artery_mask,
    postprocess_artery_mask_conditioned,
)
from utils.segmentation.pipeline_detection import (  # noqa: E402
    detect_and_evaluate_ostia,
    get_or_detect_aorta_circles,
    get_or_segment_aorta,
)
from utils.segmentation.pipeline_preprocessing import (  # noqa: E402
    get_or_compute_vesselness,
    load_and_preprocess_image,
)
from utils.utils.metrics import dice_score  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "config/pipeline_config.json"
DEFAULT_VARIANTS = SRC_DIR / "experiments/artery_vesselness_fc_variants.json"
DEFAULT_OUTPUT = REPO_ROOT / "output/segmentation/analysis/artery_vesselness_fc_sweep"

RESULT_COLUMNS = [
    "stage",
    "variant",
    "split",
    "IMG_ID",
    "segmentation_variant",
    "vesselness_profile",
    "artery_method",
    "morphology_profile",
    "morphology_mode",
    "morphology_closing_radius",
    "morphology_dilation_radius",
    "morphology_base_dilation_radius",
    "morphology_max_dilation_radius",
    "morphology_support_percentile",
    "morphology_support_factor",
    "morphology_local_max_radius",
    "ostia_status",
    "ostia_success",
    "left_dist_mm",
    "right_dist_mm",
    "dice_artery",
    "dice_artery_before_morphology",
    "dice_artery_after_morphology",
    "dice_artery_morphology_delta",
    "artery_voxels",
    "artery_voxels_before_morphology",
    "artery_voxels_after_morphology",
    "morphology_added_voxels",
    "recall_before_morphology",
    "recall_after_morphology",
    "precision_before_morphology",
    "precision_after_morphology",
    "vesselness_seconds",
    "segmentation_seconds",
    "morphology_seconds",
    "fc_processed_voxels",
    "fc_candidate_voxels_initial",
    "fc_candidate_voxels_final",
    "fc_effective_alpha",
    "fc_object_seed_count",
    "fc_grow_each_ostium_separately",
    "fc_processed_limit_hit",
    "fc_candidate_limit_hit",
    "branch_parameter_mode",
    "left_branch_voxels",
    "right_branch_voxels",
    "recovery_triggered_branches",
    "recovery_accepted_branches",
    "recovery_added_voxels",
    "conditioned_support_threshold",
    "conditioned_shell_voxels",
    "conditioned_accepted_voxels",
    "conditioned_acceptance_rate",
    "error",
]


def parse_names(value: str | None) -> list[str] | None:
    """Converte uma lista separada por vírgulas em nomes não vazios."""
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def select_named(mapping: dict[str, Any], names: list[str] | None) -> dict[str, Any]:
    """Seleciona entradas nomeadas preservando a ordem solicitada."""
    if names is None:
        return mapping
    missing = [name for name in names if name not in mapping]
    if missing:
        raise ValueError(f"Configurações desconhecidas: {missing}")
    return {name: mapping[name] for name in names}


def load_definitions(path: Path) -> dict[str, Any]:
    """Carrega perfis de vesselness, FC, refinamento e morfologia."""
    with path.open("r", encoding="utf-8") as file_handle:
        definitions = json.load(file_handle)
    if not definitions.get("vesselness_profiles"):
        raise ValueError("O arquivo não contém vesselness_profiles.")
    if not definitions.get("fc_variants"):
        raise ValueError("O arquivo não contém fc_variants.")
    if not definitions.get("refinement_variants"):
        raise ValueError("O arquivo não contém refinement_variants.")
    if not definitions.get("optimization_variants"):
        raise ValueError("O arquivo não contém optimization_variants.")
    if not definitions.get("morphology_profiles"):
        raise ValueError("O arquivo não contém morphology_profiles.")
    return definitions


def build_parser() -> argparse.ArgumentParser:
    """Cria a CLI do experimento."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["vesselness", "fc", "refinement", "optimization"],
        required=True,
    )
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--ids", default=None, help="IDs separados por vírgula.")
    parser.add_argument("--resolution", choices=["mid", "high"], default="mid")
    parser.add_argument(
        "--threshold-method", choices=["normal", "fuzzy"], default="normal"
    )
    parser.add_argument(
        "--aorta-ostia-method",
        choices=["standard", "bilateral_thin"],
        default="bilateral_thin",
    )
    parser.add_argument("--profiles", default=None)
    parser.add_argument(
        "--vesselness-profile",
        default="current",
        help="Perfil único usado no estágio fc.",
    )
    parser.add_argument("--fc-variants", default=None)
    parser.add_argument(
        "--refinement-variants",
        default=None,
        help="Variantes base separadas por vírgula para o estágio refinement.",
    )
    parser.add_argument(
        "--morphology-profiles",
        default=None,
        help="Perfis morfológicos separados por vírgula para refinement.",
    )
    parser.add_argument(
        "--optimization-variants",
        default=None,
        help="Variantes base separadas por vírgula para o estágio optimization.",
    )
    parser.add_argument("--variant-limit", type=int, default=None)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--definitions", type=Path, default=DEFAULT_VARIANTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu", dest="use_gpu", action="store_true", default=None)
    gpu_group.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    return parser


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Monta a configuração fixa compartilhada por todas as variantes."""
    config = load_config_json(str(args.config_path), {})
    if args.resolution == "high":
        config["DOWNSCALE_FACTORS"] = [1, 1, 1]
    config.setdefault("THRESHOLDING", {})["method"] = args.threshold_method
    if args.use_gpu is not None:
        config["USE_GPU"] = bool(args.use_gpu)
    config["LOAD_CACHE"] = False
    config["SAVE_CACHE"] = False
    config = apply_aorta_ostia_method(config, args.aorta_ostia_method)
    return scale_config_to_resolution(config)


def build_variants(
    args: argparse.Namespace,
    definitions: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Expande as variantes do estágio selecionado."""
    current_morphology = {
        "name": "current",
        "closing_radius": None,
        "dilation_radius": None,
        "append_to_variant": False,
    }
    variants: list[dict[str, Any]] = []
    if args.stage == "vesselness":
        requested_profiles = parse_names(args.profiles)
        if requested_profiles is None:
            requested_profiles = definitions.get("vesselness_stage_profiles")
        profiles = select_named(
            definitions["vesselness_profiles"], requested_profiles
        )
        for profile_name in profiles:
            for method in ("region_growing", "fuzzy_connectedness"):
                suffix = "rg" if method == "region_growing" else "fc"
                variants.append(
                    {
                        "name": f"{profile_name}_{suffix}",
                        "vesselness_profile": profile_name,
                        "artery_method": method,
                        "fc_overrides": {},
                        "morphology_profiles": [current_morphology],
                    }
                )
    elif args.stage == "fc":
        if args.vesselness_profile not in definitions["vesselness_profiles"]:
            raise ValueError(
                f"Perfil de vesselness desconhecido: {args.vesselness_profile}"
            )
        profiles = {
            args.vesselness_profile: definitions["vesselness_profiles"][
                args.vesselness_profile
            ]
        }
        fc_variants = select_named(
            definitions["fc_variants"], parse_names(args.fc_variants)
        )
        variants = [
            {
                "name": name,
                "vesselness_profile": args.vesselness_profile,
                "artery_method": "fuzzy_connectedness",
                "fc_overrides": overrides,
                "morphology_profiles": [current_morphology],
            }
            for name, overrides in fc_variants.items()
        ]
    else:
        variant_group = (
            "refinement_variants"
            if args.stage == "refinement"
            else "optimization_variants"
        )
        requested_variants = (
            parse_names(args.refinement_variants)
            if args.stage == "refinement"
            else parse_names(args.optimization_variants)
        )
        selected_variants = select_named(
            definitions[variant_group],
            requested_variants,
        )
        cli_morphologies = parse_names(args.morphology_profiles)
        default_morphologies = cli_morphologies
        if default_morphologies is None:
            default_morphologies = definitions.get(
                f"{args.stage}_morphology_profiles"
            )

        def morphology_profiles_for(values: dict[str, Any]) -> list[dict[str, Any]]:
            """Resolve perfis globais ou específicos da variante."""
            requested = cli_morphologies
            if requested is None:
                requested = values.get("morphology_profiles", default_morphologies)
            selected = select_named(
                definitions["morphology_profiles"],
                requested,
            )
            return [
                {
                    "name": name,
                    **profile,
                    "append_to_variant": True,
                }
                for name, profile in selected.items()
            ]
        used_profile_names = {
            values["vesselness_profile"] for values in selected_variants.values()
        }
        missing_profiles = used_profile_names - set(definitions["vesselness_profiles"])
        if missing_profiles:
            raise ValueError(
                "Perfis de vesselness desconhecidos no refinement: "
                f"{sorted(missing_profiles)}"
            )
        profiles = {
            name: definitions["vesselness_profiles"][name]
            for name in definitions["vesselness_profiles"]
            if name in used_profile_names
        }
        variants = [
            {
                "name": name,
                "vesselness_profile": values["vesselness_profile"],
                "artery_method": values["artery_method"],
                "fc_overrides": values.get("fc_overrides", {}),
                "fc_branch_overrides": values.get("fc_branch_overrides"),
                "rg_overrides": values.get("rg_overrides", {}),
                "rg_branch_overrides": values.get("rg_branch_overrides"),
                "recovery": values.get("recovery"),
                "morphology_profiles": morphology_profiles_for(values),
            }
            for name, values in selected_variants.items()
        ]

    if args.variant_limit is not None:
        variants = variants[: args.variant_limit]
    if not variants:
        raise ValueError("Nenhuma variante foi selecionada.")
    return variants, profiles


def output_variant_name(
    variant: dict[str, Any], morphology: dict[str, Any]
) -> str:
    """Monta o nome persistido para uma segmentação e uma morfologia."""
    if not morphology.get("append_to_variant", False):
        return str(variant["name"])
    return f"{variant['name']}__{morphology['name']}"


def expected_output_variant_names(variants: list[dict[str, Any]]) -> set[str]:
    """Lista todas as combinações que devem existir por imagem."""
    return {
        output_variant_name(variant, morphology)
        for variant in variants
        for morphology in variant["morphology_profiles"]
    }


def ostia_status(ostia_eval: dict[str, Any]) -> tuple[str, bool]:
    """Converte a avaliação dos óstios em status compacto."""
    if ostia_eval["both_correct"]:
        return "both_correct", True
    if ostia_eval["both_tolerable"]:
        return "both_tolerable", True
    return "found_but_wrong", False


def prepare_common_case(
    img_id: int,
    base_path: Path,
    run_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Executa uma vez as etapas compartilhadas antes da segmentação arterial."""
    image_data = load_and_preprocess_image(str(img_id), str(base_path), config)
    lcc_image = image_data["lcc_image"]
    spacing = image_data["scaled_spacing"]
    vesselness_spacing = (spacing[1], spacing[0], spacing[2])
    cache_dir = run_dir / "runtime_cache"

    vesselness_ostios = get_or_compute_vesselness(
        str(img_id),
        lcc_image,
        cache_dir=str(cache_dir / "vesselness_ostios"),
        vesselness_config=config["VESSELNESS_AORTA"],
        load_cache=False,
        save_cache=False,
        use_gpu=config.get("USE_GPU", False),
        spacing=vesselness_spacing,
    )
    circles = get_or_detect_aorta_circles(
        str(img_id),
        lcc_image,
        image_data["downscale_factors"],
        spacing,
        config["CIRCLE_DETECTION"],
        str(cache_dir),
        load_cache=False,
        save_cache=False,
    )
    aorta_mask = get_or_segment_aorta(
        str(img_id),
        lcc_image,
        circles,
        config["LEVEL_SET"],
        str(cache_dir),
        load_cache=False,
        save_cache=False,
        use_gpu=config.get("USE_GPU", False),
    )
    ostia_eval = detect_and_evaluate_ostia(
        aorta_mask,
        vesselness_ostios,
        image_data["label"],
        spacing,
        config,
        detected_circles=circles,
    )
    status, success = ostia_status(ostia_eval)
    min_threshold = float(
        image_data.get("preprocessing_details", {}).get(
            "min_threshold", config.get("MIN_THRESHOLD", -300)
        )
    )
    return {
        **image_data,
        "lcc_mask": np.asarray(lcc_image) > min_threshold,
        "vesselness_spacing": vesselness_spacing,
        "ostia_eval": ostia_eval,
        "ostia_status": status,
        "ostia_success": success,
    }


def compute_artery_vesselness(
    img_id: int,
    case: dict[str, Any],
    run_dir: Path,
    config: dict[str, Any],
    profile_name: str,
    profile: dict[str, Any],
) -> tuple[np.ndarray, float]:
    """Calcula uma vez o mapa arterial de um perfil para a imagem."""
    vesselness_config = copy.deepcopy(config["VESSELNESS_ARTERY"])
    vesselness_config.update(profile)
    started = time.perf_counter()
    vesselness = get_or_compute_vesselness(
        str(img_id),
        case["lcc_image"],
        cache_dir=str(run_dir / "runtime_cache" / profile_name),
        vesselness_config=vesselness_config,
        load_cache=False,
        save_cache=False,
        use_gpu=config.get("USE_GPU", False),
        spacing=case["vesselness_spacing"],
    )
    return vesselness, time.perf_counter() - started


def run_segmentation(
    variant: dict[str, Any],
    vesselness: np.ndarray,
    case: dict[str, Any],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Executa RG ou FC sobre um mapa arterial já calculado."""
    ostia_eval = case["ostia_eval"]
    recovery = variant.get("recovery")

    if variant["artery_method"] == "region_growing":
        rg_config = copy.deepcopy(config)
        rg_config["REGION_GROWING"].update(variant.get("rg_overrides", {}))
        branch_overrides = variant.get("rg_branch_overrides")
        if branch_overrides:
            branch_masks = []
            branch_counts = []
            for side, seed in zip(
                ("left", "right"),
                (ostia_eval["ostia_left"], ostia_eval["ostia_right"]),
            ):
                side_config = copy.deepcopy(rg_config)
                side_config["REGION_GROWING"].update(
                    branch_overrides.get(side, {})
                )
                branch_mask = normal_region_growing_from_ostia(
                    vesselness,
                    seed,
                    None,
                    side_config,
                )
                branch_masks.append(branch_mask)
                branch_counts.append(int(branch_mask.sum()))

            raw_mask = np.zeros_like(vesselness, dtype=np.uint8)
            for branch_mask in branch_masks:
                raw_mask |= np.asarray(branch_mask, dtype=np.uint8)
            return raw_mask, raw_mask, {
                "branch_parameter_mode": "side_specific",
                "branch_voxel_counts": branch_counts,
            }
        if not recovery:
            raw_mask = normal_region_growing_from_ostia(
                vesselness,
                ostia_eval["ostia_left"],
                ostia_eval["ostia_right"],
                rg_config,
            )
            return raw_mask, raw_mask, {}

        relaxed_config = copy.deepcopy(rg_config)
        relaxed_config["REGION_GROWING"].update(recovery.get("rg_overrides", {}))
        raw_mask, recovery_details = recover_small_ostia_branches(
            [ostia_eval["ostia_left"], ostia_eval["ostia_right"]],
            min_branch_voxels=int(recovery.get("min_branch_voxels", 500)),
            run_branch=lambda seed, relaxed: normal_region_growing_from_ostia(
                vesselness,
                seed,
                None,
                relaxed_config if relaxed else rg_config,
            ),
            max_growth_ratio=float(recovery.get("max_growth_ratio", 10.0)),
            max_branch_voxels=int(recovery.get("max_branch_voxels", 50_000)),
        )
        return raw_mask, raw_mask, recovery_details

    fc_params = copy.deepcopy(config["FUZZY_CONNECTEDNESS"])
    fc_params.update(variant.get("fc_overrides", {}))
    max_candidate = fc_params.pop("max_candidate_voxels", 500_000)
    max_processed = fc_params.pop("max_processed_voxels", 500_000)
    branch_overrides = variant.get("fc_branch_overrides")

    if branch_overrides:
        branch_masks = []
        branch_counts = []
        branch_details = []
        for side, seed in zip(
            ("left", "right"),
            (ostia_eval["ostia_left"], ostia_eval["ostia_right"]),
        ):
            if seed is None:
                branch_mask = np.zeros_like(vesselness, dtype=np.uint8)
                details = {}
            else:
                side_params = copy.deepcopy(fc_params)
                side_params.update(branch_overrides.get(side, {}))
                side_result = segment_artery_fuzzy_connectedness(
                    case["lcc_image"],
                    vesselness,
                    [seed],
                    case["lcc_mask"],
                    config,
                    params=side_params,
                    max_candidate_voxels=max_candidate,
                    max_processed_voxels=max_processed,
                    apply_postprocessing=False,
                )
                branch_mask = side_result["raw_mask"]
                details = side_result["details"]
            branch_masks.append(np.asarray(branch_mask, dtype=np.uint8))
            branch_counts.append(int(np.sum(branch_mask)))
            branch_details.append(details)

        raw_mask = np.zeros_like(vesselness, dtype=np.uint8)
        for branch_mask in branch_masks:
            raw_mask |= branch_mask
        return raw_mask, raw_mask, {
            "branch_parameter_mode": "side_specific",
            "branch_voxel_counts": branch_counts,
            "processed_voxels": sum(
                int(details.get("processed_voxels", 0))
                for details in branch_details
            ),
            "candidate_voxels_initial": max(
                (
                    int(details.get("candidate_voxels_initial", 0))
                    for details in branch_details
                ),
                default=0,
            ),
            "candidate_voxels_final": max(
                (
                    int(details.get("candidate_voxels_final", 0))
                    for details in branch_details
                ),
                default=0,
            ),
            "object_seed_count": sum(
                int(details.get("object_seed_count", 0))
                for details in branch_details
            ),
            "max_candidate_voxels": max_candidate,
            "max_processed_voxels": (
                None
                if max_processed is None
                else int(max_processed) * len(branch_details)
            ),
            "grow_each_ostium_separately": True,
        }

    def run_fc_branch(seed: Any, relaxed: bool) -> np.ndarray:
        if seed is None:
            return np.zeros_like(vesselness, dtype=np.uint8)
        params = copy.deepcopy(fc_params)
        if relaxed:
            params.update(recovery.get("fc_overrides", {}))
        result = segment_artery_fuzzy_connectedness(
            case["lcc_image"],
            vesselness,
            [seed],
            case["lcc_mask"],
            config,
            params=params,
            max_candidate_voxels=max_candidate,
            max_processed_voxels=max_processed,
            apply_postprocessing=False,
        )
        return result["raw_mask"]

    if recovery:
        raw_mask, details = recover_small_ostia_branches(
            [ostia_eval["ostia_left"], ostia_eval["ostia_right"]],
            min_branch_voxels=int(recovery.get("min_branch_voxels", 500)),
            run_branch=run_fc_branch,
            max_growth_ratio=float(recovery.get("max_growth_ratio", 10.0)),
            max_branch_voxels=int(recovery.get("max_branch_voxels", 50_000)),
        )
        details["max_candidate_voxels"] = max_candidate
        details["max_processed_voxels"] = max_processed
        return raw_mask, raw_mask, details

    result = segment_artery_fuzzy_connectedness(
        case["lcc_image"],
        vesselness,
        [ostia_eval["ostia_left"], ostia_eval["ostia_right"]],
        case["lcc_mask"],
        config,
        params=fc_params,
        max_candidate_voxels=max_candidate,
        max_processed_voxels=max_processed,
        apply_postprocessing=False,
    )
    details = {
        **result["details"],
        "max_candidate_voxels": max_candidate,
        "max_processed_voxels": max_processed,
    }
    return result["raw_mask"], result["raw_mask"], details


def recover_small_ostia_branches(
    ostia_seeds: list[Any],
    *,
    min_branch_voxels: int,
    run_branch: Any,
    max_growth_ratio: float,
    max_branch_voxels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Repete uma vez ramos pequenos com sementes/parâmetros relaxados."""
    branch_masks: list[np.ndarray] = []
    before_counts: list[int] = []
    after_counts: list[int] = []
    triggered = 0
    accepted = 0

    for seed in ostia_seeds:
        base_mask = np.asarray(run_branch(seed, False), dtype=np.uint8)
        base_count = int(base_mask.sum())
        selected_mask = base_mask
        before_counts.append(base_count)

        if seed is not None and base_count < int(min_branch_voxels):
            triggered += 1
            recovered_mask = np.asarray(run_branch(seed, True), dtype=np.uint8)
            recovered_count = int(recovered_mask.sum())
            growth_ratio = recovered_count / max(base_count, 1)
            plausible_growth = (
                recovered_count > base_count
                and recovered_count <= int(max_branch_voxels)
                and (base_count == 0 or growth_ratio <= float(max_growth_ratio))
            )
            if plausible_growth:
                selected_mask = recovered_mask
                accepted += 1

        branch_masks.append(selected_mask)
        after_counts.append(int(selected_mask.sum()))

    if not branch_masks:
        raise ValueError("Nenhum ramo pôde ser inicializado pelos óstios.")

    combined = np.zeros_like(branch_masks[0], dtype=np.uint8)
    for branch_mask in branch_masks:
        combined |= branch_mask
    return combined, {
        "recovery_triggered_branches": triggered,
        "recovery_accepted_branches": accepted,
        "recovery_added_voxels": int(sum(after_counts) - sum(before_counts)),
        "branch_voxels_before_recovery": before_counts,
        "branch_voxels_after_recovery": after_counts,
    }


def binary_overlap_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    """Calcula sensibilidade e precisão de uma máscara binária."""
    prediction_mask = np.asarray(prediction) > 0
    target_mask = np.asarray(target) > 0
    true_positive = int(np.logical_and(prediction_mask, target_mask).sum())
    predicted = int(prediction_mask.sum())
    target_count = int(target_mask.sum())
    return {
        "recall": true_positive / target_count if target_count else 0.0,
        "precision": true_positive / predicted if predicted else 0.0,
    }


def apply_morphology_profile(
    raw_mask: np.ndarray,
    config: dict[str, Any],
    morphology: dict[str, Any],
    vesselness: np.ndarray,
    candidate_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Aplica um perfil morfológico à mesma máscara bruta."""
    if morphology.get("mode", "standard") == "vesselness_conditioned":
        return postprocess_artery_mask_conditioned(
            raw_mask,
            config,
            vesselness,
            candidate_mask=candidate_mask,
            closing_radius=int(morphology.get("closing_radius", 3)),
            base_dilation_radius=int(morphology.get("base_dilation_radius", 1)),
            max_dilation_radius=int(morphology.get("max_dilation_radius", 2)),
            support_percentile=float(morphology.get("support_percentile", 10)),
            support_factor=float(morphology.get("support_factor", 0.5)),
            local_max_radius=int(morphology.get("local_max_radius", 1)),
        )
    return (
        postprocess_artery_mask(
            raw_mask,
            config,
            closing_radius=morphology.get("closing_radius"),
            dilation_radius=morphology.get("dilation_radius"),
        ),
        {},
    )


def result_row(
    args: argparse.Namespace,
    variant: dict[str, Any],
    img_id: int,
    case: dict[str, Any],
    raw_artery_mask: np.ndarray,
    artery_mask: np.ndarray,
    vesselness_seconds: float,
    segmentation_seconds: float,
    morphology_seconds: float,
    details: dict[str, Any],
    morphology: dict[str, Any],
) -> dict[str, Any]:
    """Monta uma linha por imagem e variante."""
    ostia_eval = case["ostia_eval"]
    max_processed = details.get("max_processed_voxels")
    max_candidate = details.get("max_candidate_voxels")
    processed = details.get("processed_voxels")
    candidate_final = details.get("candidate_voxels_final")
    dice_before = float(dice_score(raw_artery_mask, ostia_eval["label_artery"]))
    dice_after = float(dice_score(artery_mask, ostia_eval["label_artery"]))
    metrics_before = binary_overlap_metrics(
        raw_artery_mask, ostia_eval["label_artery"]
    )
    metrics_after = binary_overlap_metrics(artery_mask, ostia_eval["label_artery"])
    voxels_before = int(np.sum(raw_artery_mask))
    voxels_after = int(np.sum(artery_mask))
    return {
        "stage": args.stage,
        "variant": output_variant_name(variant, morphology),
        "split": args.split,
        "IMG_ID": img_id,
        "segmentation_variant": variant["name"],
        "vesselness_profile": variant["vesselness_profile"],
        "artery_method": variant["artery_method"],
        "morphology_profile": morphology["name"],
        "morphology_mode": morphology.get("mode", "standard"),
        "morphology_closing_radius": morphology.get("closing_radius"),
        "morphology_dilation_radius": morphology.get("dilation_radius"),
        "morphology_base_dilation_radius": morphology.get(
            "base_dilation_radius"
        ),
        "morphology_max_dilation_radius": morphology.get("max_dilation_radius"),
        "morphology_support_percentile": morphology.get("support_percentile"),
        "morphology_support_factor": morphology.get("support_factor"),
        "morphology_local_max_radius": morphology.get("local_max_radius"),
        "ostia_status": case["ostia_status"],
        "ostia_success": case["ostia_success"],
        "left_dist_mm": ostia_eval["left_info"]["physical_dist"],
        "right_dist_mm": ostia_eval["right_info"]["physical_dist"],
        "dice_artery": dice_after,
        "dice_artery_before_morphology": dice_before,
        "dice_artery_after_morphology": dice_after,
        "dice_artery_morphology_delta": dice_after - dice_before,
        "artery_voxels": voxels_after,
        "artery_voxels_before_morphology": voxels_before,
        "artery_voxels_after_morphology": voxels_after,
        "morphology_added_voxels": voxels_after - voxels_before,
        "recall_before_morphology": metrics_before["recall"],
        "recall_after_morphology": metrics_after["recall"],
        "precision_before_morphology": metrics_before["precision"],
        "precision_after_morphology": metrics_after["precision"],
        "vesselness_seconds": vesselness_seconds,
        "segmentation_seconds": segmentation_seconds,
        "morphology_seconds": morphology_seconds,
        "fc_processed_voxels": processed,
        "fc_candidate_voxels_initial": details.get("candidate_voxels_initial"),
        "fc_candidate_voxels_final": candidate_final,
        "fc_effective_alpha": details.get("effective_alpha"),
        "fc_object_seed_count": details.get("object_seed_count"),
        "fc_grow_each_ostium_separately": details.get(
            "grow_each_ostium_separately"
        ),
        "fc_processed_limit_hit": bool(
            max_processed is not None
            and processed is not None
            and processed >= max_processed
        ),
        "fc_candidate_limit_hit": bool(
            max_candidate is not None
            and candidate_final is not None
            and candidate_final >= max_candidate
        ),
        "branch_parameter_mode": details.get("branch_parameter_mode", "shared"),
        "left_branch_voxels": (
            details.get("branch_voxel_counts", [None, None]) + [None, None]
        )[0],
        "right_branch_voxels": (
            details.get("branch_voxel_counts", [None, None]) + [None, None]
        )[1],
        "recovery_triggered_branches": details.get(
            "recovery_triggered_branches", 0
        ),
        "recovery_accepted_branches": details.get("recovery_accepted_branches", 0),
        "recovery_added_voxels": details.get("recovery_added_voxels", 0),
        "conditioned_support_threshold": details.get(
            "conditioned_support_threshold"
        ),
        "conditioned_shell_voxels": details.get("conditioned_shell_voxels"),
        "conditioned_accepted_voxels": details.get(
            "conditioned_accepted_voxels"
        ),
        "conditioned_acceptance_rate": details.get("conditioned_acceptance_rate"),
        "error": None,
    }


def error_rows(
    args: argparse.Namespace,
    variants: list[dict[str, Any]],
    img_id: int,
    exc: Exception,
) -> list[dict[str, Any]]:
    """Representa uma falha comum sem perder as demais imagens do sweep."""
    error = f"{type(exc).__name__}: {exc}"
    rows = []
    for variant in variants:
        for morphology in variant["morphology_profiles"]:
            rows.append(
                {
                    **{column: None for column in RESULT_COLUMNS},
                    "stage": args.stage,
                    "variant": output_variant_name(variant, morphology),
                    "split": args.split,
                    "IMG_ID": img_id,
                    "segmentation_variant": variant["name"],
                    "vesselness_profile": variant["vesselness_profile"],
                    "artery_method": variant["artery_method"],
                    "morphology_profile": morphology["name"],
                    "morphology_closing_radius": morphology.get(
                        "closing_radius"
                    ),
                    "morphology_dilation_radius": morphology.get(
                        "dilation_radius"
                    ),
                    "ostia_success": False,
                    "error": error,
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Resume Dice, óstios, limites do FC e tempo por variante."""
    df = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
    summaries = []
    for variant, group in df.groupby("variant", sort=False):
        dice = pd.to_numeric(group["dice_artery"], errors="coerce")
        dice_before = pd.to_numeric(
            group["dice_artery_before_morphology"], errors="coerce"
        )
        morphology_delta = pd.to_numeric(
            group["dice_artery_morphology_delta"], errors="coerce"
        )
        recall_before = pd.to_numeric(
            group["recall_before_morphology"], errors="coerce"
        )
        recall_after = pd.to_numeric(
            group["recall_after_morphology"], errors="coerce"
        )
        precision_before = pd.to_numeric(
            group["precision_before_morphology"], errors="coerce"
        )
        precision_after = pd.to_numeric(
            group["precision_after_morphology"], errors="coerce"
        )
        success = group["ostia_success"].fillna(False).astype(bool)
        success_dice = dice[success]
        summaries.append(
            {
                "variant": variant,
                "images": int(len(group)),
                "valid_dice_images": int(dice.notna().sum()),
                "ostia_success_rate": float(success.mean()),
                "mean_dice": float(dice.mean()) if dice.notna().any() else None,
                "mean_dice_before_morphology": (
                    float(dice_before.mean()) if dice_before.notna().any() else None
                ),
                "median_dice_before_morphology": (
                    float(dice_before.median())
                    if dice_before.notna().any()
                    else None
                ),
                "mean_dice_after_morphology": (
                    float(dice.mean()) if dice.notna().any() else None
                ),
                "mean_dice_morphology_delta": (
                    float(morphology_delta.mean())
                    if morphology_delta.notna().any()
                    else None
                ),
                "median_dice_morphology_delta": (
                    float(morphology_delta.median())
                    if morphology_delta.notna().any()
                    else None
                ),
                "mean_recall_before_morphology": recall_before.mean(),
                "mean_recall_after_morphology": recall_after.mean(),
                "mean_precision_before_morphology": precision_before.mean(),
                "mean_precision_after_morphology": precision_after.mean(),
                "median_dice": float(dice.median()) if dice.notna().any() else None,
                "std_dice": float(dice.std()) if dice.notna().any() else None,
                "mean_dice_success_ostia": (
                    float(success_dice.mean())
                    if success_dice.notna().any()
                    else None
                ),
                "zero_dice_rate": float((dice.fillna(0) <= 0).mean()),
                "processed_limit_hit_rate": float(
                    group["fc_processed_limit_hit"].fillna(False).astype(bool).mean()
                ),
                "candidate_limit_hit_rate": float(
                    group["fc_candidate_limit_hit"].fillna(False).astype(bool).mean()
                ),
                "error_count": int(group["error"].notna().sum()),
                "mean_vesselness_seconds": pd.to_numeric(
                    group["vesselness_seconds"], errors="coerce"
                ).mean(),
                "mean_segmentation_seconds": pd.to_numeric(
                    group["segmentation_seconds"], errors="coerce"
                ).mean(),
                "mean_morphology_seconds": pd.to_numeric(
                    group["morphology_seconds"], errors="coerce"
                ).mean(),
                "mean_recovery_triggered_branches": pd.to_numeric(
                    group["recovery_triggered_branches"], errors="coerce"
                ).mean(),
                "mean_recovery_accepted_branches": pd.to_numeric(
                    group["recovery_accepted_branches"], errors="coerce"
                ).mean(),
                "mean_recovery_added_voxels": pd.to_numeric(
                    group["recovery_added_voxels"], errors="coerce"
                ).mean(),
                "mean_conditioned_acceptance_rate": pd.to_numeric(
                    group["conditioned_acceptance_rate"], errors="coerce"
                ).mean(),
            }
        )
    return pd.DataFrame(summaries).sort_values(
        ["mean_dice_success_ostia", "mean_dice", "median_dice"],
        ascending=False,
        na_position="last",
    )


def pairwise_summary(rows: list[dict[str, Any]], stage: str) -> pd.DataFrame:
    """Compara cada variante à referência correspondente."""
    df = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
    if df.empty:
        return pd.DataFrame()
    available_variants = set(df["variant"].dropna())
    records = []
    for variant in df["variant"].dropna().unique():
        if stage == "fc":
            reference = "fc_current"
        elif stage == "refinement":
            reference = "rg_gamma55__current_c3_d2"
        elif stage == "optimization":
            reference = "rg_gamma55_current__current_c3_d2"
            if reference not in available_variants:
                reference = "validation_baseline_rg__current_c3_d2"
        else:
            reference = (
                "current_fc" if str(variant).endswith("_fc") else "current_rg"
            )
        if variant == reference or reference not in available_variants:
            continue
        left = df[df["variant"] == reference][["IMG_ID", "dice_artery"]].rename(
            columns={"dice_artery": "reference_dice"}
        )
        right = df[df["variant"] == variant][["IMG_ID", "dice_artery"]].rename(
            columns={"dice_artery": "variant_dice"}
        )
        paired = left.merge(right, on="IMG_ID", how="inner")
        delta = pd.to_numeric(paired["variant_dice"], errors="coerce") - pd.to_numeric(
            paired["reference_dice"], errors="coerce"
        )
        records.append(
            {
                "reference": reference,
                "variant": variant,
                "paired_images": int(delta.notna().sum()),
                "mean_dice_delta": float(delta.mean()),
                "median_dice_delta": float(delta.median()),
                "improved_gt_0_01": int((delta > 0.01).sum()),
                "worsened_lt_minus_0_01": int((delta < -0.01).sum()),
            }
        )
    if not records:
        return pd.DataFrame(
            columns=[
                "reference",
                "variant",
                "paired_images",
                "mean_dice_delta",
                "median_dice_delta",
                "improved_gt_0_01",
                "worsened_lt_minus_0_01",
            ]
        )
    return pd.DataFrame(records).sort_values("mean_dice_delta", ascending=False)


def save_outputs(run_dir: Path, rows: list[dict[str, Any]], stage: str) -> None:
    """Atualiza resultados parciais e resumos compactos."""
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
    result_df = result_df.sort_values(["IMG_ID", "variant"])
    csv_safe(result_df).to_csv(results_dir / "image_results.csv", index=False)
    csv_safe(summarize(rows)).to_csv(results_dir / "ranking.csv", index=False)
    csv_safe(pairwise_summary(rows, stage)).to_csv(
        results_dir / "pairwise_vs_reference.csv", index=False
    )


def save_variant_parameters(
    run_dir: Path,
    variants: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> None:
    """Salva os parâmetros efetivos de cada variante em formato tabular."""
    rows = []
    for variant in variants:
        vesselness = copy.deepcopy(config["VESSELNESS_ARTERY"])
        vesselness.update(profiles[variant["vesselness_profile"]])
        fc = copy.deepcopy(config["FUZZY_CONNECTEDNESS"])
        fc.update(variant["fc_overrides"])
        rg = copy.deepcopy(config["REGION_GROWING"])
        rg.update(variant.get("rg_overrides", {}))
        for morphology in variant["morphology_profiles"]:
            rows.append(
                {
                    "variant": output_variant_name(variant, morphology),
                    "segmentation_variant": variant["name"],
                    "vesselness_profile": variant["vesselness_profile"],
                    "artery_method": variant["artery_method"],
                    "morphology_profile": morphology["name"],
                    "morphology_mode": morphology.get("mode", "standard"),
                    "morphology_closing_radius": morphology.get(
                        "closing_radius"
                    ),
                    "morphology_dilation_radius": morphology.get(
                        "dilation_radius"
                    ),
                    "morphology_base_dilation_radius": morphology.get(
                        "base_dilation_radius"
                    ),
                    "morphology_max_dilation_radius": morphology.get(
                        "max_dilation_radius"
                    ),
                    "morphology_support_percentile": morphology.get(
                        "support_percentile"
                    ),
                    "morphology_support_factor": morphology.get(
                        "support_factor"
                    ),
                    "morphology_local_max_radius": morphology.get(
                        "local_max_radius"
                    ),
                    "vesselness_sigmas": json.dumps(list(vesselness["sigmas"])),
                    "vesselness_alpha": vesselness["alpha"],
                    "vesselness_beta": vesselness["beta"],
                    "vesselness_gamma": vesselness["gamma"],
                    "rg_min_vesselness_fraction": rg.get(
                        "min_vesselness_fraction"
                    ),
                    "rg_threshold_divisor": rg.get("threshold_divisor"),
                    "rg_comparison_window": rg.get("comparison_window"),
                    "rg_relaxed_floor_factor": rg.get("relaxed_floor_factor"),
                    "rg_neighborhood": rg.get("neighborhood", 26),
                    "rg_reference_scope": rg.get("reference_scope", "global"),
                    "rg_reference_radius": rg.get("reference_radius", 3),
                    "rg_reference_percentile": rg.get(
                        "reference_percentile", 95.0
                    ),
                    "rg_branch_overrides": json.dumps(
                        variant.get("rg_branch_overrides"), sort_keys=True
                    ),
                    "fc_alpha": fc.get("alpha"),
                    "fc_sigma_hu": fc.get("sigma_hu"),
                    "fc_candidate_min_vesselness": fc.get(
                        "candidate_min_vesselness"
                    ),
                    "fc_vesselness_floor": fc.get("vesselness_floor"),
                    "fc_vesselness_weight": fc.get("vesselness_weight"),
                    "fc_seed_search_radius": fc.get("seed_search_radius"),
                    "fc_max_seeds_per_ostium": fc.get("max_seeds_per_ostium"),
                    "fc_max_candidate_voxels": fc.get("max_candidate_voxels"),
                    "fc_max_processed_voxels": fc.get("max_processed_voxels"),
                    "fc_grow_each_ostium_separately": fc.get(
                        "grow_each_ostium_separately"
                    ),
                    "fc_branch_overrides": json.dumps(
                        variant.get("fc_branch_overrides"), sort_keys=True
                    ),
                    "recovery": json.dumps(variant.get("recovery"), sort_keys=True),
                }
            )
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_safe(pd.DataFrame(rows)).to_csv(
        results_dir / "variant_parameters.csv", index=False
    )


def completed_ids(
    rows: list[dict[str, Any]], variants: list[dict[str, Any]]
) -> set[int]:
    """Identifica imagens com todas as variantes já salvas para retomada."""
    if not rows:
        return set()
    expected = expected_output_variant_names(variants)
    df = pd.DataFrame(rows)
    return {
        int(img_id)
        for img_id, group in df.groupby("IMG_ID")
        if set(group["variant"].dropna()) >= expected
    }


def main() -> None:
    """Executa o sweep solicitado e salva após cada imagem."""
    args = build_parser().parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size deve ser maior que zero.")

    definitions = load_definitions(args.definitions)
    variants, profiles = build_variants(args, definitions)
    config = build_config(args)
    if args.dry_run:
        base_path = Path(".")
        image_ids = (
            [int(value) for value in parse_names(args.ids) or []]
            if args.ids
            else []
        )
    else:
        base_path = resolve_imagecas_base_path()
        image_ids = select_ids(
            args.split, args.sample_size, args.start_index, args.ids, base_path
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.resume_dir or args.output_root / sanitize_name(
        args.run_name or f"{args.stage}_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results/image_results.csv"
    rows = (
        pd.read_csv(results_path).to_dict("records")
        if args.resume_dir and results_path.exists()
        else []
    )
    done = completed_ids(rows, variants)

    write_json(
        run_dir / "run_config.json",
        {
            "stage": args.stage,
            "split": args.split,
            "sample_size": args.sample_size,
            "start_index": args.start_index,
            "ids": image_ids,
            "resolution": args.resolution,
            "threshold_method": args.threshold_method,
            "aorta_ostia_method": args.aorta_ostia_method,
            "use_gpu": config.get("USE_GPU"),
            "profiles": profiles,
            "variants": variants,
            "output_variants": sorted(expected_output_variant_names(variants)),
            "definitions": str(args.definitions),
        },
    )
    save_variant_parameters(run_dir, variants, profiles, config)
    print(f"Run: {run_dir}")
    output_variant_count = len(expected_output_variant_names(variants))
    image_count = args.sample_size if args.dry_run and not image_ids else len(image_ids)
    print(
        f"Imagens: {image_count} | Segmentações: {len(variants)} | "
        f"Avaliações por imagem: {output_variant_count}"
    )
    print(f"Segmentações: {[variant['name'] for variant in variants]}")
    if args.dry_run:
        return

    for image_index, img_id in enumerate(image_ids, start=1):
        if int(img_id) in done:
            print(f"[{image_index}/{len(image_ids)}] IMG_ID={img_id} já concluída")
            continue
        print(f"[{image_index}/{len(image_ids)}] IMG_ID={img_id}")
        try:
            case = prepare_common_case(img_id, base_path, run_dir, config)
        except Exception as exc:
            rows.extend(error_rows(args, variants, img_id, exc))
            print(f"  ERRO comum: {type(exc).__name__}: {exc}")
            save_outputs(run_dir, rows, args.stage)
            continue

        variants_by_profile: dict[str, list[dict[str, Any]]] = {}
        for variant in variants:
            variants_by_profile.setdefault(variant["vesselness_profile"], []).append(
                variant
            )

        for profile_name, profile_variants in variants_by_profile.items():
            try:
                vesselness, vesselness_seconds = compute_artery_vesselness(
                    img_id,
                    case,
                    run_dir,
                    config,
                    profile_name,
                    profiles[profile_name],
                )
            except Exception as exc:
                rows.extend(error_rows(args, profile_variants, img_id, exc))
                print(
                    f"  ERRO vesselness {profile_name}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            for variant in profile_variants:
                try:
                    started = time.perf_counter()
                    raw_artery_mask, _, details = run_segmentation(
                        variant, vesselness, case, config
                    )
                    segmentation_seconds = time.perf_counter() - started
                    for morphology in variant["morphology_profiles"]:
                        morphology_started = time.perf_counter()
                        if (
                            morphology.get("closing_radius") is None
                            and morphology.get("dilation_radius") is None
                        ):
                            artery_mask = postprocess_artery_mask(
                                raw_artery_mask, config
                            )
                            morphology_details = {}
                        else:
                            artery_mask, morphology_details = apply_morphology_profile(
                                raw_artery_mask,
                                config,
                                morphology,
                                vesselness,
                                case["lcc_mask"],
                            )
                        morphology_seconds = time.perf_counter() - morphology_started
                        rows.append(
                            result_row(
                                args,
                                variant,
                                img_id,
                                case,
                                raw_artery_mask,
                                artery_mask,
                                vesselness_seconds,
                                segmentation_seconds,
                                morphology_seconds,
                                {**details, **morphology_details},
                                morphology,
                            )
                        )
                except Exception as exc:
                    rows.extend(error_rows(args, [variant], img_id, exc))
                    print(
                        f"  ERRO {variant['name']}: "
                        f"{type(exc).__name__}: {exc}"
                    )
            del vesselness
        save_outputs(run_dir, rows, args.stage)

    print("\nRanking final:")
    print(summarize(rows).to_string(index=False))
    print(f"\nResultados: {run_dir / 'results'}")


if __name__ == "__main__":
    main()
