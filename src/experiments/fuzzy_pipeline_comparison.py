"""Run fuzzy pipeline comparison variants from the command line.

This script is the server-friendly version of
``src/experiments/fuzzy_pipeline_comparison.ipynb``. It compares the normal
pipeline, contextual-object fuzzy thresholding, contextual fuzzy vesselness
weighting, region growing and fuzzy connectedness.

Example:
    uv run python src/experiments/fuzzy_pipeline_comparison.py --split train --sample-size 30
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.experiments.fuzzy_pipeline_comparison import (  # noqa: E402
    build_base_config,
    parameter_row,
    run_image,
    save_outputs,
    split_overrides,
    summarize_variant,
)
from utils.experiments.sweep_common import (  # noqa: E402
    apply_overrides,
    csv_safe,
    sanitize_name,
    select_ids,
    write_json,
)
from utils.project.config import scale_config_to_resolution  # noqa: E402
from utils.project.notebook_env import resolve_imagecas_base_path  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output/segmentation/analysis/fuzzy_pipeline_comparison"
DEFAULT_CONFIG_PATH = REPO_ROOT / "config/pipeline_config.json"


FC_PARAMS = {
    "fc.alpha": 0.18,
    "fc.sigma_hu": 80,
    "fc.neighborhood": 26,
    "fc.candidate_min_vesselness": 0.02,
    "fc.seed_search_radius": 2,
    "fc.max_seeds_per_ostium": 4,
    "fc.seed_min_vesselness": 0.02,
    "fc.min_seed_distance_voxels": 1.0,
    "fc.vesselness_affinity_mode": "geometric_mean",
    "fc.vesselness_floor": 0.02,
    "fc.edge_affinity_mode": "weighted_product",
    "fc.vesselness_weight": 0.9,
    "fc.mask_strategy": "alpha",
}

FC_PERMISSIVE_PARAMS = {
    **FC_PARAMS,
    "fc.alpha": 0.14,
    "fc.sigma_hu": 120,
    "fc.candidate_min_vesselness": 0.015,
    "fc.seed_min_vesselness": 0.015,
    "fc.vesselness_floor": 0.015,
}

FC_SEMI_PERMISSIVE_PARAMS = {
    **FC_PARAMS,
    "fc.alpha": 0.16,
    "fc.sigma_hu": 100,
    "fc.candidate_min_vesselness": 0.018,
    "fc.seed_min_vesselness": 0.018,
    "fc.vesselness_floor": 0.018,
}

FC_STRICT_PARAMS = {
    **FC_PARAMS,
    "fc.alpha": 0.22,
    "fc.sigma_hu": 60,
    "fc.candidate_min_vesselness": 0.03,
    "fc.seed_min_vesselness": 0.02,
    "fc.vesselness_floor": 0.03,
}

CONTEXTUAL_PARAMS = {
    "contextual.weight_floor": 0.15,
    "contextual.dense_power": 2.0,
    "contextual.weight_mode": "dense_only",
    "contextual.soft_margin_hu": 160,
    "contextual.object_percentile": 99.8,
    "contextual.dense_percentile": 99.95,
    "contextual.smooth_radius": 1,
    "contextual.smooth_mode": "mean",
}

CONTEXTUAL_MODERATE_PARAMS = {
    "contextual.weight_floor": 0.10,
    "contextual.dense_power": 3.0,
    "contextual.weight_mode": "object_dense",
    "contextual.soft_margin_hu": 160,
    "contextual.object_percentile": 99.7,
    "contextual.dense_percentile": 99.9,
    "contextual.smooth_radius": 1,
    "contextual.smooth_mode": "mean",
}

CONTEXTUAL_STRONG_PARAMS = {
    "contextual.weight_floor": 0.05,
    "contextual.dense_power": 4.0,
    "contextual.weight_mode": "object_dense",
    "contextual.soft_margin_hu": 160,
    "contextual.object_percentile": 99.5,
    "contextual.dense_percentile": 99.9,
    "contextual.smooth_radius": 2,
    "contextual.smooth_mode": "mean",
}

FUZZY_THRESHOLD_PARAMS = {
    # Mesmo modelo do contextual fuzzy, mas usado como threshold:
    # mantém voxels cuja maior pertinência local é a classe objeto.
    "threshold_mode": "contextual_object",
    **CONTEXTUAL_PARAMS,
}

FUZZY_THRESHOLD_BALANCED_PARAMS = {
    "threshold_mode": "contextual_object",
    **CONTEXTUAL_MODERATE_PARAMS,
}

FUZZY_THRESHOLD_CONSERVATIVE_PARAMS = {
    "threshold_mode": "contextual_object",
    **CONTEXTUAL_STRONG_PARAMS,
}


def variant(
    name: str,
    description: str,
    overrides: dict[str, Any],
    *,
    threshold_rule: str,
    vesselness_rule: str,
) -> dict[str, Any]:
    """Create a variant dictionary with a consistent shape."""
    return {
        "name": name,
        "description": description,
        "threshold_rule": threshold_rule,
        "vesselness_rule": vesselness_rule,
        "overrides": overrides,
    }


def default_variants() -> list[dict[str, Any]]:
    """Return the compact, article-oriented comparison set."""
    normal_threshold = "-300 <= I <= P99.7"
    fuzzy_original = "contextual object argmax, base parameters"
    fuzzy_balanced = "contextual object argmax, moderate parameters"
    fuzzy_conservative = "contextual object argmax, strong parameters"
    no_weight = "no contextual weighting"
    artery_weight = "contextual weighting on artery vesselness"

    return [
        variant(
            "normal_rg",
            "Normal threshold + region growing baseline.",
            {
                "threshold_mode": "normal",
                "contextual_apply_to": "none",
                "artery_method": "region_growing",
            },
            threshold_rule=normal_threshold,
            vesselness_rule=no_weight,
        ),
        variant(
            "fuzzy_threshold_rg",
            "Contextual-object fuzzy threshold + region growing.",
            {
                **FUZZY_THRESHOLD_PARAMS,
                "contextual_apply_to": "none",
                "artery_method": "region_growing",
            },
            threshold_rule=fuzzy_original,
            vesselness_rule=no_weight,
        ),
        variant(
            "fuzzy_threshold_balanced_rg",
            "Moderate contextual-object fuzzy threshold + region growing.",
            {
                **FUZZY_THRESHOLD_BALANCED_PARAMS,
                "contextual_apply_to": "none",
                "artery_method": "region_growing",
            },
            threshold_rule=fuzzy_balanced,
            vesselness_rule=no_weight,
        ),
        variant(
            "contextual_strong_rg",
            "Strong contextual fuzzy weighting + region growing.",
            {
                "threshold_mode": "normal",
                "contextual_apply_to": "artery",
                "artery_method": "region_growing",
                **CONTEXTUAL_STRONG_PARAMS,
            },
            threshold_rule=normal_threshold,
            vesselness_rule=artery_weight,
        ),
        variant(
            "contextual_strong_object_threshold_rg",
            "Strong contextual-object threshold + strong contextual fuzzy weighting + RG.",
            {
                **FUZZY_THRESHOLD_CONSERVATIVE_PARAMS,
                "contextual_apply_to": "artery",
                "artery_method": "region_growing",
                **CONTEXTUAL_STRONG_PARAMS,
            },
            threshold_rule=fuzzy_conservative,
            vesselness_rule=artery_weight,
        ),
        variant(
            "normal_threshold_fc",
            "Normal threshold + base fuzzy connectedness.",
            {
                "threshold_mode": "normal",
                "contextual_apply_to": "none",
                "artery_method": "fuzzy_connectedness",
                **FC_PARAMS,
            },
            threshold_rule=normal_threshold,
            vesselness_rule=no_weight,
        ),
        variant(
            "fuzzy_threshold_balanced_fc_semi_permissive",
            "Moderate contextual-object threshold + semi-permissive fuzzy connectedness.",
            {
                **FUZZY_THRESHOLD_BALANCED_PARAMS,
                "contextual_apply_to": "none",
                "artery_method": "fuzzy_connectedness",
                **FC_SEMI_PERMISSIVE_PARAMS,
            },
            threshold_rule=fuzzy_balanced,
            vesselness_rule=no_weight,
        ),
        variant(
            "contextual_strong_object_threshold_fc",
            "Strong contextual-object threshold + strong contextual fuzzy weighting + FC.",
            {
                **FUZZY_THRESHOLD_CONSERVATIVE_PARAMS,
                "contextual_apply_to": "artery",
                "artery_method": "fuzzy_connectedness",
                **CONTEXTUAL_STRONG_PARAMS,
                **FC_PARAMS,
            },
            threshold_rule=fuzzy_conservative,
            vesselness_rule=artery_weight,
        ),
    ]


def select_variants(
    all_variants: list[dict[str, Any]],
    names: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Select variants by comma-separated names and/or a simple limit."""
    selected = all_variants
    if names:
        requested = [name.strip() for name in names.split(",") if name.strip()]
        by_name = {item["name"]: item for item in all_variants}
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise ValueError(f"Unknown variants: {missing}")
        selected = [by_name[name] for name in requested]
    if limit is not None:
        selected = selected[:limit]
    return selected


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compare fuzzy/contextual/FC variants on the coronary pipeline.",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--ids",
        default=None,
        help="Comma-separated image IDs. Overrides --split/--sample-size.",
    )
    parser.add_argument("--resolution", choices=["mid", "high"], default="mid")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output folder name. Defaults to the current timestamp.",
    )
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated variant names. Defaults to all built-in variants.",
    )
    parser.add_argument(
        "--variant-limit",
        type=int,
        default=None,
        help="Run only the first N variants, useful for smoke tests.",
    )
    parser.add_argument("--cache", dest="load_cache", action="store_true")
    parser.add_argument("--save-cache", action="store_true")
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu", dest="use_gpu", action="store_true", default=None)
    gpu_group.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    return parser


def make_diagnostics(run_dir: Path, image_rows: list[dict[str, Any]]) -> None:
    """Write extra CSVs that help decide when FC or RG is better."""
    df = pd.DataFrame(image_rows)
    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    best_idx = df.groupby("IMG_ID")["dice_artery"].idxmax()
    best_by_image = df.loc[
        best_idx,
        [
            "IMG_ID",
            "variant",
            "dice_artery",
            "ostia_status",
            "ostia_success",
            "artery_voxels",
        ],
    ].sort_values("IMG_ID")
    csv_safe(best_by_image).to_csv(diagnostics_dir / "best_by_image.csv", index=False)

    pairs = [
        ("normal_rg", "normal_threshold_fc"),
        ("contextual_strong_rg", "contextual_strong_fc"),
        ("fuzzy_threshold_balanced_rg", "fuzzy_threshold_balanced_fc_semi_permissive"),
        ("contextual_strong_object_threshold_rg", "contextual_strong_object_threshold_fc"),
    ]
    delta_frames = []
    available = set(df["variant"])
    for rg_variant, fc_variant in pairs:
        if rg_variant not in available or fc_variant not in available:
            continue
        pair_df = df[df["variant"].isin([rg_variant, fc_variant])]
        wide = pair_df.pivot_table(
            index="IMG_ID",
            columns="variant",
            values="dice_artery",
            aggfunc="first",
        ).reset_index()
        wide["rg_variant"] = rg_variant
        wide["fc_variant"] = fc_variant
        wide["fc_minus_rg"] = wide[fc_variant] - wide[rg_variant]
        status_df = df[df["variant"] == rg_variant][
            ["IMG_ID", "ostia_status", "left_dist_mm", "right_dist_mm"]
        ]
        delta_frames.append(wide.merge(status_df, on="IMG_ID", how="left"))

    if delta_frames:
        deltas = pd.concat(delta_frames, ignore_index=True)
        csv_safe(deltas).to_csv(diagnostics_dir / "fc_vs_rg_deltas.csv", index=False)


def main() -> None:
    """Run all selected variants and save compact CSV outputs."""
    args = build_parser().parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = sanitize_name(args.run_name or timestamp)
    run_dir = args.output_root / run_name
    cache_dir = run_dir / "cache"
    run_dir.mkdir(parents=True, exist_ok=True)

    base_path = resolve_imagecas_base_path()
    base_config_args = SimpleNamespace(
        config_path=args.config_path,
        resolution=args.resolution,
        load_cache=args.load_cache,
        save_cache=args.save_cache,
        use_gpu=args.use_gpu,
    )
    base_config = build_base_config(base_config_args)
    image_ids = select_ids(
        args.split,
        args.sample_size,
        args.start_index,
        args.ids,
        base_path,
    )
    variants = select_variants(default_variants(), args.variants, args.variant_limit)

    write_json(
        run_dir / "run_config.json",
        {
            "split": args.split,
            "sample_size": args.sample_size,
            "start_index": args.start_index,
            "ids": image_ids,
            "resolution": args.resolution,
            "config_path": str(args.config_path),
            "base_path": str(base_path),
            "load_cache": args.load_cache,
            "save_cache": args.save_cache,
            "use_gpu": base_config.get("USE_GPU"),
            "variants": variants,
            "run_dir": str(run_dir),
        },
    )

    summaries: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []

    print(f"Run dir: {run_dir}")
    print(f"Images ({len(image_ids)}): {image_ids}")
    print(f"Variants ({len(variants)}): {[item['name'] for item in variants]}")

    for variant_index, current_variant in enumerate(variants, start=1):
        variant_name = sanitize_name(current_variant["name"])
        overrides = current_variant.get("overrides", {})
        config_overrides, experiment = split_overrides(overrides)
        config = apply_overrides(base_config, config_overrides)
        config = scale_config_to_resolution(config)
        parameter_rows.append(parameter_row(variant_name, overrides, config, experiment))

        print(f"\n[{variant_index}/{len(variants)}] {variant_name}")
        start_time = time.time()
        variant_rows = []
        for img_index, img_id in enumerate(image_ids, start=1):
            print(f"  [{img_index}/{len(image_ids)}] IMG_ID={img_id}")
            row = run_image(
                img_id,
                variant_name,
                args.split,
                base_path,
                cache_dir,
                config,
                experiment,
            )
            variant_rows.append(row)
            image_rows.append(row)

        runtime_seconds = time.time() - start_time
        summaries.append(summarize_variant(variant_name, variant_rows, runtime_seconds))
        save_outputs(run_dir, summaries, image_rows, parameter_rows)
        make_diagnostics(run_dir, image_rows)

    summary_df = pd.DataFrame(summaries).sort_values(
        [
            "selection_score",
            "ostia_success_rate",
            "mean_dice_success_ostia",
            "mean_dice",
        ],
        ascending=[False, False, False, False],
        na_position="last",
    )
    print("\nRanking:")
    print(summary_df.to_string(index=False))
    print(f"\nCSV ranking: {run_dir / 'summary' / 'ranking.csv'}")
    print(f"CSV por imagem: {run_dir / 'results' / 'image_results.csv'}")
    print(f"CSV parâmetros: {run_dir / 'parameters' / 'variant_parameters.csv'}")
    print(f"CSV melhor por imagem: {run_dir / 'diagnostics' / 'best_by_image.csv'}")
    print(f"CSV FC vs RG: {run_dir / 'diagnostics' / 'fc_vs_rg_deltas.csv'}")


if __name__ == "__main__":
    main()
