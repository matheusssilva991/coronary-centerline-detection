"""Run fuzzy pipeline comparison variants from the command line.

It compares the four retained combinations of normal/fuzzy thresholding and
region growing/fuzzy connectedness.

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
    "fc.alpha": 0.16,
    "fc.sigma_hu": 100,
    "fc.neighborhood": 26,
    "fc.candidate_min_vesselness": 0.018,
    "fc.seed_search_radius": 2,
    "fc.max_seeds_per_ostium": 4,
    "fc.seed_min_vesselness": 0.018,
    "fc.min_seed_distance_voxels": 1.0,
    "fc.vesselness_floor": 0.018,
    "fc.vesselness_weight": 0.9,
}

FUZZY_PARAMS = {
    "LOWER_THRESHOLD.percentile": 10.5,
    "fuzzy.soft_margin_hu": 100,
    "fuzzy.object_percentile": 99.8,
    "fuzzy.dense_percentile": 99.96,
    "fuzzy.smooth_radius": 0,
    "fuzzy.smooth_mode": "mean",
}

FUZZY_THRESHOLD_PARAMS = {
    "threshold_mode": "fuzzy",
    **FUZZY_PARAMS,
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
    normal_threshold = "P10.75 <= I <= P99.8"
    fuzzy_threshold = "P10.5 + fuzzy object argmax (P99.8/P99.96)"
    no_weight = "no vesselness weighting"

    return [
        variant(
            "normal_rg",
            "Normal threshold + region growing baseline.",
            {
                "threshold_mode": "normal",
                "artery_method": "region_growing",
            },
            threshold_rule=normal_threshold,
            vesselness_rule=no_weight,
        ),
        variant(
            "fuzzy_threshold_rg",
            "Fuzzy threshold + region growing.",
            {
                **FUZZY_THRESHOLD_PARAMS,
                "artery_method": "region_growing",
            },
            threshold_rule=fuzzy_threshold,
            vesselness_rule=no_weight,
        ),
        variant(
            "normal_threshold_fc",
            "Normal threshold + fuzzy connectedness.",
            {
                "threshold_mode": "normal",
                "artery_method": "fuzzy_connectedness",
                **FC_PARAMS,
            },
            threshold_rule=normal_threshold,
            vesselness_rule=no_weight,
        ),
        variant(
            "fuzzy_threshold_fc",
            "Fuzzy threshold + fuzzy connectedness.",
            {
                **FUZZY_THRESHOLD_PARAMS,
                "artery_method": "fuzzy_connectedness",
                **FC_PARAMS,
            },
            threshold_rule=fuzzy_threshold,
            vesselness_rule=no_weight,
        ),
    ]


def failure_correction_variants() -> list[dict[str, Any]]:
    """Return focused, interpretable corrections around the current defaults."""
    variants = default_variants()
    normal_threshold = "P10.75 <= I <= P99.8"
    fuzzy_dense_threshold = "P10.5 + fuzzy object argmax (P99.8/P99.98)"
    no_weight = "no vesselness weighting"

    variants.extend(
        [
            variant(
                "normal_rg_relaxed",
                "RG with a lower vesselness floor and broader local seeds.",
                {
                    "threshold_mode": "normal",
                    "artery_method": "region_growing",
                    "REGION_GROWING.min_vesselness_fraction": 0.065,
                    "REGION_GROWING.threshold_divisor": 6,
                    "REGION_GROWING.seed_candidate_radius": 3,
                    "REGION_GROWING.max_seed_candidates": 8,
                },
                threshold_rule=normal_threshold,
                vesselness_rule=no_weight,
            ),
            variant(
                "normal_fc_alpha014",
                "FC with a lower final connectivity threshold.",
                {
                    "threshold_mode": "normal",
                    "artery_method": "fuzzy_connectedness",
                    **FC_PARAMS,
                    "fc.alpha": 0.14,
                },
                threshold_rule=normal_threshold,
                vesselness_rule=no_weight,
            ),
            variant(
                "normal_fc_relaxed",
                "FC with permissive candidate floor and broader multiseed search.",
                {
                    "threshold_mode": "normal",
                    "artery_method": "fuzzy_connectedness",
                    **FC_PARAMS,
                    "fc.alpha": 0.14,
                    "fc.candidate_min_vesselness": 0.014,
                    "fc.seed_min_vesselness": 0.014,
                    "fc.vesselness_floor": 0.014,
                    "fc.seed_search_radius": 4,
                    "fc.max_seeds_per_ostium": 8,
                },
                threshold_rule=normal_threshold,
                vesselness_rule=no_weight,
            ),
            variant(
                "fuzzy_threshold_fc_relaxed",
                "Fuzzy threshold with the permissive FC recovery parameters.",
                {
                    **FUZZY_THRESHOLD_PARAMS,
                    "artery_method": "fuzzy_connectedness",
                    **FC_PARAMS,
                    "fc.alpha": 0.14,
                    "fc.candidate_min_vesselness": 0.014,
                    "fc.seed_min_vesselness": 0.014,
                    "fc.vesselness_floor": 0.014,
                    "fc.seed_search_radius": 4,
                    "fc.max_seeds_per_ostium": 8,
                },
                threshold_rule="P10.5 + fuzzy object argmax (P99.8/P99.96)",
                vesselness_rule=no_weight,
            ),
            variant(
                "fuzzy_threshold_rg_dense9998",
                "Fuzzy threshold with a more permissive dense-background center.",
                {
                    **FUZZY_THRESHOLD_PARAMS,
                    "fuzzy.dense_percentile": 99.98,
                    "artery_method": "region_growing",
                },
                threshold_rule=fuzzy_dense_threshold,
                vesselness_rule=no_weight,
            ),
            variant(
                "fuzzy_threshold_fc_dense9998",
                "Fuzzy P99.98 dense center with the current FC parameters.",
                {
                    **FUZZY_THRESHOLD_PARAMS,
                    "fuzzy.dense_percentile": 99.98,
                    "artery_method": "fuzzy_connectedness",
                    **FC_PARAMS,
                },
                threshold_rule=fuzzy_dense_threshold,
                vesselness_rule=no_weight,
            ),
        ]
    )
    return variants


def variants_for_set(name: str) -> list[dict[str, Any]]:
    """Resolve a named experiment set without changing article defaults."""
    if name in {"article", "baseline"}:
        return default_variants()
    if name == "corrections":
        return failure_correction_variants()
    raise ValueError(f"Unknown variant set: {name}")


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


def parse_split_sizes(value: str) -> list[tuple[str, int]]:
    """Parse ``train:30,val:30`` into ordered split/size pairs."""
    split_sizes: list[tuple[str, int]] = []
    valid_splits = {"train", "val", "test"}
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid --split-sizes item {item!r}. Use the form train:30,val:30."
            )
        split_name, raw_size = [part.strip() for part in item.split(":", 1)]
        if split_name not in valid_splits:
            raise ValueError(
                f"Invalid split {split_name!r}. Expected one of {sorted(valid_splits)}."
            )
        size = int(raw_size)
        if size <= 0:
            raise ValueError("Split sizes must be > 0.")
        split_sizes.append((split_name, size))
    if not split_sizes:
        raise ValueError("--split-sizes cannot be empty.")
    return split_sizes


def select_image_items(
    split: str,
    sample_size: int,
    start_index: int,
    ids_arg: str | None,
    split_sizes_arg: str | None,
    base_path: Path,
) -> list[tuple[str, int]]:
    """Select images while preserving the split name for each ID."""
    if ids_arg:
        image_ids = select_ids(split, sample_size, start_index, ids_arg, base_path)
        allowed_ids = set(select_ids(split, 10_000, 0, None, base_path))
        invalid_ids = sorted(set(image_ids) - allowed_ids)
        if invalid_ids:
            raise ValueError(
                f"IDs outside split {split!r}: {invalid_ids}. "
                "Parameter selection must remain inside the requested split."
            )
        return [(split, img_id) for img_id in image_ids]

    if split_sizes_arg:
        image_items: list[tuple[str, int]] = []
        for split_name, requested_size in parse_split_sizes(split_sizes_arg):
            selected_ids = select_ids(
                split_name,
                requested_size,
                start_index,
                None,
                base_path,
            )
            if len(selected_ids) != requested_size:
                raise ValueError(
                    f"Split {split_name!r} returned {len(selected_ids)} images, "
                    f"but {requested_size} were requested. "
                    "Use a smaller size or another split."
                )
            image_items.extend((split_name, img_id) for img_id in selected_ids)
        return image_items

    image_ids = select_ids(split, sample_size, start_index, None, base_path)
    return [(split, img_id) for img_id in image_ids]


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compare fuzzy threshold, RG and FC variants on the coronary pipeline.",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--ids",
        default=None,
        help="Comma-separated image IDs. Overrides --split/--sample-size.",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help="CSV containing an IMG_ID column. Cannot be combined with --ids.",
    )
    parser.add_argument(
        "--split-sizes",
        default=None,
        help=(
            "Comma-separated split sizes, for example train:30,val:30. "
            "Ignored when --ids is provided."
        ),
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
        "--variant-set",
        choices=["article", "baseline", "corrections"],
        default="article",
        help="Named variant set. 'corrections' adds focused RG/FC/fuzzy tests.",
    )
    parser.add_argument(
        "--variant-limit",
        type=int,
        default=None,
        help="Run only the first N variants, useful for smoke tests.",
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu", dest="use_gpu", action="store_true", default=None)
    gpu_group.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate selection/configuration and write run_config.json only.",
    )
    return parser


def load_ids_csv(path: Path) -> pd.DataFrame:
    """Read and validate an ID/cohort CSV."""
    frame = pd.read_csv(path)
    if "IMG_ID" not in frame.columns:
        raise ValueError(f"{path} must contain an IMG_ID column.")
    ids = pd.to_numeric(frame["IMG_ID"], errors="raise").astype(int).tolist()
    if not ids:
        raise ValueError(f"{path} does not contain image IDs.")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate IMG_ID values.")
    frame = frame.copy()
    frame["IMG_ID"] = ids
    return frame


def make_diagnostics(run_dir: Path, image_rows: list[dict[str, Any]]) -> None:
    """Write extra CSVs that help decide when FC or RG is better."""
    df = pd.DataFrame(image_rows)
    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    df["dice_artery"] = pd.to_numeric(df["dice_artery"], errors="coerce")
    result_columns = [
        "IMG_ID",
        "variant",
        "dice_artery",
        "dice_artery_before_morphology",
        "dice_artery_after_morphology",
        "dice_artery_morphology_delta",
        "ostia_status",
        "ostia_success",
        "artery_voxels",
    ]

    # Some images can fail in every variant and therefore have only NaN Dice.
    # Keep diagnostics generation robust and list those images separately.
    valid_dice_df = df.dropna(subset=["dice_artery"])
    missing_dice_df = df[
        df.groupby("IMG_ID")["dice_artery"].transform(lambda values: values.notna().sum())
        == 0
    ]
    if not missing_dice_df.empty:
        missing_images = (
            missing_dice_df[
                ["IMG_ID", "variant", "ostia_status", "ostia_success", "artery_voxels"]
            ]
            .sort_values(["IMG_ID", "variant"])
            .reset_index(drop=True)
        )
        csv_safe(missing_images).to_csv(
            diagnostics_dir / "images_without_valid_dice.csv",
            index=False,
        )

    if not valid_dice_df.empty:
        best_idx = valid_dice_df.groupby("IMG_ID")["dice_artery"].idxmax()
        best_by_image = df.loc[best_idx, result_columns].sort_values("IMG_ID")
        csv_safe(best_by_image).to_csv(
            diagnostics_dir / "best_by_image.csv",
            index=False,
        )

    pairs = [
        ("normal_rg", "normal_threshold_fc"),
        ("fuzzy_threshold_rg", "fuzzy_threshold_fc"),
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
            dropna=False,
        ).reindex(columns=[rg_variant, fc_variant])
        wide = wide.reset_index()
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

    correction_pairs = [
        ("normal_rg", "normal_rg_relaxed"),
        ("normal_threshold_fc", "normal_fc_alpha014"),
        ("normal_threshold_fc", "normal_fc_relaxed"),
        ("fuzzy_threshold_fc", "fuzzy_threshold_fc_relaxed"),
        ("fuzzy_threshold_rg", "fuzzy_threshold_rg_dense9998"),
        ("fuzzy_threshold_fc", "fuzzy_threshold_fc_dense9998"),
    ]
    correction_frames = []
    for baseline, correction in correction_pairs:
        if baseline not in available or correction not in available:
            continue
        pair = df[df["variant"].isin([baseline, correction])].pivot_table(
            index="IMG_ID",
            columns="variant",
            values="dice_artery",
            aggfunc="first",
            dropna=False,
        ).reindex(columns=[baseline, correction])
        pair["baseline_variant"] = baseline
        pair["correction_variant"] = correction
        pair["dice_delta"] = pair[correction] - pair[baseline]
        correction_frames.append(pair.reset_index())
    if correction_frames:
        csv_safe(pd.concat(correction_frames, ignore_index=True)).to_csv(
            diagnostics_dir / "correction_deltas.csv",
            index=False,
        )

    if "cohort_roles" in df and df["cohort_roles"].notna().any():
        by_role = df.copy()
        by_role["cohort_role"] = by_role["cohort_roles"].fillna("").str.split(";")
        by_role = by_role.explode("cohort_role")
        by_role = by_role.loc[by_role["cohort_role"].ne("")]
        role_summary = (
            by_role.groupby(["cohort_role", "variant"], as_index=False)
            .agg(
                images=("IMG_ID", "nunique"),
                mean_dice=("dice_artery", "mean"),
                median_dice=("dice_artery", "median"),
                ostia_success_rate=("ostia_success", "mean"),
                mean_artery_voxels=("artery_voxels", "mean"),
            )
            .sort_values(["cohort_role", "mean_dice"], ascending=[True, False])
        )
        csv_safe(role_summary).to_csv(
            diagnostics_dir / "summary_by_failure_role.csv",
            index=False,
        )


def main() -> None:
    """Run all selected variants and save compact CSV outputs."""
    args = build_parser().parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = sanitize_name(args.run_name or timestamp)
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    base_path = resolve_imagecas_base_path()
    base_config_args = SimpleNamespace(
        config_path=args.config_path,
        resolution=args.resolution,
        use_gpu=args.use_gpu,
    )
    base_config = build_base_config(base_config_args)
    if args.ids and args.ids_file:
        raise ValueError("Use only one of --ids or --ids-file.")
    if args.ids_file and args.split_sizes:
        raise ValueError("--ids-file cannot be combined with --split-sizes.")
    ids_frame = load_ids_csv(args.ids_file) if args.ids_file else None
    ids_arg = (
        ",".join(str(image_id) for image_id in ids_frame["IMG_ID"])
        if ids_frame is not None
        else args.ids
    )
    cohort_metadata = (
        ids_frame.set_index("IMG_ID")
        .reindex(columns=["cohort_kind", "cohort_roles"])
        .to_dict(orient="index")
        if ids_frame is not None
        else {}
    )
    image_items = select_image_items(
        args.split,
        args.sample_size,
        args.start_index,
        ids_arg,
        args.split_sizes,
        base_path,
    )
    image_ids = [img_id for _, img_id in image_items]
    variants = select_variants(
        variants_for_set(args.variant_set),
        args.variants,
        args.variant_limit,
    )

    write_json(
        run_dir / "run_config.json",
        {
            "split": args.split,
            "split_sizes": args.split_sizes,
            "sample_size": args.sample_size,
            "start_index": args.start_index,
            "ids": image_ids,
            "ids_file": str(args.ids_file) if args.ids_file else None,
            "image_items": [
                {"split": split_name, "IMG_ID": img_id}
                for split_name, img_id in image_items
            ],
            "resolution": args.resolution,
            "variant_set": args.variant_set,
            "config_path": str(args.config_path),
            "base_path": str(base_path),
            "use_gpu": base_config.get("USE_GPU"),
            "variants": variants,
            "effective_base_config": base_config,
            "run_dir": str(run_dir),
        },
    )

    summaries: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []

    print(f"Run dir: {run_dir}")
    print(f"Images ({len(image_items)}): {image_items}")
    print(f"Variants ({len(variants)}): {[item['name'] for item in variants]}")
    if args.dry_run:
        print("Dry run complete; no image was processed.")
        return

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
        for img_index, (split_name, img_id) in enumerate(image_items, start=1):
            print(
                f"  [{img_index}/{len(image_items)}] split={split_name} IMG_ID={img_id}"
            )
            row = run_image(
                img_id,
                variant_name,
                split_name,
                base_path,
                config,
                experiment,
            )
            row.update(cohort_metadata.get(img_id, {}))
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
