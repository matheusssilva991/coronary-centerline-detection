"""Cataloga falhas das variantes baseline e fuzzy sem reexecutar o pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.comparison_utils.failure_analysis import (  # noqa: E402
    build_failure_case_catalog,
    compact_focused_failure_cohort,
    select_focused_failure_cohort,
    summarize_failure_categories,
)
from utils.visualization.variant_comparison import load_variant_results  # noqa: E402


DEFAULT_RESULTS = REPO_ROOT / "output/segmentation/runs/mid_res/fuzzy_comparison"
DEFAULT_OUTPUT = REPO_ROOT / "output/segmentation/analysis/pipeline_failure_analysis"
VARIANT_ORDER = ["normal_rg", "th_fuzzy_rg", "normal_fc", "th_fuzzy_fc"]
PRETTY_NAMES = {
    "normal_rg": "Normal threshold + RG",
    "th_fuzzy_rg": "Fuzzy threshold + RG",
    "normal_fc": "Normal threshold + FC",
    "th_fuzzy_fc": "Fuzzy threshold + FC",
}
LEGACY_DETAIL_FILES = (
    "case_catalog.csv",
    "selected_cases.csv",
    "variant_summary.csv",
)


def build_parser() -> argparse.ArgumentParser:
    """Constrói a interface de linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--allow-test-analysis", action="store_true")
    parser.add_argument("--low-dice-threshold", type=float, default=0.4)
    parser.add_argument("--meaningful-delta", type=float, default=0.05)
    parser.add_argument("--volume-ratio", type=float, default=1.5)
    parser.add_argument("--focused-max-per-category", type=int, default=6)
    parser.add_argument("--focused-shared-ostia", type=int, default=6)
    parser.add_argument("--focused-controls", type=int, default=10)
    return parser


def main() -> None:
    """Gera tabelas compactas de falha a partir dos runs concluídos."""
    args = build_parser().parse_args()
    if args.split == "test" and not args.allow_test_analysis:
        raise ValueError(
            "O split test está protegido contra seleção de parâmetros. "
            "Use validação ou informe --allow-test-analysis somente para descrição."
        )

    results_df, _ = load_variant_results(
        args.result_root,
        split=args.split,
        preferred_order=VARIANT_ORDER,
        pretty_names=PRETTY_NAMES,
        repo_root=REPO_ROOT,
    )
    catalog = build_failure_case_catalog(
        results_df,
        low_dice_threshold=args.low_dice_threshold,
        meaningful_delta=args.meaningful_delta,
        volume_ratio=args.volume_ratio,
    )
    category_summary = summarize_failure_categories(catalog)
    focused_cohort = compact_focused_failure_cohort(
        select_focused_failure_cohort(
            catalog,
            max_per_category=args.focused_max_per_category,
            shared_ostia_cases=args.focused_shared_ostia,
            stable_controls=args.focused_controls,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    category_summary.to_csv(args.output_dir / "category_summary.csv", index=False)
    focused_cohort.to_csv(args.output_dir / "focused_cohort.csv", index=False)
    for filename in LEGACY_DETAIL_FILES:
        (args.output_dir / filename).unlink(missing_ok=True)
    with (args.output_dir / "analysis_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "result_root": str(
                    args.result_root.resolve().relative_to(REPO_ROOT.resolve())
                    if args.result_root.resolve().is_relative_to(REPO_ROOT.resolve())
                    else args.result_root.resolve()
                ),
                "split": args.split,
                "image_count": len(catalog),
                "low_dice_threshold": args.low_dice_threshold,
                "meaningful_delta": args.meaningful_delta,
                "volume_ratio": args.volume_ratio,
                "test_analysis_only": args.split == "test",
                "focused_cohort_size": len(focused_cohort),
                "focused_selection": {
                    "max_per_category": args.focused_max_per_category,
                    "shared_ostia_cases": args.focused_shared_ostia,
                    "stable_controls": args.focused_controls,
                },
            },
            handle,
            indent=2,
        )

    print(f"Images compared: {len(catalog)}")
    print(f"Focused improvement cohort: {len(focused_cohort)} images")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
