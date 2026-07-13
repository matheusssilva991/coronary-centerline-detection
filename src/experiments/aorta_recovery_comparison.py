"""Compara ajustes focados da recuperação inicial da localização da aorta."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config/pipeline_config.json"
DEFAULT_SPLITS = REPO_ROOT / "config/imagecas_splits.json"
DEFAULT_OUTPUT = REPO_ROOT / "output/segmentation/analysis/aorta_recovery_comparison"

VARIANTS = [
    {
        "name": "recovery_standard",
        "search_slices": 8,
        "require_min_circles": False,
        "description": "Recuperação validada com busca em 8 fatias.",
    },
    {
        "name": "recovery_guard",
        "search_slices": 8,
        "require_min_circles": True,
        "description": "Descarta recuperações que continuam com menos de 10 círculos.",
    },
    {
        "name": "recovery_extended",
        "search_slices": 16,
        "require_min_circles": False,
        "description": "Amplia a busca de reinicialização de 8 para 16 fatias.",
    },
]


def load_json(path: Path) -> dict[str, Any]:
    """Carrega um JSON da raiz do projeto."""
    resolved = path if path.is_absolute() else REPO_ROOT / path
    return json.loads(resolved.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Salva JSON legível criando os diretórios necessários."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def build_sample_split(
    source_path: Path,
    split: str,
    sample_size: int,
    output_path: Path,
) -> list[int]:
    """Cria split temporário completo com apenas a amostra no split avaliado."""
    source = load_json(source_path)
    source_splits = source["splits"]
    selected = [int(value) for value in source_splits[split][:sample_size]]
    if len(selected) != sample_size:
        raise ValueError(
            f"O split {split!r} possui apenas {len(source_splits[split])} IDs; "
            f"não é possível selecionar {sample_size}."
        )

    all_ids = [
        int(value)
        for split_values in source_splits.values()
        for value in split_values
    ]
    selected_set = set(selected)
    remaining = [value for value in all_ids if value not in selected_set]
    target_splits = {"train": [], "val": [], "test": []}
    target_splits[split] = selected
    holding_split = "train" if split != "train" else "test"
    target_splits[holding_split] = remaining
    write_json(
        output_path,
        {
            "metadata": {
                "purpose": "aorta recovery comparison",
                "source": str(source_path),
                "sample_size": sample_size,
                "target_split": split,
            },
            "splits": target_splits,
        },
    )
    return selected


def latest_pipeline_run(output_root: Path, resolution: str) -> Path | None:
    """Localiza o run mais recente criado dentro da pasta da variante."""
    runs_root = output_root / "segmentation" / "runs" / f"{resolution}_res"
    if not runs_root.exists():
        return None
    runs = [path for path in runs_root.iterdir() if path.is_dir()]
    return max(runs, key=lambda path: path.stat().st_mtime) if runs else None


def read_summary(run_dir: Path | None, split: str) -> pd.DataFrame:
    """Carrega o resultado por imagem do pipeline."""
    if run_dir is None:
        return pd.DataFrame()
    path = run_dir / "numeric" / f"ostios_{split}_summary.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def summarize(
    variant: dict[str, Any],
    df: pd.DataFrame,
    run_dir: Path | None,
    duration_seconds: float,
    return_code: int,
) -> dict[str, Any]:
    """Resume Dice, óstios e frequência de recuperação de uma variante."""
    row = {
        **variant,
        "run_dir": None if run_dir is None else str(run_dir),
        "return_code": return_code,
        "duration_seconds": duration_seconds,
        "duration_minutes": duration_seconds / 60.0,
        "n_images": len(df),
    }
    if df.empty:
        return row

    dice = pd.to_numeric(df.get("artery_dice"), errors="coerce")
    status = df.get("ostia_detection_status", pd.Series(index=df.index, dtype=str))
    success = status.astype(str).isin(
        {"both correct", "both tolerable", "both ostia correct", "both ostia tolerable"}
    )
    recovered = df.get(
        "aorta_recovered_initialization",
        pd.Series(False, index=df.index),
    ).astype(str).str.lower().isin({"yes", "true", "1"})
    row.update(
        {
            "ostia_success_rate": float(success.mean()),
            "ostia_success_n": int(success.sum()),
            "both_correct_n": int(
                status.isin({"both correct", "both ostia correct"}).sum()
            ),
            "both_tolerable_n": int(
                status.isin({"both tolerable", "both ostia tolerable"}).sum()
            ),
            "not_found_n": int(status.isin({"not found", "ostia not found"}).sum()),
            "mean_dice": float(dice.mean()),
            "median_dice": float(dice.median()),
            "recovered_images_n": int(recovered.sum()),
        }
    )
    return row


def save_pairwise(image_results: pd.DataFrame, output_path: Path) -> None:
    """Salva deltas por imagem das variantes contra a recuperação padrão."""
    if image_results.empty or "recovery_standard" not in set(image_results["variant"]):
        return
    value_columns = ["IMG_ID", "artery_dice", "ostia_detection_status"]
    baseline = image_results[image_results["variant"] == "recovery_standard"][
        value_columns
    ].rename(
        columns={
            "artery_dice": "artery_dice_standard",
            "ostia_detection_status": "ostia_status_standard",
        }
    )
    frames = []
    for variant_name in sorted(set(image_results["variant"]) - {"recovery_standard"}):
        comparison = image_results[image_results["variant"] == variant_name][
            value_columns
        ].rename(
            columns={
                "artery_dice": "artery_dice_variant",
                "ostia_detection_status": "ostia_status_variant",
            }
        )
        pair = baseline.merge(comparison, on="IMG_ID", how="inner")
        pair.insert(1, "variant", variant_name)
        pair["dice_delta_vs_standard"] = (
            pd.to_numeric(pair["artery_dice_variant"], errors="coerce")
            - pd.to_numeric(pair["artery_dice_standard"], errors="coerce")
        )
        frames.append(pair)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(output_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    """Cria a CLI do comparativo focado."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--resolution", choices=["mid", "high"], default="mid")
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split-config-source", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", dest="use_gpu", action="store_true", default=True)
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    """Executa as três variantes e salva resultados compactos."""
    args = build_parser().parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size deve ser maior que zero.")

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.output_root / run_name
    configs_dir = run_dir / "configs"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    split_config = run_dir / "sample_split_config.json"
    selected_ids = build_sample_split(
        args.split_config_source,
        args.split,
        args.sample_size,
        split_config,
    )
    base_config = load_json(args.config_path)

    summary_rows: list[dict[str, Any]] = []
    image_frames: list[pd.DataFrame] = []
    for index, variant in enumerate(VARIANTS, start=1):
        config = json.loads(json.dumps(base_config))
        circle = config.setdefault("CIRCLE_DETECTION", {})
        circle.update(
            {
                "early_track_recovery": True,
                "early_recovery_search_slices": variant["search_slices"],
                "early_recovery_min_circles": 10,
                "early_recovery_require_min_circles": variant[
                    "require_min_circles"
                ],
            }
        )
        config_path = configs_dir / f"{variant['name']}.json"
        write_json(config_path, config)
        variant_output = run_dir / "pipeline_runs" / variant["name"]
        command = [
            sys.executable,
            str(REPO_ROOT / "src/segmentation_pipeline.py"),
            "--split",
            args.split,
            "--resolution",
            args.resolution,
            "--num-batches",
            str(args.num_batches),
            "--config-file",
            str(config_path),
            "--split-config",
            str(split_config),
            "--output-dir",
            str(variant_output),
            "--no-save-cache",
            "--gpu" if args.use_gpu else "--no-gpu",
        ]
        print(f"[{index}/{len(VARIANTS)}] {variant['name']}")
        print(" ".join(command))
        start = time.perf_counter()
        return_code = 0
        if not args.dry_run:
            return_code = subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
        duration = time.perf_counter() - start
        pipeline_run = (
            None
            if args.dry_run
            else latest_pipeline_run(variant_output, args.resolution)
        )
        image_df = read_summary(pipeline_run, args.split)
        if not image_df.empty:
            image_df.insert(0, "variant", variant["name"])
            image_frames.append(image_df)
        summary_rows.append(
            summarize(variant, image_df, pipeline_run, duration, return_code)
        )
        if return_code != 0:
            break

    summary_df = pd.DataFrame(summary_rows)
    if "ostia_success_rate" in summary_df:
        summary_df = summary_df.sort_values(
            ["ostia_success_rate", "mean_dice"],
            ascending=False,
        )
    summary_df.to_csv(results_dir / "summary.csv", index=False)
    image_results = (
        pd.concat(image_frames, ignore_index=True) if image_frames else pd.DataFrame()
    )
    image_results.to_csv(results_dir / "image_results.csv", index=False)
    save_pairwise(image_results, results_dir / "pairwise_vs_standard.csv")
    write_json(
        run_dir / "run_config.json",
        {
            "split": args.split,
            "sample_size": args.sample_size,
            "selected_ids": selected_ids,
            "resolution": args.resolution,
            "variants": VARIANTS,
            "dry_run": args.dry_run,
        },
    )
    print(f"Resultados: {results_dir}")


if __name__ == "__main__":
    main()
