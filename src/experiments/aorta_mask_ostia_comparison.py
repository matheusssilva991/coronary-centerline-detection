"""Compara refinamentos da máscara da aorta e da seleção dos óstios."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.experiments.sweep_common import (  # noqa: E402
    apply_overrides,
    load_json_file,
    write_json,
)


DEFAULT_CONFIG = REPO_ROOT / "config/pipeline_config.json"
DEFAULT_SPLITS = REPO_ROOT / "config/imagecas_splits.json"
DEFAULT_OUTPUT = REPO_ROOT / "output/segmentation/analysis/aorta_mask_ostia_comparison"

QUICK_IDS = [
    23,
    11,
    134,
    248,
    362,
    376,
    606,
    705,
    798,
    867,
    450,
    788,
    10,
    605,
    553,
    922,
    682,
    934,
    737,
    510,
    2,
    420,
    881,
    906,
]

VARIANTS = [
    {
        "name": "baseline",
        "description": "Configuração atual sem refinamento experimental.",
        "overrides": {},
    },
    {
        "name": "short_z_region",
        "description": "Busca candidatos nos 60% iniciais da extensão da aorta.",
        "overrides": {"OSTIA_DETECTION.lower_fraction": 0.60},
    },
    {
        "name": "thin_surface",
        "description": "Usa superfície mais fina, com erosão de raio 2.",
        "overrides": {"OSTIA_DETECTION.erosion_radius": 2},
    },
    {
        "name": "trajectory_only_f150",
        "description": "Aplica somente o envelope da trajetória com fator 1.50.",
        "overrides": {"LEVEL_SET.trajectory_radius_factor": 1.50},
    },
    {
        "name": "trajectory_only_f175",
        "description": "Aplica somente o envelope da trajetória com fator 1.75.",
        "overrides": {"LEVEL_SET.trajectory_radius_factor": 1.75},
    },
    {
        "name": "trajectory_only_f200",
        "description": "Aplica somente o envelope da trajetória com fator 2.00.",
        "overrides": {"LEVEL_SET.trajectory_radius_factor": 2.00},
    },
    {
        "name": "trajectory_f175_thin_surface",
        "description": "Combina envelope 1.75 com superfície de erosão 2.",
        "overrides": {
            "LEVEL_SET.trajectory_radius_factor": 1.75,
            "OSTIA_DETECTION.erosion_radius": 2,
        },
    },
    {
        "name": "local_vesselness",
        "description": "Ordena candidatos pela média local do vesselness.",
        "overrides": {
            "OSTIA_DETECTION.candidate_score_mode": "local_mean",
            "OSTIA_DETECTION.candidate_score_radius": 2,
        },
    },
    {
        "name": "external_vesselness",
        "description": "Prioriza suporte de vesselness fora da máscara da aorta.",
        "overrides": {
            "OSTIA_DETECTION.candidate_score_mode": "external_mean",
            "OSTIA_DETECTION.candidate_score_radius": 2,
        },
    },
    {
        "name": "physical_xy_pair",
        "description": "Valida separação do par no plano XY em milímetros.",
        "overrides": {"OSTIA_DETECTION.pair_distance_mode": "physical_xy"},
    },
    {
        "name": "combined_local",
        "description": "Combina região curta, superfície fina e média local.",
        "overrides": {
            "OSTIA_DETECTION.lower_fraction": 0.60,
            "OSTIA_DETECTION.erosion_radius": 2,
            "OSTIA_DETECTION.candidate_score_mode": "local_mean",
            "OSTIA_DETECTION.candidate_score_radius": 2,
        },
    },
    {
        "name": "combined_external",
        "description": "Combina região curta, superfície fina e suporte externo.",
        "overrides": {
            "OSTIA_DETECTION.lower_fraction": 0.60,
            "OSTIA_DETECTION.erosion_radius": 2,
            "OSTIA_DETECTION.candidate_score_mode": "external_mean",
            "OSTIA_DETECTION.candidate_score_radius": 2,
        },
    },
    {
        "name": "trajectory_local",
        "description": "Restringe a máscara aos círculos e usa seleção local combinada.",
        "overrides": {
            "LEVEL_SET.trajectory_radius_factor": 1.5,
            "OSTIA_DETECTION.lower_fraction": 0.60,
            "OSTIA_DETECTION.erosion_radius": 2,
            "OSTIA_DETECTION.candidate_score_mode": "local_mean",
            "OSTIA_DETECTION.candidate_score_radius": 2,
        },
    },
    {
        "name": "physical_shell_15mm",
        "description": "Extrai uma casca física de 1.5 mm da máscara da aorta.",
        "overrides": {
            "OSTIA_DETECTION.surface_mode": "physical_distance",
            "OSTIA_DETECTION.surface_thickness_mm": 1.5,
        },
    },
    {
        "name": "physical_shell_20mm",
        "description": "Extrai uma casca física de 2.0 mm da máscara da aorta.",
        "overrides": {
            "OSTIA_DETECTION.surface_mode": "physical_distance",
            "OSTIA_DETECTION.surface_thickness_mm": 2.0,
        },
    },
    {
        "name": "candidate_nms_3mm",
        "description": "Mantém máximos locais separados por aproximadamente 3 mm.",
        "overrides": {
            "OSTIA_DETECTION.candidate_suppression_radius_mm": 3.0,
        },
    },
    {
        "name": "candidate_nms_4mm",
        "description": "Mantém máximos locais separados por aproximadamente 4 mm.",
        "overrides": {
            "OSTIA_DETECTION.candidate_suppression_radius_mm": 4.0,
        },
    },
    {
        "name": "joint_pair",
        "description": "Seleciona globalmente o par válido de maior score.",
        "overrides": {
            "OSTIA_DETECTION.pair_selection_mode": "joint",
            "OSTIA_DETECTION.joint_pair_top_k": 100,
        },
    },
    {
        "name": "nms4_joint_pair",
        "description": "Combina máximos locais de 4 mm e seleção conjunta.",
        "overrides": {
            "OSTIA_DETECTION.candidate_suppression_radius_mm": 4.0,
            "OSTIA_DETECTION.pair_selection_mode": "joint",
            "OSTIA_DETECTION.joint_pair_top_k": 100,
        },
    },
    {
        "name": "robust_score_p90_w30",
        "description": "Combina 70% do voxel com 30% do percentil local 90.",
        "overrides": {
            "OSTIA_DETECTION.candidate_score_mode": "robust_percentile",
            "OSTIA_DETECTION.candidate_score_radius": 2,
            "OSTIA_DETECTION.candidate_local_percentile": 90.0,
            "OSTIA_DETECTION.candidate_point_weight": 0.7,
        },
    },
    {
        "name": "robust_nms4_joint",
        "description": "Combina score robusto, máximos de 4 mm e par conjunto.",
        "overrides": {
            "OSTIA_DETECTION.candidate_score_mode": "robust_percentile",
            "OSTIA_DETECTION.candidate_score_radius": 2,
            "OSTIA_DETECTION.candidate_local_percentile": 90.0,
            "OSTIA_DETECTION.candidate_point_weight": 0.7,
            "OSTIA_DETECTION.candidate_suppression_radius_mm": 4.0,
            "OSTIA_DETECTION.pair_selection_mode": "joint",
            "OSTIA_DETECTION.joint_pair_top_k": 100,
        },
    },
    {
        "name": "thin_nms4_joint",
        "description": "Combina superfície fina, máximos de 4 mm e par conjunto.",
        "overrides": {
            "OSTIA_DETECTION.erosion_radius": 2,
            "OSTIA_DETECTION.candidate_suppression_radius_mm": 4.0,
            "OSTIA_DETECTION.pair_selection_mode": "joint",
            "OSTIA_DETECTION.joint_pair_top_k": 100,
        },
    },
    {
        "name": "conditional_mask_a175",
        "description": "Corrige fatias com área acima de 1.75 vezes o círculo.",
        "overrides": {
            "LEVEL_SET.trajectory_area_ratio_threshold": 1.75,
            "LEVEL_SET.trajectory_correction_radius_factor": 1.75,
        },
    },
    {
        "name": "conditional_mask_a200",
        "description": "Corrige fatias com área acima de 2.00 vezes o círculo.",
        "overrides": {
            "LEVEL_SET.trajectory_area_ratio_threshold": 2.0,
            "LEVEL_SET.trajectory_correction_radius_factor": 1.75,
        },
    },
    {
        "name": "thin_conditional_a200",
        "description": "Combina erosão 2 e correção condicional de área 2.00.",
        "overrides": {
            "OSTIA_DETECTION.erosion_radius": 2,
            "LEVEL_SET.trajectory_area_ratio_threshold": 2.0,
            "LEVEL_SET.trajectory_correction_radius_factor": 1.75,
        },
    },
    {
        "name": "bilateral_pair",
        "description": "Seleciona um candidato de cada lado do centro da aorta.",
        "overrides": {
            "OSTIA_DETECTION.pair_selection_mode": "bilateral",
            "OSTIA_DETECTION.bilateral_top_k_per_side": 50,
        },
    },
    {
        "name": "bilateral_thin",
        "description": "Combina seleção bilateral com superfície de erosão 2.",
        "overrides": {
            "OSTIA_DETECTION.erosion_radius": 2,
            "OSTIA_DETECTION.pair_selection_mode": "bilateral",
            "OSTIA_DETECTION.bilateral_top_k_per_side": 50,
        },
    },
    {
        "name": "bilateral_thin_conditional",
        "description": "Combina seleção bilateral, erosão 2 e correção condicional.",
        "overrides": {
            "OSTIA_DETECTION.erosion_radius": 2,
            "OSTIA_DETECTION.pair_selection_mode": "bilateral",
            "OSTIA_DETECTION.bilateral_top_k_per_side": 50,
            "LEVEL_SET.trajectory_area_ratio_threshold": 2.0,
            "LEVEL_SET.trajectory_correction_radius_factor": 1.75,
        },
    },
    {
        "name": "physical20_nms4_joint",
        "description": "Combina casca de 2 mm, máximos de 4 mm e par conjunto.",
        "overrides": {
            "OSTIA_DETECTION.surface_mode": "physical_distance",
            "OSTIA_DETECTION.surface_thickness_mm": 2.0,
            "OSTIA_DETECTION.candidate_suppression_radius_mm": 4.0,
            "OSTIA_DETECTION.pair_selection_mode": "joint",
            "OSTIA_DETECTION.joint_pair_top_k": 100,
        },
    },
]


def parse_ids(value: str | None) -> list[int] | None:
    """Converte uma lista de IDs separada por vírgulas."""
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_sample_split(
    source_path: Path,
    split: str,
    sample_size: int,
    start_index: int,
    explicit_ids: list[int] | None,
    output_path: Path,
) -> list[int]:
    """Cria um arquivo de splits completo com a amostra escolhida."""
    source = load_json_file(source_path)
    source_splits = source["splits"]
    available = [int(value) for value in source_splits[split]]
    if start_index < 0:
        raise ValueError("start_index deve ser >= 0")
    selected = (
        explicit_ids
        if explicit_ids is not None
        else available[start_index : start_index + sample_size]
    )
    if explicit_ids is None and len(selected) != sample_size:
        raise ValueError(
            f"O split {split} não possui {sample_size} IDs a partir do índice "
            f"{start_index}."
        )
    invalid = sorted(set(selected) - set(available))
    if invalid:
        raise ValueError(f"IDs fora do split {split}: {invalid}")
    if len(selected) != len(set(selected)):
        raise ValueError("A lista de IDs contém duplicatas.")
    if not selected:
        raise ValueError("Nenhuma imagem foi selecionada.")

    all_ids = [
        int(value)
        for split_values in source_splits.values()
        for value in split_values
    ]
    selected_set = set(selected)
    remaining = [value for value in all_ids if value not in selected_set]
    target_splits = {"train": [], "val": [], "test": []}
    target_splits[split] = selected
    target_splits["train" if split != "train" else "test"] = remaining
    write_json(
        output_path,
        {
            "metadata": {
                "purpose": "aorta mask and ostia comparison",
                "source": str(source_path),
                "sample_size": len(selected),
                "start_index": start_index,
                "target_split": split,
            },
            "splits": target_splits,
        },
    )
    return selected


def latest_pipeline_run(output_root: Path, resolution: str) -> Path | None:
    """Localiza o run mais recente criado para uma variante."""
    runs_root = output_root / "segmentation" / "runs" / f"{resolution}_res"
    if not runs_root.exists():
        return None
    runs = [path for path in runs_root.iterdir() if path.is_dir()]
    return max(runs, key=lambda path: path.stat().st_mtime) if runs else None


def read_pipeline_summary(run_dir: Path | None, split: str) -> pd.DataFrame:
    """Carrega a tabela por imagem produzida pelo pipeline."""
    if run_dir is None:
        return pd.DataFrame()
    path = run_dir / "numeric" / f"ostios_{split}_summary.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def ostia_success(status: pd.Series) -> pd.Series:
    """Converte o status textual no sucesso inclusivo de 7 mm."""
    return status.astype(str).isin(
        {"both correct", "both tolerable", "both ostia correct", "both ostia tolerable"}
    )


def summarize_variant(
    variant: dict[str, Any],
    image_df: pd.DataFrame,
    duration_seconds: float,
    return_code: int,
    run_dir: Path | None,
) -> dict[str, Any]:
    """Resume seleção dos óstios, Dice e tamanho da máscara da aorta."""
    row = {
        "variant": variant["name"],
        "description": variant["description"],
        "return_code": return_code,
        "duration_minutes": duration_seconds / 60.0,
        "n_images": len(image_df),
        "pipeline_run": None if run_dir is None else str(run_dir),
    }
    if image_df.empty:
        return row

    status = image_df["ostia_detection_status"]
    success = ostia_success(status)
    dice = pd.to_numeric(image_df["artery_dice"], errors="coerce")
    mask_voxels = pd.to_numeric(
        image_df.get("aorta_mask_voxel_count"), errors="coerce"
    )
    row.update(
        {
            "ostia_success_n": int(success.sum()),
            "ostia_success_rate": float(success.mean()),
            "both_correct_n": int(status.isin({"both correct", "both ostia correct"}).sum()),
            "both_tolerable_n": int(
                status.isin({"both tolerable", "both ostia tolerable"}).sum()
            ),
            "mean_dice": float(dice.mean()),
            "median_dice": float(dice.median()),
            "mean_aorta_mask_voxels": float(mask_voxels.mean()),
            "median_aorta_mask_voxels": float(mask_voxels.median()),
        }
    )
    return row


def save_pairwise(image_results: pd.DataFrame, results_dir: Path) -> None:
    """Compara cada variante com o baseline por imagem e por desfecho."""
    if image_results.empty or "baseline" not in set(image_results["variant"]):
        return
    columns = ["IMG_ID", "artery_dice", "ostia_detection_status"]
    baseline = image_results[image_results["variant"] == "baseline"][columns].copy()
    baseline = baseline.rename(
        columns={
            "artery_dice": "baseline_dice",
            "ostia_detection_status": "baseline_ostia_status",
        }
    )
    baseline["baseline_ostia_success"] = ostia_success(
        baseline["baseline_ostia_status"]
    )

    comparisons = []
    for name in sorted(set(image_results["variant"]) - {"baseline"}):
        current = image_results[image_results["variant"] == name][columns].copy()
        current = current.rename(
            columns={
                "artery_dice": "variant_dice",
                "ostia_detection_status": "variant_ostia_status",
            }
        )
        current["variant_ostia_success"] = ostia_success(
            current["variant_ostia_status"]
        )
        pair = baseline.merge(current, on="IMG_ID", how="inner")
        pair.insert(1, "variant", name)
        pair["dice_delta"] = (
            pd.to_numeric(pair["variant_dice"], errors="coerce")
            - pd.to_numeric(pair["baseline_dice"], errors="coerce")
        )
        pair["ostia_outcome"] = "unchanged_failure"
        pair.loc[
            pair["baseline_ostia_success"] & pair["variant_ostia_success"],
            "ostia_outcome",
        ] = "unchanged_success"
        pair.loc[
            ~pair["baseline_ostia_success"] & pair["variant_ostia_success"],
            "ostia_outcome",
        ] = "improved"
        pair.loc[
            pair["baseline_ostia_success"] & ~pair["variant_ostia_success"],
            "ostia_outcome",
        ] = "worsened"
        comparisons.append(pair)

    if not comparisons:
        return
    pairwise = pd.concat(comparisons, ignore_index=True)
    pairwise.to_csv(results_dir / "pairwise_by_image.csv", index=False)
    summary = (
        pairwise.groupby(["variant", "ostia_outcome"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    dice_summary = pairwise.groupby("variant", observed=True)["dice_delta"].agg(
        mean_dice_delta="mean",
        median_dice_delta="median",
    )
    summary.merge(dice_summary, on="variant", how="left").to_csv(
        results_dir / "pairwise_summary.csv",
        index=False,
    )


def build_parser() -> argparse.ArgumentParser:
    """Cria a CLI do experimento."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--ids", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--variants", default=None)
    parser.add_argument("--resolution", choices=["mid", "high"], default="mid")
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split-config-source", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", dest="use_gpu", action="store_true", default=True)
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def selected_variants(names: str | None) -> list[dict[str, Any]]:
    """Filtra variantes por nomes separados por vírgula."""
    if not names:
        return VARIANTS
    requested = {name.strip() for name in names.split(",") if name.strip()}
    known = {variant["name"] for variant in VARIANTS}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"Variantes desconhecidas: {unknown}. Disponíveis: {sorted(known)}")
    return [variant for variant in VARIANTS if variant["name"] in requested]


def main() -> None:
    """Executa as variantes e salva tabelas compactas."""
    args = build_parser().parse_args()
    variants = selected_variants(args.variants)
    explicit_ids = QUICK_IDS if args.quick else parse_ids(args.ids)
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
        args.start_index,
        explicit_ids,
        split_config,
    )
    base_config = load_json_file(args.config_path)

    summary_rows: list[dict[str, Any]] = []
    image_frames: list[pd.DataFrame] = []
    for index, variant in enumerate(variants, start=1):
        config = apply_overrides(base_config, variant["overrides"])
        config["LOAD_CACHE"] = not args.no_cache
        config["SAVE_CACHE"] = not args.no_cache
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
            "--gpu" if args.use_gpu else "--no-gpu",
        ]
        if not args.no_cache:
            command.append("--cache")
        else:
            command.append("--no-save-cache")

        print(f"[{index}/{len(variants)}] {variant['name']}")
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
        image_df = read_pipeline_summary(pipeline_run, args.split)
        if not image_df.empty:
            image_df.insert(0, "variant", variant["name"])
            image_frames.append(image_df)
        summary_rows.append(
            summarize_variant(
                variant,
                image_df,
                duration,
                return_code,
                pipeline_run,
            )
        )
        if return_code != 0:
            break

    summary = pd.DataFrame(summary_rows)
    if "ostia_success_rate" in summary:
        summary = summary.sort_values(
            ["ostia_success_rate", "mean_dice"], ascending=False
        )
    summary.to_csv(results_dir / "summary.csv", index=False)
    image_results = (
        pd.concat(image_frames, ignore_index=True) if image_frames else pd.DataFrame()
    )
    image_results.to_csv(results_dir / "image_results.csv", index=False)
    save_pairwise(image_results, results_dir)
    write_json(
        run_dir / "run_config.json",
        {
            "split": args.split,
            "selected_ids": selected_ids,
            "sample_size": len(selected_ids),
            "start_index": args.start_index,
            "resolution": args.resolution,
            "variants": variants,
            "cache_enabled": not args.no_cache,
            "dry_run": args.dry_run,
        },
    )
    print(f"Resultados: {results_dir}")


if __name__ == "__main__":
    main()
