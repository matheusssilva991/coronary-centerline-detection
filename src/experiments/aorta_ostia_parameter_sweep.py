"""Executa sweeps focados na localização da aorta e detecção dos óstios.

O experimento roda o pipeline oficial com pequenas variações em:

- modo da LCC antes da localização da aorta;
- quantidade de misses consecutivos permitidos;
- tratamento de candidato fora da tolerância;
- estratégia de seleção do círculo candidato.

As saídas incluem um resumo por variante e uma tabela dedicada à relação entre
quantidade de fatias da imagem e quantidade/cobertura dos círculos da aorta.

Exemplo:
    uv run python src/experiments/aorta_ostia_parameter_sweep.py \\
      --split train \\
      --resolution mid \\
      --run-name aorta_ostia_train_quick \\
      --gpu \\
      --no-save-cache
"""

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
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output/segmentation/analysis/aorta_ostia_sweep"
DEFAULT_CONFIG_PATH = REPO_ROOT / "config/pipeline_config.json"
SUCCESS_STATUSES = {"both ostia correct", "both ostia tolerable"}


def parse_csv_ints(value: str) -> list[int]:
    """Converte ``2,3,5`` em lista de inteiros."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("A lista de inteiros não pode ser vazia.")
    return [int(item) for item in items]


def parse_csv_strings(value: str) -> list[str]:
    """Converte ``a,b`` em lista textual."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("A lista textual não pode ser vazia.")
    return items


def parse_bool_modes(value: str) -> list[bool]:
    """Converte modos textuais de tolerância em booleanos."""
    aliases = {
        "stop": False,
        "stop_on_out": False,
        "false": False,
        "0": False,
        "miss": True,
        "out_as_miss": True,
        "true": True,
        "1": True,
    }
    modes = []
    for item in parse_csv_strings(value):
        key = item.lower().replace("-", "_")
        if key not in aliases:
            raise argparse.ArgumentTypeError(
                "Modo inválido em --tolerance-modes. Use: stop,miss."
            )
        modes.append(aliases[key])
    return modes


def normalize_lcc_mode(value: str) -> str:
    """Normaliza modo de LCC para ``per_slice`` ou ``per_volume``."""
    normalized = value.lower().replace("-", "_")
    aliases = {
        "slice": "per_slice",
        "per_slice": "per_slice",
        "volume": "per_volume",
        "per_volume": "per_volume",
    }
    if normalized not in aliases:
        raise argparse.ArgumentTypeError(
            "Modo inválido em --lcc-modes. Use: per_slice,per_volume."
        )
    return aliases[normalized]


def parse_lcc_modes(value: str) -> list[str]:
    """Converte lista textual de modos da LCC."""
    return [normalize_lcc_mode(item) for item in parse_csv_strings(value)]


def parse_candidate_strategies(value: str) -> list[str]:
    """Converte e valida estratégias de seleção do círculo candidato."""
    valid = {"closest", "score"}
    strategies = [item.lower() for item in parse_csv_strings(value)]
    invalid = [item for item in strategies if item not in valid]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Estratégias inválidas: {invalid}. Use: {sorted(valid)}."
        )
    return strategies


def _bool_text(value: bool) -> str:
    """Texto curto para nomes de variantes."""
    return "miss" if value else "stop"


def variant_name(
    *,
    lcc_mode: str,
    miss_count: int,
    out_of_tolerance_as_miss: bool,
    candidate_strategy: str,
) -> str:
    """Gera nome curto e legível para uma variante."""
    return (
        f"lcc_{lcc_mode}"
        f"_out_{_bool_text(out_of_tolerance_as_miss)}"
        f"_miss{miss_count}"
        f"_cand_{candidate_strategy}"
    )


def default_variants() -> list[dict[str, Any]]:
    """Retorna um grid pequeno para evitar runs muito longos."""
    return [
        {
            "lcc_mode": "per_slice",
            "miss_count": 5,
            "out_of_tolerance_as_miss": False,
            "candidate_strategy": "closest",
        },
        {
            "lcc_mode": "per_slice",
            "miss_count": 2,
            "out_of_tolerance_as_miss": True,
            "candidate_strategy": "closest",
        },
        {
            "lcc_mode": "per_slice",
            "miss_count": 3,
            "out_of_tolerance_as_miss": True,
            "candidate_strategy": "closest",
        },
        {
            "lcc_mode": "per_slice",
            "miss_count": 2,
            "out_of_tolerance_as_miss": True,
            "candidate_strategy": "score",
        },
        {
            "lcc_mode": "per_volume",
            "miss_count": 5,
            "out_of_tolerance_as_miss": False,
            "candidate_strategy": "closest",
        },
        {
            "lcc_mode": "per_volume",
            "miss_count": 2,
            "out_of_tolerance_as_miss": True,
            "candidate_strategy": "score",
        },
    ]


def build_grid_variants(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Monta o grid completo a partir dos argumentos."""
    variants = []
    for lcc_mode in args.lcc_modes:
        for miss_count in args.miss_counts:
            for out_of_tolerance_as_miss in args.tolerance_modes:
                for candidate_strategy in args.candidate_strategies:
                    variants.append(
                        {
                            "lcc_mode": lcc_mode,
                            "miss_count": miss_count,
                            "out_of_tolerance_as_miss": out_of_tolerance_as_miss,
                            "candidate_strategy": candidate_strategy,
                        }
                    )
    return variants


def resolve_repo_path(path: Path) -> Path:
    """Resolve caminhos relativos a partir da raiz do repositório."""
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    """Retorna caminho relativo ao repo quando possível."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    """Carrega arquivo JSON."""
    return json.loads(resolve_repo_path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Escreve JSON indentado."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Merge recursivo simples para overrides de configuração."""
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def variant_overrides(
    variant: dict[str, Any],
    *,
    use_gpu: bool | None,
    threshold_preset: str,
    score_weights: tuple[float, float, float],
) -> dict[str, Any]:
    """Converte uma variante em overrides do pipeline."""
    accum_weight, distance_weight, radius_weight = score_weights
    overrides: dict[str, Any] = {
        "LCC_PER_SLICE": variant["lcc_mode"] == "per_slice",
        "ARTERY_SEGMENTATION": {"method": "region_growing"},
        "CIRCLE_DETECTION": {
            "max_slice_miss_threshold": int(variant["miss_count"]),
            "out_of_tolerance_as_miss": bool(variant["out_of_tolerance_as_miss"]),
            "candidate_selection_strategy": variant["candidate_strategy"],
            "candidate_score_accum_weight": float(accum_weight),
            "candidate_score_distance_weight": float(distance_weight),
            "candidate_score_radius_weight": float(radius_weight),
        },
    }
    if threshold_preset == "best_normal":
        overrides.update(
            {
                "MAX_THRESHOLD_PERCENTILE": 99.8,
                "LOWER_THRESHOLD": {
                    "method": "percentile",
                    "percentile": 10.75,
                    "clip_min_hu": -400.0,
                    "clip_max_hu": 500.0,
                },
                "THRESHOLDING": {"method": "normal"},
            }
        )
    if use_gpu is not None:
        overrides["USE_GPU"] = bool(use_gpu)
    return overrides


def pipeline_command(
    args: argparse.Namespace,
    variant_output_root: Path,
    config_file: Path,
) -> list[str]:
    """Monta o comando do pipeline para uma variante."""
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
        str(config_file),
        "--output-dir",
        str(variant_output_root),
    ]
    if args.no_save_cache:
        command.append("--no-save-cache")
    if args.cache:
        command.append("--cache")
    if args.use_gpu is True:
        command.append("--gpu")
    elif args.use_gpu is False:
        command.append("--no-gpu")
    if args.base_path is not None:
        command.extend(["--base-path", str(args.base_path)])
    if args.base_save_path is not None:
        command.extend(["--base-save-path", str(args.base_save_path)])
    if args.downscale_method is not None:
        command.extend(["--downscale-method", args.downscale_method])
    if args.opencv_interpolation is not None:
        command.extend(["--opencv-interpolation", args.opencv_interpolation])
    return command


def latest_pipeline_run(variant_output_root: Path, resolution: str) -> Path | None:
    """Encontra o run criado pelo pipeline dentro da pasta da variante."""
    runs_root = variant_output_root / "segmentation" / "runs" / f"{resolution}_res"
    if not runs_root.exists():
        return None
    candidates = [path for path in runs_root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_split_summary(run_dir: Path | None, split: str) -> pd.DataFrame:
    """Carrega o CSV consolidado de um split."""
    if run_dir is None:
        return pd.DataFrame()
    path = run_dir / "numeric" / f"ostios_{split}_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def status_success(series: pd.Series) -> pd.Series:
    """Marca status aceitos como sucesso dos óstios."""
    return series.isin(SUCCESS_STATUSES)


def summarize_run(
    variant: dict[str, Any],
    run_dir: Path | None,
    split: str,
    duration_seconds: float,
    return_code: int | str,
    timed_out: bool,
) -> dict[str, Any]:
    """Resume uma variante em uma linha."""
    df = read_split_summary(run_dir, split)
    row = {
        "variant": variant["name"],
        "lcc_mode": variant["lcc_mode"],
        "aorta_miss_count": variant["miss_count"],
        "out_of_tolerance_as_miss": variant["out_of_tolerance_as_miss"],
        "candidate_selection_strategy": variant["candidate_strategy"],
        "run_dir": None if run_dir is None else display_path(run_dir),
        "return_code": return_code,
        "timed_out": timed_out,
        "duration_seconds": duration_seconds,
        "duration_minutes": duration_seconds / 60,
        "n_images": len(df),
    }
    if df.empty:
        return row

    status = df["status"] if "status" in df.columns else pd.Series([], dtype=str)
    success = status_success(status)
    row.update(
        {
            "ostia_success_rate": float(success.mean()) if len(success) else None,
            "both_correct_n": int((status == "both ostia correct").sum()),
            "both_tolerable_n": int((status == "both ostia tolerable").sum()),
            "one_correct_n": int((status == "one ostium correct").sum()),
            "none_correct_n": int((status == "no ostium correct").sum()),
            "not_found_n": int((status == "ostia not found").sum()),
            "mean_dice": df.get("artery_dice", pd.Series(dtype=float)).mean(),
            "median_dice": df.get("artery_dice", pd.Series(dtype=float)).median(),
            "mean_image_slices": df.get(
                "image_slice_count", pd.Series(dtype=float)
            ).mean(),
            "mean_aorta_circles": df.get(
                "aorta_circle_count", pd.Series(dtype=float)
            ).mean(),
            "mean_detected_circles": df.get(
                "aorta_detected_circle_count", pd.Series(dtype=float)
            ).mean(),
            "mean_interpolated_circles": df.get(
                "aorta_interpolated_circle_count", pd.Series(dtype=float)
            ).mean(),
            "mean_aorta_circle_coverage": df.get(
                "aorta_circle_coverage", pd.Series(dtype=float)
            ).mean(),
        }
    )
    return row


def build_circle_slice_metrics(
    variant: dict[str, Any],
    run_dir: Path | None,
    split: str,
) -> pd.DataFrame:
    """Extrai métricas por imagem sobre fatias e círculos da aorta."""
    df = read_split_summary(run_dir, split)
    if df.empty:
        return pd.DataFrame()
    columns = [
        "IMG_ID",
        "status",
        "artery_dice",
        "image_slice_count",
        "aorta_circle_count",
        "aorta_detected_circle_count",
        "aorta_interpolated_circle_count",
        "aorta_circle_coverage",
        "aorta_circle_first_slice",
        "aorta_circle_last_slice",
        "left_ostium_distance_mm",
        "right_ostium_distance_mm",
    ]
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    out = df[columns].copy()
    out.insert(0, "variant", variant["name"])
    out.insert(1, "lcc_mode", variant["lcc_mode"])
    out.insert(2, "aorta_miss_count", variant["miss_count"])
    out.insert(3, "out_of_tolerance_as_miss", variant["out_of_tolerance_as_miss"])
    out.insert(4, "candidate_selection_strategy", variant["candidate_strategy"])
    out["ostia_success"] = status_success(out["status"])
    out["aorta_detected_circle_coverage"] = (
        out["aorta_detected_circle_count"] / out["image_slice_count"]
    )
    out["aorta_missing_circle_slices"] = (
        out["image_slice_count"] - out["aorta_circle_count"]
    )
    return out


def build_circle_slice_stats(circle_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula correlações por variante entre fatias e círculos."""
    if circle_df.empty:
        return pd.DataFrame()
    rows = []
    metric_columns = [
        "aorta_circle_count",
        "aorta_detected_circle_count",
        "aorta_circle_coverage",
        "aorta_detected_circle_coverage",
        "aorta_missing_circle_slices",
    ]
    for variant, group in circle_df.groupby("variant"):
        row = {"variant": variant, "n_images": len(group)}
        for metric in metric_columns:
            if group[metric].notna().sum() >= 2:
                row[f"pearson_image_slices_vs_{metric}"] = group[
                    "image_slice_count"
                ].corr(group[metric], method="pearson")
                row[f"spearman_image_slices_vs_{metric}"] = group[
                    "image_slice_count"
                ].corr(group[metric], method="spearman")
        rows.append(row)
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    """Cria parser da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Sweep de parâmetros de aorta/óstios usando o pipeline oficial."
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--resolution", choices=["mid", "high"], default="mid")
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Usa produto cartesiano dos parâmetros em vez do grid enxuto.",
    )
    parser.add_argument("--miss-counts", type=parse_csv_ints, default=parse_csv_ints("2,3"))
    parser.add_argument(
        "--tolerance-modes",
        type=parse_bool_modes,
        default=parse_bool_modes("stop,miss"),
        help="Modos separados por vírgula: stop,miss.",
    )
    parser.add_argument(
        "--lcc-modes",
        type=parse_lcc_modes,
        default=parse_lcc_modes("per_slice,per_volume"),
        help="Modos separados por vírgula: per_slice,per_volume.",
    )
    parser.add_argument(
        "--candidate-strategies",
        type=parse_candidate_strategies,
        default=parse_candidate_strategies("closest,score"),
    )
    parser.add_argument(
        "--threshold-preset",
        choices=["config", "best_normal"],
        default="best_normal",
        help=(
            "config usa pipeline_config.json como está; best_normal fixa o "
            "melhor threshold normal do sweep anterior."
        ),
    )
    parser.add_argument("--score-accum-weight", type=float, default=1.0)
    parser.add_argument("--score-distance-weight", type=float, default=1.0)
    parser.add_argument("--score-radius-weight", type=float, default=1.0)
    parser.add_argument(
        "--timeout-minutes",
        type=float,
        default=180.0,
        help="Tempo máximo por variante. Use 0 para desabilitar.",
    )
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--no-save-cache", action="store_true")
    parser.add_argument("--base-path", type=Path, default=None)
    parser.add_argument("--base-save-path", type=Path, default=None)
    parser.add_argument("--downscale-method", choices=["scipy", "opencv"], default=None)
    parser.add_argument(
        "--opencv-interpolation",
        choices=["nearest", "linear", "cubic", "area", "lanczos4"],
        default=None,
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu", dest="use_gpu", action="store_true", default=None)
    gpu_group.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main() -> None:
    """Executa o sweep."""
    args = build_parser().parse_args()
    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.output_root / run_name
    configs_dir = run_dir / "configs"
    variants_root = run_dir / "pipeline_runs"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    variants = build_grid_variants(args) if args.full_grid else default_variants()
    for variant in variants:
        variant["name"] = variant_name(
            lcc_mode=variant["lcc_mode"],
            miss_count=variant["miss_count"],
            out_of_tolerance_as_miss=variant["out_of_tolerance_as_miss"],
            candidate_strategy=variant["candidate_strategy"],
        )

    base_config = load_json(args.config_path)
    score_weights = (
        args.score_accum_weight,
        args.score_distance_weight,
        args.score_radius_weight,
    )
    write_json(
        run_dir / "sweep_config.json",
        {
            "split": args.split,
            "resolution": args.resolution,
            "num_batches": args.num_batches,
            "full_grid": args.full_grid,
            "variants": variants,
            "threshold_preset": args.threshold_preset,
            "score_weights": score_weights,
            "timeout_minutes": args.timeout_minutes,
            "config_path": display_path(resolve_repo_path(args.config_path)),
            "dry_run": args.dry_run,
        },
    )

    variant_rows = []
    run_rows = []
    summary_rows = []
    circle_rows = []
    timeout_seconds = None if args.timeout_minutes <= 0 else args.timeout_minutes * 60

    for index, variant in enumerate(variants, start=1):
        variant_dir = variants_root / variant["name"]
        config_file = configs_dir / f"{variant['name']}.json"
        overrides = variant_overrides(
            variant,
            use_gpu=args.use_gpu,
            threshold_preset=args.threshold_preset,
            score_weights=score_weights,
        )
        variant_config = deep_merge(base_config, overrides)
        write_json(config_file, variant_config)
        command = pipeline_command(args, variant_dir, config_file)
        command_text = " ".join(command)

        print(f"[{index}/{len(variants)}] {variant['name']}")
        print(command_text)
        variant_rows.append(
            {
                **variant,
                "config_file": display_path(config_file),
                "command": command_text,
            }
        )

        if args.dry_run:
            run_rows.append(
                {
                    "variant": variant["name"],
                    "return_code": None,
                    "timed_out": False,
                    "duration_seconds": 0.0,
                    "run_dir": None,
                    "command": command_text,
                }
            )
            continue

        start = time.perf_counter()
        timed_out = False
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=False,
                timeout=timeout_seconds,
            )
            return_code: int | str = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            return_code = "timeout"
            print(
                f"Timeout após {args.timeout_minutes:.1f} min: {variant['name']}"
            )
        duration = time.perf_counter() - start
        pipeline_run_dir = latest_pipeline_run(variant_dir, args.resolution)

        run_rows.append(
            {
                "variant": variant["name"],
                "return_code": return_code,
                "timed_out": timed_out,
                "duration_seconds": duration,
                "run_dir": None
                if pipeline_run_dir is None
                else display_path(pipeline_run_dir),
                "command": command_text,
            }
        )
        summary_rows.append(
            summarize_run(
                variant,
                pipeline_run_dir,
                args.split,
                duration,
                return_code,
                timed_out,
            )
        )
        circle_df = build_circle_slice_metrics(variant, pipeline_run_dir, args.split)
        if not circle_df.empty:
            circle_rows.append(circle_df)

        pd.DataFrame(summary_rows).to_csv(results_dir / "summary.csv", index=False)
        pd.DataFrame(run_rows).to_csv(results_dir / "runs.csv", index=False)
        if circle_rows:
            circle_all = pd.concat(circle_rows, ignore_index=True)
            circle_all.to_csv(results_dir / "circle_slice_metrics.csv", index=False)
            build_circle_slice_stats(circle_all).to_csv(
                results_dir / "circle_slice_correlations.csv",
                index=False,
            )

        if return_code not in (0, None) and args.stop_on_error:
            raise SystemExit(124 if timed_out else int(return_code))

    pd.DataFrame(variant_rows).to_csv(results_dir / "variants.csv", index=False)
    pd.DataFrame(run_rows).to_csv(results_dir / "runs.csv", index=False)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(
            ["ostia_success_rate", "mean_dice"],
            ascending=False,
            na_position="last",
        )
        summary_df.to_csv(results_dir / "summary.csv", index=False)
        print("\nResumo salvo em:", results_dir / "summary.csv")
        columns = [
            "variant",
            "ostia_success_rate",
            "mean_dice",
            "mean_aorta_circle_coverage",
            "duration_minutes",
            "timed_out",
        ]
        available = [column for column in columns if column in summary_df.columns]
        print(summary_df[available].to_string(index=False))

    if circle_rows:
        circle_all = pd.concat(circle_rows, ignore_index=True)
        circle_all.to_csv(results_dir / "circle_slice_metrics.csv", index=False)
        build_circle_slice_stats(circle_all).to_csv(
            results_dir / "circle_slice_correlations.csv",
            index=False,
        )

    if args.dry_run:
        print("\nDry run concluído. Configs salvas em:", configs_dir)


if __name__ == "__main__":
    main()
