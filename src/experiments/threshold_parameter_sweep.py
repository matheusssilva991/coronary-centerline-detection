"""Executa vários runs do pipeline variando parâmetros de threshold.

O script chama ``src/segmentation_pipeline.py`` para cada variante, mantendo o
pipeline oficial intacto. As variantes focam em ``threshold + RG`` para avaliar
o impacto do piso inferior, do teto superior e dos parâmetros do threshold fuzzy.

Exemplos:
    uv run python src/experiments/threshold_parameter_sweep.py --split train --dry-run

    uv run python src/experiments/threshold_parameter_sweep.py \\
      --split train \\
      --methods fixed,percentile \\
      --percentiles 9,10,11,14,15,16 \\
      --max-threshold-percentiles 99.7 \\
      --object-percentiles 99.5 \\
      --num-batches 5 \\
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
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output/segmentation/analysis/threshold_sweep"
DEFAULT_CONFIG_PATH = REPO_ROOT / "config/pipeline_config.json"


def parse_csv_floats(value: str) -> list[float]:
    """Converte ``1,2,5`` em lista de floats."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("A lista de percentis não pode ser vazia.")
    return [float(item) for item in items]


def parse_csv_methods(value: str) -> list[str]:
    """Converte lista textual de métodos e valida nomes conhecidos."""
    valid = {"fixed", "percentile", "object_relative_percentile"}
    methods = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [method for method in methods if method not in valid]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Métodos inválidos: {invalid}. Use: {sorted(valid)}."
        )
    return methods


def parse_csv_threshold_methods(value: str) -> list[str]:
    """Converte lista textual de modos de threshold."""
    valid = {"normal", "fuzzy"}
    methods = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [method for method in methods if method not in valid]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Thresholds inválidos: {invalid}. Use: {sorted(valid)}."
        )
    return methods


def parse_csv_ints(value: str) -> list[int]:
    """Converte ``1,2,3`` em lista de inteiros."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("A lista de inteiros não pode ser vazia.")
    return [int(item) for item in items]


def parse_csv_strings(value: str) -> list[str]:
    """Converte ``mean,median`` em lista de strings."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("A lista textual não pode ser vazia.")
    return items


def _percent_text(value: float | None) -> str:
    """Formata percentil para nomes de variantes."""
    return str(value).replace(".", "p")


def fuzzy_suffix(fuzzy_config: dict[str, Any] | None) -> str:
    """Gera sufixo curto para variantes de threshold fuzzy."""
    if not fuzzy_config:
        return ""
    strategy = fuzzy_config.get("mask_strategy", "object_argmax")
    dense_threshold = fuzzy_config.get("dense_membership_threshold", 0.5)
    return (
        "_thfuzzy"
        f"_objp{_percent_text(fuzzy_config['object_percentile'])}"
        f"_densep{_percent_text(fuzzy_config['dense_percentile'])}"
        f"_m{_percent_text(fuzzy_config['soft_margin_hu'])}"
        f"_r{fuzzy_config['smooth_radius']}"
        f"_{fuzzy_config['smooth_mode']}"
        f"_{strategy}"
        f"_dt{_percent_text(dense_threshold)}"
    )


def variant_name(
    method: str,
    percentile: float | None = None,
    object_percentile: float | None = None,
    max_threshold_percentile: float = 99.7,
    threshold_method: str = "normal",
    fuzzy_config: dict[str, Any] | None = None,
) -> str:
    """Gera nome curto para a variante."""
    if method == "fixed":
        name = "fixed_m300"
    else:
        percent_text = _percent_text(percentile)
        if method == "object_relative_percentile":
            name = f"object_relative_p{percent_text}"
            if object_percentile is not None and object_percentile != 99.5:
                name = f"{name}_objp{_percent_text(object_percentile)}"
        else:
            name = f"percentile_p{percent_text}"

    if max_threshold_percentile != 99.7:
        name = f"{name}_maxp{_percent_text(max_threshold_percentile)}"
    if threshold_method == "fuzzy":
        name = f"{name}{fuzzy_suffix(fuzzy_config)}"
    return name


def build_variants(
    methods: list[str],
    threshold_methods: list[str],
    percentiles: list[float],
    object_percentiles: list[float],
    max_threshold_percentiles: list[float],
    fuzzy_configs: list[dict[str, Any] | None],
    clip_min_hu: float,
    clip_max_hu: float,
) -> list[dict[str, Any]]:
    """Monta variantes de limiar inferior e superior."""
    variants: list[dict[str, Any]] = []
    if "fixed" in methods:
        for threshold_method in threshold_methods:
            method_fuzzy_configs = fuzzy_configs if threshold_method == "fuzzy" else [None]
            for fuzzy_config in method_fuzzy_configs:
                method_max_threshold_percentiles = (
                    max_threshold_percentiles if threshold_method == "normal" else [99.7]
                )
                for max_threshold_percentile in method_max_threshold_percentiles:
                    thresholding = {"method": threshold_method}
                    if fuzzy_config:
                        thresholding["fuzzy"] = fuzzy_config
                    variants.append(
                        {
                            "name": variant_name(
                                "fixed",
                                max_threshold_percentile=max_threshold_percentile,
                                threshold_method=threshold_method,
                                fuzzy_config=fuzzy_config,
                            ),
                            "method": "fixed",
                            "threshold_method": threshold_method,
                            "percentile": None,
                            "object_percentile": None,
                            "max_threshold_percentile": max_threshold_percentile,
                            "fuzzy_config": fuzzy_config,
                            "overrides": {
                                "MAX_THRESHOLD_PERCENTILE": max_threshold_percentile,
                                "THRESHOLDING": thresholding,
                                "LOWER_THRESHOLD": {
                                    "method": "fixed",
                                    "fixed_hu": -300,
                                },
                            },
                        },
                    )

    for method in methods:
        if method == "fixed":
            continue
        for threshold_method in threshold_methods:
            method_fuzzy_configs = fuzzy_configs if threshold_method == "fuzzy" else [None]
            for fuzzy_config in method_fuzzy_configs:
                for percentile in percentiles:
                    method_object_percentiles = (
                        object_percentiles
                        if method == "object_relative_percentile"
                        else [None]
                    )
                    for object_percentile in method_object_percentiles:
                        method_max_threshold_percentiles = (
                            max_threshold_percentiles
                            if threshold_method == "normal"
                            else [99.7]
                        )
                        for max_threshold_percentile in method_max_threshold_percentiles:
                            thresholding = {"method": threshold_method}
                            if fuzzy_config:
                                thresholding["fuzzy"] = fuzzy_config
                            lower_threshold_config = {
                                "method": method,
                                "percentile": percentile,
                                "clip_min_hu": clip_min_hu,
                                "clip_max_hu": clip_max_hu,
                            }
                            if object_percentile is not None:
                                lower_threshold_config["object_percentile"] = (
                                    object_percentile
                                )
                            variants.append(
                                {
                                    "name": variant_name(
                                        method,
                                        percentile,
                                        object_percentile,
                                        max_threshold_percentile,
                                        threshold_method,
                                        fuzzy_config,
                                    ),
                                    "method": method,
                                    "threshold_method": threshold_method,
                                    "percentile": percentile,
                                    "object_percentile": object_percentile,
                                    "max_threshold_percentile": max_threshold_percentile,
                                    "fuzzy_config": fuzzy_config,
                                    "overrides": {
                                        "MAX_THRESHOLD_PERCENTILE": max_threshold_percentile,
                                        "THRESHOLDING": thresholding,
                                        "LOWER_THRESHOLD": lower_threshold_config,
                                    },
                                }
                            )
    return variants

def base_pipeline_overrides(use_gpu: bool | None) -> dict[str, Any]:
    """Força o experimento para region growing, deixando o threshold por variante."""
    overrides: dict[str, Any] = {
        "ARTERY_SEGMENTATION": {"method": "region_growing"},
    }
    if use_gpu is not None:
        overrides["USE_GPU"] = bool(use_gpu)
    return overrides


def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Merge recursivo simples para overrides JSON."""
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Escreve JSON indentado."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_repo_path(path: Path) -> Path:
    """Resolve caminhos relativos a partir da raiz do repositório."""
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    """Carrega um JSON de configuração."""
    resolved = resolve_repo_path(path)
    return json.loads(resolved.read_text(encoding="utf-8"))


def display_path(path: Path) -> str:
    """Retorna caminho relativo ao repo quando possível."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def latest_pipeline_run(variant_output_root: Path, resolution: str) -> Path | None:
    """Encontra o run criado pelo pipeline dentro da pasta da variante."""
    runs_root = variant_output_root / "segmentation" / "runs" / f"{resolution}_res"
    if not runs_root.exists():
        return None
    candidates = [path for path in runs_root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_split_summary(run_dir: Path, split: str) -> pd.DataFrame:
    """Carrega o CSV consolidado do split."""
    path = run_dir / "numeric" / f"ostios_{split}_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def summarize_run(
    variant: dict[str, Any],
    run_dir: Path | None,
    split: str,
    duration_seconds: float,
    return_code: int,
) -> dict[str, Any]:
    """Gera uma linha de resumo por variante."""
    row = {
        "variant": variant["name"],
        "threshold_method": variant.get("threshold_method", "normal"),
        "lower_threshold_method": variant["method"],
        "lower_threshold_percentile": variant["percentile"],
        "lower_threshold_object_percentile": variant.get("object_percentile"),
        "max_threshold_percentile": variant.get("max_threshold_percentile"),
        "fuzzy_object_percentile": None,
        "fuzzy_dense_percentile": None,
        "fuzzy_soft_margin_hu": None,
        "fuzzy_smooth_radius": None,
        "fuzzy_smooth_mode": None,
        "fuzzy_mask_strategy": None,
        "fuzzy_dense_membership_threshold": None,
        "run_dir": None if run_dir is None else display_path(run_dir),
        "return_code": return_code,
        "duration_seconds": duration_seconds,
        "duration_minutes": duration_seconds / 60,
        "n_images": None,
        "mean_min_threshold_hu": None,
        "mean_max_threshold_hu": None,
        "std_min_threshold_hu": None,
        "mean_object_center_hu": None,
        "mean_dice": None,
        "median_dice": None,
        "ostia_success_rate": None,
        "both_correct_n": None,
        "both_tolerable_n": None,
        "found_wrong_n": None,
        "not_found_n": None,
    }
    fuzzy_config = variant.get("fuzzy_config") or {}
    row.update(
        {
            "fuzzy_object_percentile": fuzzy_config.get("object_percentile"),
            "fuzzy_dense_percentile": fuzzy_config.get("dense_percentile"),
            "fuzzy_soft_margin_hu": fuzzy_config.get("soft_margin_hu"),
            "fuzzy_smooth_radius": fuzzy_config.get("smooth_radius"),
            "fuzzy_smooth_mode": fuzzy_config.get("smooth_mode"),
            "fuzzy_mask_strategy": fuzzy_config.get("mask_strategy"),
            "fuzzy_dense_membership_threshold": fuzzy_config.get(
                "dense_membership_threshold"
            ),
        }
    )
    if run_dir is None or return_code != 0:
        return row

    df = read_split_summary(run_dir, split)
    if df.empty:
        return row

    dice = pd.to_numeric(df.get("artery_dice"), errors="coerce")
    min_threshold = pd.to_numeric(df.get("min_threshold_hu"), errors="coerce")
    max_threshold = (
        pd.to_numeric(df["max_threshold_hu"], errors="coerce")
        if "max_threshold_hu" in df.columns
        else pd.Series([pd.NA] * len(df), dtype="Float64")
    )
    object_center = pd.to_numeric(
        df.get("lower_threshold_object_center_hu"),
        errors="coerce",
    )
    status = df.get("ostia_detection_status", pd.Series(dtype=str)).astype(str)
    success = status.isin({"both correct", "both tolerable"})
    mean_max_threshold = max_threshold.mean()

    row.update(
        {
            "n_images": int(len(df)),
            "mean_min_threshold_hu": float(min_threshold.mean()),
            "mean_max_threshold_hu": (
                None if pd.isna(mean_max_threshold) else float(mean_max_threshold)
            ),
            "std_min_threshold_hu": float(min_threshold.std()),
            "mean_object_center_hu": float(object_center.mean()),
            "mean_dice": float(dice.mean()),
            "median_dice": float(dice.median()),
            "ostia_success_rate": float(success.mean()),
            "both_correct_n": int((status == "both correct").sum()),
            "both_tolerable_n": int((status == "both tolerable").sum()),
            "found_wrong_n": int((status == "found but incorrect").sum()),
            "not_found_n": int((status == "not found").sum()),
        }
    )
    return row


def build_parser() -> argparse.ArgumentParser:
    """Cria o parser do sweep."""
    parser = argparse.ArgumentParser(
        description=(
            "Sweep de limiar HU usando RG. Pode comparar threshold normal e fuzzy."
        ),
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--resolution", choices=["mid", "high"], default="mid")
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--methods",
        type=parse_csv_methods,
        default=parse_csv_methods("fixed,percentile"),
        help=(
            "Métodos separados por vírgula. Opções: fixed, percentile, "
            "object_relative_percentile."
        ),
    )
    parser.add_argument(
        "--threshold-methods",
        type=parse_csv_threshold_methods,
        default=parse_csv_threshold_methods("normal"),
        help=(
            "Modos de threshold separados por vírgula. Use normal,fuzzy para "
            "comparar o threshold original com o fuzzy."
        ),
    )
    parser.add_argument(
        "--percentiles",
        type=parse_csv_floats,
        default=parse_csv_floats("9,10,11,14,15,16"),
        help=(
            "Percentis baixos testados nos métodos adaptativos. O default "
            "densifica a busca perto de p10 e faz uma checagem curta perto "
            "de p15, que foram as melhores faixas do sweep anterior."
        ),
    )
    parser.add_argument(
        "--object-percentiles",
        type=parse_csv_floats,
        default=parse_csv_floats("99.5"),
        help=(
            "Percentis usados como centro do objeto no método "
            "object_relative_percentile. Use 99.0,99.5,99.7 para uma busca "
            "expandida."
        ),
    )
    parser.add_argument(
        "--object-percentile",
        type=float,
        default=None,
        help=(
            "Compatibilidade com sweeps antigos: se definido, substitui "
            "--object-percentiles por um único valor."
        ),
    )
    parser.add_argument(
        "--max-threshold-percentiles",
        type=parse_csv_floats,
        default=parse_csv_floats("99.7"),
        help=(
            "Percentis do limiar superior HU. Use, por exemplo, "
            "99.5,99.7,99.9 para testar estruturas densas."
        ),
    )
    parser.add_argument(
        "--fuzzy-object-percentiles",
        type=parse_csv_floats,
        default=parse_csv_floats("99.5"),
        help="Percentis usados como centro da classe objeto no threshold fuzzy.",
    )
    parser.add_argument(
        "--fuzzy-dense-percentiles",
        type=parse_csv_floats,
        default=parse_csv_floats("99.9"),
        help="Percentis usados como centro da classe fundo denso no threshold fuzzy.",
    )
    parser.add_argument(
        "--fuzzy-soft-margins",
        type=parse_csv_floats,
        default=parse_csv_floats("160"),
        help="Margens HU abaixo do limiar mínimo para o centro do fundo mole.",
    )
    parser.add_argument(
        "--fuzzy-smooth-radii",
        type=parse_csv_ints,
        default=parse_csv_ints("2"),
        help="Raios da suavização contextual das pertinências fuzzy.",
    )
    parser.add_argument(
        "--fuzzy-smooth-modes",
        type=parse_csv_strings,
        default=parse_csv_strings("mean"),
        help="Modos de suavização fuzzy. Opções úteis: mean, median.",
    )
    parser.add_argument(
        "--fuzzy-mask-strategies",
        type=parse_csv_strings,
        default=parse_csv_strings("object_argmax"),
        help=(
            "Estratégias de máscara fuzzy. Opções: object_argmax, "
            "dense_suppression, normal_dense_suppression."
        ),
    )
    parser.add_argument(
        "--fuzzy-dense-membership-thresholds",
        type=parse_csv_floats,
        default=parse_csv_floats("0.5"),
        help=(
            "Limiar máximo de pertinência densa aceito nas estratégias de "
            "supressão densa."
        ),
    )
    parser.add_argument("--clip-min-hu", type=float, default=-400.0)
    parser.add_argument("--clip-max-hu", type=float, default=500.0)
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra os comandos e grava configs, mas não executa o pipeline.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Interrompe o sweep se uma variante falhar.",
    )
    return parser


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
        "--artery-method",
        "rg",
        "--config-file",
        str(config_file),
        "--output-dir",
        str(variant_output_root),
    ]
    if args.cache:
        command.append("--cache")
    if args.no_save_cache:
        command.append("--no-save-cache")
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


def build_fuzzy_configs(args: argparse.Namespace) -> list[dict[str, Any] | None]:
    """Monta o grid de parâmetros fuzzy usado apenas em variantes fuzzy."""
    if "fuzzy" not in args.threshold_methods:
        return [None]
    configs: list[dict[str, Any] | None] = []
    for object_percentile in args.fuzzy_object_percentiles:
        for dense_percentile in args.fuzzy_dense_percentiles:
            if dense_percentile <= object_percentile:
                raise ValueError(
                    "Cada fuzzy dense percentile precisa ser maior que o "
                    "fuzzy object percentile."
                )
            for soft_margin_hu in args.fuzzy_soft_margins:
                for smooth_radius in args.fuzzy_smooth_radii:
                    for smooth_mode in args.fuzzy_smooth_modes:
                        for mask_strategy in args.fuzzy_mask_strategies:
                            for dense_threshold in (
                                args.fuzzy_dense_membership_thresholds
                            ):
                                configs.append(
                                    {
                                        "object_percentile": object_percentile,
                                        "dense_percentile": dense_percentile,
                                        "soft_margin_hu": soft_margin_hu,
                                        "smooth_radius": smooth_radius,
                                        "smooth_mode": smooth_mode,
                                        "mask_strategy": mask_strategy,
                                        "dense_membership_threshold": dense_threshold,
                                    }
                                )
    return configs


def main() -> None:
    """Executa o sweep."""
    args = build_parser().parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name or timestamp
    run_dir = args.output_root / run_name
    configs_dir = run_dir / "configs"
    variants_root = run_dir / "pipeline_runs"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    object_percentiles = (
        [float(args.object_percentile)]
        if args.object_percentile is not None
        else args.object_percentiles
    )
    fuzzy_configs = build_fuzzy_configs(args)

    variants = build_variants(
        args.methods,
        args.threshold_methods,
        args.percentiles,
        object_percentiles,
        args.max_threshold_percentiles,
        fuzzy_configs,
        args.clip_min_hu,
        args.clip_max_hu,
    )
    base_config = load_json(args.config_path)
    base_overrides = base_pipeline_overrides(args.use_gpu)
    write_json(
        run_dir / "sweep_config.json",
        {
            "split": args.split,
            "resolution": args.resolution,
            "num_batches": args.num_batches,
            "methods": args.methods,
            "threshold_methods": args.threshold_methods,
            "percentiles": args.percentiles,
            "object_percentiles": object_percentiles,
            "max_threshold_percentiles": args.max_threshold_percentiles,
            "fuzzy_configs": fuzzy_configs,
            "clip_min_hu": args.clip_min_hu,
            "clip_max_hu": args.clip_max_hu,
            "config_path": str(resolve_repo_path(args.config_path)),
            "dry_run": args.dry_run,
        },
    )

    variant_rows = []
    run_rows = []
    summary_rows = []
    for index, variant in enumerate(variants, start=1):
        variant_dir = variants_root / variant["name"]
        config_file = configs_dir / f"{variant['name']}.json"
        overrides = deep_merge(base_overrides, variant["overrides"])
        variant_config = deep_merge(base_config, overrides)
        write_json(config_file, variant_config)
        command = pipeline_command(args, variant_dir, config_file)
        command_text = " ".join(command)

        print(f"[{index}/{len(variants)}] {variant['name']}")
        print(command_text)
        variant_rows.append(
            {
                "variant": variant["name"],
                "threshold_method": variant.get("threshold_method", "normal"),
                "lower_threshold_method": variant["method"],
                "lower_threshold_percentile": variant["percentile"],
                "lower_threshold_object_percentile": variant.get(
                    "object_percentile"
                ),
                "max_threshold_percentile": variant.get(
                    "max_threshold_percentile"
                ),
                "fuzzy_config": variant.get("fuzzy_config"),
                "config_file": display_path(config_file),
                "command": command_text,
            }
        )

        if args.dry_run:
            run_rows.append(
                {
                    "variant": variant["name"],
                    "return_code": None,
                    "duration_seconds": 0.0,
                    "run_dir": None,
                    "command": command_text,
                }
            )
            continue

        start = time.perf_counter()
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        duration = time.perf_counter() - start
        pipeline_run_dir = latest_pipeline_run(variant_dir, args.resolution)
        run_rows.append(
            {
                "variant": variant["name"],
                "return_code": completed.returncode,
                "duration_seconds": duration,
                "run_dir": (
                    None
                    if pipeline_run_dir is None
                    else display_path(pipeline_run_dir)
                ),
                "command": command_text,
            }
        )
        summary_rows.append(
            summarize_run(
                variant,
                pipeline_run_dir,
                args.split,
                duration,
                completed.returncode,
            )
        )
        pd.DataFrame(summary_rows).to_csv(results_dir / "summary.csv", index=False)
        if completed.returncode != 0 and args.stop_on_error:
            raise SystemExit(completed.returncode)

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
        print(
            summary_df[
                [
                    "variant",
                    "threshold_method",
                    "mean_dice",
                    "ostia_success_rate",
                    "mean_min_threshold_hu",
                    "mean_max_threshold_hu",
                ]
            ]
        )
    else:
        print("\nDry run concluído. Configs salvas em:", configs_dir)


if __name__ == "__main__":
    main()
