"""Executa o pipeline híbrido: óstios em mid e artérias em high resolution.

Exemplo rápido:
    uv run python src/experiments/hybrid_resolution_pipeline.py \
        --split train --sample-size 5 --gpu

O resultado é salvo incrementalmente para permitir retomada após interrupções.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.experiments import process_hybrid_resolution_variants  # noqa: E402
from utils.experiments.sweep_common import (  # noqa: E402
    csv_safe,
    resolve_cli_path,
    sanitize_name,
    write_json,
)
from utils.project.config import (  # noqa: E402
    load_config_json,
    scale_config_to_resolution,
    serialize_config_for_json,
)
from utils.project.dataset import get_data_splits  # noqa: E402
from utils.project.notebook_env import resolve_imagecas_base_path  # noqa: E402


DEFAULT_CONFIG_PATH = REPO_ROOT / "config/pipeline_config.json"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "output/segmentation/analysis/hybrid_resolution_pipeline"
)
BASELINE_VARIANT = "baseline_high_scaled"
RECOMMENDED_VARIANTS = (
    BASELINE_VARIANT,
    "morphology_mid_radii",
    "rg_mid_thresholds",
    "rg_mid_thresholds_morphology_mid",
    "artery_sigmas_physical_x2",
)


def build_parser() -> argparse.ArgumentParser:
    """Cria a interface de linha de comando do experimento híbrido."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30,
        help="Número de exames; use 0 para processar todo o split.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--ids",
        default=None,
        help="IDs explícitos separados por vírgula; substitui split e sample-size.",
    )
    parser.add_argument("--split-config", type=Path, default=None)
    parser.add_argument("--base-path", type=Path, default=None)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="Diretório do experimento anterior; IDs já salvos serão ignorados.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Ao retomar, executa novamente linhas que possuem erro.",
    )
    parser.add_argument(
        "--upper-threshold-percentile",
        type=float,
        default=None,
        help="Sobrescreve MAX_THRESHOLD_PERCENTILE em mid e high.",
    )
    parser.add_argument(
        "--threshold-method",
        choices=["normal", "fuzzy"],
        default=None,
    )
    parser.add_argument(
        "--artery-method",
        choices=["rg", "fc", "region_growing", "fuzzy_connectedness"],
        default=None,
        help="Método usado somente na segmentação arterial high resolution.",
    )
    parser.add_argument(
        "--variants",
        default=BASELINE_VARIANT,
        help=(
            "Variantes high separadas por vírgula ou 'recommended'. "
            f"Disponíveis: {', '.join(RECOMMENDED_VARIANTS)}."
        ),
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu", dest="use_gpu", action="store_true", default=None)
    gpu_group.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve_dataset_path(path: Path | None) -> Path:
    """Usa o caminho explícito ou os candidatos conhecidos do ImageCAS."""
    if path is None:
        return resolve_imagecas_base_path()
    resolved = resolve_cli_path(path)
    if resolved is None or not resolved.is_dir():
        raise FileNotFoundError(f"Dataset não encontrado: {resolved}")
    return resolved


def _select_image_ids(args: argparse.Namespace, base_path: Path) -> list[int]:
    """Seleciona IDs explícitos ou uma faixa determinística do split fixo."""
    if args.ids:
        image_ids = [
            int(value.strip()) for value in args.ids.split(",") if value.strip()
        ]
        if not image_ids:
            raise ValueError("--ids não contém nenhum identificador válido.")
        return image_ids
    if args.start_index < 0:
        raise ValueError("--start-index deve ser >= 0.")
    if args.sample_size < 0:
        raise ValueError("--sample-size deve ser >= 0.")

    split_config = resolve_cli_path(args.split_config)
    train_ids, val_ids, test_ids, _ = get_data_splits(
        str(base_path),
        split_config_path=split_config,
    )
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}[args.split]
    available = split_ids[args.start_index :]
    return available if args.sample_size == 0 else available[: args.sample_size]


def _load_unscaled_config(config_path: Path) -> dict[str, Any]:
    """Carrega a configuração escolhida sobre os defaults do projeto."""
    default_config = load_config_json(str(DEFAULT_CONFIG_PATH), {})
    resolved = resolve_cli_path(config_path)
    if resolved is None or not resolved.is_file():
        raise FileNotFoundError(f"Configuração não encontrada: {resolved}")
    if resolved.resolve() == DEFAULT_CONFIG_PATH.resolve():
        return default_config
    return load_config_json(str(resolved), default_config)


def _build_resolution_config(
    base_config: dict[str, Any],
    factors: tuple[int, int, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Aplica overrides e escala uma cópia da configuração uma única vez."""
    config = copy.deepcopy(base_config)
    config.pop("SAVE_CACHE", None)
    config.pop("LOAD_CACHE", None)
    config["DOWNSCALE_FACTORS"] = list(factors)
    if args.use_gpu is not None:
        config["USE_GPU"] = bool(args.use_gpu)
    if args.upper_threshold_percentile is not None:
        config["MAX_THRESHOLD_PERCENTILE"] = float(args.upper_threshold_percentile)
    if args.threshold_method is not None:
        config.setdefault("THRESHOLDING", {})["method"] = args.threshold_method
    if args.artery_method is not None:
        artery_method = {
            "rg": "region_growing",
            "fc": "fuzzy_connectedness",
        }.get(args.artery_method, args.artery_method)
        config.setdefault("ARTERY_SEGMENTATION", {})["method"] = artery_method

    return scale_config_to_resolution(config)


def build_hybrid_configs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Constrói as configurações efetivas mid e high do mesmo experimento."""
    base_config = _load_unscaled_config(args.config_path)
    mid_config = _build_resolution_config(base_config, (2, 2, 1), args)
    high_config = _build_resolution_config(base_config, (1, 1, 1), args)
    return mid_config, high_config


def _selected_variant_names(value: str) -> list[str]:
    """Interpreta a seleção da CLI preservando a ordem informada."""
    names = (
        list(RECOMMENDED_VARIANTS)
        if value.strip().lower() == "recommended"
        else [name.strip() for name in value.split(",") if name.strip()]
    )
    unknown = [name for name in names if name not in RECOMMENDED_VARIANTS]
    if unknown:
        raise ValueError(f"Variantes híbridas desconhecidas: {', '.join(unknown)}.")
    if not names:
        raise ValueError("--variants deve selecionar ao menos uma variante.")
    return list(dict.fromkeys(names))


def build_high_variants(
    mid_config: dict[str, Any],
    high_config: dict[str, Any],
    selected_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Cria variantes high que isolam vesselness, RG e morfologia."""
    variants: dict[str, dict[str, Any]] = {}
    for name in selected_names:
        config = copy.deepcopy(high_config)
        if name in {"morphology_mid_radii", "rg_mid_thresholds_morphology_mid"}:
            config["POSTPROCESSING"] = copy.deepcopy(mid_config["POSTPROCESSING"])
        if name in {"rg_mid_thresholds", "rg_mid_thresholds_morphology_mid"}:
            config["REGION_GROWING"]["threshold_divisor"] = mid_config[
                "REGION_GROWING"
            ]["threshold_divisor"]
            config["REGION_GROWING"]["min_vesselness_fraction"] = mid_config[
                "REGION_GROWING"
            ]["min_vesselness_fraction"]
        if name == "artery_sigmas_physical_x2":
            config["VESSELNESS_ARTERY"]["sigmas"] = [
                2.0 * float(value)
                for value in mid_config["VESSELNESS_ARTERY"]["sigmas"]
            ]
        variants[name] = config
    return variants


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Persiste o progresso sem deixar um CSV parcial em caso de interrupção."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    csv_safe(pd.DataFrame(rows)).to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _load_existing_rows(path: Path, retry_errors: bool) -> list[dict[str, Any]]:
    """Carrega resultados anteriores e opcionalmente remove linhas com erro."""
    if not path.is_file() or path.stat().st_size == 0:
        return []
    rows = pd.read_csv(path).to_dict("records")
    if not retry_errors:
        return rows

    def completed_without_error(row: dict[str, Any]) -> bool:
        error = row.get("error")
        return (
            error is None
            or (isinstance(error, float) and pd.isna(error))
            or not str(error)
        )

    return [row for row in rows if completed_without_error(row)]


def _build_summary(rows: list[dict[str, Any]], split: str) -> pd.DataFrame:
    """Resume localização dos óstios, Dice high e tempo do pipeline híbrido."""
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    for column in (
        "dice_artery",
        "dice_artery_before_morphology",
        "dice_artery_after_morphology",
        "total_seconds",
    ):
        if column not in df:
            df[column] = pd.NA
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ("mid_ostia_success", "high_ostia_success"):
        if column not in df:
            df[column] = False
        df[column] = df[column].fillna(False).astype(bool)
    for column in (
        "artery_voxels",
        "high_label_artery_voxels",
        "dice_artery_morphology_delta",
        "incremental_sweep_seconds",
        "shared_preparation_seconds",
    ):
        if column not in df:
            df[column] = pd.NA

    if "variant" not in df:
        df["variant"] = BASELINE_VARIANT
    if "error" not in df:
        df["error"] = None

    # Calcula deltas pareados somente quando o baseline está no mesmo run.
    baseline = df.loc[df["variant"] == BASELINE_VARIANT, ["IMG_ID", "dice_artery"]]
    baseline = baseline.rename(columns={"dice_artery": "baseline_dice"})
    df = df.merge(baseline, on="IMG_ID", how="left")
    df["dice_delta_vs_baseline"] = df["dice_artery"] - df["baseline_dice"]
    df["prediction_to_label_ratio"] = pd.to_numeric(
        df["artery_voxels"], errors="coerce"
    ) / pd.to_numeric(df["high_label_artery_voxels"], errors="coerce")
    actual_by_image = df.groupby("IMG_ID", sort=False).agg(
        shared_seconds=("shared_preparation_seconds", "max"),
        incremental_seconds=("incremental_sweep_seconds", "sum"),
    )
    actual_sweep_seconds = (
        actual_by_image["shared_seconds"].fillna(0).sum()
        + actual_by_image["incremental_seconds"].fillna(0).sum()
    )

    summaries: list[dict[str, Any]] = []
    for variant, group in df.groupby("variant", sort=False):
        summaries.append(
            {
                "split": split,
                "variant": variant,
                "images": int(group["IMG_ID"].nunique()),
                "errors": int(group["error"].notna().sum()),
                "mid_ostia_success_percent": 100 * group["mid_ostia_success"].mean(),
                "high_rescaled_ostia_success_percent": 100
                * group["high_ostia_success"].mean(),
                "mean_dice_high": group["dice_artery"].mean(),
                "median_dice_high": group["dice_artery"].median(),
                "mean_dice_before_morphology_high": group[
                    "dice_artery_before_morphology"
                ].mean(),
                "mean_morphology_delta_high": group[
                    "dice_artery_morphology_delta"
                ].mean(),
                "mean_dice_delta_vs_baseline": group[
                    "dice_delta_vs_baseline"
                ].mean(),
                "wins_vs_baseline": int((group["dice_delta_vs_baseline"] > 0).sum()),
                "mean_prediction_to_label_ratio": group[
                    "prediction_to_label_ratio"
                ].mean(),
                "mean_isolated_time_seconds": group["total_seconds"].mean(),
                "actual_sweep_total_seconds": actual_sweep_seconds,
            }
        )
    return pd.DataFrame(summaries).sort_values(
        "mean_dice_high", ascending=False, ignore_index=True
    )


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    """Cria um run novo ou valida o diretório solicitado para retomada."""
    if args.resume_dir is not None:
        run_dir = resolve_cli_path(args.resume_dir)
        if run_dir is None or not run_dir.is_dir():
            raise FileNotFoundError(f"Run para retomada não encontrado: {run_dir}")
        return run_dir

    output_root = resolve_cli_path(args.output_root)
    if output_root is None:
        raise ValueError("--output-root inválido.")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = sanitize_name(args.run_name or timestamp)
    run_dir = output_root / run_name
    if run_dir.exists():
        raise FileExistsError(
            f"Run já existe: {run_dir}. Use --resume-dir para continuar."
        )
    run_dir.mkdir(parents=True)
    return run_dir


def _save_or_validate_run_config(
    run_dir: Path,
    args: argparse.Namespace,
    image_ids: list[int],
    base_path: Path,
    mid_config: dict[str, Any],
    high_variants: dict[str, dict[str, Any]],
) -> None:
    """Congela a coorte/configuração e impede retomadas incompatíveis."""
    config_dir = run_dir / "config"
    run_config_path = config_dir / "run_config.json"
    payload = {
        "split": args.split,
        "image_ids": image_ids,
        "base_path": str(base_path),
        "config_path": str(resolve_cli_path(args.config_path)),
        "pipeline_mode": "mid_ostia_high_artery",
        "mid_config": serialize_config_for_json(mid_config),
        "high_variants": serialize_config_for_json(high_variants),
    }
    if run_config_path.is_file():
        existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        legacy_payload = dict(payload)
        if list(high_variants) == [BASELINE_VARIANT]:
            legacy_payload["high_config"] = legacy_payload.pop("high_variants")[
                BASELINE_VARIANT
            ]
        if existing not in (payload, legacy_payload):
            raise ValueError(
                "A configuração/coorte informada não corresponde ao run retomado."
            )
        return

    write_json(run_config_path, payload)
    write_json(config_dir / "effective_mid_config.json", mid_config)
    write_json(config_dir / "effective_high_variants.json", high_variants)


def main() -> None:
    """Executa os IDs selecionados e salva resultados incrementalmente."""
    args = build_parser().parse_args()
    base_path = _resolve_dataset_path(args.base_path)
    image_ids = _select_image_ids(args, base_path)
    if not image_ids:
        raise ValueError("Nenhuma imagem selecionada para o experimento.")
    mid_config, high_config = build_hybrid_configs(args)
    variant_names = _selected_variant_names(args.variants)
    high_variants = build_high_variants(mid_config, high_config, variant_names)

    print("Pipeline híbrido: localização mid -> segmentação arterial high")
    print(f"Split: {args.split} | imagens: {len(image_ids)}")
    print(f"Dataset: {base_path}")
    print(f"Fatores mid: {mid_config['DOWNSCALE_FACTORS']}")
    print(f"Fatores high: {high_config['DOWNSCALE_FACTORS']}")
    print(f"Segmentação arterial high: {high_config['ARTERY_SEGMENTATION']['method']}")
    print(f"Variantes: {', '.join(variant_names)}")
    if args.dry_run:
        print(f"IDs: {image_ids}")
        return

    run_dir = _resolve_run_dir(args)
    _save_or_validate_run_config(
        run_dir,
        args,
        image_ids,
        base_path,
        mid_config,
        high_variants,
    )
    results_path = run_dir / "results/image_results.csv"
    rows = _load_existing_rows(results_path, args.retry_errors)
    completed_pairs = {
        (int(row["IMG_ID"]), str(row.get("variant", BASELINE_VARIANT)))
        for row in rows
    }
    pending_ids = [
        image_id
        for image_id in image_ids
        if any((image_id, variant) not in completed_pairs for variant in variant_names)
    ]
    print(f"Run: {run_dir}")
    print(
        f"Imagens concluídas: {len(image_ids) - len(pending_ids)} "
        f"| pendentes: {len(pending_ids)}"
    )

    for image_id in tqdm(pending_ids, desc="Pipeline híbrido", unit="imagem"):
        missing_variants = {
            name: config
            for name, config in high_variants.items()
            if (image_id, name) not in completed_pairs
        }
        image_results = process_hybrid_resolution_variants(
            image_id,
            mid_config,
            missing_variants,
            base_path,
        )
        for result in image_results:
            result["split"] = args.split
            rows.append(result)
        variant_order = {name: index for index, name in enumerate(variant_names)}
        rows.sort(
            key=lambda row: (
                image_ids.index(int(row["IMG_ID"])),
                variant_order[str(row.get("variant", BASELINE_VARIANT))],
            )
        )
        _atomic_write_csv(results_path, rows)

    summary = _build_summary(rows, args.split)
    summary_path = run_dir / "results/summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nResumo:")
    print(summary.round(4).to_string(index=False))
    print(f"\nResultados: {results_path}")


if __name__ == "__main__":
    main()
