"""Executa a análise OFAT de sensibilidade descrita no artigo.

O teste preserva threshold normal e region growing. A referência usa os valores
centrais declarados no estudo e cada variante altera um único parâmetro.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.experiments import (  # noqa: E402
    parameter_validation_variants,
    validate_parameter_validation_append,
)
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
    load_json_file,
    sanitize_name,
    select_ids,
    write_json,
)
from utils.project.config import (  # noqa: E402
    apply_aorta_ostia_method,
    scale_config_to_resolution,
)
from utils.project.notebook_env import resolve_imagecas_base_path  # noqa: E402


DEFAULT_CONFIG_PATH = REPO_ROOT / "config/article_sensitivity_reference.json"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "output/segmentation/analysis/pipeline_parameter_validation/runs"
)


def build_parser() -> argparse.ArgumentParser:
    """Cria a CLI do experimento."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resolution", choices=["mid", "high"], default="mid")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--variants",
        default=None,
        help="Nomes separados por vírgula. Por padrão executa todas as variantes.",
    )
    parser.add_argument(
        "--aorta-ostia-method",
        choices=["standard", "bilateral_thin"],
        default="standard",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Anexa variantes a um --run-name existente, preservando resultados "
            "já concluídos e validando a compatibilidade da execução."
        ),
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu", dest="use_gpu", action="store_true", default=None)
    gpu_group.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def select_variants(names: str | None) -> list[dict]:
    """Seleciona variantes mantendo a ordem declarada."""
    variants = parameter_validation_variants()
    if not names:
        return variants
    requested = [item.strip() for item in names.split(",") if item.strip()]
    by_name = {item["name"]: item for item in variants}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"Variantes desconhecidas: {missing}")
    return [by_name[name] for name in requested]


def _load_csv_records(path: Path) -> list[dict]:
    """Carrega um CSV existente como registros ou retorna uma lista vazia."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).to_dict("records")


def main() -> None:
    """Executa as variantes e salva resultados parciais após cada uma."""
    args = build_parser().parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size deve ser maior que zero.")

    run_name = sanitize_name(
        args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    run_dir = args.output_root / run_name
    run_config_path = run_dir / "run_config.json"
    if args.append and not run_config_path.exists():
        raise FileNotFoundError(
            f"--append requer um run existente: {run_config_path}"
        )
    if not args.append and run_config_path.exists():
        raise FileExistsError(
            f"O run já existe: {run_dir}. Use outro --run-name ou --append."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    base_path = resolve_imagecas_base_path()
    base_args = SimpleNamespace(
        config_path=args.config_path,
        resolution=args.resolution,
        use_gpu=args.use_gpu,
    )
    base_config = apply_aorta_ostia_method(
        build_base_config(base_args),
        args.aorta_ostia_method,
    )
    image_ids = select_ids(
        args.split,
        args.sample_size,
        args.start_index,
        None,
        base_path,
    )
    requested_variants = select_variants(args.variants)

    summaries: list[dict] = []
    image_rows: list[dict] = []
    parameter_rows: list[dict] = []
    existing_variants: list[dict] = []
    completed_variants: set[str] = set()
    if args.append:
        existing_config = load_json_file(run_config_path)
        validate_parameter_validation_append(
            existing_config,
            split=args.split,
            image_ids=image_ids,
            resolution=args.resolution,
            aorta_ostia_method=args.aorta_ostia_method,
            config_path=args.config_path,
            use_gpu=bool(base_config.get("USE_GPU")),
        )
        summary_path = run_dir / "summary/sensitivity_summary.csv"
        if not summary_path.exists():
            summary_path = run_dir / "summary/ranking.csv"
        summaries = _load_csv_records(summary_path)
        image_rows = _load_csv_records(run_dir / "results/image_results.csv")
        parameter_rows = _load_csv_records(
            run_dir / "parameters/variant_parameters.csv"
        )
        existing_variants = list(existing_config.get("variants", []))
        completed_variants = {
            str(row["variant"])
            for row in summaries
            if row.get("variant") is not None
        }

    variants = [
        item for item in requested_variants if item["name"] not in completed_variants
    ]
    skipped = [
        item["name"]
        for item in requested_variants
        if item["name"] in completed_variants
    ]
    combined_variants = list(existing_variants)
    known_variant_names = {item.get("name") for item in combined_variants}
    combined_variants.extend(
        item
        for item in requested_variants
        if item["name"] not in known_variant_names
    )

    write_json(
        run_config_path,
        {
            "split": args.split,
            "sample_size": args.sample_size,
            "start_index": args.start_index,
            "ids": image_ids,
            "resolution": args.resolution,
            "aorta_ostia_method": args.aorta_ostia_method,
            "config_path": str(args.config_path),
            "base_path": str(base_path),
            "use_gpu": base_config.get("USE_GPU"),
            "variants": combined_variants,
            "effective_base_config": base_config,
        },
    )

    print(f"Run: {run_dir}")
    print(f"Split: {args.split}; imagens: {len(image_ids)}")
    print(f"Variantes: {[item['name'] for item in variants]}")
    if skipped:
        print(f"Variantes já concluídas, ignoradas: {skipped}")
    if args.dry_run:
        print("Dry run concluído; nenhuma imagem foi processada.")
        return
    if not variants:
        print("Nenhuma variante pendente para processar.")
        return

    for variant_index, current_variant in enumerate(variants, start=1):
        variant_name = current_variant["name"]
        overrides = current_variant["overrides"]
        config_overrides, experiment = split_overrides(overrides)
        config = scale_config_to_resolution(
            apply_overrides(base_config, config_overrides)
        )
        parameters = parameter_row(variant_name, overrides, config, experiment)
        parameters.update(
            {
                "parameter_group": current_variant["parameter_group"],
                "description": current_variant["description"],
            }
        )
        parameter_rows.append(parameters)

        print(f"\n[{variant_index}/{len(variants)}] {variant_name}")
        started = time.perf_counter()
        variant_rows = []
        for image_index, image_id in enumerate(image_ids, start=1):
            print(f"  [{image_index}/{len(image_ids)}] IMG_ID={image_id}")
            row = run_image(
                image_id,
                variant_name,
                args.split,
                base_path,
                config,
                experiment,
            )
            # Uma falha completa do pipeline conta como Dice zero, seguindo a
            # convenção dos resultados canônicos usados como referência.
            if row.get("error") and pd.isna(row.get("dice_artery")):
                row.update(
                    {
                        "dice_artery": 0.0,
                        "dice_artery_before_morphology": 0.0,
                        "dice_artery_after_morphology": 0.0,
                        "dice_artery_morphology_delta": 0.0,
                    }
                )
            row["parameter_group"] = current_variant["parameter_group"]
            variant_rows.append(row)
            image_rows.append(row)

        summaries.append(
            {
                **summarize_variant(
                    variant_name,
                    variant_rows,
                    time.perf_counter() - started,
                ),
                "parameter_group": current_variant["parameter_group"],
                "description": current_variant["description"],
            }
        )
        save_outputs(run_dir, summaries, image_rows, parameter_rows)

    summary = pd.DataFrame(summaries)
    summary.to_csv(run_dir / "summary/sensitivity_summary.csv", index=False)
    print("\nResumo da análise de sensibilidade:")
    print(
        summary[
            ["variant", "ostia_success_rate", "mean_dice", "median_dice"]
        ].to_string(index=False)
    )
    print(f"\nResultados: {run_dir}")


if __name__ == "__main__":
    main()
