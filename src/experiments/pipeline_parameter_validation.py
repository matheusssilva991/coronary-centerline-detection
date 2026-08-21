"""Executa análises OFAT de sensibilidade e escala por resolução.

O estudo padrão preserva a análise do artigo. ``resolution_scaling`` isola os
grupos aplicados por ``scale_config_to_resolution`` para diagnosticar high-res.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.experiments import (  # noqa: E402
    image_load_cache_key,
    parameter_validation_variants,
    prepared_context_cache_key,
    resolution_scaling_variants,
    validate_parameter_validation_append,
)
from utils.experiments.fuzzy_pipeline_comparison import (  # noqa: E402
    build_base_config,
    evaluate_prepared_image,
    load_downsampled_case,
    make_image_result_row,
    parameter_row,
    prepare_image_context,
    save_outputs,
    set_row_error,
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
    RESOLUTION_SCALING_GROUPS,
    apply_aorta_ostia_method,
    scale_config_to_resolution,
)
from utils.project.notebook_env import resolve_imagecas_base_path  # noqa: E402


DEFAULT_CONFIG_PATH = REPO_ROOT / "config/article_cbeb_sensitivity.json"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "output/segmentation/analysis/pipeline_parameter_validation/runs"
)


def build_parser() -> argparse.ArgumentParser:
    """Cria a CLI do experimento."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument(
        "--study",
        choices=["article_sensitivity", "resolution_scaling"],
        default="article_sensitivity",
        help="Família de variantes executada pelo experimento.",
    )
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--ids",
        default=None,
        help="IMG_IDs separados por vírgula; quando informado, ignora a amostragem.",
    )
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
    parser.add_argument(
        "--ostia-only",
        action="store_true",
        help="Interrompe após avaliar os óstios, sem vesselness arterial ou RG/FC.",
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu", dest="use_gpu", action="store_true", default=None)
    gpu_group.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def select_variants(names: str | None, study: str) -> list[dict]:
    """Seleciona variantes mantendo a ordem declarada."""
    variants = (
        resolution_scaling_variants()
        if study == "resolution_scaling"
        else parameter_validation_variants()
    )
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


def _zero_failed_dice(row: dict) -> None:
    """Aplica a convenção de Dice zero para falhas completas do pipeline."""
    if row.get("error") and pd.isna(row.get("dice_artery")):
        row.update(
            {
                "dice_artery": 0.0,
                "dice_artery_before_morphology": 0.0,
                "dice_artery_after_morphology": 0.0,
                "dice_artery_morphology_delta": 0.0,
            }
        )


def _prepare_variant_specs(
    variants: list[dict],
    base_config: dict,
    *,
    ostia_only: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Materializa configs, parâmetros e chaves de reaproveitamento."""
    specs: list[dict] = []
    parameter_rows: list[dict] = []
    for current_variant in variants:
        variant_name = current_variant["name"]
        overrides = current_variant["overrides"]
        config_overrides, experiment = split_overrides(overrides)
        if ostia_only:
            experiment["ostia_only"] = True
        disabled_groups = set(current_variant.get("disabled_scaling_groups", []))
        enabled_groups = RESOLUTION_SCALING_GROUPS.difference(disabled_groups)
        config = scale_config_to_resolution(
            apply_overrides(base_config, config_overrides),
            enabled_groups=enabled_groups,
        )
        post_scale_overrides = current_variant.get("post_scale_overrides", {})
        config = apply_overrides(config, post_scale_overrides)
        parameters = parameter_row(variant_name, overrides, config, experiment)
        parameters.update(
            {
                "parameter_group": current_variant["parameter_group"],
                "description": current_variant["description"],
                "disabled_scaling_groups": sorted(disabled_groups),
                "post_scale_overrides": post_scale_overrides,
            }
        )
        parameter_rows.append(parameters)
        specs.append(
            {
                **current_variant,
                "config": config,
                "experiment": experiment,
                "disabled_scaling_groups": sorted(disabled_groups),
                "post_scale_overrides": post_scale_overrides,
                "load_key": image_load_cache_key(config),
                "context_key": prepared_context_cache_key(config, experiment),
            }
        )
    return specs, parameter_rows


def _summaries_for_specs(
    specs: list[dict],
    rows_by_variant: dict[str, list[dict]],
    runtime_by_variant: dict[str, float],
) -> list[dict]:
    """Resume resultados parciais ou finais das variantes novas."""
    summaries = []
    for spec in specs:
        name = spec["name"]
        summaries.append(
            {
                **summarize_variant(
                    name,
                    rows_by_variant[name],
                    runtime_by_variant[name],
                ),
                "parameter_group": spec["parameter_group"],
                "description": spec["description"],
                "runtime_mode": "shared_stages_allocated",
            }
        )
    return summaries


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
        raise FileNotFoundError(f"--append requer um run existente: {run_config_path}")
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
        args.ids,
        base_path,
    )
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("--ids não pode conter IMG_IDs repetidos.")
    if args.ids:
        split_ids = set(select_ids(args.split, 10_000, 0, None, base_path))
        invalid_ids = sorted(set(image_ids).difference(split_ids))
        if invalid_ids:
            raise ValueError(
                f"IDs fora do split {args.split!r}: {invalid_ids}. "
                "A seleção de parâmetros deve permanecer no split solicitado."
            )
    requested_variants = select_variants(args.variants, args.study)

    summaries: list[dict] = []
    image_rows: list[dict] = []
    parameter_rows: list[dict] = []
    existing_variants: list[dict] = []
    completed_variants: set[str] = set()
    resumed_runtime_by_variant: dict[str, float] = {}
    if args.append:
        existing_config = load_json_file(run_config_path)
        if existing_config.get("study", "article_sensitivity") != args.study:
            raise ValueError("--append requer o mesmo --study do run existente.")
        if bool(existing_config.get("ostia_only", False)) != args.ostia_only:
            raise ValueError("--append requer o mesmo modo --ostia-only.")
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
        loaded_summaries = _load_csv_records(summary_path)
        image_rows = _load_csv_records(run_dir / "results/image_results.csv")
        parameter_rows = _load_csv_records(
            run_dir / "parameters/variant_parameters.csv"
        )
        existing_variants = list(existing_config.get("variants", []))
        resumed_runtime_by_variant = {
            str(row["variant"]): float(row.get("runtime_seconds") or 0.0)
            for row in loaded_summaries
            if row.get("variant") is not None
        }
        completed_variants = {
            str(row["variant"])
            for row in loaded_summaries
            if row.get("variant") is not None
            and int(row.get("images") or 0) >= len(image_ids)
        }
        summaries = [
            row
            for row in loaded_summaries
            if str(row.get("variant")) in completed_variants
        ]

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
        item for item in requested_variants if item["name"] not in known_variant_names
    )

    write_json(
        run_config_path,
        {
            "study": args.study,
            "ostia_only": args.ostia_only,
            "split": args.split,
            "sample_size": args.sample_size,
            "start_index": args.start_index,
            "ids_argument": args.ids,
            "ids": image_ids,
            "resolution": args.resolution,
            "aorta_ostia_method": args.aorta_ostia_method,
            "config_path": str(args.config_path),
            "base_path": str(base_path),
            "use_gpu": base_config.get("USE_GPU"),
            "execution_order": "image_first",
            "reuse_shared_stages": True,
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

    specs, new_parameter_rows = _prepare_variant_specs(
        variants,
        base_config,
        ostia_only=args.ostia_only,
    )
    existing_parameter_names = {
        str(row.get("variant")) for row in parameter_rows if row.get("variant")
    }
    parameter_rows.extend(
        row
        for row in new_parameter_rows
        if str(row.get("variant")) not in existing_parameter_names
    )
    rows_by_variant: dict[str, list[dict]] = {
        spec["name"]: [
            row for row in image_rows if str(row.get("variant")) == spec["name"]
        ]
        for spec in specs
    }
    runtime_by_variant: dict[str, float] = defaultdict(
        float,
        {
            spec["name"]: resumed_runtime_by_variant.get(spec["name"], 0.0)
            for spec in specs
        },
    )
    processed_pairs = {
        (str(row.get("variant")), int(row["IMG_ID"]))
        for row in image_rows
        if row.get("variant") is not None and pd.notna(row.get("IMG_ID"))
    }

    # Os membros permitem repartir o custo comum entre as variantes do resumo.
    load_members: dict[tuple, list[str]] = defaultdict(list)
    context_members: dict[tuple, list[str]] = defaultdict(list)
    for spec in specs:
        load_members[spec["load_key"]].append(spec["name"])
        context_members[spec["context_key"]].append(spec["name"])

    print(
        "Reaproveitamento por imagem: "
        f"{len(load_members)} carregamento(s), "
        f"{len(context_members)} contexto(s) de threshold/aorta/vesselness."
    )

    for image_index, image_id in enumerate(image_ids, start=1):
        print(f"\n[{image_index}/{len(image_ids)}] IMG_ID={image_id}")
        case_cache: dict[tuple, dict] = {}
        case_errors: dict[tuple, Exception] = {}
        context_cache: dict[tuple, dict] = {}
        context_errors: dict[tuple, Exception] = {}

        for spec in specs:
            name = spec["name"]
            if (name, int(image_id)) in processed_pairs:
                continue
            config = spec["config"]
            experiment = spec["experiment"]
            row = make_image_result_row(image_id, name, args.split, experiment)

            load_key = spec["load_key"]
            if load_key not in case_cache and load_key not in case_errors:
                started = time.perf_counter()
                try:
                    case_cache[load_key] = load_downsampled_case(
                        image_id,
                        base_path,
                        config,
                    )
                except Exception as exc:
                    case_errors[load_key] = exc
                elapsed = time.perf_counter() - started
                share = elapsed / len(load_members[load_key])
                for member in load_members[load_key]:
                    runtime_by_variant[member] += share

            if load_key in case_errors:
                set_row_error(row, case_errors[load_key])
            else:
                context_key = spec["context_key"]
                if (
                    context_key not in context_cache
                    and context_key not in context_errors
                ):
                    started = time.perf_counter()
                    try:
                        context_cache[context_key] = prepare_image_context(
                            case_cache[load_key],
                            config,
                            experiment,
                        )
                    except Exception as exc:
                        context_errors[context_key] = exc
                    elapsed = time.perf_counter() - started
                    share = elapsed / len(context_members[context_key])
                    for member in context_members[context_key]:
                        runtime_by_variant[member] += share

                if context_key in context_errors:
                    set_row_error(row, context_errors[context_key])
                else:
                    started = time.perf_counter()
                    try:
                        evaluate_prepared_image(
                            context_cache[context_key],
                            row,
                            config,
                            experiment,
                        )
                    except Exception as exc:
                        set_row_error(row, exc)
                    runtime_by_variant[name] += time.perf_counter() - started

            _zero_failed_dice(row)
            row["parameter_group"] = spec["parameter_group"]
            rows_by_variant[name].append(row)
            image_rows.append(row)

        # Cada imagem forma um checkpoint compacto, útil em quedas longas.
        current_summaries = [
            *summaries,
            *_summaries_for_specs(specs, rows_by_variant, runtime_by_variant),
        ]
        save_outputs(run_dir, current_summaries, image_rows, parameter_rows)
        pd.DataFrame(current_summaries).to_csv(
            run_dir / "summary/sensitivity_summary.csv",
            index=False,
        )

    summaries = [
        *summaries,
        *_summaries_for_specs(specs, rows_by_variant, runtime_by_variant),
    ]

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
