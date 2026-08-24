# ============================================================================
# IMPORTS
# ============================================================================
import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, cast

import pandas as pd

# Usa GPU 0 por padrão quando a variável não for definida externamente.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from utils.processing.gpu_utils import use_gpu
from utils.project.config import (
    apply_aorta_ostia_method,
    load_config_json,
    scale_config_to_resolution,
)
from utils.project.dataset import get_data_splits
from utils.project.results import (
    create_timestamped_output_dir,
    load_batch_timing_records,
    make_json_safe,
    make_result_dataframe,
    merge_batch_results,
    save_metadata,
    summarize_batch_timing_records,
)
from utils.segmentation.pipeline_cli import parse_pipeline_args
from utils.segmentation.pipeline_orchestration import run_pipeline
from utils.segmentation.pipeline_reporting import print_split_summary, print_statistics

# ============================================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================================

logger = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

GPU_ENABLED = use_gpu()
if GPU_ENABLED:
    logger.info("GPU detectada! Operações aceleradas por GPU ativadas.")
else:
    logger.warning("GPU não disponível. Acelerações CPU usadas.")

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = Path("/media/matheus/HD/DatasetsCCTA/ImageCAS/1-1000")
BASE_PATH_FALLBACK = Path("/data04/home/mpmaia/ImageCAS/database/1-1000")
OUTPUT_DIR = REPO_ROOT / "output"
PIPELINE_CONFIG_PATH = REPO_ROOT / "config" / "pipeline_config.json"


def has_imagecas_images(path: Path) -> bool:
    """Retorna True se o diretório contém imagens ImageCAS."""
    return path.exists() and any(path.glob("*.img.nii.gz"))


def resolve_base_path(base_path: Path) -> Path:
    """Resolve o caminho do ImageCAS usando fallback só para o default."""
    if has_imagecas_images(base_path):
        return base_path

    if base_path == BASE_PATH and has_imagecas_images(BASE_PATH_FALLBACK):
        print(
            "⚠️  Dataset não encontrado no caminho padrão local. "
            f"Usando fallback: {BASE_PATH_FALLBACK}"
        )
        logger.info(
            "Dataset não encontrado em %s; usando fallback %s",
            base_path,
            BASE_PATH_FALLBACK,
        )
        return BASE_PATH_FALLBACK

    return base_path


def load_default_config(config_path=PIPELINE_CONFIG_PATH):
    """Carrega a configuração base do pipeline."""
    try:
        config = load_config_json(str(config_path), {})
        logger.info("Config carregada de pipeline_config.json: %s", config_path)
        return config
    except Exception as exc:
        logger.warning(
            "Falha ao carregar %s: %s. Usando defaults mínimos.",
            config_path,
            exc,
        )
        return {
            "USE_GPU": GPU_ENABLED,
            "NUM_BATCHES": 5,
        }


CONFIG = load_default_config()
CONFIG_MID_RES = copy.deepcopy(CONFIG)
CONFIG_HIGH_RES = copy.deepcopy(CONFIG)
CONFIG_HIGH_RES["DOWNSCALE_FACTORS"] = [1, 1, 1]


def select_resolution_config(args):
    """Seleciona a configuração base para mid/high resolution."""
    if args.resolution == "high":
        print(
            "🔍 Resolução: HIGH "
            "(sem downscaling, DOWNSCALE_FACTORS = "
            f"{CONFIG_HIGH_RES['DOWNSCALE_FACTORS']})"
        )
        return CONFIG_HIGH_RES

    print(
        "🔍 Resolução: MID "
        f"(downscale 2x, DOWNSCALE_FACTORS = {CONFIG_MID_RES['DOWNSCALE_FACTORS']})"
    )
    return CONFIG_MID_RES


def _apply_execution_overrides(config, args):
    """Aplica opções operacionais da CLI à configuração efetiva."""
    # Configurações antigas podem conter opções de cache já removidas.
    config.pop("SAVE_CACHE", None)
    config.pop("LOAD_CACHE", None)

    direct_overrides = {
        "DOWNSCALE_METHOD": args.downscale_method,
        "OPENCV_INTERPOLATION": args.opencv_interpolation,
        "USE_GPU": args.use_gpu,
    }
    for config_key, value in direct_overrides.items():
        if value is not None:
            config[config_key] = value
    config["SAVE_SEGMENTATION_VISUALS"] = bool(
        getattr(args, "save_segmentation_visuals", False)
    )

    nested_overrides = (
        ("ARTERY_SEGMENTATION", "method", args.artery_segmentation_method),
        ("REGION_GROWING", "comparison_window", args.rg_comparison_window),
    )
    for section, config_key, value in nested_overrides:
        if value is not None:
            config.setdefault(section, {})[config_key] = value

    level_set_mode = getattr(args, "aorta_level_set_mode", None)
    if level_set_mode is not None:
        config.setdefault("LEVEL_SET", {})["iteration_mode"] = level_set_mode

    circle_filter = getattr(args, "aorta_circle_filter", None)
    circle_filter_config = config.setdefault("CIRCLE_DETECTION", {}).setdefault(
        "trajectory_filter", {}
    )
    if circle_filter is not None:
        circle_filter_config["method"] = circle_filter
    circle_filter_min_coverage = getattr(
        args, "aorta_circle_filter_min_coverage", None
    )
    if circle_filter_min_coverage is not None:
        circle_filter_config["min_tail_coverage"] = circle_filter_min_coverage
    circle_filter_interpolate = getattr(
        args, "aorta_circle_filter_interpolate", None
    )
    if circle_filter_interpolate is not None:
        circle_filter_config["interpolate_isolated_outliers"] = bool(
            circle_filter_interpolate
        )
    reject_oversegmented = getattr(
        args,
        "aorta_circle_filter_reject_oversegmented",
        None,
    )
    if reject_oversegmented is not None:
        circle_filter_config["reject_oversegmented_result"] = bool(
            reject_oversegmented
        )

    ostia_overrides = {
        "surface_mode": getattr(args, "ostia_surface_mode", None),
        "surface_thickness_mm": getattr(args, "ostia_surface_thickness_mm", None),
        "candidate_score_mode": getattr(args, "ostia_candidate_score_mode", None),
        "pair_selection_mode": getattr(args, "ostia_pair_selection_mode", None),
    }
    ostia_config = config.setdefault("OSTIA_DETECTION", {})
    for config_key, value in ostia_overrides.items():
        if value is not None:
            ostia_config[config_key] = value

    leak_correction = getattr(args, "aorta_leak_correction", None)
    if leak_correction is not None:
        level_set_config = config.setdefault("LEVEL_SET", {})
        level_set_config.setdefault("experimental_leak_correction", {})[
            "method"
        ] = leak_correction

    pruning_overrides = {
        "erosion_radius": getattr(
            args, "aorta_neck_pruning_erosion_radius", None
        ),
        "core_radius_factor": getattr(
            args, "aorta_neck_pruning_core_radius_factor", None
        ),
        "max_volume_loss_fraction": getattr(
            args, "aorta_neck_pruning_max_volume_loss", None
        ),
    }
    if any(value is not None for value in pruning_overrides.values()):
        pruning_config = (
            config.setdefault("LEVEL_SET", {})
            .setdefault("experimental_leak_correction", {})
            .setdefault("circle_seeded_neck_pruning", {})
        )
        for config_key, value in pruning_overrides.items():
            if value is not None:
                pruning_config[config_key] = value


def _apply_aorta_ostia_override(config, args):
    """Aplica o perfil de aorta/óstios configurado ou informado pela CLI."""
    configured_method = config.get("AORTA_OSTIA_METHOD", {}).get(
        "method",
        "standard",
    )
    if args.aorta_ostia_method is None and configured_method == "standard":
        return config
    return apply_aorta_ostia_method(config, method=args.aorta_ostia_method)


def _apply_threshold_overrides(config, args):
    """Aplica opções de threshold normal/fuzzy e piso inferior."""
    thresholding_config = config.setdefault("THRESHOLDING", {})
    if args.threshold_method is not None:
        thresholding_config["method"] = args.threshold_method
    if args.upper_threshold_percentile is not None:
        config["MAX_THRESHOLD_PERCENTILE"] = args.upper_threshold_percentile

    lower_threshold_config = config.setdefault("LOWER_THRESHOLD", {})
    lower_overrides = {
        "method": args.lower_threshold_method,
        "percentile": args.lower_threshold_percentile,
        "clip_min_hu": args.lower_threshold_clip_min,
        "clip_max_hu": args.lower_threshold_clip_max,
    }
    for config_key, value in lower_overrides.items():
        if value is not None:
            lower_threshold_config[config_key] = value

    if (
        args.lower_threshold_percentile is not None
        and thresholding_config.get("method") == "fuzzy"
    ):
        thresholding_config.setdefault("fuzzy", {})["lower_percentile"] = (
            args.lower_threshold_percentile
        )


def build_effective_config(args):
    """Aplica resolução, arquivo extra, flags CLI e escala espacial da configuração."""
    effective_config = copy.deepcopy(select_resolution_config(args))

    if args.config_file:
        effective_config = load_config_json(args.config_file, effective_config)
        print(f"⚙️  Configuração carregada de: {args.config_file}")

    # A resolução escolhida na CLI prevalece sobre fatores salvos em snapshots
    # de configuração produzidos originalmente para mid resolution.
    if args.resolution == "high":
        effective_config["DOWNSCALE_FACTORS"] = [1, 1, 1]

    _apply_execution_overrides(effective_config, args)
    effective_config = _apply_aorta_ostia_override(effective_config, args)
    _apply_threshold_overrides(effective_config, args)

    effective_config["NUM_BATCHES"] = args.num_batches
    return scale_config_to_resolution(effective_config)


def print_run_settings(args, config, base_path):
    """Mostra as configurações operacionais principais da execução."""
    if config["DOWNSCALE_METHOD"] == "opencv":
        print(
            "🔧 Método de downscale: "
            f"{config['DOWNSCALE_METHOD']} "
            f"(interpolação: {config['OPENCV_INTERPOLATION']})"
        )
    else:
        print(f"🔧 Método de downscale: {config['DOWNSCALE_METHOD']}")

    print(f"🗂️  Dataset: {base_path}")
    print(
        "🖥️  GPU nas etapas compatíveis: "
        f"{'habilitada' if config.get('USE_GPU', False) else 'desabilitada'}"
    )
    circle_config = config.get("CIRCLE_DETECTION", {})
    circle_filter_config = circle_config.get("trajectory_filter", {})
    print("🔎 LCC: por fatia")
    print(
        "⭕ Localização da aorta: "
        f"miss_count={circle_config.get('max_slice_miss_threshold')}, "
        f"recuperação inicial={circle_config.get('early_track_recovery', True)}, "
        "filtro de trajetória="
        f"{circle_filter_config.get('method', 'none')} "
        f"(cobertura mínima={circle_filter_config.get('min_tail_coverage', 0.8)}, "
        "fallback sobresegmentado="
        f"{circle_filter_config.get('reject_oversegmented_result', False)})"
    )
    artery_config = config.get("ARTERY_SEGMENTATION", {})
    thresholding_config = config.get("THRESHOLDING", {})
    lower_threshold_config = config.get("LOWER_THRESHOLD", {})
    print(f"🫀 Segmentação arterial: {artery_config.get('method', 'region_growing')}")
    print(
        "🫁 Aorta/óstios: "
        f"{config.get('AORTA_OSTIA_METHOD', {}).get('method', 'standard')}"
    )
    ostia_config = config.get("OSTIA_DETECTION", {})
    print(
        "🎯 Superfície/seleção dos óstios: "
        f"{ostia_config.get('surface_mode', 'erosion')} / "
        f"{ostia_config.get('pair_selection_mode', 'greedy')}"
    )
    print(
        "🔄 Level set da aorta: "
        f"{config.get('LEVEL_SET', {}).get('iteration_mode', 'fixed')}"
    )
    print(f"🧩 Threshold: {thresholding_config.get('method', 'normal')}")
    print(
        "🧱 Piso inferior: "
        f"{lower_threshold_config.get('method', 'fixed')} "
        f"(p={lower_threshold_config.get('percentile', 'N/A')})"
    )
    print(f"📦 Processamento em {args.num_batches} lotes")
    print(
        "🖼️  Visualizações 3D: "
        f"{'salvas' if config.get('SAVE_SEGMENTATION_VISUALS') else 'desativadas'}"
    )

    if args.resume_batch > 0:
        print(f"🔄 Retomando a partir do lote {args.resume_batch}")
    if args.resume_batches:
        print(
            "🔄 Retomada por subset: "
            f"train={args.resume_batches_by_split['train']}, "
            f"val={args.resume_batches_by_split['val']}, "
            f"test={args.resume_batches_by_split['test']}"
        )


def resolve_output_dir(args, output_root_dir):
    """Cria ou reutiliza os diretórios de saída da execução."""
    if (args.resume_requested or args.merge_only) and args.resume_dir:
        if args.resume_dir.exists():
            print(f"\n📁 Usando diretório anterior: {args.resume_dir}\n")
            return resolve_existing_output_dirs(args.resume_dir)
        print(f"❌ Erro: Diretório não encontrado: {args.resume_dir}")
        print("   Use --resume-dir com o caminho do diretório anterior")
        raise SystemExit(1)

    if args.resume_requested:
        print("❌ Erro: para retomar a partir de um lote, informe --resume-dir")
        print(
            "   Exemplo: --resume-batch 11 "
            "--resume-dir output/segmentation/2026-05-19_17-08-33"
        )
        raise SystemExit(1)

    experiment_name = f"segmentation/runs/{args.resolution}_res"
    if args.run_group:
        experiment_name = f"{experiment_name}/{Path(args.run_group).as_posix()}"
    run_dir = create_timestamped_output_dir(
        output_root_dir,
        experiment_name=experiment_name,
    )
    output_dirs = build_structured_output_dirs(run_dir)
    print(f"📁 Diretório da execução: {output_dirs['run_dir']}")
    print(f"📊 Resultados numéricos: {output_dirs['numeric_dir']}")
    print(f"🖼️  Exemplos visuais: {output_dirs['visual_dir']} (criada sob demanda)\n")
    return output_dirs


def build_structured_output_dirs(run_dir):
    """Cria a estrutura padrão de uma execução nova."""
    run_dir = Path(run_dir)
    output_dirs = {
        "run_dir": run_dir,
        "numeric_dir": run_dir / "numeric",
        "config_dir": run_dir / "config",
        "visual_dir": run_dir / "visual",
        "logs_dir": run_dir / "logs",
    }
    for output_path in (
        output_dirs["run_dir"],
        output_dirs["numeric_dir"],
        output_dirs["config_dir"],
        output_dirs["logs_dir"],
    ):
        output_path.mkdir(parents=True, exist_ok=True)
    return output_dirs


def resolve_existing_output_dirs(resume_dir):
    """Resolve diretórios para retomada em layout novo ou legado."""
    resume_dir = Path(resume_dir)
    if resume_dir.name == "numeric":
        run_dir = resume_dir.parent
        numeric_dir = resume_dir
    elif (resume_dir / "numeric").exists():
        run_dir = resume_dir
        numeric_dir = resume_dir / "numeric"
    else:
        run_dir = resume_dir
        numeric_dir = resume_dir

    output_dirs = {
        "run_dir": run_dir,
        "numeric_dir": numeric_dir,
        "config_dir": run_dir / "config",
        "visual_dir": run_dir / "visual",
        "logs_dir": run_dir / "logs",
    }
    output_dirs["numeric_dir"].mkdir(parents=True, exist_ok=True)
    output_dirs["logs_dir"].mkdir(parents=True, exist_ok=True)
    return output_dirs


def setup_file_logging(logs_dir):
    """Adiciona um arquivo de log dentro do diretório da execução."""
    try:
        fh_path = Path(logs_dir) / "pipeline.log"
        fh = logging.FileHandler(fh_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(LOG_FORMAT))
        logging.getLogger().addHandler(fh)
        logger.info("Logs também serão gravados em: %s", fh_path)
    except Exception:
        logger.warning("Não foi possível criar arquivo de log no diretório de saída.")


def save_run_snapshots(output_dirs, config, splits_to_run, split_config_path=None):
    """Salva config efetiva e IDs por split dentro da pasta config da execução."""
    config_dir = output_dirs["config_dir"]
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / "effective_pipeline_config.json"
    with config_path.open("w", encoding="utf-8") as file_handle:
        json.dump(make_json_safe(config), file_handle, indent=2, ensure_ascii=False)

    split_ids = {
        split_name: ids for split_name, ids in splits_to_run if ids is not None
    }
    if split_ids:
        split_payload = {
            "source": split_config_path,
            "splits": split_ids,
        }
        split_path = config_dir / "split_ids.json"
        with split_path.open("w", encoding="utf-8") as file_handle:
            json.dump(
                make_json_safe(split_payload),
                file_handle,
                indent=2,
                ensure_ascii=False,
            )


def split_names_from_args(args):
    """Resolve a lista de splits solicitada pela CLI."""
    if "all" in args.split:
        return ["train", "val", "test"]
    return args.split


def build_splits_to_run(args, base_path):
    """Retorna pares (split_name, ids), ou ids=None no modo merge-only."""
    split_names_to_run = split_names_from_args(args)
    if args.merge_only:
        return [(name, None) for name in split_names_to_run]

    train_ids, val_ids, test_ids, all_ids = get_data_splits(
        base_path,
        split_config_path=args.split_config,
    )
    print_statistics(train_ids, val_ids, test_ids, all_ids)
    split_map = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }
    return [(name, split_map[name]) for name in split_names_to_run]


def save_split_metadata(
    split_name,
    output_dir,
    config,
    ids,
    details,
    execution_time,
    base_path,
    output_root_dir,
    current_run_execution_time=None,
    batch_timings=None,
    batch_timing_summary=None,
):
    """Salva metadados e registra o caminho no logger."""
    metadata_path = save_metadata(
        split_name,
        output_dir,
        config,
        ids,
        details,
        execution_time,
        current_run_execution_time=current_run_execution_time,
        batch_timings=batch_timings,
        batch_timing_summary=batch_timing_summary,
        base_path=base_path,
        root_output_dir=output_root_dir,
    )
    logger.info("Metadados salvos em: %s", metadata_path)
    return metadata_path


def run_merge_only_split(
    split_name,
    output_dir,
    config,
    base_path,
    output_root_dir,
):
    """Consolida lotes já existentes e recria metadata/summary."""
    final_path = merge_batch_results(split_name, output_dir)
    if final_path is None:
        print(f"❌ Nenhum lote encontrado para o split '{split_name}'")
        raise SystemExit(1)

    df = pd.read_csv(final_path)
    details = cast(list[dict[str, Any]], df.to_dict("records"))
    metadata_ids = df["IMG_ID"].dropna().tolist() if "IMG_ID" in df.columns else []
    batch_timings = load_batch_timing_records(output_dir, split_name)
    batch_timing_summary = summarize_batch_timing_records(batch_timings)
    execution_time = batch_timing_summary.get("total_known_duration_seconds")
    save_split_metadata(
        split_name,
        output_dir,
        config,
        metadata_ids,
        details,
        execution_time=execution_time,
        current_run_execution_time=None,
        batch_timings=batch_timings,
        batch_timing_summary=batch_timing_summary,
        base_path=base_path,
        output_root_dir=output_root_dir,
    )
    print_split_summary(
        df,
        split_name,
        config,
        execution_time,
        timing_summary=batch_timing_summary,
    )


def run_processing_split(
    split_name,
    ids,
    output_dir,
    config,
    args,
    base_path,
    output_root_dir,
    visual_dir=None,
):
    """Processa um split e salva CSV final + metadata."""
    summary = run_pipeline(
        ids,
        split_name,
        config,
        base_path,
        output_dir,
        resume_from_batch=args.resume_batches_by_split.get(
            split_name, args.resume_batch
        ),
        visual_output_dir=(
            Path(visual_dir) / split_name
            if config.get("SAVE_SEGMENTATION_VISUALS") and visual_dir is not None
            else None
        ),
    )
    current_run_execution_time = summary.get("execution_time")
    batch_timing_summary = summary.get("batch_timing_summary") or {}
    execution_time = (
        batch_timing_summary.get("total_known_duration_seconds")
        or current_run_execution_time
    )

    logger.info("Finalizando processamento em lotes...")
    merge_batch_results(split_name, output_dir)
    output_path = Path(output_dir) / f"ostios_{split_name}_summary.csv"
    logger.info("Resumo final salvo em: %s", output_path)

    summary_details = summary.get("details")
    if summary_details is None:
        details = cast(
            list[dict[str, Any]],
            pd.read_csv(output_path).to_dict("records"),
        )
    else:
        details = cast(list[dict[str, Any]], summary_details)

    save_split_metadata(
        split_name,
        output_dir,
        config,
        ids,
        details,
        execution_time=execution_time,
        base_path=base_path,
        output_root_dir=output_root_dir,
        current_run_execution_time=current_run_execution_time,
        batch_timings=summary.get("batch_timings"),
        batch_timing_summary=batch_timing_summary,
    )

    df = (
        pd.read_csv(output_path)
        if summary_details is None
        else make_result_dataframe(details)
    )
    print_split_summary(
        df,
        split_name,
        config,
        execution_time,
        timing_summary=batch_timing_summary,
        current_run_execution_time=current_run_execution_time,
    )


def run_requested_splits(
    args,
    splits_to_run,
    output_dir,
    config,
    base_path,
    output_root_dir,
    visual_dir=None,
):
    """Executa ou consolida todos os splits solicitados."""
    for split_name, ids in splits_to_run:
        print(f"\n{'=' * 60}")
        action_label = "Consolidando" if args.merge_only else "Processando"
        print(f"🔬 {action_label} conjunto: {split_name.upper()}")
        print(f"{'=' * 60}")

        if args.merge_only:
            run_merge_only_split(
                split_name,
                output_dir,
                config,
                base_path,
                output_root_dir,
            )
        else:
            run_processing_split(
                split_name,
                ids,
                output_dir,
                config,
                args,
                base_path,
                output_root_dir,
                visual_dir,
            )


def main():
    """Função principal com argumentos de linha de comando."""
    args = parse_pipeline_args(BASE_PATH, OUTPUT_DIR)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Logging verbose habilitado (DEBUG)")

    base_path = resolve_base_path(args.base_path)
    output_root_dir = args.output_dir
    effective_config = build_effective_config(args)

    print_run_settings(args, effective_config, base_path)
    output_dirs = resolve_output_dir(args, output_root_dir)
    setup_file_logging(output_dirs["logs_dir"])
    splits_to_run = build_splits_to_run(args, base_path)
    if not (args.resume_requested or args.merge_only):
        save_run_snapshots(
            output_dirs,
            effective_config,
            splits_to_run,
            split_config_path=args.split_config,
        )
    run_requested_splits(
        args,
        splits_to_run,
        output_dirs["numeric_dir"],
        effective_config,
        base_path,
        output_root_dir,
        output_dirs["visual_dir"],
    )

    print(f"\n{'=' * 60}")
    print("✨ Processamento concluído!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
