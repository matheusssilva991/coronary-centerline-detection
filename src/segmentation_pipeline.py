# ============================================================================
# IMPORTS
# ============================================================================
import copy
import json
import logging
import os
from pathlib import Path

import pandas as pd

# Usa GPU 1 por padrão quando a variável não for definida externamente.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from utils.config_utils import load_config_json
from utils.dataset_utils import get_data_splits
from utils.processing.gpu_utils import use_gpu
from utils.results_utils import (
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
BASE_SAVE_PATH = Path("/media/matheus/HD/DatasetsCCTA/Processed_ImageCAS")
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
            f"(sem downscaling, DOWNSCALE_FACTORS = {CONFIG_HIGH_RES['DOWNSCALE_FACTORS']})"
        )
        return CONFIG_HIGH_RES

    print(
        "🔍 Resolução: MID "
        f"(downscale 2x, DOWNSCALE_FACTORS = {CONFIG_MID_RES['DOWNSCALE_FACTORS']})"
    )
    return CONFIG_MID_RES


def build_effective_config(args):
    """Aplica resolução, arquivo extra e flags CLI sobre a configuração base."""
    effective_config = copy.deepcopy(select_resolution_config(args))

    if args.config_file:
        effective_config = load_config_json(args.config_file, effective_config)
        print(f"⚙️  Configuração carregada de: {args.config_file}")

    if args.cache:
        effective_config["LOAD_CACHE"] = True
        print("⚙️  Carregamento de cache habilitado")

    if args.no_save_cache:
        effective_config["SAVE_CACHE"] = False
    elif "SAVE_CACHE" not in effective_config:
        effective_config["SAVE_CACHE"] = True

    if effective_config.get("SAVE_CACHE"):
        print("💾 Salvamento de cache habilitado")
    else:
        print("⚠️  Salvamento de cache desabilitado")

    if args.downscale_method is not None:
        effective_config["DOWNSCALE_METHOD"] = args.downscale_method
    if args.opencv_interpolation is not None:
        effective_config["OPENCV_INTERPOLATION"] = args.opencv_interpolation
    if args.use_gpu is not None:
        effective_config["USE_GPU"] = args.use_gpu

    effective_config["NUM_BATCHES"] = args.num_batches
    return effective_config


def print_run_settings(args, config, base_path, base_save_path):
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
    print(f"💽 Cache/artefatos: {base_save_path}")
    print(
        "🖥️  GPU nas etapas compatíveis: "
        f"{'habilitada' if config.get('USE_GPU', False) else 'desabilitada'}"
    )
    print(f"📦 Processamento em {args.num_batches} lotes")

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

    run_dir = create_timestamped_output_dir(
        output_root_dir,
        experiment_name=f"segmentation/runs/{args.resolution}_res",
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
        split_name: ids
        for split_name, ids in splits_to_run
        if ids is not None
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
    base_save_path,
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
        base_save_path=base_save_path,
        root_output_dir=output_root_dir,
    )
    logger.info("Metadados salvos em: %s", metadata_path)
    return metadata_path


def run_merge_only_split(
    split_name,
    output_dir,
    config,
    base_path,
    base_save_path,
    output_root_dir,
):
    """Consolida lotes já existentes e recria metadata/summary."""
    final_path = merge_batch_results(split_name, output_dir)
    if final_path is None:
        print(f"❌ Nenhum lote encontrado para o split '{split_name}'")
        raise SystemExit(1)

    df = pd.read_csv(final_path)
    details = df.to_dict("records")
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
        base_save_path=base_save_path,
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
    base_save_path,
    output_root_dir,
):
    """Processa um split e salva CSV final + metadata."""
    summary = run_pipeline(
        ids,
        split_name,
        config,
        base_path,
        base_save_path,
        output_dir,
        resume_from_batch=args.resume_batches_by_split.get(
            split_name, args.resume_batch
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

    if summary.get("details") is None:
        details = pd.read_csv(output_path).to_dict("records")
    else:
        details = summary["details"]

    save_split_metadata(
        split_name,
        output_dir,
        config,
        ids,
        details,
        execution_time=execution_time,
        base_path=base_path,
        base_save_path=base_save_path,
        output_root_dir=output_root_dir,
        current_run_execution_time=current_run_execution_time,
        batch_timings=summary.get("batch_timings"),
        batch_timing_summary=batch_timing_summary,
    )

    df = pd.read_csv(output_path) if summary.get("details") is None else make_result_dataframe(details)
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
    base_save_path,
    output_root_dir,
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
                base_save_path,
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
                base_save_path,
                output_root_dir,
            )


def main():
    """Função principal com argumentos de linha de comando."""
    args = parse_pipeline_args(BASE_PATH, BASE_SAVE_PATH, OUTPUT_DIR)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Logging verbose habilitado (DEBUG)")

    base_path = resolve_base_path(args.base_path)
    base_save_path = args.base_save_path
    output_root_dir = args.output_dir
    effective_config = build_effective_config(args)

    print_run_settings(args, effective_config, base_path, base_save_path)
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
        base_save_path,
        output_root_dir,
    )

    print(f"\n{'=' * 60}")
    print("✨ Processamento concluído!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
