# ============================================================================
# IMPORTS
# ============================================================================
# Biblioteca padrão
import os
import copy
import logging
from pathlib import Path

# Terceiros - Machine Learning
import pandas as pd

# Usa GPU 1 por padrão quando a variável não for definida externamente.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

# Locais
from utils import (
    use_gpu,
    load_config_json,
    get_data_splits,
    create_timestamped_output_dir,
    make_result_dataframe,
    merge_batch_results,
    save_metadata,
)
from utils.segmentation.pipeline_cli import parse_pipeline_args
from utils.segmentation.pipeline_orchestration import (
    run_pipeline,
)
from utils.segmentation.pipeline_reporting import print_split_summary, print_statistics


# ============================================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================================

# Informações sobre aceleração GPU
logger = logging.getLogger(__name__)
# Formato de logging mais rico: timestamp, nível, logger, arquivo:linha
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

GPU_ENABLED = use_gpu()
if GPU_ENABLED:
    logger.info("GPU detectada! Operações aceleradas por GPU ativadas.")
else:
    logger.warning("GPU não disponível. Acelerações CPU usadas.")

# Caminhos padrão (usar pathlib)
# BASE_PATH = Path("/media/matheus/HD/DatasetsCCTA/ImageCAS/1-1000")
BASE_PATH = Path("/data04/home/mpmaia/ImageCAS/database/1-1000")
BASE_SAVE_PATH = Path("/media/matheus/HD/DatasetsCCTA/Processed_ImageCAS")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Carregar apenas pipeline_config.json (usuário)
pipeline_config_path = (
    Path(__file__).resolve().parent.parent / "config" / "pipeline_config.json"
)
try:
    CONFIG = load_config_json(str(pipeline_config_path), {})
    logger.info(f"Config carregada de pipeline_config.json: {pipeline_config_path}")
except Exception as e:
    logger.warning(
        f"Falha ao carregar {pipeline_config_path}: {e}. Usando defaults mínimos."
    )
    CONFIG = {
        "USE_GPU": GPU_ENABLED,
        "NUM_BATCHES": 5,
    }

# ============================================================================
# CONFIGURAÇÕES POR RESOLUÇÃO
# ============================================================================
# MID/HIGH configs: permitir sobreposição via deepcopy
CONFIG_MID_RES = copy.deepcopy(CONFIG)
CONFIG_HIGH_RES = copy.deepcopy(CONFIG)
CONFIG_HIGH_RES["DOWNSCALE_FACTORS"] = [1, 1, 1]

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================


def main():
    """Função principal com argumentos de linha de comando."""
    args = parse_pipeline_args(BASE_PATH, BASE_SAVE_PATH, OUTPUT_DIR)
    base_path = args.base_path
    base_save_path = args.base_save_path
    output_root_dir = args.output_dir

    # Ajustar nível de logging se solicitado
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Logging verbose habilitado (DEBUG)")

    # Selecionar configuração baseada na resolução escolhida
    if args.resolution == "high":
        base_config = CONFIG_HIGH_RES
        print(
            f"🔍 Resolução: HIGH (sem downscaling, DOWNSCALE_FACTORS = {base_config['DOWNSCALE_FACTORS']})"
        )
    else:
        base_config = CONFIG_MID_RES
        print(
            f"🔍 Resolução: MID (downscale 2x, DOWNSCALE_FACTORS = {base_config['DOWNSCALE_FACTORS']})"
        )

    effective_config = copy.deepcopy(base_config)

    if args.config_file:
        effective_config = load_config_json(args.config_file, effective_config)
        print(f"⚙️  Configuração carregada de: {args.config_file}")

    # Atualizar configurações via CLI
    if args.cache:
        effective_config["LOAD_CACHE"] = True
        print("⚙️  Carregamento de cache habilitado")

    if args.no_save_cache:
        effective_config["SAVE_CACHE"] = False
        print("⚠️  Salvamento de cache desabilitado")
    else:
        if "SAVE_CACHE" not in effective_config:
            effective_config["SAVE_CACHE"] = True
        print("💾 Salvamento de cache habilitado")

    if args.downscale_method is not None:
        effective_config["DOWNSCALE_METHOD"] = args.downscale_method
    if args.opencv_interpolation is not None:
        effective_config["OPENCV_INTERPOLATION"] = args.opencv_interpolation

    if effective_config["DOWNSCALE_METHOD"] == "opencv":
        print(
            f"🔧 Método de downscale: {effective_config['DOWNSCALE_METHOD']} (interpolação: {effective_config['OPENCV_INTERPOLATION']})"
        )
    else:
        print(f"🔧 Método de downscale: {effective_config['DOWNSCALE_METHOD']}")
    print(f"🗂️  Dataset: {base_path}")
    print(f"💽 Cache/artefatos: {base_save_path}")

    # Configurar batch processing
    effective_config["NUM_BATCHES"] = args.num_batches
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

    # Criar ou reusar diretório
    if (args.resume_requested or args.merge_only) and args.resume_dir:
        # Modo retomada/merge: usar diretório anterior
        if args.resume_dir.exists():
            timestamped_output_dir = args.resume_dir
            print(f"\n📁 Usando diretório anterior: {timestamped_output_dir}\n")
        else:
            print(f"❌ Erro: Diretório não encontrado: {args.resume_dir}")
            print("   Use --resume-dir com o caminho do diretório anterior")
            exit(1)
    elif args.resume_requested:
        print("❌ Erro: para retomar a partir de um lote, informe --resume-dir")
        print(
            "   Exemplo: --resume-batch 11 "
            "--resume-dir output/segmentation/2026-05-19_17-08-33"
        )
        exit(1)
    else:
        # Modo normal: criar novo diretório com timestamp
        timestamped_output_dir = create_timestamped_output_dir(
            output_root_dir, experiment_name="segmentation"
        )
        print(f"📁 Diretório de saída: {timestamped_output_dir}\n")

    timestamped_output_dir = Path(timestamped_output_dir)

    # Configurar FileHandler de logging no diretório de saída (debug)
    try:
        fh_path = Path(timestamped_output_dir) / "pipeline.log"
        fh = logging.FileHandler(fh_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(LOG_FORMAT))
        logging.getLogger().addHandler(fh)
        logger.info(f"Logs também serão gravados em: {fh_path}")
    except Exception:
        logger.warning("Não foi possível criar arquivo de log no diretório de saída.")

    # Determinar quais conjuntos processar
    if "all" in args.split:
        split_names_to_run = ["train", "val", "test"]
    else:
        split_names_to_run = args.split

    if args.merge_only:
        splits_to_run = [(name, None) for name in split_names_to_run]
    else:
        # Obter splits de dados
        train_ids, val_ids, test_ids, all_ids = get_data_splits(base_path)
        print_statistics(train_ids, val_ids, test_ids, all_ids)

        split_map = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }
        splits_to_run = [(name, split_map[name]) for name in split_names_to_run]

    # Processar cada conjunto
    for split_name, ids in splits_to_run:
        print(f"\n{'=' * 60}")
        action_label = "Consolidando" if args.merge_only else "Processando"
        print(f"🔬 {action_label} conjunto: {split_name.upper()}")
        print(f"{'=' * 60}")

        if args.merge_only:
            final_path = merge_batch_results(split_name, timestamped_output_dir)
            if final_path is None:
                print(f"❌ Nenhum lote encontrado para o split '{split_name}'")
                exit(1)

            df = pd.read_csv(final_path)
            details_for_metadata = df.to_dict("records")
            metadata_ids = (
                df["IMG_ID"].dropna().tolist() if "IMG_ID" in df.columns else []
            )
            metadata_path = save_metadata(
                split_name,
                timestamped_output_dir,
                effective_config,
                metadata_ids,
                details_for_metadata,
                execution_time=None,
                base_path=base_path,
                base_save_path=base_save_path,
                root_output_dir=output_root_dir,
            )
            logger.info(f"Metadados salvos em: {metadata_path}")
            print_split_summary(df, split_name, effective_config)
            continue

        summary = run_pipeline(
            ids,
            split_name,
            effective_config,
            base_path,
            base_save_path,
            timestamped_output_dir,
            resume_from_batch=args.resume_batches_by_split.get(
                split_name, args.resume_batch
            ),
        )
        execution_time = summary.get("execution_time")

        # Salvar/conciliar resultados CSV
        logger.info("Finalizando processamento em lotes...")
        merge_batch_results(split_name, timestamped_output_dir)
        output_path = Path(timestamped_output_dir) / f"ostios_{split_name}_summary.csv"
        logger.info(f"Resumo final salvo em: {output_path}")

        # Salvar metadados JSON (carregar detalhes do CSV se necessário)
        if summary.get("details") is None:
            details_for_metadata = pd.read_csv(output_path).to_dict("records")
        else:
            details_for_metadata = summary["details"]

        metadata_path = save_metadata(
            split_name,
            timestamped_output_dir,
            effective_config,
            ids,
            details_for_metadata,
            execution_time,
            base_path=base_path,
            base_save_path=base_save_path,
            root_output_dir=output_root_dir,
        )
        logger.info(f"Metadados salvos em: {metadata_path}")

        # Estatísticas do conjunto
        if summary.get("details") is None:
            df = pd.read_csv(output_path)
        else:
            df = make_result_dataframe(summary["details"])
        print_split_summary(df, split_name, effective_config, execution_time)

    print(f"\n{'=' * 60}")
    print("✨ Processamento concluído!")
    print(f"{'=' * 60}\n")


# ============================================================================
# EXECUÇÃO
# ============================================================================


if __name__ == "__main__":
    main()
