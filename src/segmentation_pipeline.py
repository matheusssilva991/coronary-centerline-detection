# ============================================================================
# IMPORTS
# ============================================================================
# Biblioteca padrão
import argparse
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
from utils.segmentation.pipeline_orchestration import (
    parse_resume_batches,
    print_statistics,
    run_pipeline,
)


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
    parser = argparse.ArgumentParser(
        description="Pipeline de segmentação coronária",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Processar todos os conjuntos
  python segmentation_pipeline.py

  # Processar apenas treino
  python segmentation_pipeline.py --split train

  # Processar validação e teste
  python segmentation_pipeline.py --split val test

  # Com cache habilitado (carregar caches existentes)
  python segmentation_pipeline.py --split train --cache

  # Sem salvar cache (não recomendado, apenas para testes rápidos)
  python segmentation_pipeline.py --split val --no-save-cache

  # Usar OpenCV para downscaling com interpolação AREA
  python segmentation_pipeline.py --downscale-method opencv

  # Usar OpenCV com interpolação LINEAR
  python segmentation_pipeline.py --downscale-method opencv --opencv-interpolation linear

  # Combinação: carregar cache, usar OpenCV (cubic) e processar só validação
  python segmentation_pipeline.py --split val --cache --downscale-method opencv --opencv-interpolation cubic

  # Usar resolução alta (sem downscaling)
  python segmentation_pipeline.py --resolution high

  # Usar resolução média (downscale 2x)
  python segmentation_pipeline.py --resolution mid --split val

  # PROCESSAMENTO EM LOTES (salvamento incremental):
    # Processar em 10 lotes (divide as imagens entre 10 blocos)
    python segmentation_pipeline.py --num-batches 10

    # Processar teste em 5 lotes
    python segmentation_pipeline.py --split test --num-batches 5

  # Combinar: teste em lotes com cache
    python segmentation_pipeline.py --split test --num-batches 10 --cache

  # RETOMADA DE LOTES (em caso de falha):
  # Primeira execução - cria novo diretório
    python segmentation_pipeline.py --split test --num-batches 70
  # Saída: output/segmentation/2026-03-14_10-30-00/

  # Se falhar no lote 3, retomar no MESMO diretório:
    python segmentation_pipeline.py --split test --num-batches 70 --resume-batch 3 --resume-dir output/segmentation/2026-03-14_10-30-00

    # Retomada explícita por subset:
    python segmentation_pipeline.py --split all --num-batches 70 --resume-batches train=0,val=3,test=0

  # Versão curta (se no mesmo diretório):
    python segmentation_pipeline.py --split test --num-batches 70 --resume-batch 3 --resume-dir ./output/segmentation/2026-03-14_10-30-00

Arquivos de saída:
  - ostios_{split}_summary.csv: Resultados consolidados ao final (ou após merge)
  - ostios_{split}_lote_1.csv, ostios_{split}_lote_2.csv, etc: Resultados de cada lote (modo batch)
  - ostios_{split}_metadata.json: Metadados completos (configurações, estatísticas, timestamp)
        """,
    )

    parser.add_argument(
        "--split",
        nargs="+",
        choices=["train", "val", "test", "all"],
        default=["all"],
        help="Conjunto(s) para processar (padrão: all)",
    )

    parser.add_argument(
        "--resolution",
        type=str,
        choices=["mid", "high"],
        default="mid",
        help="Resolução da imagem: 'mid' (downscale 2x) ou 'high' (sem downscale) (padrão: mid)",
    )

    parser.add_argument(
        "--cache", action="store_true", help="Habilitar carregamento de cache"
    )

    parser.add_argument(
        "--no-save-cache",
        action="store_true",
        help="Desabilitar salvamento de cache (não recomendado)",
    )

    parser.add_argument(
        "--downscale-method",
        type=str,
        choices=["scipy", "opencv"],
        default=None,
        help="Método de downscaling: scipy (ndi.zoom) ou opencv (cv2.resize)",
    )

    parser.add_argument(
        "--opencv-interpolation",
        type=str,
        choices=["nearest", "linear", "cubic", "area", "lanczos4"],
        default=None,
        help="Método de interpolação do OpenCV (usado apenas se --downscale-method=opencv)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Diretório de saída (padrão: {OUTPUT_DIR})",
    )

    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Arquivo JSON com configurações para sobrescrever valores padrão",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Habilitar logging detalhado (DEBUG)",
    )

    parser.add_argument(
        "--num-batches",
        type=int,
        default=5,
        help="Número de lotes para dividir o conjunto de imagens (ex: 5 divide as 700 imagens em 5 lotes)",
    )

    parser.add_argument(
        "--resume-batch",
        type=int,
        default=0,
        help="Número do lote para retomar (padrão: 0 = começar do início). Use se um processamento foi interrompido.",
    )

    parser.add_argument(
        "--resume-batches",
        type=str,
        default=None,
        help="Retomada explícita por subset no formato 'train=1,val=0,test=3'. Se informado, sobrescreve --resume-batch para os splits listados.",
    )

    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Diretório anterior para retomar (ex: output/segmentation/2026-03-14_10-30-00). Se não fornecido e --resume-batch > 0, cria novo diretório.",
    )

    args = parser.parse_args()

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

    # Configurar batch processing
    effective_config["NUM_BATCHES"] = args.num_batches
    print(f"📦 Processamento em {args.num_batches} lotes")
    if args.resume_batch > 0:
        print(f"🔄 Retomando a partir do lote {args.resume_batch}")

    try:
        resume_batches_by_split = parse_resume_batches(args.resume_batches)
    except ValueError as exc:
        print(f"❌ Erro: {exc}")
        exit(1)

    if args.resume_batches:
        print(
            "🔄 Retomada por subset: "
            f"train={resume_batches_by_split['train']}, "
            f"val={resume_batches_by_split['val']}, "
            f"test={resume_batches_by_split['test']}"
        )

    # Criar ou reusar diretório
    if args.resume_batch > 0 and args.resume_dir:
        # Modo retomada: usar diretório anterior
        if os.path.exists(args.resume_dir):
            timestamped_output_dir = args.resume_dir
            print(f"\n📁 Usando diretório anterior: {timestamped_output_dir}\n")
        else:
            print(f"❌ Erro: Diretório não encontrado: {args.resume_dir}")
            print("   Use --resume-dir com o caminho do diretório anterior")
            exit(1)
    else:
        # Modo normal: criar novo diretório com timestamp
        timestamped_output_dir = create_timestamped_output_dir(
            args.output_dir, experiment_name="segmentation"
        )
        if args.resume_batch > 0:
            print("⚠️  Dica: Para retomar no mesmo diretório, use:")
            print(
                f"   --resume-batch {args.resume_batch} --resume-dir {timestamped_output_dir}\n"
            )
        print(f"📁 Diretório de saída: {timestamped_output_dir}\n")

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

    # Obter splits de dados
    train_ids, val_ids, test_ids, all_ids = get_data_splits(BASE_PATH)
    print_statistics(train_ids, val_ids, test_ids, all_ids)

    # Determinar quais conjuntos processar
    splits_to_run = []
    if "all" in args.split:
        splits_to_run = [
            ("train", train_ids),
            ("val", val_ids),
            ("test", test_ids),
        ]
    else:
        split_map = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }
        splits_to_run = [(name, split_map[name]) for name in args.split]

    # Processar cada conjunto
    for split_name, ids in splits_to_run:
        print(f"\n{'=' * 60}")
        print(f"🔬 Processando conjunto: {split_name.upper()}")
        print(f"{'=' * 60}")

        summary = run_pipeline(
            ids,
            split_name,
            effective_config,
            BASE_PATH,
            BASE_SAVE_PATH,
            timestamped_output_dir,
            resume_from_batch=resume_batches_by_split.get(
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
            base_path=BASE_PATH,
            base_save_path=BASE_SAVE_PATH,
            root_output_dir=OUTPUT_DIR,
        )
        logger.info(f"Metadados salvos em: {metadata_path}")

        # Estatísticas do conjunto
        if summary.get("details") is None:
            df = pd.read_csv(output_path)
        else:
            df = make_result_dataframe(summary["details"])
        if not df.empty:
            both_correct_series = df["both_correct"].fillna(False)
            both_tolerable_series = df["both_tolerable"].fillna(False)
            ostia_found_series = df["ostia_found"].fillna(False)
            ostia_not_found_series = df["ostia_status"].eq("not_found")
            segmentation_attempted_series = df["segmentation_attempted"].fillna(False)
            proceeded_with_bad_ostia_series = df["proceeded_with_bad_ostia"].fillna(
                False
            )
            tolerance_mm = effective_config["OSTIA_VALIDATION"]["distance_threshold_mm"]

            print(f"\n📊 Estatísticas do conjunto {split_name}:")
            print(
                f"   - Óstios encontrados:         {ostia_found_series.sum():3d} ({ostia_found_series.mean() * 100:5.1f}%)"
            )
            print(
                f"   - Óstios não encontrados:     {ostia_not_found_series.sum():3d} ({ostia_not_found_series.mean() * 100:5.1f}%)"
            )
            print(
                f"   - Ambos corretos (estrito): {both_correct_series.sum():3d} ({both_correct_series.mean() * 100:5.1f}%)"
            )
            print(
                f"   - Tolerável apenas:         {both_tolerable_series.sum():3d} ({both_tolerable_series.mean() * 100:5.1f}%)"
            )
            print(
                f"   - Segmentação tentada:      {segmentation_attempted_series.sum():3d} ({segmentation_attempted_series.mean() * 100:5.1f}%)"
            )
            print(
                f"   - Prosseguiu com óstio ruim:{proceeded_with_bad_ostia_series.sum():3d} ({proceeded_with_bad_ostia_series.mean() * 100:5.1f}%)"
            )
            print(
                f"   - Total sucesso (<= {tolerance_mm}mm): {(both_correct_series | both_tolerable_series).sum():3d} ({(both_correct_series | both_tolerable_series).mean() * 100:5.1f}%)"
            )
            if "dice_artery" in df.columns and df["dice_artery"].notna().any():
                print(f"   - Dice médio:       {df['dice_artery'].mean():.4f}")
            if execution_time:
                print(
                    f"   - Tempo de execução: {execution_time:.1f}s ({execution_time / 60:.1f}min)"
                )

    print(f"\n{'=' * 60}")
    print("✨ Processamento concluído!")
    print(f"{'=' * 60}\n")


# ============================================================================
# EXECUÇÃO
# ============================================================================


if __name__ == "__main__":
    main()
