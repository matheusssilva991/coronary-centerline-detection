# ============================================================================
# IMPORTS
# ============================================================================
# Biblioteca padrão
import argparse
import os
import time
import copy

# Terceiros - Processamento Numérico
import numpy as np

# Terceiros - Machine Learning
from tqdm import tqdm
import pandas as pd

# Usa GPU 1 por padrão quando a variável não for definida externamente.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

# Locais
from utils import (
    use_gpu,
    load_config_json,
    scale_config_to_resolution,
    get_data_splits,
    create_timestamped_output_dir,
    make_result_dataframe,
    merge_batch_results,
    save_results,
    save_metadata,
    load_and_preprocess_image,
    get_or_compute_vesselness,
    get_or_detect_aorta_circles,
    get_or_segment_aorta,
    detect_and_evaluate_ostia,
    segment_arteries_from_ostia,
)


# ============================================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================================

# Informações sobre aceleração GPU
GPU_ENABLED = use_gpu()
if GPU_ENABLED:
    print("✓ GPU detectada! Operações aceleradas por GPU ativadas.")
    print("  - Binary morphology operations (closing, dilation)")
    print("  - Connected components labeling")
    print("  - Image downscaling")
else:
    print("⚠ GPU não disponível. Acelerações CPU usadas.")

# Caminhos padrão
BASE_PATH = "/media/matheus/HD/DatasetsCCTA/ImageCAS/1-1000"
# BASE_PATH = "/data04/home/mpmaia/ImageCAS/database/1-1000"
BASE_SAVE_PATH = "/media/matheus/HD/DatasetsCCTA/Processed_ImageCAS"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))

# Parâmetros de processamento padrão
CONFIG = {
    # GPU acceleration
    "USE_GPU": GPU_ENABLED,
    # Cache
    "LOAD_CACHE": False,
    "SAVE_CACHE": False,
    # Downscaling (com GPU quando possível)
    "DOWNSCALE_METHOD": "scipy",  # "scipy" ou "opencv" (GPU automática se disponível)
    "OPENCV_INTERPOLATION": "area",  # "nearest", "linear", "cubic", "area", "lanczos4"
    "DOWNSCALE_FACTORS": (2, 2, 1),
    "MAX_THRESHOLD_PERCENTILE": 99.7,
    # Vesselness - Detecção de Óstios (Aorta)
    "VESSELNESS_AORTA": {
        "sigmas": np.arange(2.5, 3.5, 1),
        "black_ridges": False,
        "alpha": 0.5,
        "beta": 1,
        "gamma": 30,
        "normalization": "none",
    },
    # Vesselness - Segmentação de Artérias
    "VESSELNESS_ARTERY": {
        "sigmas": np.arange(1.5, 3.5, 0.5),
        "black_ridges": False,
        "alpha": 0.5,
        "beta": 0.5,
        "gamma": 55,
        "normalization": "none",
    },
    # Detecção de Círculos (Transformada de Hough)
    "CIRCLE_DETECTION": {
        "radii_start_px": 36,
        "radii_end_px": 62,
        "radius_step_px": 1,
        "tol_radius_mm": 9.0,
        "tol_distance_mm": 20.0,
        "quadrant_offset": (30, 30),
        "max_slice_miss_threshold": 5,
        "neighbor_distance_threshold": 5,
        "total_num_peaks_initial": 15,
        "total_num_peaks": 15,
        "canny_sigma": 3,
        "use_local_roi": True,
        "local_roi_padding": 30,
    },
    # Level Set Segmentation
    "LEVEL_SET": {
        "radius_reduction_factor": 0.15,
        "num_iter": 31,
        "balloon": 0.8,
        "smoothing": 2,
        "leak_removal_radius": 2,
    },
    # Detecção de Óstios
    "OSTIA_DETECTION": {
        "top_n": 10000,
        "max_z_diff_mm": 40.0,
        "lower_fraction": 0.80,
        "min_center_distance_factor": 0.85,
        "min_lateral_factor": 0.4,
        "erosion_radius": 4,
    },
    # Validação de óstios
    "OSTIA_VALIDATION": {
        "distance_threshold_mm": 8.0,
    },
    # Region Growing - Segmentação de Artérias
    "REGION_GROWING": {
        "max_volume": 100000,
        "switch_at_voxels": 2000,
        "min_vesselness_fraction": 0.078,
        "threshold_divisor": 7,
        "relaxed_floor_factor": 0.98,
        "comparison_window": 1,
        "smooth_relaxation": True,
        "verbose": False,
    },
    # Pós-processamento (Closing e Dilation)
    "POSTPROCESSING": {
        "closing_radius": 3,
        "dilation_radius": 2,
    },
    # Processamento em Lotes
    "BATCH_SIZE": None,  # None = processar todas, ou número inteiro para lotes (ex: 10)
}

# ============================================================================
# CONFIGURAÇÕES POR RESOLUÇÃO
# ============================================================================
# MID RESOLUTION (padrão): Downscale 2x em X e Y, mantém Z
CONFIG_MID_RES = copy.deepcopy(CONFIG)

# HIGH RESOLUTION: Mantém resolução original (sem downscale)
CONFIG_HIGH_RES = copy.deepcopy(CONFIG)
CONFIG_HIGH_RES["DOWNSCALE_FACTORS"] = (1, 1, 1)

# ============================================================================
# PROCESSAMENTO DE IMAGEM
# ============================================================================


def process_image(IMG_ID, config=CONFIG):
    """
    Processa uma imagem completa: pré-processamento, segmentação de aorta,
    detecção de óstios e segmentação de artérias.

    Args:
        IMG_ID: ID da imagem a processar
        config: Dicionário de configurações

    Returns:
        dict: Resultados do processamento
    """
    result = {
        "IMG_ID": IMG_ID,
        "ostia_left": None,
        "ostia_right": None,
        "artery_voxels": None,
        "dice_artery": None,
        "ostia_found": False,
        "ostia_status": "not_evaluated",
        "segmentation_attempted": False,
        "proceeded_with_bad_ostia": False,
        "skip_reason": None,
        "ostia_error": None,
        "left_intersects": False,
        "right_intersects": False,
        "left_dist_voxels": None,
        "right_dist_voxels": None,
        "left_dist_mm": None,
        "right_dist_mm": None,
        "both_correct": False,
        "both_tolerable": False,
    }

    try:
        image_data = load_and_preprocess_image(IMG_ID, BASE_PATH, config)
        lcc_image = image_data["lcc_image"]
        label = image_data["label"]
        scaled_spacing = image_data["scaled_spacing"]
        downscale_factors = image_data["downscale_factors"]

        image_data = None  # Liberar memória

        vesselness_ostios = get_or_compute_vesselness(
            IMG_ID,
            lcc_image,
            cache_dir=f"{BASE_SAVE_PATH}/vesselness_ostios_cache",
            vesselness_config=config["VESSELNESS_AORTA"],
            load_cache=config["LOAD_CACHE"],
            save_cache=config["SAVE_CACHE"],
        )

        detected_circles = get_or_detect_aorta_circles(
            IMG_ID,
            lcc_image,
            downscale_factors,
            scaled_spacing,
            config["CIRCLE_DETECTION"],
            BASE_SAVE_PATH,
            load_cache=config["LOAD_CACHE"],
            save_cache=config["SAVE_CACHE"],
        )

        aorta_mask = get_or_segment_aorta(
            IMG_ID,
            lcc_image,
            detected_circles,
            config["LEVEL_SET"],
            BASE_SAVE_PATH,
            load_cache=config["LOAD_CACHE"],
            save_cache=config["SAVE_CACHE"],
        )

        try:
            ostia_eval = detect_and_evaluate_ostia(
                aorta_mask,
                vesselness_ostios,
                label,
                scaled_spacing,
                config,
            )

            del aorta_mask  # Liberar memória
        except ValueError as ostia_exc:
            result["ostia_status"] = "not_found"
            result["ostia_error"] = str(ostia_exc)
            result["skip_reason"] = "ostia_not_found"
            result["dice_artery"] = 0.0
            return result

        result["ostia_left"] = (
            tuple(map(int, ostia_eval["ostia_left"]))
            if ostia_eval["ostia_left"] is not None
            else None
        )
        result["ostia_right"] = (
            tuple(map(int, ostia_eval["ostia_right"]))
            if ostia_eval["ostia_right"] is not None
            else None
        )
        result["ostia_found"] = True
        result["left_intersects"] = ostia_eval["left_info"]["intersects"]
        result["right_intersects"] = ostia_eval["right_info"]["intersects"]
        result["left_dist_voxels"] = ostia_eval["left_info"]["euclidean_dist"]
        result["right_dist_voxels"] = ostia_eval["right_info"]["euclidean_dist"]
        result["left_dist_mm"] = ostia_eval["left_info"]["physical_dist"]
        result["right_dist_mm"] = ostia_eval["right_info"]["physical_dist"]
        result["both_correct"] = ostia_eval["both_correct"]
        result["both_tolerable"] = ostia_eval["both_tolerable"]

        if result["both_correct"]:
            result["ostia_status"] = "both_correct"
        elif result["both_tolerable"]:
            result["ostia_status"] = "both_tolerable"
        else:
            result["ostia_status"] = "found_but_wrong"

        is_bad_ostia = not (result["both_correct"] or result["both_tolerable"])

        if is_bad_ostia:
            result["proceeded_with_bad_ostia"] = True

        result["segmentation_attempted"] = True
        artery_metrics = segment_arteries_from_ostia(
            IMG_ID,
            lcc_image,
            ostia_eval["label_artery"],
            ostia_eval["ostia_left"],
            ostia_eval["ostia_right"],
            config,
            BASE_SAVE_PATH,
        )
        result.update(artery_metrics)

    except Exception as e:
        result["error"] = str(e)

    return result


def run_pipeline(ids, split_name, config=CONFIG, output_dir=None, resume_from_batch=0):
    """
    Processa um conjunto de imagens, opcionalmente em lotes com salvamento incremental.

    Args:
        ids: Lista de IDs das imagens
        split_name: Nome do conjunto (train, val, test)
        config: Dicionário de configurações
        output_dir: Diretório de saída para salvamento incremental (se None, retorna tudo ao final)
        resume_from_batch: Número do lote para retomar (0 = começar do início)

    Returns:
        dict: Dicionário com lista de resultados, tempo de execução e info de lotes
    """
    start_time = time.time()

    # Escalar parâmetros em voxels uma vez para toda a execução
    scaled_config = scale_config_to_resolution(config)

    batch_size = config.get("BATCH_SIZE")
    all_results = []
    batches_processed = []

    # Se não há batch_size definido ou output_dir não fornecido, processar tudo de uma vez
    if batch_size is None or output_dir is None:
        for img_id in tqdm(ids, desc=f"Processando {split_name}"):
            all_results.append(process_image(img_id, scaled_config))
    else:
        # Processar em lotes com salvamento incremental
        num_batches = (len(ids) + batch_size - 1) // batch_size

        # Se retomando, carregar lotes anteriores
        if resume_from_batch > 0:
            print(f"🔄 Retomando a partir do lote {resume_from_batch}...")
            for batch_num in range(0, resume_from_batch):
                batch_csv_path = os.path.join(
                    output_dir, f"ostios_{split_name}_lote_{batch_num + 1}.csv"
                )
                if os.path.exists(batch_csv_path):
                    df_batch = pd.read_csv(batch_csv_path)
                    # Converter DataFrame para lista de dicts (formato dos resultados)
                    batch_data = df_batch.to_dict("records")
                    all_results.extend(batch_data)
                    batches_processed.append(batch_num + 1)
                    print(
                        f"   ✓ Lote {batch_num + 1} carregado ({len(batch_data)} registros)"
                    )

        for batch_num in range(resume_from_batch, num_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(ids))
            batch_ids = ids[start_idx:end_idx]

            print(
                f"\n📦 Processando lote {batch_num + 1}/{num_batches} ({len(batch_ids)} imagens)"
            )
            batch_results = []

            for img_id in tqdm(
                batch_ids, desc=f"Lote {batch_num + 1}/{num_batches}", leave=False
            ):
                batch_results.append(process_image(img_id, scaled_config))

            all_results.extend(batch_results)
            batches_processed.append(batch_num + 1)

            # Salvar resultados intermediários
            batch_output_path = save_results(
                batch_results,
                f"{split_name}_lote_{batch_num + 1}",
                output_dir,
                config=scaled_config,
            )
            print(f"✅ Lote {batch_num + 1} salvo: {batch_output_path}")

    execution_time = time.time() - start_time

    return {
        "details": all_results,
        "execution_time": execution_time,
        "batches_processed": batches_processed,
        "is_batched": batch_size is not None and output_dir is not None,
    }


# ============================================================================
# FUNÇÕES AUXILIARES - PIPELINE
# ============================================================================


def print_statistics(train_ids, val_ids, test_ids, all_ids):
    """Imprime estatísticas dos conjuntos de dados."""
    print("\n" + "=" * 50)
    print("ESTATÍSTICAS DOS CONJUNTOS")
    print("=" * 50)
    print(
        f"Treino:    {len(train_ids):3d} imagens ({len(train_ids) / len(all_ids) * 100:5.1f}%)"
    )
    print(
        f"Validação: {len(val_ids):3d} imagens ({len(val_ids) / len(all_ids) * 100:5.1f}%)"
    )
    print(
        f"Teste:     {len(test_ids):3d} imagens ({len(test_ids) / len(all_ids) * 100:5.1f}%)"
    )
    print(f"Total:     {len(all_ids):3d} imagens")
    print("=" * 50 + "\n")


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
  # Processar 10 imagens por vez (salva cada lote, depois prossegue)
  python segmentation_pipeline.py --batch-size 10

  # Processar teste em lotes de 5 imagens
  python segmentation_pipeline.py --split test --batch-size 5

  # Combinar: teste em lotes com cache
  python segmentation_pipeline.py --split test --batch-size 10 --cache

  # RETOMADA DE LOTES (em caso de falha):
  # Se processamento falhou no lote 3, retomar a partir dali
  python segmentation_pipeline.py --batch-size 10 --resume-batch 3

  # Retomar treino do lote 2
  python segmentation_pipeline.py --split train --batch-size 10 --resume-batch 2

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
        "--batch-size",
        type=int,
        default=None,
        help="Tamanho do lote para processamento incremental com salvamento. Ex: 10 (processa 10 imagens, salva, depois mais 10, etc.)",
    )

    parser.add_argument(
        "--resume-batch",
        type=int,
        default=0,
        help="Número do lote para retomar (padrão: 0 = começar do início). Use se um processamento foi interrompido.",
    )

    args = parser.parse_args()

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
    if args.batch_size is not None:
        effective_config["BATCH_SIZE"] = args.batch_size
        print(f"📦 Processamento em lotes: {args.batch_size} imagens por lote")
        if args.resume_batch > 0:
            print(f"🔄 Retomando a partir do lote {args.resume_batch}")
    else:
        print("🔧 Processamento em imagem única (sem lotes)")

    # Criar diretório com timestamp
    timestamped_output_dir = create_timestamped_output_dir(
        args.output_dir, experiment_name="segmentation"
    )
    print(f"\n📁 Diretório de saída: {timestamped_output_dir}\n")

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
            timestamped_output_dir,
            resume_from_batch=args.resume_batch,
        )
        execution_time = summary.get("execution_time")
        is_batched = summary.get("is_batched", False)

        # Salvar resultados CSV
        if is_batched:
            # Modo batch: mesclar todos os lotes em um arquivo final
            print("\n🔄 Finalizando processamento em lotes...")
            merge_batch_results(split_name, timestamped_output_dir)
        else:
            # Modo normal: salvar resultado único
            output_path = save_results(
                summary["details"],
                split_name,
                timestamped_output_dir,
                config=effective_config,
            )
            print(f"✅ Resumo CSV salvo em: {output_path}")

        # Salvar metadados JSON
        metadata_path = save_metadata(
            split_name,
            timestamped_output_dir,
            effective_config,
            ids,
            summary["details"],
            execution_time,
            base_path=BASE_PATH,
            base_save_path=BASE_SAVE_PATH,
            root_output_dir=OUTPUT_DIR,
        )
        print(f"📊 Metadados salvos em: {metadata_path}")

        # Estatísticas do conjunto
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
