# ============================================================================
# IMPORTS
# ============================================================================
# Biblioteca padrão
import os
import json
import argparse
import time
import platform
from glob import glob
from datetime import datetime

# Terceiros - Processamento Numérico
import numpy as np
import pandas as pd
from skimage.morphology import ball

# Terceiros - Machine Learning
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

# Locais
from utils import (
    get_vesselness,
    load_vesselness_cache,
    save_vesselness_cache,
    load_raw_img_and_label,
    run_core_preprocessing_pipeline,
    keep_largest_component,
    detect_aorta_circles,
    level_set_segmentation,
    remove_leaks_morphology,
    find_ostia,
    check_ostium_intersection,
    save_npy_array,
    region_growing_segmentation,
    dice_score,
    downscale_image_ndi,
    downscale_image,
    binary_closing,
    binary_dilation,
    use_gpu,
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
#BASE_PATH = "/media/matheus/HD/DatasetsCCTA/ImageCAS"
BASE_PATH = "/data04/home/mpmaia/ImageCAS/database/1-1000"
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
    # Avaliação
    "TOLERABLE_DISTANCE_MM": 7.0,
    # Vesselness - Detecção de Óstios (Aorta)
    "VESSELNESS_AORTA": {
        "sigmas": np.arange(2.5, 3.5, 1),
        "alpha": 0.5,
        "beta": 1,
        "gamma": 30,
    },
    # Vesselness - Segmentação de Artérias
    "VESSELNESS_ARTERY": {
        "sigmas": np.arange(1.5, 3.5, 0.5),
        "alpha": 0.5,
        "beta": 0.5,
        "gamma": 55,
    },
    # Detecção de Círculos (Transformada de Hough)
    "CIRCLE_DETECTION": {
        "radii_start_mm": 36,
        "radii_end_mm": 62,
        "tol_radius_mm": 9.0,
        "tol_distance_mm": 20.0,
        "max_slice_miss_threshold": 5,
        "neighbor_distance_threshold": 5,
        "total_num_peaks_initial": 10,
        "total_num_peaks": 15,
        "canny_sigma": 3,
    },
    # Level Set Segmentation
    "LEVEL_SET": {
        "radius_reduction_factor": 0.15,
        "num_iter": 35,
        "balloon": 0.8,
        "smoothing": 2,
    },
    # Detecção de Óstios
    "OSTIA_DETECTION": {
        "top_n": 2000,
        "max_z_diff_mm": 40.0,
        "lower_fraction": 0.80,
        "min_center_distance_factor": 0.85,
        "min_lateral_factor": 0.4,
        "erosion_radius": 4,
    },
    # Pós-processamento (Closing e Dilation)
    "POSTPROCESSING": {
        "closing_radius": 3,
        "dilation_radius": 2,
    },
}


# ============================================================================
# FUNÇÕES AUXILIARES - DIVISÃO DE DADOS
# ============================================================================


def create_timestamped_output_dir(base_output_dir, experiment_name="segmentation"):
    """
    Cria um diretório de saída com estrutura: base_output_dir/experiment_name/YYYY-MM-DD_HH-MM-SS

    Args:
        base_output_dir: Diretório base de saída
        experiment_name: Nome do experimento (padrão: "segmentation")

    Returns:
        str: Caminho do diretório criado
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(base_output_dir, experiment_name, timestamp)
    os.makedirs(output_path, exist_ok=True)
    return output_path


def get_data_splits(base_path, test_size=0.7, val_size=0.1, random_state=42):
    """
    Divide o dataset em treino, validação e teste.

    Args:
        base_path: Caminho base dos dados
        test_size: Proporção para teste (padrão: 0.7 = 70%)
        val_size: Proporção de validação dentro do treino (padrão: 0.1 = 10%)
        random_state: Seed para reprodutibilidade

    Returns:
        tuple: (train_ids, val_ids, test_ids, all_ids)
    """
    img_files = sorted(glob(os.path.join(base_path, "*.img.nii.gz")))
    all_ids = [int(os.path.basename(f).split(".")[0]) for f in img_files]

    # Dividir em treino+validação e teste
    train_val_ids, test_ids = train_test_split(
        all_ids, test_size=test_size, random_state=random_state
    )

    # Dividir treino em treino e validação
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_size, random_state=random_state
    )

    return train_ids, val_ids, test_ids, all_ids


# ============================================================================
# FUNÇÕES AUXILIARES - FORMATAÇÃO DE RESULTADOS
# ============================================================================


def make_result_dataframe(results):
    """
    Converte lista de resultados em DataFrame formatado.

    Args:
        results: Lista de dicionários com resultados do processamento

    Returns:
        pd.DataFrame: DataFrame com resultados formatados
    """
    rows = []
    for r in results:
        row = {
            "IMG_ID": r.get("IMG_ID"),
            "dice_artery": r.get("dice_artery"),
            "artery_voxels": r.get("artery_voxels"),
            "both_correct": r.get("both_correct", False),
            "both_tolerable": r.get("both_tolerable", False),
            "left_intersects": r.get("left_intersects", False),
            "right_intersects": r.get("right_intersects", False),
            "left_dist_mm": r.get("left_dist_mm"),
            "right_dist_mm": r.get("right_dist_mm"),
            "ostia_left": r.get("ostia_left"),
            "ostia_right": r.get("ostia_right"),
            "error": r.get("error", None),
        }

        # Status detalhado
        if r.get("both_correct", False):
            row["status"] = "ambos corretos"
        elif r.get("both_tolerable", False):
            row["status"] = "ambos toleráveis"
        elif r.get("left_intersects", False) or r.get("right_intersects", False):
            row["status"] = "um correto"
        elif r.get("error"):
            row["status"] = "erro"
        else:
            row["status"] = "nenhum correto"

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# FUNÇÕES AUXILIARES - SALVAMENTO DE RESULTADOS
# ============================================================================


def save_results(results, split_name, output_dir, config=None):
    """
    Salva resultados em CSV.

    Args:
        results: Lista de resultados do processamento
        split_name: Nome do conjunto (train, val, test)
        output_dir: Diretório de saída
        config: Dicionário de configurações usado (opcional)

    Returns:
        str: Caminho do arquivo salvo
    """
    df = make_result_dataframe(results)

    # Adicionar informações de configuração ao DataFrame se fornecidas
    if config is not None:
        df["downscale_method"] = config.get("DOWNSCALE_METHOD", "N/A")
        df["opencv_interpolation"] = (
            config.get("OPENCV_INTERPOLATION", "N/A")
            if config.get("DOWNSCALE_METHOD") == "opencv"
            else "N/A"
        )
        df["downscale_factors"] = str(config.get("DOWNSCALE_FACTORS", "N/A"))
        df["max_threshold_percentile"] = config.get("MAX_THRESHOLD_PERCENTILE", "N/A")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"ostios_{split_name}_summary.csv")
    df.to_csv(output_path, index=False)
    return output_path


def save_metadata(split_name, output_dir, config, ids, results, execution_time=None):
    """
    Salva metadados da execução em arquivo JSON.

    Args:
        split_name: Nome do conjunto (train, val, test)
        output_dir: Diretório de saída
        config: Dicionário de configurações usado
        ids: Lista de IDs processados
        results: Lista de resultados do processamento
        execution_time: Tempo de execução em segundos (opcional)

    Returns:
        str: Caminho do arquivo de metadados salvo
    """
    # Calcular estatísticas
    df = make_result_dataframe(results)

    both_correct_series = df["both_correct"].fillna(False)
    both_tolerable_series = df["both_tolerable"].fillna(False)

    metadata = {
        "execution_info": {
            "timestamp": datetime.now().isoformat(),
            "split_name": split_name,
            "num_images": len(ids),
            "image_ids": ids,
            "execution_time_seconds": execution_time,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "preprocessing_config": {
            "downscale_method": config.get("DOWNSCALE_METHOD"),
            "opencv_interpolation": config.get("OPENCV_INTERPOLATION")
            if config.get("DOWNSCALE_METHOD") == "opencv"
            else None,
            "downscale_factors": config.get("DOWNSCALE_FACTORS"),
            "max_threshold_percentile": config.get("MAX_THRESHOLD_PERCENTILE"),
        },
        "vesselness_config": {
            "ostios": {
                "sigmas": config["VESSELNESS_AORTA"]["sigmas"].tolist()
                if hasattr(config["VESSELNESS_AORTA"]["sigmas"], "tolist")
                else list(config["VESSELNESS_AORTA"]["sigmas"]),
                "alpha": config["VESSELNESS_AORTA"]["alpha"],
                "beta": config["VESSELNESS_AORTA"]["beta"],
                "gamma": config["VESSELNESS_AORTA"]["gamma"],
            },
            "artery": {
                "sigmas": config["VESSELNESS_ARTERY"]["sigmas"].tolist()
                if hasattr(config["VESSELNESS_ARTERY"]["sigmas"], "tolist")
                else list(config["VESSELNESS_ARTERY"]["sigmas"]),
                "alpha": config["VESSELNESS_ARTERY"]["alpha"],
                "beta": config["VESSELNESS_ARTERY"]["beta"],
                "gamma": config["VESSELNESS_ARTERY"]["gamma"],
            },
        },
        "circle_detection_config": config.get("CIRCLE_DETECTION"),
        "level_set_config": config.get("LEVEL_SET"),
        "ostia_detection_config": config.get("OSTIA_DETECTION"),
        "postprocessing_config": config.get("POSTPROCESSING"),
        "cache_config": {
            "load_cache": config.get("LOAD_CACHE"),
            "save_cache": config.get("SAVE_CACHE"),
        },
        "evaluation_config": {
            "tolerable_distance_mm": config.get("TOLERABLE_DISTANCE_MM"),
        },
        "results_summary": {
            "total_processed": len(df),
            "both_correct": int(both_correct_series.sum()),
            "both_correct_percent": float(both_correct_series.mean() * 100),
            # both_tolerable é exclusivo: tolerável, mas não correto estrito
            "both_tolerable": int(both_tolerable_series.sum()),
            "both_tolerable_percent": float(both_tolerable_series.mean() * 100),
            # Mantido por compatibilidade com análises antigas
            "both_tolerable_only": int(both_tolerable_series.sum()),
            "both_tolerable_only_percent": float(both_tolerable_series.mean() * 100),
            # Sucesso total sem dupla contagem
            "total_success": int((both_correct_series | both_tolerable_series).sum()),
            "total_success_percent": float(
                (both_correct_series | both_tolerable_series).mean() * 100
            ),
            "left_correct": int(df["left_intersects"].sum()),
            "right_correct": int(df["right_intersects"].sum()),
            "errors": int(df["error"].notna().sum()),
            "dice_artery_mean": float(df["dice_artery"].mean())
            if df["dice_artery"].notna().any()
            else None,
            "dice_artery_std": float(df["dice_artery"].std())
            if df["dice_artery"].notna().any()
            else None,
            "dice_artery_median": float(df["dice_artery"].median())
            if df["dice_artery"].notna().any()
            else None,
        },
        "paths": {
            "base_path": BASE_PATH,
            "base_save_path": BASE_SAVE_PATH,
            "output_dir": OUTPUT_DIR,
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, f"ostios_{split_name}_metadata.json")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata_path


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
        # Carregar imagem e label
        nii_img, nii_label = load_raw_img_and_label(
            f"{BASE_PATH}/{IMG_ID}.img.nii.gz", f"{BASE_PATH}/{IMG_ID}.label.nii.gz"
        )
        spacing = nii_img.header.get_zooms()
        img = np.array(nii_img.get_fdata())
        label = np.array(nii_label.get_fdata()).astype(np.uint8)

        # Pré-processamento
        downscale_factors = config["DOWNSCALE_FACTORS"]
        use_opencv = config["DOWNSCALE_METHOD"] == "opencv"

        # Mapear string de interpolação para constante cv2
        interpolation_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos4": cv2.INTER_LANCZOS4,
        }
        opencv_interpolation = interpolation_map.get(
            config["OPENCV_INTERPOLATION"], cv2.INTER_AREA
        )

        down_image, thresh_image, lcc_image, thresh_vals = (
            run_core_preprocessing_pipeline(
                img,
                downscale_factors=downscale_factors,
                lcc_per_slice=True,
                max_threshold_percentile=config["MAX_THRESHOLD_PERCENTILE"],
                use_opencv=use_opencv,
                opencv_interpolation=opencv_interpolation,
            )
        )
        label = downscale_image_ndi(label, downscale_factors, order=0)

        # Calcular spacing ajustado
        dx, dy, dz = (
            spacing[0] * downscale_factors[0],
            spacing[1] * downscale_factors[1],
            spacing[2] * downscale_factors[2],
        )

        # Vesselness para detecção de óstios
        cache = load_vesselness_cache(
            IMG_ID, cache_dir=f"{BASE_SAVE_PATH}/vesselness_ostios_cache"
        )
        if cache is not None and config["LOAD_CACHE"]:
            vesselness_ostios = cache
        else:
            vesselness_config = config["VESSELNESS_AORTA"]
            vesselness_ostios = get_vesselness(
                lcc_image,
                sigmas=vesselness_config["sigmas"],
                black_ridges=False,
                alpha=vesselness_config["alpha"],
                beta=vesselness_config["beta"],
                gamma=vesselness_config["gamma"],
                normalization="none",
            )
            if config["SAVE_CACHE"]:
                save_vesselness_cache(
                    vesselness_ostios,
                    IMG_ID,
                    cache_dir=f"{BASE_SAVE_PATH}/vesselness_ostios_cache",
                )

        # Detecção de círculos (aorta)
        saved_dir_circles = f"{BASE_SAVE_PATH}/detected_circles"
        json_path = os.path.join(saved_dir_circles, f"{IMG_ID}_detected_circles.json")

        if os.path.exists(json_path) and config["LOAD_CACHE"]:
            with open(json_path, "r") as f:
                detected_circles = json.load(f)
        else:
            circle_config = config["CIRCLE_DETECTION"]
            radii_start = circle_config["radii_start_mm"] / downscale_factors[0]
            radii_end = circle_config["radii_end_mm"] / downscale_factors[0]
            hough_radii = np.arange(radii_start, radii_end, 1)
            pixel_spacing = (dx + dy) / 2.0

            detected_circles = detect_aorta_circles(
                lcc_image,
                hough_radii,
                pixel_spacing,
                tol_radius_mm=circle_config["tol_radius_mm"],
                tol_distance_mm=circle_config["tol_distance_mm"],
                max_slice_miss_threshold=circle_config["max_slice_miss_threshold"],
                neighbor_distance_threshold=circle_config[
                    "neighbor_distance_threshold"
                ],
                total_num_peaks_initial=circle_config["total_num_peaks_initial"],
                total_num_peaks=circle_config["total_num_peaks"],
                canny_sigma=circle_config["canny_sigma"],
            )
            if config["SAVE_CACHE"]:
                os.makedirs(saved_dir_circles, exist_ok=True)
                with open(json_path, "w") as f:
                    json.dump(detected_circles, f, indent=4)

        # Segmentação da aorta com Level Set
        saved_dir_aorta = f"{BASE_SAVE_PATH}/segmented_aorta"
        mask_path = os.path.join(saved_dir_aorta, f"{IMG_ID}_mask_aorta.npy")

        if os.path.exists(mask_path) and config["LOAD_CACHE"]:
            aorta_mask = np.load(mask_path)
        else:
            ls_config = config["LEVEL_SET"]
            mask_refined = level_set_segmentation(
                lcc_image,
                detected_circles,
                radius_reduction_factor=ls_config["radius_reduction_factor"],
                num_iter=ls_config["num_iter"],
                balloon=ls_config["balloon"],
                smoothing=ls_config["smoothing"],
            )
            aorta_mask = remove_leaks_morphology(mask_refined, radius=2)
            aorta_mask = keep_largest_component(aorta_mask)  # GPU-accelerated
            aorta_mask = aorta_mask.astype(np.uint8)
            if config["SAVE_CACHE"]:
                os.makedirs(saved_dir_aorta, exist_ok=True)
                save_npy_array(aorta_mask, mask_path)

        # Encontrar óstios
        ostia_config = config["OSTIA_DETECTION"]
        ostia_left, ostia_right = find_ostia(
            aorta_mask,
            vesselness_ostios,
            spacing=(dx, dy, dz),
            top_n=ostia_config["top_n"],
            max_z_diff_mm=ostia_config["max_z_diff_mm"],
            lower_fraction=ostia_config["lower_fraction"],
            min_center_distance_factor=ostia_config["min_center_distance_factor"],
            min_lateral_factor=ostia_config["min_lateral_factor"],
            erosion_radius=ostia_config["erosion_radius"],
        )
        result["ostia_left"] = tuple(map(int, ostia_left))
        result["ostia_right"] = tuple(map(int, ostia_right))

        # Avaliar óstios
        label_artery = (label == 1).astype(np.uint8)
        left_info = check_ostium_intersection(
            ostia_left, label_artery, spacing=(dy, dx, dz), ostium_name="Óstio esquerdo"
        )
        right_info = check_ostium_intersection(
            ostia_right, label_artery, spacing=(dy, dx, dz), ostium_name="Óstio direito"
        )

        result["left_intersects"] = left_info["intersects"]
        result["right_intersects"] = right_info["intersects"]
        result["left_dist_voxels"] = left_info["euclidean_dist"]
        result["right_dist_voxels"] = right_info["euclidean_dist"]
        result["left_dist_mm"] = left_info["physical_dist"]
        result["right_dist_mm"] = right_info["physical_dist"]

        # Critérios de avaliação
        tolerable = config["TOLERABLE_DISTANCE_MM"]
        result["both_correct"] = left_info["intersects"] and right_info["intersects"]
        both_tolerable_inclusive = (
            left_info["intersects"] or left_info["physical_dist"] <= tolerable
        ) and (right_info["intersects"] or right_info["physical_dist"] <= tolerable)
        # Torna ambas as métricas mutuamente exclusivas no CSV
        result["both_tolerable"] = both_tolerable_inclusive and (not result["both_correct"])

        # Segmentação das artérias (apenas se óstios corretos/toleráveis)
        if result["both_correct"] or result["both_tolerable"]:
            cache_artery = load_vesselness_cache(
                IMG_ID, cache_dir=f"{BASE_SAVE_PATH}/vesselness_artery_cache"
            )
            if cache_artery is not None and config["LOAD_CACHE"]:
                vesselness_artery = cache_artery
            else:
                vesselness_artery_config = config["VESSELNESS_ARTERY"]
                vesselness_artery = get_vesselness(
                    lcc_image,
                    sigmas=vesselness_artery_config["sigmas"],
                    black_ridges=False,
                    alpha=vesselness_artery_config["alpha"],
                    beta=vesselness_artery_config["beta"],
                    gamma=vesselness_artery_config["gamma"],
                    normalization="none",
                )
                if config["SAVE_CACHE"]:
                    save_vesselness_cache(
                        vesselness_artery,
                        IMG_ID,
                        cache_dir=f"{BASE_SAVE_PATH}/vesselness_artery_cache",
                    )

            # Region Growing
            region_growing_params = {
                "threshold": (vesselness_artery.max() - vesselness_artery.min()) / 5,
                "max_volume": 100000,
                "min_vesselness": vesselness_artery.max() * 0.078,
                "relaxed_floor_factor": 0.97,
                "switch_at_voxels": 2000,
                "comparison_window": 1,
                "smooth_relaxation": True,
                "verbose": False,
            }

            left_mask = region_growing_segmentation(
                vesselness_artery,
                seed_point=ostia_left,
                **region_growing_params,
            )

            right_mask = region_growing_segmentation(
                vesselness_artery,
                seed_point=ostia_right,
                **region_growing_params,
            )

            artery_mask = (left_mask + right_mask).astype(np.uint8)

            # Pós-processamento (com aceleração GPU)
            post_config = config["POSTPROCESSING"]
            closed_mask = binary_closing(
                artery_mask > 0, structure=ball(post_config["closing_radius"])
            )
            dilated_mask = binary_dilation(
                closed_mask, structure=ball(post_config["dilation_radius"])
            )
            artery_mask = dilated_mask

            result["artery_voxels"] = int(np.sum(artery_mask))
            result["dice_artery"] = float(dice_score(artery_mask, label_artery))

    except Exception as e:
        result["error"] = str(e)

    return result


def run_pipeline(ids, split_name, config=CONFIG):
    """
    Processa um conjunto de imagens.

    Args:
        ids: Lista de IDs das imagens
        split_name: Nome do conjunto (train, val, test)
        config: Dicionário de configurações

    Returns:
        dict: Dicionário com lista de resultados e tempo de execução
    """
    start_time = time.time()

    results = []
    for img_id in tqdm(ids, desc=f"Processando {split_name}"):
        results.append(process_image(img_id, config))

    execution_time = time.time() - start_time

    return {
        "details": results,
        "execution_time": execution_time,
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

Arquivos de saída:
  - ostios_{split}_summary.csv: Resultados detalhados por imagem com parâmetros usados
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
        default="scipy",
        help="Método de downscaling: scipy (ndi.zoom) ou opencv (cv2.resize)",
    )

    parser.add_argument(
        "--opencv-interpolation",
        type=str,
        choices=["nearest", "linear", "cubic", "area", "lanczos4"],
        default="area",
        help="Método de interpolação do OpenCV (usado apenas se --downscale-method=opencv)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Diretório de saída (padrão: {OUTPUT_DIR})",
    )

    args = parser.parse_args()

    # Atualizar configurações
    if args.cache:
        CONFIG["LOAD_CACHE"] = True
        print("⚙️  Carregamento de cache habilitado")

    if args.no_save_cache:
        CONFIG["SAVE_CACHE"] = False
        print("⚠️  Salvamento de cache desabilitado")
    else:
        print("💾 Salvamento de cache habilitado")

    CONFIG["DOWNSCALE_METHOD"] = args.downscale_method
    CONFIG["OPENCV_INTERPOLATION"] = args.opencv_interpolation

    if args.downscale_method == "opencv":
        print(
            f"🔧 Método de downscale: {args.downscale_method} (interpolação: {args.opencv_interpolation})"
        )
    else:
        print(f"🔧 Método de downscale: {args.downscale_method}")

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

        summary = run_pipeline(ids, split_name, CONFIG)
        execution_time = summary.get("execution_time")

        # Salvar resultados CSV
        output_path = save_results(
            summary["details"], split_name, timestamped_output_dir, config=CONFIG
        )
        print(f"✅ Resumo CSV salvo em: {output_path}")

        # Salvar metadados JSON
        metadata_path = save_metadata(
            split_name,
            timestamped_output_dir,
            CONFIG,
            ids,
            summary["details"],
            execution_time,
        )
        print(f"📊 Metadados salvos em: {metadata_path}")

        # Estatísticas do conjunto
        df = make_result_dataframe(summary["details"])
        if not df.empty:
            both_correct_series = df["both_correct"].fillna(False)
            both_tolerable_series = df["both_tolerable"].fillna(False)

            print(f"\n📊 Estatísticas do conjunto {split_name}:")
            print(
                f"   - Ambos corretos (estrito): {both_correct_series.sum():3d} ({both_correct_series.mean() * 100:5.1f}%)"
            )
            print(
                f"   - Tolerável apenas:         {both_tolerable_series.sum():3d} ({both_tolerable_series.mean() * 100:5.1f}%)"
            )
            print(
                f"   - Total sucesso (<= {CONFIG['TOLERABLE_DISTANCE_MM']}mm): {(both_correct_series | both_tolerable_series).sum():3d} ({(both_correct_series | both_tolerable_series).mean() * 100:5.1f}%)"
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
