import os
import json
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import binary_closing, binary_dilation
from skimage.morphology import ball

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
    downscale_image,
)


# Configurações
LOAD_CACHE = False
base_path = "/media/matheus/HD/DatasetsCCTA/ImageCAS"
base_save_path = "/media/matheus/HD/DatasetsCCTA/Processed_ImageCAS"
output_report_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "output", "ostios_report.json")
)

# Listar todas as imagens disponíveis
img_files = sorted(glob(os.path.join(base_path, "*.img.nii.gz")))
ids = [int(os.path.basename(f).split(".")[0]) for f in img_files]

# Dividir em treino e teste (80/20)
train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=42)


def process_image(IMG_ID):
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
        nii_img, nii_label = load_raw_img_and_label(
            f"{base_path}/{IMG_ID}.img.nii.gz", f"{base_path}/{IMG_ID}.label.nii.gz"
        )
        spacing = nii_img.header.get_zooms()
        img = np.array(nii_img.get_fdata())
        label = np.array(nii_label.get_fdata()).astype(np.uint8)
        downscale_factors = (2, 2, 1)
        down_image, thresh_image, lcc_image, thresh_vals = (
            run_core_preprocessing_pipeline(
                img,
                downscale_factors=downscale_factors,
                lcc_per_slice=True,
                max_threshold_percentile=99.7,
            )
        )
        label = downscale_image(label, downscale_factors, order=0)
        dx, dy, dz = (
            spacing[0] * downscale_factors[0],
            spacing[1] * downscale_factors[1],
            spacing[2] * downscale_factors[2],
        )
        affine_downscaled = nii_img.affine.copy()
        affine_downscaled[0, 0] /= downscale_factors[0]
        affine_downscaled[1, 1] /= downscale_factors[1]
        affine_downscaled[2, 2] /= downscale_factors[2]

        cache = load_vesselness_cache(
            IMG_ID, cache_dir=f"{base_save_path}/vesselness_cache"
        )
        if cache is not None and LOAD_CACHE:
            vesselness_i = cache
        else:
            vesselness_i = get_vesselness(
                lcc_image,
                sigmas=np.arange(2.5, 3.5, 1),
                black_ridges=False,
                alpha=0.5,  # controla a distinção entre uma estrutura Tubular (artéria) e uma estrutura Plana
                beta=1,  # controla a distinção entre um Tubo (comprido) e uma Bola (redonda/blob)
                gamma=30,  # controla a sensibilidade ao Contraste/Intensidade. Ele distingue entre Estrutura Real e Ruído de Fundo
                normalization="none",
            )
            save_vesselness_cache(
                vesselness_i, IMG_ID, cache_dir=f"{base_save_path}/vesselness_cache"
            )

        # Segmentação da aorta
        detected_circles = []
        saved_dir_circles = f"{base_save_path}/detected_circles"
        os.makedirs(saved_dir_circles, exist_ok=True)
        json_path = os.path.join(saved_dir_circles, f"{IMG_ID}_detected_circles.json")
        img_target = lcc_image
        radii_start = 36 / downscale_factors[0]
        radii_end = 62 / downscale_factors[0]
        hough_radii = np.arange(radii_start, radii_end, 1)
        pixel_spacing = (dx + dy) / 2.0

        # Carregar círculos detectados
        if os.path.exists(json_path) and LOAD_CACHE:
            with open(json_path, "r") as f:
                detected_circles = json.load(f)
        else:
            detected_circles = detect_aorta_circles(
                img_target,
                hough_radii,
                pixel_spacing,
                tol_radius_mm=9.0,
                tol_distance_mm=18.0,
                max_slice_miss_threshold=5,
                neighbor_distance_threshold=5,
                total_num_peaks_initial=10,
                total_num_peaks=15,
                canny_sigma=3,
            )
            with open(json_path, "w") as f:
                json.dump(detected_circles, f, indent=4)

        saved_dir_aorta = f"{base_save_path}/segmented_aorta"
        os.makedirs(saved_dir_aorta, exist_ok=True)
        mask_path = os.path.join(saved_dir_aorta, f"{IMG_ID}_mask_aorta.npy")
        # Carregar ou gerar máscara da aorta
        if os.path.exists(mask_path) and LOAD_CACHE:
            aorta_mask = np.load(mask_path)
        else:
            mask_refined = level_set_segmentation(
                img_target,
                detected_circles,
                radius_reduction_factor=0.15,
                num_iter=31,
                balloon=0.8,
                smoothing=2,
            )
            aorta_mask = remove_leaks_morphology(mask_refined, radius=2)
            aorta_mask = keep_largest_component(aorta_mask)
            aorta_mask = aorta_mask.astype(np.uint8)
            save_npy_array(aorta_mask, mask_path)

        # Encontrar óstios
        ostia_left, ostia_right = find_ostia(
            aorta_mask,
            vesselness_i,
            top_n=2000,
            max_z_diff=52,
            lower_fraction=0.80,  # 0.80
            min_center_distance_factor=0.70,
            min_lateral_factor=0.50,
            erosion_radius=4,
        )
        result["ostia_left"] = tuple(map(int, ostia_left))
        result["ostia_right"] = tuple(map(int, ostia_right))

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
        result["both_correct"] = left_info["intersects"] and right_info["intersects"]
        tolerable = 7.0  # mm
        result["both_tolerable"] = (
            left_info["intersects"] or left_info["physical_dist"] <= tolerable
        ) and (right_info["intersects"] or right_info["physical_dist"] <= tolerable)

        # Segmentação das artérias por Region Growing (apenas se ambos corretos/toleráveis)
        if result["both_correct"] or result["both_tolerable"]:
            cache_ii = load_vesselness_cache(
                IMG_ID, cache_dir=f"{base_save_path}/vesselness_cache2"
            )
            if cache_ii is not None and LOAD_CACHE:
                vesselness_ii = cache_ii
            else:
                vesselness_ii = get_vesselness(
                    lcc_image,
                    sigmas=np.arange(1.5, 2.5, 0.5),
                    black_ridges=False,
                    alpha=0.5,  # controla a distinção entre Tubo e Plano
                    beta=0.5,  # controla a distinção entre Tubo e Blob
                    gamma=55,  # sensibilidade ao contraste
                    normalization="none",
                )
                save_vesselness_cache(
                    vesselness_ii,
                    IMG_ID,
                    cache_dir=f"{base_save_path}/vesselness_cache2",
                )

            region_growing_params = {
                "threshold": (vesselness_ii.max() - vesselness_ii.min()) / 5,
                "max_volume": 100000,
                "min_vesselness": vesselness_ii.max() * 0.078,
                "relaxed_floor_factor": 0.97,
                "switch_at_voxels": 2000,
                "comparison_window": 1,
                "smooth_relaxation": True,
                "verbose": False,
            }

            left_mask = region_growing_segmentation(
                vesselness_ii,
                seed_point=ostia_left,  # (y, x, z)
                **region_growing_params,
            )

            right_mask = region_growing_segmentation(
                vesselness_ii,
                seed_point=ostia_right,  # (y, x, z)
                **region_growing_params,
            )

            artery_mask = (left_mask + right_mask).astype(np.uint8)

            # Pós-processamento leve
            closing_radius = 3
            closed_mask = binary_closing(
                artery_mask > 0, structure=ball(closing_radius)
            ).astype(np.uint8)
            dilation_radius = 2
            dilated_mask = binary_dilation(
                closed_mask, structure=ball(dilation_radius)
            ).astype(np.uint8)
            artery_mask = dilated_mask

            result["artery_voxels"] = int(np.sum(artery_mask))

            # Avaliação (Dice)
            label_artery = (label == 1).astype(np.uint8)
            result["dice_artery"] = float(dice_score(artery_mask, label_artery))
    except Exception as e:
        result["error"] = str(e)
    return result


def run_pipeline(ids, split_name):
    results = []
    for img_id in tqdm(ids, desc=f"Processando {split_name}"):
        results.append(process_image(img_id))
    return {"details": results}


if __name__ == "__main__":
    # train_ids = [1, 2, 3, 4, 134, 195, 401, 487, 491, 548, 649]
    train_summary = run_pipeline(train_ids, "train")

    # Gerar DataFrame resumido
    def make_df(results):
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

    df_train = make_df(train_summary["details"])
    output_csv_path = os.path.join(
        os.path.dirname(output_report_path), "ostios_train_summary.csv"
    )
    df_train.to_csv(output_csv_path, index=False)
    print(f"Resumo em CSV salvo em {output_csv_path}")
