"""Benchmark do pipeline completo de extração de artéria coronária.

Mede custo computacional por etapa:
- tempo (s)
- pico de memória RAM do processo (MB)
- uso de memória GPU (MB), quando CuPy estiver disponível
"""

import argparse
import json
import os
import resource
import sys
import time

import cv2
import numpy as np
from skimage.morphology import ball

# Adiciona src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils import (
    binary_closing,
    binary_dilation,
    check_ostium_intersection,
    detect_aorta_circles,
    dice_score,
    downscale_image_ndi,
    find_ostia,
    get_vesselness,
    keep_largest_component,
    level_set_segmentation,
    load_raw_img_and_label,
    region_growing_segmentation,
    remove_leaks_morphology,
    run_core_preprocessing_pipeline,
    use_gpu,
    normalize_runtime_config,
    serialize_config_for_json,
    load_config_json,
    save_config_json,
)

try:
    from utils.gpu_utils import cp
except Exception:
    cp = None

def bytes_to_mb(value):
    return float(value) / (1024.0 * 1024.0)


def get_cpu_peak_mb():
    # Linux: ru_maxrss em KB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def get_gpu_memory_mb():
    if not use_gpu() or cp is None:
        return None

    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        used_bytes = total_bytes - free_bytes
        pool_used = cp.get_default_memory_pool().used_bytes()
        return {
            "gpu_used_mb": bytes_to_mb(used_bytes),
            "gpu_pool_mb": bytes_to_mb(pool_used),
        }
    except Exception:
        return None


class StageProfiler:
    def __init__(self):
        self.rows = []
        self.gpu_peak_mb = 0.0
        self.start_t = time.perf_counter()

    def _snapshot(self):
        snap = {
            "cpu_peak_mb": get_cpu_peak_mb(),
        }
        gpu = get_gpu_memory_mb()
        if gpu is not None:
            snap.update(gpu)
            self.gpu_peak_mb = max(self.gpu_peak_mb, gpu["gpu_used_mb"])
        else:
            snap["gpu_used_mb"] = None
            snap["gpu_pool_mb"] = None
        return snap

    def run(self, name, fn, *args, **kwargs):
        before = self._snapshot()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        after = self._snapshot()

        row = {
            "stage": name,
            "time_s": elapsed,
            "cpu_peak_mb": after["cpu_peak_mb"],
            "cpu_peak_delta_mb": max(0.0, after["cpu_peak_mb"] - before["cpu_peak_mb"]),
            "gpu_used_mb": after["gpu_used_mb"],
            "gpu_used_delta_mb": None
            if after["gpu_used_mb"] is None or before["gpu_used_mb"] is None
            else after["gpu_used_mb"] - before["gpu_used_mb"],
            "gpu_pool_mb": after["gpu_pool_mb"],
            "gpu_pool_delta_mb": None
            if after["gpu_pool_mb"] is None or before["gpu_pool_mb"] is None
            else after["gpu_pool_mb"] - before["gpu_pool_mb"],
        }
        self.rows.append(row)
        return out

    def total_time(self):
        return time.perf_counter() - self.start_t


def print_stage_table(rows):
    print("\n" + "=" * 110)
    print("CUSTO COMPUTACIONAL POR ETAPA")
    print("=" * 110)
    print(
        f"{'Etapa':32s} {'Tempo (s)':>10s} {'CPU pico (MB)':>14s} {'dCPU (MB)':>10s} "
        f"{'GPU uso (MB)':>13s} {'dGPU (MB)':>10s} {'GPU pool (MB)':>14s}"
    )
    print("-" * 110)

    for row in rows:
        gpu_used = "-" if row["gpu_used_mb"] is None else f"{row['gpu_used_mb']:.1f}"
        gpu_delta = "-" if row["gpu_used_delta_mb"] is None else f"{row['gpu_used_delta_mb']:+.1f}"
        gpu_pool = "-" if row["gpu_pool_mb"] is None else f"{row['gpu_pool_mb']:.1f}"
        print(
            f"{row['stage'][:32]:32s} {row['time_s']:10.3f} {row['cpu_peak_mb']:14.1f} "
            f"{row['cpu_peak_delta_mb']:10.1f} {gpu_used:13s} {gpu_delta:10s} {gpu_pool:14s}"
        )
    print("=" * 110)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark do pipeline completo de extração de artéria")
    parser.add_argument("--img-id", type=int, default=1, help="ID da imagem do dataset")
    parser.add_argument(
        "--base-path",
        type=str,
        default="/data04/home/mpmaia/ImageCAS/database/1-1000",
        help="Caminho do dataset ImageCAS",
    )
    parser.add_argument("--downscale-factors", nargs=3, type=float, default=None)
    parser.add_argument("--downscale-method", choices=["scipy", "opencv"], default=None)
    parser.add_argument(
        "--opencv-interpolation",
        choices=["nearest", "linear", "cubic", "area", "lanczos4"],
        default=None,
    )
    parser.add_argument("--max-threshold-percentile", type=float, default=None)
    parser.add_argument("--config-file", type=str, default=None, help="Arquivo JSON de configuração")
    parser.add_argument("--save-config", type=str, default=None, help="Salva configuração efetiva em JSON")
    parser.add_argument(
        "--strict-ostia-gate",
        action="store_true",
        help="Se ligado, só segmenta artéria quando ambos os óstios forem corretos/toleráveis",
    )
    parser.add_argument("--save-json", type=str, default=None, help="Salva relatório em JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    # Configuração de parâmetros (alinhada ao pipeline principal)
    config = {
        "DOWNSCALE_FACTORS": (2, 2, 1),
        "DOWNSCALE_METHOD": "scipy",
        "OPENCV_INTERPOLATION": "area",
        "MAX_THRESHOLD_PERCENTILE": 99.7,
        "TOLERABLE_DISTANCE_MM": 7.0,
        "VESSELNESS_AORTA": {
            "sigmas": np.arange(2.5, 3.5, 1),
            "alpha": 0.5,
            "beta": 1,
            "gamma": 30,
        },
        "VESSELNESS_ARTERY": {
            "sigmas": np.arange(1.5, 3.5, 0.5),
            "alpha": 0.5,
            "beta": 0.5,
            "gamma": 55,
        },
        "CIRCLE_DETECTION": {
            "radii_start_px": 38,
            "radii_end_px": 62,
            "radius_step_px": 1,
            "tol_radius_mm": 9.0,
            "tol_distance_mm": 20.0,
            "max_slice_miss_threshold": 5,
            "neighbor_distance_threshold": 5,
            "quadrant_offset": (30, 30),
            "total_num_peaks_initial": 15,
            "total_num_peaks": 15,
            "canny_sigma": 3,
            "use_local_roi": True,
            "local_roi_padding": 20,
        },
        "LEVEL_SET": {
            "radius_reduction_factor": 0.15,
            "num_iter": 35,
            "balloon": 0.8,
            "smoothing": 2,
            "leak_removal_radius": 2,
        },
        "OSTIA_DETECTION": {
            "top_n": 2000,
            "max_z_diff_mm": 40.0,
            "lower_fraction": 0.80,
            "min_center_distance_factor": 0.85,
            "min_lateral_factor": 0.4,
            "erosion_radius": 4,
        },
        "REGION_GROWING": {
            "threshold_divisor": 5,
            "max_volume": 100000,
            "min_vesselness_fraction": 0.078,
            "relaxed_floor_factor": 0.97,
            "switch_at_voxels": 2000,
            "comparison_window": 1,
            "smooth_relaxation": True,
        },
        "POSTPROCESSING": {
            "closing_radius": 3,
            "dilation_radius": 2,
        },
    }

    if args.config_file:
        config = load_config_json(args.config_file, config)

    if args.downscale_factors is not None:
        config["DOWNSCALE_FACTORS"] = tuple(args.downscale_factors)
    if args.downscale_method is not None:
        config["DOWNSCALE_METHOD"] = args.downscale_method
    if args.opencv_interpolation is not None:
        config["OPENCV_INTERPOLATION"] = args.opencv_interpolation
    if args.max_threshold_percentile is not None:
        config["MAX_THRESHOLD_PERCENTILE"] = args.max_threshold_percentile

    config = normalize_runtime_config(config)

    if args.save_config:
        save_config_json(config, args.save_config)
        print(f"Configuração salva em: {args.save_config}")

    print("=" * 70)
    print("BENCHMARK - PIPELINE COMPLETO DE EXTRAÇÃO DE ARTÉRIA")
    print("=" * 70)
    print(f"Imagem: {args.img_id}")
    print(f"GPU disponível: {use_gpu()}")
    print(f"Downscale factors: {config['DOWNSCALE_FACTORS']}")
    print(f"Downscale method: {config['DOWNSCALE_METHOD']}")

    profiler = StageProfiler()

    # 1) Carregamento
    nii_img, nii_label = profiler.run(
        "load_raw_img_and_label",
        load_raw_img_and_label,
        f"{args.base_path}/{args.img_id}.img.nii.gz",
        f"{args.base_path}/{args.img_id}.label.nii.gz",
    )

    img = np.array(nii_img.get_fdata())
    label = np.array(nii_label.get_fdata()).astype(np.uint8)
    spacing = nii_img.header.get_zooms()

    print(f"Shape original: {img.shape}")
    print(f"Spacing original: {spacing[:3]}")
    print(f"RAM do volume bruto: {img.nbytes / (1024 * 1024):.1f} MB")

    # 2) Pré-processamento
    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos4": cv2.INTER_LANCZOS4,
    }
    opencv_interpolation = interpolation_map[config["OPENCV_INTERPOLATION"]]

    down_image, thresh_image, lcc_image, thresh_vals = profiler.run(
        "run_core_preprocessing_pipeline",
        run_core_preprocessing_pipeline,
        img,
        downscale_factors=config["DOWNSCALE_FACTORS"],
        lcc_per_slice=True,
        max_threshold_percentile=config["MAX_THRESHOLD_PERCENTILE"],
        use_opencv=config["DOWNSCALE_METHOD"] == "opencv",
        opencv_interpolation=opencv_interpolation,
    )

    label_ds = profiler.run(
        "downscale_label",
        downscale_image_ndi,
        label,
        config["DOWNSCALE_FACTORS"],
        0,
    )

    dx, dy, dz = (
        spacing[0] * config["DOWNSCALE_FACTORS"][0],
        spacing[1] * config["DOWNSCALE_FACTORS"][1],
        spacing[2] * config["DOWNSCALE_FACTORS"][2],
    )

    # 3) Vesselness para óstios
    vesselness_ostios = profiler.run(
        "get_vesselness_ostia",
        get_vesselness,
        lcc_image,
        sigmas=config["VESSELNESS_AORTA"]["sigmas"],
        black_ridges=False,
        alpha=config["VESSELNESS_AORTA"]["alpha"],
        beta=config["VESSELNESS_AORTA"]["beta"],
        gamma=config["VESSELNESS_AORTA"]["gamma"],
        normalization="none",
    )

    # 4) Detecção de círculos da aorta
    circle_config = config["CIRCLE_DETECTION"]
    hough_radii = np.arange(
        circle_config["radii_start_px"] / config["DOWNSCALE_FACTORS"][0],
        circle_config["radii_end_px"] / config["DOWNSCALE_FACTORS"][0],
        circle_config.get("radius_step_px", 1) / config["DOWNSCALE_FACTORS"][0],
    )
    pixel_spacing = (dx + dy) / 2.0

    detected_circles = profiler.run(
        "detect_aorta_circles",
        detect_aorta_circles,
        lcc_image,
        hough_radii,
        pixel_spacing,
        tol_radius_mm=circle_config["tol_radius_mm"],
        tol_distance_mm=circle_config["tol_distance_mm"],
        max_slice_miss_threshold=circle_config["max_slice_miss_threshold"],
        neighbor_distance_threshold=circle_config["neighbor_distance_threshold"],
        quadrant_offset=tuple(circle_config["quadrant_offset"]),
        total_num_peaks_initial=circle_config["total_num_peaks_initial"],
        total_num_peaks=circle_config["total_num_peaks"],
        canny_sigma=circle_config["canny_sigma"],
        use_local_roi=circle_config.get("use_local_roi", True),
        local_roi_padding=circle_config.get("local_roi_padding", 20),
    )

    if len(detected_circles) == 0:
        raise RuntimeError("Nenhum círculo da aorta detectado. Não foi possível continuar o pipeline.")

    # 5) Segmentação da aorta
    ls_config = config["LEVEL_SET"]
    mask_refined = profiler.run(
        "level_set_segmentation",
        level_set_segmentation,
        lcc_image,
        detected_circles,
        radius_reduction_factor=ls_config["radius_reduction_factor"],
        num_iter=ls_config["num_iter"],
        balloon=ls_config["balloon"],
        smoothing=ls_config["smoothing"],
    )

    aorta_mask = profiler.run(
        "remove_leaks_morphology",
        remove_leaks_morphology,
        mask_refined,
        ls_config["leak_removal_radius"],
    )

    aorta_mask = profiler.run("keep_largest_component", keep_largest_component, aorta_mask)
    aorta_mask = aorta_mask.astype(np.uint8)

    # 6) Detecção de óstios
    ostia_config = config["OSTIA_DETECTION"]
    ostia_left, ostia_right = profiler.run(
        "find_ostia",
        find_ostia,
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

    # 7) Checagem de interseção dos óstios
    label_artery = (label_ds == 1).astype(np.uint8)
    left_info = profiler.run(
        "check_ostium_left",
        check_ostium_intersection,
        ostia_left,
        label_artery,
        spacing=(dy, dx, dz),
        ostium_name="ostio esquerdo",
    )
    right_info = profiler.run(
        "check_ostium_right",
        check_ostium_intersection,
        ostia_right,
        label_artery,
        spacing=(dy, dx, dz),
        ostium_name="ostio direito",
    )

    tolerable = config["TOLERABLE_DISTANCE_MM"]
    both_correct = left_info["intersects"] and right_info["intersects"]
    both_tolerable = (
        (left_info["intersects"] or left_info["physical_dist"] <= tolerable)
        and (right_info["intersects"] or right_info["physical_dist"] <= tolerable)
    )

    should_run_artery = True
    if args.strict_ostia_gate:
        should_run_artery = both_correct or both_tolerable

    artery_mask = None
    dice = None

    if should_run_artery:
        # 8) Vesselness para artéria
        vesselness_artery = profiler.run(
            "get_vesselness_artery",
            get_vesselness,
            lcc_image,
            sigmas=config["VESSELNESS_ARTERY"]["sigmas"],
            black_ridges=False,
            alpha=config["VESSELNESS_ARTERY"]["alpha"],
            beta=config["VESSELNESS_ARTERY"]["beta"],
            gamma=config["VESSELNESS_ARTERY"]["gamma"],
            normalization="none",
        )

        # 9) Region growing (esquerda e direita)
        rg = config["REGION_GROWING"]
        rg_params = {
            "threshold": (vesselness_artery.max() - vesselness_artery.min()) / rg["threshold_divisor"],
            "max_volume": rg["max_volume"],
            "min_vesselness": vesselness_artery.max() * rg["min_vesselness_fraction"],
            "relaxed_floor_factor": rg["relaxed_floor_factor"],
            "switch_at_voxels": rg["switch_at_voxels"],
            "comparison_window": rg["comparison_window"],
            "smooth_relaxation": rg["smooth_relaxation"],
            "verbose": False,
        }

        left_mask = profiler.run(
            "region_growing_left",
            region_growing_segmentation,
            vesselness_artery,
            ostia_left,
            **rg_params,
        )
        right_mask = profiler.run(
            "region_growing_right",
            region_growing_segmentation,
            vesselness_artery,
            ostia_right,
            **rg_params,
        )

        artery_mask = profiler.run("merge_artery_masks", lambda a, b: (a + b).astype(np.uint8), left_mask, right_mask)

        # 10) Pós-processamento morfológico
        post = config["POSTPROCESSING"]
        closed_mask = profiler.run(
            "binary_closing",
            binary_closing,
            artery_mask > 0,
            ball(post["closing_radius"]),
        )
        artery_mask = profiler.run(
            "binary_dilation",
            binary_dilation,
            closed_mask,
            ball(post["dilation_radius"]),
        )

        # 11) Métrica final
        dice = profiler.run("dice_score", dice_score, artery_mask, label_artery)

    # Resumo
    print_stage_table(profiler.rows)

    total_time = profiler.total_time()
    print("\nResumo final")
    print("-" * 70)
    print(f"Tempo total: {total_time:.2f}s")
    print(f"Pico RAM processo: {get_cpu_peak_mb():.1f} MB")
    if use_gpu() and cp is not None:
        gpu_now = get_gpu_memory_mb()
        if gpu_now is not None:
            print(f"Uso GPU atual: {gpu_now['gpu_used_mb']:.1f} MB")
            print(f"Pool CuPy atual: {gpu_now['gpu_pool_mb']:.1f} MB")
        print(f"Pico GPU observado (aprox): {profiler.gpu_peak_mb:.1f} MB")

    print(f"Círculos detectados: {len(detected_circles)}")
    print(f"Óstio esquerdo: {tuple(map(int, ostia_left))}")
    print(f"Óstio direito: {tuple(map(int, ostia_right))}")
    print(f"Óstios corretos (estrito): {both_correct}")
    print(f"Óstios toleráveis (<= {tolerable} mm): {both_tolerable}")

    if artery_mask is not None:
        print(f"Voxels de artéria: {int(np.sum(artery_mask))}")
        print(f"Dice artéria: {float(dice):.5f}")
    else:
        print("Segmentação de artéria não executada (strict-ostia-gate ativo e óstios fora do critério).")

    if args.save_json:
        report = {
            "img_id": args.img_id,
            "config": serialize_config_for_json(config),
            "gpu_available": use_gpu(),
            "stages": profiler.rows,
            "total_time_s": total_time,
            "cpu_peak_mb": get_cpu_peak_mb(),
            "gpu_peak_mb": profiler.gpu_peak_mb,
            "outputs": {
                "num_detected_circles": len(detected_circles),
                "ostia_left": tuple(map(int, ostia_left)),
                "ostia_right": tuple(map(int, ostia_right)),
                "both_correct": bool(both_correct),
                "both_tolerable": bool(both_tolerable),
                "artery_voxels": None if artery_mask is None else int(np.sum(artery_mask)),
                "dice_artery": None if dice is None else float(dice),
            },
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nRelatório salvo em: {args.save_json}")


if __name__ == "__main__":
    main()
