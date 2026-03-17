"""Etapas reutilizáveis do pipeline de segmentação coronária."""

import json
import os

import cv2
import numpy as np
from skimage.morphology import ball

from .aorta_localization import detect_aorta_circles
from .aorta_segmentation import level_set_segmentation, remove_leaks_morphology
from .artery_segmentation import region_growing_segmentation
from .binary_operations import binary_closing, binary_dilation, keep_largest_component
from .frangi import get_vesselness, load_vesselness_cache, save_vesselness_cache
from .ostia_detection import check_ostium_intersection, find_ostia
from .preprocessing import downscale_image_ndi, run_core_preprocessing_pipeline
from .utils import dice_score, load_raw_img_and_label, save_npy_array


def load_and_preprocess_image(img_id, base_path, config):
    """Carrega imagem/label e executa o pré-processamento básico."""
    nii_img, nii_label = load_raw_img_and_label(
        f"{base_path}/{img_id}.img.nii.gz", f"{base_path}/{img_id}.label.nii.gz"
    )
    spacing = nii_img.header.get_zooms()
    img = np.array(nii_img.get_fdata())
    label = np.array(nii_label.get_fdata()).astype(np.uint8)

    downscale_factors = config["DOWNSCALE_FACTORS"]
    use_opencv = config["DOWNSCALE_METHOD"] == "opencv"
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

    down_image, thresh_image, lcc_image, thresh_vals = run_core_preprocessing_pipeline(
        img,
        downscale_factors=downscale_factors,
        lcc_per_slice=True,
        max_threshold_percentile=config["MAX_THRESHOLD_PERCENTILE"],
        use_opencv=use_opencv,
        opencv_interpolation=opencv_interpolation,
    )
    label = downscale_image_ndi(label, downscale_factors, order=0)

    dx, dy, dz = (
        spacing[0] * downscale_factors[0],
        spacing[1] * downscale_factors[1],
        spacing[2] * downscale_factors[2],
    )

    return {
        "nii_img": nii_img,
        "nii_label": nii_label,
        "img": img,
        "down_image": down_image,
        "thresh_image": thresh_image,
        "thresh_vals": thresh_vals,
        "lcc_image": lcc_image,
        "label": label,
        "spacing": spacing,
        "scaled_spacing": (dx, dy, dz),
        "downscale_factors": downscale_factors,
    }


def get_or_compute_vesselness(
    img_id,
    image,
    cache_dir,
    vesselness_config,
    load_cache=False,
    save_cache=False,
):
    """Carrega ou calcula vesselness para um volume 3D."""
    cache = load_vesselness_cache(img_id, cache_dir=cache_dir)
    if cache is not None and load_cache:
        return cache

    vesselness = get_vesselness(
        image,
        sigmas=vesselness_config["sigmas"],
        black_ridges=False,
        alpha=vesselness_config["alpha"],
        beta=vesselness_config["beta"],
        gamma=vesselness_config["gamma"],
        normalization="none",
    )
    if save_cache:
        save_vesselness_cache(vesselness, img_id, cache_dir=cache_dir)
    return vesselness


def get_or_detect_aorta_circles(
    img_id,
    lcc_image,
    downscale_factors,
    scaled_spacing,
    circle_config,
    base_save_path,
    load_cache=False,
    save_cache=False,
):
    """Carrega ou detecta círculos da aorta."""
    saved_dir_circles = f"{base_save_path}/detected_circles"
    json_path = os.path.join(saved_dir_circles, f"{img_id}_detected_circles.json")

    if os.path.exists(json_path) and load_cache:
        with open(json_path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)

    dx, dy, _ = scaled_spacing
    radii_start = circle_config["radii_start_px"] / downscale_factors[0]
    radii_end = circle_config["radii_end_px"] / downscale_factors[0]
    radius_step = circle_config.get("radius_step_px", 1) / downscale_factors[0]
    hough_radii = np.arange(radii_start, radii_end, radius_step)
    pixel_spacing = (dx + dy) / 2.0

    detected_circles = detect_aorta_circles(
        lcc_image,
        hough_radii,
        pixel_spacing,
        tol_radius_mm=circle_config["tol_radius_mm"],
        tol_distance_mm=circle_config["tol_distance_mm"],
        quadrant_offset=tuple(circle_config["quadrant_offset"]),
        max_slice_miss_threshold=circle_config["max_slice_miss_threshold"],
        neighbor_distance_threshold=circle_config["neighbor_distance_threshold"],
        total_num_peaks_initial=circle_config["total_num_peaks_initial"],
        total_num_peaks=circle_config["total_num_peaks"],
        canny_sigma=circle_config["canny_sigma"],
        use_local_roi=circle_config.get("use_local_roi", True),
        local_roi_padding=circle_config.get("local_roi_padding", 20),
    )
    if save_cache:
        os.makedirs(saved_dir_circles, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as file_handle:
            json.dump(detected_circles, file_handle, indent=4)

    return detected_circles


def get_or_segment_aorta(
    img_id,
    lcc_image,
    detected_circles,
    level_set_config,
    base_save_path,
    load_cache=False,
    save_cache=False,
):
    """Carrega ou segmenta a aorta com level set + pós-processamento."""
    saved_dir_aorta = f"{base_save_path}/segmented_aorta"
    mask_path = os.path.join(saved_dir_aorta, f"{img_id}_mask_aorta.npy")

    if os.path.exists(mask_path) and load_cache:
        return np.load(mask_path)

    mask_refined = level_set_segmentation(
        lcc_image,
        detected_circles,
        radius_reduction_factor=level_set_config["radius_reduction_factor"],
        num_iter=level_set_config["num_iter"],
        balloon=level_set_config["balloon"],
        smoothing=level_set_config["smoothing"],
    )
    aorta_mask = remove_leaks_morphology(
        mask_refined, radius=level_set_config["leak_removal_radius"]
    )
    aorta_mask = keep_largest_component(aorta_mask)
    aorta_mask = aorta_mask.astype(np.uint8)

    if save_cache:
        os.makedirs(saved_dir_aorta, exist_ok=True)
        save_npy_array(aorta_mask, mask_path)

    return aorta_mask


def detect_and_evaluate_ostia(aorta_mask, vesselness_ostios, label, scaled_spacing, config):
    """Detecta os óstios e avalia correção/tolerância."""
    dx, dy, dz = scaled_spacing
    ostia_config = config["OSTIA_DETECTION"]
    ostia_left, ostia_right = find_ostia(
        aorta_mask,
        vesselness_ostios,
        spacing=(dy, dx, dz),
        top_n=ostia_config["top_n"],
        max_z_diff_mm=ostia_config["max_z_diff_mm"],
        lower_fraction=ostia_config["lower_fraction"],
        min_center_distance_factor=ostia_config["min_center_distance_factor"],
        min_lateral_factor=ostia_config["min_lateral_factor"],
        erosion_radius=ostia_config["erosion_radius"],
    )

    label_artery = (label == 1).astype(np.uint8)
    left_info = check_ostium_intersection(
        ostia_left, label_artery, spacing=(dy, dx, dz), ostium_name="Óstio esquerdo"
    )
    right_info = check_ostium_intersection(
        ostia_right, label_artery, spacing=(dy, dx, dz), ostium_name="Óstio direito"
    )

    tolerable = config["OSTIA_VALIDATION"]["distance_threshold_mm"]
    both_correct = left_info["intersects"] and right_info["intersects"]
    both_tolerable_inclusive = (
        left_info["intersects"] or left_info["physical_dist"] <= tolerable
    ) and (right_info["intersects"] or right_info["physical_dist"] <= tolerable)

    return {
        "ostia_left": ostia_left,
        "ostia_right": ostia_right,
        "label_artery": label_artery,
        "left_info": left_info,
        "right_info": right_info,
        "both_correct": both_correct,
        "both_tolerable": both_tolerable_inclusive and (not both_correct),
    }


def segment_arteries_from_ostia(
    img_id,
    lcc_image,
    label_artery,
    ostia_left,
    ostia_right,
    config,
    base_save_path,
):
    """Calcula vesselness arterial, executa region growing e avalia Dice."""
    vesselness_artery = get_or_compute_vesselness(
        img_id,
        lcc_image,
        cache_dir=f"{base_save_path}/vesselness_artery_cache",
        vesselness_config=config["VESSELNESS_ARTERY"],
        load_cache=config["LOAD_CACHE"],
        save_cache=config["SAVE_CACHE"],
    )

    rg_config = config["REGION_GROWING"]
    region_growing_params = {
        "threshold": (vesselness_artery.max() - vesselness_artery.min())
        / rg_config["threshold_divisor"],
        "max_volume": rg_config["max_volume"],
        "min_vesselness": vesselness_artery.max() * rg_config["min_vesselness_fraction"],
        "relaxed_floor_factor": rg_config["relaxed_floor_factor"],
        "switch_at_voxels": rg_config["switch_at_voxels"],
        "comparison_window": rg_config["comparison_window"],
        "smooth_relaxation": rg_config["smooth_relaxation"],
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
    post_config = config["POSTPROCESSING"]
    closed_mask = binary_closing(
        artery_mask > 0, structure=ball(post_config["closing_radius"])
    )
    dilated_mask = binary_dilation(
        closed_mask, structure=ball(post_config["dilation_radius"])
    )
    artery_mask = dilated_mask

    return {
        "artery_voxels": int(np.sum(artery_mask)),
        "dice_artery": float(dice_score(artery_mask, label_artery)),
    }