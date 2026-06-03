"""Orquestração do pipeline de segmentação coronária."""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..config_utils import scale_config_to_resolution
from ..results_utils import (
    duration_breakdown,
    get_batch_result_file,
    load_batch_timing_records,
    save_batch_timing_record,
    save_results,
    summarize_batch_timing_records,
)
from .pipeline_arteries import segment_arteries_from_ostia
from .pipeline_detection import (
    detect_and_evaluate_ostia,
    get_or_detect_aorta_circles,
    get_or_segment_aorta,
)
from .pipeline_preprocessing import get_or_compute_vesselness, load_and_preprocess_image


logger = logging.getLogger(__name__)


def process_image(img_id, config, base_path, base_save_path):
    """Processa uma imagem completa e retorna o dicionário de resultados."""
    result = {
        "IMG_ID": img_id,
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
        image_data = load_and_preprocess_image(img_id, base_path, config)
        lcc_image = image_data["lcc_image"]
        label = image_data["label"]
        scaled_spacing = image_data["scaled_spacing"]
        downscale_factors = image_data["downscale_factors"]

        image_data = None

        vesselness_ostios = get_or_compute_vesselness(
            img_id,
            lcc_image,
            cache_dir=f"{base_save_path}/vesselness_ostios_cache",
            vesselness_config=config["VESSELNESS_AORTA"],
            load_cache=config["LOAD_CACHE"],
            save_cache=config["SAVE_CACHE"],
        )

        detected_circles = get_or_detect_aorta_circles(
            img_id,
            lcc_image,
            downscale_factors,
            scaled_spacing,
            config["CIRCLE_DETECTION"],
            base_save_path,
            load_cache=config["LOAD_CACHE"],
            save_cache=config["SAVE_CACHE"],
            use_gpu=config.get("USE_GPU", False),
        )

        aorta_mask = get_or_segment_aorta(
            img_id,
            lcc_image,
            detected_circles,
            config["LEVEL_SET"],
            base_save_path,
            load_cache=config["LOAD_CACHE"],
            save_cache=config["SAVE_CACHE"],
            use_gpu=config.get("USE_GPU", False),
        )

        try:
            ostia_eval = detect_and_evaluate_ostia(
                aorta_mask,
                vesselness_ostios,
                label,
                scaled_spacing,
                config,
            )

            del aorta_mask
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

        if not (result["both_correct"] or result["both_tolerable"]):
            result["proceeded_with_bad_ostia"] = True

        result["segmentation_attempted"] = True
        artery_metrics = segment_arteries_from_ostia(
            img_id,
            lcc_image,
            ostia_eval["label_artery"],
            ostia_eval["ostia_left"],
            ostia_eval["ostia_right"],
            config,
            base_save_path,
        )
        artery_metrics.pop("artery_mask", None)
        result.update(artery_metrics)

    except Exception as exc:
        result["error"] = str(exc)

    return result


def run_pipeline(
    ids,
    split_name,
    config,
    base_path,
    base_save_path,
    output_dir=None,
    resume_from_batch=0,
):
    """Processa um conjunto de imagens em lotes e salva cada lote separadamente."""
    start_time = time.time()
    scaled_config = scale_config_to_resolution(config)

    if not ids:
        raise ValueError(f"Nenhuma imagem encontrada para o split '{split_name}'.")

    num_batches = config.get("NUM_BATCHES") or 5
    if num_batches <= 0:
        num_batches = 5
    num_batches = min(num_batches, len(ids))
    all_results = []
    batches_processed = []
    if output_dir is None:
        raise ValueError("output_dir é obrigatório no modo batch")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = max(1, math.ceil(len(ids) / num_batches))
    if resume_from_batch < 0:
        raise ValueError("resume_from_batch não pode ser negativo.")
    if resume_from_batch > num_batches:
        raise ValueError(
            f"resume_from_batch={resume_from_batch} é maior que o total de lotes ({num_batches})."
        )

    start_batch_index = resume_from_batch - 1 if resume_from_batch > 0 else 0

    if resume_from_batch > 0:
        logger.info(f"Retomando a partir do lote {resume_from_batch}...")
        missing_batches = []
        for batch_num in range(0, start_batch_index):
            found_path = get_batch_result_file(
                output_dir, split_name, batch_num + 1
            )

            if found_path:
                df_batch = pd.read_csv(found_path)
                batch_data = df_batch.to_dict("records")
                all_results.extend(batch_data)
                batches_processed.append(batch_num + 1)
                logger.info(
                    f"✓ Lote {batch_num + 1} carregado ({len(batch_data)} registros) (arquivo: {found_path.name})"
                )
            else:
                missing_batches.append(batch_num + 1)

        if missing_batches:
            missing_list = ", ".join(str(batch) for batch in missing_batches)
            raise FileNotFoundError(
                f"Não foi possível retomar o split '{split_name}': faltam os arquivos dos lotes {missing_list}. "
            )

    for batch_num in range(start_batch_index, num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(ids))
        batch_ids = ids[start_idx:end_idx]
        batch_number = batch_num + 1
        batch_started_at = datetime.now().isoformat(timespec="seconds")
        batch_start_time = time.time()

        logger.info(
            f"Processando lote {batch_number}/{num_batches} ({len(batch_ids)} imagens)"
        )
        batch_results = []

        for img_id in tqdm(
            batch_ids, desc=f"Lote {batch_number}/{num_batches}", leave=False
        ):
            batch_results.append(
                process_image(img_id, scaled_config, base_path, base_save_path)
            )

        all_results.extend(batch_results)
        batches_processed.append(batch_number)

        batch_output_path = save_results(
            batch_results,
            f"{split_name}_lote_{batch_number}",
            output_dir,
            config=scaled_config,
        )
        batch_duration = time.time() - batch_start_time
        batch_finished_at = datetime.now().isoformat(timespec="seconds")
        duration = duration_breakdown(batch_duration)
        timing_record = {
            "split_name": split_name,
            "batch_number": batch_number,
            "total_batches": num_batches,
            "num_images": len(batch_ids),
            "first_img_id": batch_ids[0] if batch_ids else None,
            "last_img_id": batch_ids[-1] if batch_ids else None,
            "result_file": Path(batch_output_path).name,
            "started_at": batch_started_at,
            "finished_at": batch_finished_at,
            "duration_seconds": duration["seconds"],
            "duration_minutes": duration["minutes"],
            "duration_hours": duration["hours"],
        }
        manifest_path = save_batch_timing_record(
            output_dir,
            split_name,
            timing_record,
        )
        logger.info(f"Lote {batch_number} salvo: {batch_output_path}")
        logger.info(
            "Tempo do lote %s: %.1fs (%.2fmin, %.3fh). Manifest: %s",
            batch_number,
            duration["seconds"],
            duration["minutes"],
            duration["hours"],
            manifest_path,
        )

    execution_time = time.time() - start_time
    batch_timings = load_batch_timing_records(output_dir, split_name)
    batch_timing_summary = summarize_batch_timing_records(
        batch_timings,
        expected_batches=list(range(1, num_batches + 1)),
    )
    result = {
        "details": all_results,
        "execution_time": execution_time,
        "execution_time_breakdown": duration_breakdown(execution_time),
        "batches_processed": batches_processed,
        "batch_timings": batch_timings,
        "batch_timing_summary": batch_timing_summary,
        "is_batched": True,
    }

    return result
