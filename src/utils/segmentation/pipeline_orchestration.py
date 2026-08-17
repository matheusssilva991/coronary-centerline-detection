"""Orquestração do pipeline de segmentação coronária."""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..project.results import (
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
    locate_aorta_circles,
    segment_aorta,
)
from .pipeline_preprocessing import compute_vesselness, load_and_preprocess_image

logger = logging.getLogger(__name__)


IMAGE_RESULT_DEFAULTS = {
    "ostia_left": None,
    "ostia_right": None,
    "artery_voxels": None,
    "artery_voxels_before_morphology": None,
    "artery_voxels_after_morphology": None,
    "dice_artery": None,
    "dice_artery_before_morphology": None,
    "dice_artery_after_morphology": None,
    "dice_artery_morphology_delta": None,
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
    "threshold_mode": None,
    "fuzzy_mask_strategy": None,
    "min_threshold": None,
    "max_threshold": None,
    "lower_threshold_method": None,
    "lower_threshold_percentile": None,
    "threshold_voxels": None,
    "lcc_voxels": None,
    "image_slice_count": None,
    "aorta_circle_count": None,
    "aorta_detected_circle_count": None,
    "aorta_interpolated_circle_count": None,
    "aorta_circle_first_slice": None,
    "aorta_circle_last_slice": None,
    "aorta_circle_coverage": None,
    "aorta_recovered_initialization": False,
    "aorta_mask_voxels": None,
}


def _new_image_result(img_id):
    """Cria o registro completo com os defaults de uma imagem."""
    return {"IMG_ID": img_id, **IMAGE_RESULT_DEFAULTS}


def _preprocessing_result_fields(preprocessing_details, image_slice_count):
    """Converte detalhes do pré-processamento nos campos persistidos."""
    detail_keys = (
        "threshold_mode",
        "fuzzy_mask_strategy",
        "min_threshold",
        "max_threshold",
        "lower_threshold_method",
        "lower_threshold_percentile",
        "threshold_voxels",
        "lcc_voxels",
    )
    fields = {key: preprocessing_details.get(key) for key in detail_keys}
    fields["image_slice_count"] = int(image_slice_count)
    return fields


def summarize_aorta_circles(detected_circles, image_slice_count):
    """Resume cobertura, interpolação e recuperação do rastreamento da aorta."""
    circle_slices = [
        int(circle["slice_index"])
        for circle in detected_circles
        if circle.get("slice_index") is not None
    ]
    interpolated_count = sum(
        bool(circle.get("interpolated", False)) for circle in detected_circles
    )
    circle_count = len(detected_circles)
    return {
        "aorta_circle_count": circle_count,
        "aorta_detected_circle_count": circle_count - interpolated_count,
        "aorta_interpolated_circle_count": interpolated_count,
        "aorta_circle_first_slice": min(circle_slices) if circle_slices else None,
        "aorta_circle_last_slice": max(circle_slices) if circle_slices else None,
        "aorta_circle_coverage": (
            circle_count / image_slice_count if image_slice_count else None
        ),
        "aorta_recovered_initialization": any(
            bool(circle.get("recovered_initialization", False))
            for circle in detected_circles
        ),
    }


def _circle_result_fields(detected_circles, image_slice_count):
    """Compatibilidade interna para o antigo nome do resumo de círculos."""
    return summarize_aorta_circles(detected_circles, image_slice_count)


def _ostia_result_fields(ostia_eval):
    """Converte a avaliação dos óstios nos campos persistidos."""
    both_correct = bool(ostia_eval["both_correct"])
    both_tolerable = bool(ostia_eval["both_tolerable"])
    if both_correct:
        status = "both_correct"
    elif both_tolerable:
        status = "both_tolerable"
    else:
        status = "found_but_wrong"

    return {
        "ostia_left": (
            tuple(map(int, ostia_eval["ostia_left"]))
            if ostia_eval["ostia_left"] is not None
            else None
        ),
        "ostia_right": (
            tuple(map(int, ostia_eval["ostia_right"]))
            if ostia_eval["ostia_right"] is not None
            else None
        ),
        "ostia_found": True,
        "left_intersects": ostia_eval["left_info"]["intersects"],
        "right_intersects": ostia_eval["right_info"]["intersects"],
        "left_dist_voxels": ostia_eval["left_info"]["euclidean_dist"],
        "right_dist_voxels": ostia_eval["right_info"]["euclidean_dist"],
        "left_dist_mm": ostia_eval["left_info"]["physical_dist"],
        "right_dist_mm": ostia_eval["right_info"]["physical_dist"],
        "both_correct": both_correct,
        "both_tolerable": both_tolerable,
        "ostia_status": status,
        "proceeded_with_bad_ostia": not (both_correct or both_tolerable),
    }


def process_image(img_id, config, base_path):
    """Processa uma imagem completa e retorna o dicionário de resultados.

    Fluxo por imagem:
    1. carrega e pré-processa o volume;
    2. calcula vesselness para detecção dos óstios;
    3. detecta círculos e segmenta a aorta;
    4. seleciona/avalia os óstios;
    5. segmenta as artérias a partir dos óstios.
    """
    result = _new_image_result(img_id)

    try:
        # Carrega imagem/label e gera o volume pré-processado (LCC).
        image_data = load_and_preprocess_image(img_id, base_path, config)
        lcc_image = image_data["lcc_image"]
        label = image_data["label"]
        scaled_spacing = image_data["scaled_spacing"]
        preprocessing_details = image_data.get("preprocessing_details", {})
        downscale_factors = image_data["downscale_factors"]

        image_data = None
        result.update(
            _preprocessing_result_fields(
                preprocessing_details,
                lcc_image.shape[2],
            )
        )

        # Calcula o mapa de vasos usado para selecionar candidatos de óstios.
        vesselness_ostios = compute_vesselness(
            lcc_image,
            vesselness_config=config["VESSELNESS_AORTA"],
            use_gpu=config.get("USE_GPU", False),
        )

        # Localiza a aorta por círculos em fatias consecutivas.
        detected_circles = locate_aorta_circles(
            lcc_image,
            downscale_factors,
            scaled_spacing,
            config["CIRCLE_DETECTION"],
        )
        result.update(
            summarize_aorta_circles(
                detected_circles,
                result["image_slice_count"],
            )
        )

        # Segmenta a aorta a partir dos círculos detectados.
        aorta_mask = segment_aorta(
            lcc_image,
            detected_circles,
            config["LEVEL_SET"],
            use_gpu=config.get("USE_GPU", False),
        )
        result["aorta_mask_voxels"] = int(aorta_mask.sum())

        try:
            # Seleciona os óstios e valida contra o label arterial.
            ostia_eval = detect_and_evaluate_ostia(
                aorta_mask,
                vesselness_ostios,
                label,
                scaled_spacing,
                config,
                detected_circles=detected_circles,
            )

            del aorta_mask
        except ValueError as ostia_exc:
            result["ostia_status"] = "not_found"
            result["ostia_error"] = str(ostia_exc)
            result["skip_reason"] = "ostia_not_found"
            result["dice_artery"] = 0.0
            return result

        result.update(_ostia_result_fields(ostia_eval))

        # Segmenta as artérias mesmo quando os óstios são apenas toleráveis.
        result["segmentation_attempted"] = True
        artery_metrics = segment_arteries_from_ostia(
            lcc_image,
            ostia_eval["label_artery"],
            ostia_eval["ostia_left"],
            ostia_eval["ostia_right"],
            config,
        )
        artery_metrics.pop("artery_mask", None)
        artery_metrics.pop("raw_artery_mask", None)
        result.update(artery_metrics)

    except Exception as exc:
        result["error"] = str(exc)

    return result


def _resolve_batch_plan(ids, config, resume_from_batch):
    """Calcula quantidade, tamanho e índice inicial dos lotes."""
    num_batches = config.get("NUM_BATCHES") or 5
    if num_batches <= 0:
        num_batches = 5
    num_batches = min(num_batches, len(ids))

    if resume_from_batch < 0:
        raise ValueError("resume_from_batch não pode ser negativo.")
    if resume_from_batch > num_batches:
        raise ValueError(
            f"resume_from_batch={resume_from_batch} é maior que o total de "
            f"lotes ({num_batches})."
        )

    batch_size = max(1, math.ceil(len(ids) / num_batches))
    start_batch_index = resume_from_batch - 1 if resume_from_batch > 0 else 0
    return num_batches, batch_size, start_batch_index


def _load_previous_batches(output_dir, split_name, start_batch_index):
    """Carrega lotes anteriores necessários para uma retomada consistente."""
    all_results = []
    batches_processed = []
    missing_batches = []

    # Só lotes anteriores ao ponto de retomada entram como resultados preservados.
    for batch_index in range(start_batch_index):
        batch_number = batch_index + 1
        found_path = get_batch_result_file(output_dir, split_name, batch_number)
        if found_path is None:
            missing_batches.append(batch_number)
            continue

        batch_data = pd.read_csv(found_path).to_dict("records")
        all_results.extend(batch_data)
        batches_processed.append(batch_number)
        logger.info(
            "✓ Lote %s carregado (%s registros) (arquivo: %s)",
            batch_number,
            len(batch_data),
            found_path.name,
        )

    # Impede consolidar uma execução com uma lacuna silenciosa entre os lotes.
    if missing_batches:
        missing_list = ", ".join(str(batch) for batch in missing_batches)
        raise FileNotFoundError(
            f"Não foi possível retomar o split '{split_name}': "
            f"faltam os arquivos dos lotes {missing_list}. "
        )
    return all_results, batches_processed


def _process_and_save_batch(
    batch_ids,
    batch_number,
    num_batches,
    split_name,
    config,
    base_path,
    output_dir,
):
    """Processa um lote e persiste resultados e duração imediatamente."""
    batch_started_at = datetime.now().isoformat(timespec="seconds")
    batch_start_time = time.time()
    logger.info(
        "Processando lote %s/%s (%s imagens)",
        batch_number,
        num_batches,
        len(batch_ids),
    )

    # Processa todas as imagens antes de persistir o lote de forma atômica.
    batch_results = [
        process_image(img_id, config, base_path)
        for img_id in tqdm(
            batch_ids,
            desc=f"Lote {batch_number}/{num_batches}",
            leave=False,
        )
    ]
    batch_output_path = save_results(
        batch_results,
        f"{split_name}_lote_{batch_number}",
        output_dir,
        config=config,
    )

    # O manifest separado permite recompor o tempo após queda ou retomada.
    duration = duration_breakdown(time.time() - batch_start_time)
    timing_record = {
        "split_name": split_name,
        "batch_number": batch_number,
        "total_batches": num_batches,
        "num_images": len(batch_ids),
        "first_img_id": batch_ids[0] if batch_ids else None,
        "last_img_id": batch_ids[-1] if batch_ids else None,
        "result_file": Path(batch_output_path).name,
        "started_at": batch_started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": duration["seconds"],
        "duration_minutes": duration["minutes"],
        "duration_hours": duration["hours"],
    }
    manifest_path = save_batch_timing_record(output_dir, split_name, timing_record)
    logger.info("Lote %s salvo: %s", batch_number, batch_output_path)
    logger.info(
        "Tempo do lote %s: %.1fs (%.2fmin, %.3fh). Manifest: %s",
        batch_number,
        duration["seconds"],
        duration["minutes"],
        duration["hours"],
        manifest_path,
    )
    return batch_results


def run_pipeline(
    ids,
    split_name,
    config,
    base_path,
    output_dir=None,
    resume_from_batch=0,
):
    """Processa imagens em lotes com uma config runtime já escalada para a resolução."""
    start_time = time.time()

    if not ids:
        raise ValueError(f"Nenhuma imagem encontrada para o split '{split_name}'.")

    if output_dir is None:
        raise ValueError("output_dir é obrigatório no modo batch")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_batches, batch_size, start_batch_index = _resolve_batch_plan(
        ids,
        config,
        resume_from_batch,
    )

    # Em retomadas, recupera resultados anteriores antes de processar o lote solicitado.
    if resume_from_batch > 0:
        logger.info("Retomando a partir do lote %s...", resume_from_batch)
        all_results, batches_processed = _load_previous_batches(
            output_dir,
            split_name,
            start_batch_index,
        )
    else:
        all_results, batches_processed = [], []

    for batch_num in range(start_batch_index, num_batches):
        # Define o intervalo de IDs pertencente ao lote atual.
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(ids))
        batch_ids = ids[start_idx:end_idx]
        batch_number = batch_num + 1
        batch_results = _process_and_save_batch(
            batch_ids,
            batch_number,
            num_batches,
            split_name,
            config,
            base_path,
            output_dir,
        )

        all_results.extend(batch_results)
        batches_processed.append(batch_number)

    # Consolida tempos persistidos, incluindo lotes executados em processos anteriores.
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
