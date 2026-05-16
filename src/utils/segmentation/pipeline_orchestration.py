"""Orquestração do pipeline de segmentação coronária."""

from __future__ import annotations

import glob
import logging
import math
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..groups.io import save_results, scale_config_to_resolution
from .pipeline_steps import (
    detect_and_evaluate_ostia,
    get_or_compute_vesselness,
    get_or_detect_aorta_circles,
    get_or_segment_aorta,
    load_and_preprocess_image,
    segment_arteries_from_ostia,
)


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
        )

        aorta_mask = get_or_segment_aorta(
            img_id,
            lcc_image,
            detected_circles,
            config["LEVEL_SET"],
            base_save_path,
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

    num_batches = config.get("NUM_BATCHES") or 5
    if num_batches <= 0:
        num_batches = 5
    all_results = []
    batches_processed = []
    if output_dir is None:
        raise ValueError("output_dir é obrigatório no modo batch")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = max(1, math.ceil(len(ids) / num_batches))

    if resume_from_batch > 0:
        logger.info(f"Retomando a partir do lote {resume_from_batch}...")
        missing_batches = []
        for batch_num in range(0, resume_from_batch):
            candidate1 = (
                output_dir / f"ostios_{split_name}_lote_{batch_num + 1}_summary.csv"
            )
            candidate2 = output_dir / f"ostios_{split_name}_lote_{batch_num + 1}.csv"

            found_path = None
            if candidate1.exists():
                found_path = candidate1
            elif candidate2.exists():
                found_path = candidate2
            else:
                pattern = str(
                    output_dir / f"ostios_{split_name}_lote_{batch_num + 1}*.csv"
                )
                matches = sorted(glob.glob(pattern))
                if matches:
                    found_path = Path(matches[0])

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

    for batch_num in range(resume_from_batch, num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(ids))
        batch_ids = ids[start_idx:end_idx]

        logger.info(
            f"Processando lote {batch_num + 1}/{num_batches} ({len(batch_ids)} imagens)"
        )
        batch_results = []

        for img_id in tqdm(
            batch_ids, desc=f"Lote {batch_num + 1}/{num_batches}", leave=False
        ):
            batch_results.append(
                process_image(img_id, scaled_config, base_path, base_save_path)
            )

        all_results.extend(batch_results)
        batches_processed.append(batch_num + 1)

        batch_output_path = save_results(
            batch_results,
            f"{split_name}_lote_{batch_num + 1}",
            output_dir,
            config=scaled_config,
        )
        logger.info(f"Lote {batch_num + 1} salvo: {batch_output_path}")

    execution_time = time.time() - start_time
    result = {
        "details": all_results,
        "execution_time": execution_time,
        "batches_processed": batches_processed,
        "is_batched": True,
    }

    return result


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


def parse_resume_batches(resume_batches_arg):
    """Converte um argumento no formato 'train=1,val=0,test=3' em um dicionário."""
    resume_map = {"train": 0, "val": 0, "test": 0}

    if not resume_batches_arg:
        return resume_map

    entries = [
        entry.strip() for entry in resume_batches_arg.split(",") if entry.strip()
    ]
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                "Formato inválido para --resume-batches. Use algo como 'train=1,val=0,test=3'."
            )

        split_name, batch_text = entry.split("=", 1)
        split_name = split_name.strip()
        batch_text = batch_text.strip()

        if split_name not in resume_map:
            raise ValueError(
                f"Split inválido em --resume-batches: {split_name}. Use train, val ou test."
            )

        try:
            resume_map[split_name] = int(batch_text)
        except ValueError as exc:
            raise ValueError(
                f"Valor inválido para o split '{split_name}' em --resume-batches: {batch_text}"
            ) from exc

    return resume_map
