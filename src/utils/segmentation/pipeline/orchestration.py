"""Orquestração do pipeline de segmentação coronária."""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils.project.evaluation.artery_distances import pending_artery_distance_metrics
from utils.project.results import (
    duration_breakdown,
    get_batch_result_file,
    load_batch_timing_records,
    save_batch_timing_record,
    save_results,
    summarize_batch_timing_records,
)
from utils.project.results.artery_masks import (
    ArteryMaskExport,
    pending_artery_mask_export,
)
from utils.project.runtime.resource_monitor import (
    ResourceMonitor,
    persist_resource_diagnostics,
)
from utils.segmentation.pipeline.arteries import segment_arteries_from_ostia
from utils.segmentation.pipeline.detection import (
    detect_and_evaluate_ostia,
    locate_and_segment_aorta,
)
from utils.segmentation.pipeline.preprocessing import (
    compute_vesselness,
    load_and_preprocess_image,
)
from utils.visualization.pipeline.pipeline_artifacts import save_segmentation_visual

logger = logging.getLogger(__name__)


IMAGE_RESULT_DEFAULTS = {
    "ostia_left": None,
    "ostia_right": None,
    "artery_voxels": None,
    "artery_voxels_before_morphology": None,
    "artery_voxels_after_closing": None,
    "artery_voxels_after_morphology": None,
    "dice_artery": None,
    "dice_artery_before_morphology": None,
    "dice_artery_after_closing": None,
    "dice_artery_closing_delta": None,
    "dice_artery_dilation_delta": None,
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
    "effective_upper_threshold_hu": None,
    "lower_threshold_method": None,
    "lower_threshold_percentile": None,
    "threshold_voxels": None,
    "lcc_voxels": None,
    "image_slice_count": None,
    "image_voxels": None,
    "aorta_circle_count": None,
    "aorta_detected_circle_count": None,
    "aorta_interpolated_circle_count": None,
    "aorta_circle_first_slice": None,
    "aorta_circle_last_slice": None,
    "aorta_circle_coverage": None,
    "aorta_circle_radius_min_px": None,
    "aorta_circle_radius_max_px": None,
    "aorta_circle_radius_mean_px": None,
    "aorta_circle_radius_median_px": None,
    "aorta_circle_radius_std_px": None,
    "aorta_circle_radius_p10_px": None,
    "aorta_circle_radius_p90_px": None,
    "aorta_circle_radius_min_mm": None,
    "aorta_circle_radius_max_mm": None,
    "aorta_circle_radius_mean_mm": None,
    "aorta_circle_radius_median_mm": None,
    "aorta_circle_radius_std_mm": None,
    "aorta_circle_radius_p10_mm": None,
    "aorta_circle_radius_p90_mm": None,
    "aorta_detected_circle_radius_median_mm": None,
    "aorta_interpolated_circle_radius_median_mm": None,
    "aorta_circle_radius_first_mm": None,
    "aorta_circle_radius_last_mm": None,
    "aorta_circle_radius_max_step_change_mm": None,
    "aorta_circle_radius_p90_step_change_mm": None,
    "aorta_circle_mean_hough_accumulator": None,
    "aorta_circle_lower_radius_bound_fraction": None,
    "aorta_circle_upper_radius_bound_fraction": None,
    "aorta_circle_filter_method": "none",
    "aorta_circle_filter_applied": False,
    "aorta_circle_original_count": None,
    "aorta_circle_used_count": None,
    "aorta_circle_filter_synthetic_tail_count": 0,
    "aorta_circle_filter_trimmed_tail_count": 0,
    "aorta_circle_filter_trim_start_slice": None,
    "aorta_circle_filter_detected_tail_start_slice": None,
    "aorta_circle_filter_original_coverage": None,
    "aorta_circle_filter_used_coverage": None,
    "aorta_circle_filter_reason": None,
    "aorta_mask_voxels": None,
    "aorta_segmented_slice_count": None,
    "aorta_voxels_per_segmented_slice": None,
    "aorta_volume_fraction": None,
    "aorta_level_set_initial_voxel_count": None,
    "aorta_level_set_raw_voxel_count": None,
    "aorta_level_set_initial_volume_fraction": None,
    "aorta_level_set_raw_volume_fraction": None,
    "aorta_level_set_iterations_used": None,
    "aorta_level_set_circle_fill_q25": None,
    "aorta_level_set_circle_area_ratio_p90": None,
    "aorta_slice_area_jump_p95": None,
    "aorta_segmentation_feedback": "insufficient_data",
}


def _new_image_result(img_id):
    """Cria o registro completo com os defaults de uma imagem.

    Args:
        img_id (int | str): Identificador do exame no ImageCAS.

    Returns:
        dict[str, Any]: Dicionário inicializado com ``IMG_ID`` e campos padrão.
    """
    return {"IMG_ID": img_id, **IMAGE_RESULT_DEFAULTS}


def _preprocessing_result_fields(
    preprocessing_details,
    image_slice_count,
    image_voxel_count=None,
):
    """Converte detalhes do pré-processamento nos campos persistidos.

    Args:
        preprocessing_details (dict[str, Any]): Metadados retornados por
            ``load_and_preprocess_image``.
        image_slice_count (int): Número de fatias axiais do volume pós-downscaling.
        image_voxel_count (int | None): Total opcional de voxels do volume processado.

    Returns:
        dict[str, Any]: Subconjunto de campos tabulares de pré-processamento e
            dimensões do exame.
    """
    # Copia as métricas do threshold/LCC e registra as dimensões do volume processado.
    detail_keys = (
        "threshold_mode",
        "fuzzy_mask_strategy",
        "min_threshold",
        "max_threshold",
        "effective_upper_threshold_hu",
        "lower_threshold_method",
        "lower_threshold_percentile",
        "threshold_voxels",
        "lcc_voxels",
    )
    fields = {key: preprocessing_details.get(key) for key in detail_keys}
    fields["image_slice_count"] = int(image_slice_count)
    fields["image_voxels"] = (
        int(image_voxel_count) if image_voxel_count is not None else None
    )
    return fields


def _ostia_result_fields(ostia_eval):
    """Converte a avaliação dos óstios nos campos persistidos.

    Args:
        ostia_eval (dict[str, Any]): Resultado retornado por
            ``detect_and_evaluate_ostia``.

    Returns:
        dict[str, Any]: Campos tabulares de coordenadas, distâncias e status dos óstios.
    """
    # Determina o status consolidado do par de óstios em relação à referência.
    both_correct = bool(ostia_eval["both_correct"])
    both_tolerable = bool(ostia_eval["both_tolerable"])
    if both_correct:
        status = "both_correct"
    elif both_tolerable:
        status = "both_tolerable"
    else:
        status = "found_but_wrong"

    # Normaliza coordenadas para tuplas inteiras e extrai distâncias e flags.
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


def process_image(
    img_id,
    config,
    base_path,
    visual_output_dir=None,
    *,
    artery_mask_export: ArteryMaskExport | None = None,
):
    """Executa o fluxo completo do pipeline para um exame ImageCAS e agrega métricas.

    Sequência: pré-processamento, localização/segmentação da aorta, vesselness e
    detecção/avaliação de óstios e segmentação arterial.

    Args:
        img_id (int | str): Identificador do exame no ImageCAS.
        config (dict[str, Any]): Configuração efetiva normalizada e já escalada para a
            resolução.
        base_path (Path | str): Diretório raiz contendo as imagens e rótulos do
            dataset.
        visual_output_dir (Path | str | None): Diretório opcional onde visualizações
            3D HTML são salvas.
        artery_mask_export (ArteryMaskExport | None): Destino opcional da máscara
            arterial final na grade processada.

    Returns:
        dict[str, Any]: Dicionário com métricas, diagnósticos de etapas, telemetria e
            eventual mensagem de erro.
    """
    # Inicializa o registro com defaults e inicia o monitoramento de RAM/tempo.
    result = _new_image_result(img_id)
    result.update(pending_artery_distance_metrics(config))
    if artery_mask_export is not None:
        result.update(pending_artery_mask_export())
    monitor = ResourceMonitor(config)
    monitor.start()
    try:
        # Carrega imagem/label e gera o volume pré-processado (LCC).
        image_data = load_and_preprocess_image(img_id, base_path, config)
        lcc_image = image_data["lcc_image"]
        label = image_data["label"]
        scaled_spacing = image_data["scaled_spacing"]
        preprocessing_details = image_data.get("preprocessing_details", {})
        downscale_factors = image_data["downscale_factors"]
        source_path = image_data.get("source_path")

        image_data = None
        result.update(
            _preprocessing_result_fields(
                preprocessing_details,
                lcc_image.shape[2],
                lcc_image.size,
            )
        )

        # Localiza a aorta, filtra a trajetória, segmenta e consolida diagnósticos.
        aorta_stage = locate_and_segment_aorta(
            lcc_image,
            downscale_factors,
            scaled_spacing,
            config,
            circle_summary_source="original",
            require_non_empty=False,
        )
        aorta_mask = aorta_stage.aorta_mask
        result.update(aorta_stage.diagnostics)

        # Calcula o mapa de vasos só após segmentar a aorta para reduzir pico de RAM.
        vesselness_ostios = compute_vesselness(
            lcc_image,
            vesselness_config=config["VESSELNESS_AORTA"],
            use_gpu=config.get("USE_GPU", False),
        )

        try:
            # Seleciona os óstios e valida contra o label arterial.
            ostia_eval = detect_and_evaluate_ostia(
                aorta_mask,
                vesselness_ostios,
                label,
                scaled_spacing,
                config,
            )
        except ValueError as ostia_exc:
            # Registra falha na busca de óstios e gera visual diagnóstico se habilitado.
            result["ostia_status"] = "not_found"
            result["ostia_error"] = str(ostia_exc)
            result["skip_reason"] = "ostia_not_found"
            result["dice_artery"] = 0.0
            if visual_output_dir is not None:
                save_segmentation_visual(
                    visual_output_dir,
                    img_id,
                    aorta_mask=aorta_mask,
                    ostia_left=None,
                    ostia_right=None,
                    artery_mask=None,
                    label_artery=(label == 1).astype("uint8"),
                    spacing=scaled_spacing,
                )
            return result
        finally:
            del vesselness_ostios

        result.update(_ostia_result_fields(ostia_eval))

        # Segmenta as artérias mesmo quando os óstios são apenas toleráveis.
        result["segmentation_attempted"] = True
        artery_metrics = segment_arteries_from_ostia(
            lcc_image,
            ostia_eval["label_artery"],
            ostia_eval["ostia_left"],
            ostia_eval["ostia_right"],
            config,
            spacing=scaled_spacing,
        )
        if (
            artery_mask_export is not None
            and artery_metrics.get("artery_mask") is not None
        ):
            result.update(
                artery_mask_export.save(
                    str(img_id),
                    artery_metrics["artery_mask"],
                    source_path,
                    config,
                )
            )
        if visual_output_dir is not None:
            save_segmentation_visual(
                visual_output_dir,
                img_id,
                aorta_mask=aorta_mask,
                ostia_left=ostia_eval["ostia_left"],
                ostia_right=ostia_eval["ostia_right"],
                artery_mask=artery_metrics.get("artery_mask"),
                label_artery=ostia_eval["label_artery"],
                spacing=scaled_spacing,
            )

        # Descarta máscaras volumétricas antes de retornar os campos tabulares.
        del aorta_mask
        artery_metrics.pop("artery_mask", None)
        artery_metrics.pop("raw_artery_mask", None)
        result.update(artery_metrics)

    except Exception as exc:
        result["error"] = str(exc)
    finally:
        monitor.finish(result)

    return result


def _resolve_batch_plan(ids, config, resume_from_batch):
    """Calcula quantidade, tamanho e índice inicial dos lotes.

    Args:
        ids (Sequence[int | str]): Lista ordenada de IDs de exames do split.
        config (dict[str, Any]): Configuração efetiva contendo ``NUM_BATCHES``.
        resume_from_batch (int): Número do lote (base 1) para retomada, ou ``0`` para
            executar do início.

    Returns:
        tuple[int, int, int]: Tupla ``(num_batches, batch_size, start_batch_index)``.
    """
    # Ajusta a quantidade de lotes ao total de exames e valida o ponto de retomada.
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

    # Calcula o tamanho de cada lote e o índice base-0 de início.
    batch_size = max(1, math.ceil(len(ids) / num_batches))
    start_batch_index = resume_from_batch - 1 if resume_from_batch > 0 else 0
    return num_batches, batch_size, start_batch_index


def _load_previous_batches(output_dir, split_name, start_batch_index):
    """Carrega lotes anteriores necessários para uma retomada consistente.

    Args:
        output_dir (Path | str): Diretório numérico onde os CSVs de lote estão salvos.
        split_name (str): Nome do split usado nos arquivos de resultado.
        start_batch_index (int): Índice base-0 do primeiro lote a ser reprocessado.

    Returns:
        tuple[list[dict[str, Any]], list[int]]: Tupla ``(all_results,
            batches_processed)`` com os registros e números dos lotes recuperados.
    """
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
    visual_output_dir=None,
    *,
    artery_mask_export: ArteryMaskExport | None = None,
):
    """Processa um lote e persiste resultados e duração imediatamente.

    Args:
        batch_ids (Sequence[int | str]): IDs de exames pertencentes ao lote atual.
        batch_number (int): Número do lote atual (base 1).
        num_batches (int): Total de lotes planejados para o split.
        split_name (str): Nome do split usado nos arquivos e manifests.
        config (dict[str, Any]): Configuração efetiva normalizada da execução.
        base_path (Path | str): Diretório raiz do dataset ImageCAS.
        output_dir (Path | str): Diretório numérico de saída para CSVs e manifests.
        visual_output_dir (Path | str | None): Diretório opcional para HTMLs 3D.
        artery_mask_export (ArteryMaskExport | None): Destino das máscaras.

    Returns:
        list[dict[str, Any]]: Lista de dicionários de resultado das imagens do lote.
    """
    # Registra o instante inicial do lote para calcular a duração após a escrita.
    batch_started_at = datetime.now().isoformat(timespec="seconds")
    batch_start_time = time.time()
    logger.info(
        "Processando lote %s/%s (%s imagens)",
        batch_number,
        num_batches,
        len(batch_ids),
    )

    # Processa todas as imagens antes de persistir o lote de forma atômica.
    batch_results = []
    for img_id in tqdm(
        batch_ids, desc=f"Lote {batch_number}/{num_batches}", leave=False
    ):
        image_result = process_image(
            img_id,
            config,
            base_path,
            visual_output_dir=visual_output_dir,
            artery_mask_export=artery_mask_export,
        )
        persist_resource_diagnostics(
            image_result,
            Path(output_dir) / f"resource_monitoring_{split_name}.json",
            str(img_id),
        )
        batch_results.append(image_result)
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
    visual_output_dir=None,
    *,
    artery_mask_export: ArteryMaskExport | None = None,
):
    """Executa o pipeline em lotes e consolida resultados e tempos de execução.

    Args:
        ids (Sequence[int | str]): Lista ordenada de IDs de exames ImageCAS a
            processar.
        split_name (str): Nome do split (usado nos arquivos de lote, manifests e logs).
        config (dict[str, Any]): Configuração runtime normalizada e já escalada para a
            resolução.
        base_path (Path | str): Diretório raiz do dataset ImageCAS.
        output_dir (Path | str | None): Diretório numérico obrigatório onde os CSVs e
            manifests de lote são salvos.
        resume_from_batch (int): Número do lote (base 1) para retomada, ou ``0`` para
            execução do início.
        visual_output_dir (Path | str | None): Diretório opcional onde visualizações
            3D HTML são gravadas.
        artery_mask_export (ArteryMaskExport | None): Destino das máscaras.

    Returns:
        dict[str, Any]: Dicionário com ``details``, ``execution_time``,
            ``execution_time_breakdown``, ``batches_processed``, ``batch_timings``,
            ``batch_timing_summary`` e ``is_batched``.
    """
    # Valida pré-condições de entrada e define o plano de divisão em lotes.
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
            visual_output_dir,
            artery_mask_export=artery_mask_export,
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
