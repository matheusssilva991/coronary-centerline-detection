"""Agregação sob demanda e validação de integridade de runs do pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .results_metadata import make_json_safe
from .results_schema import add_internal_result_aliases, summarize_results_df


SUMMARY_SCHEMA_VERSION = 2

SUMMARY_SCIENTIFIC_METRICS = (
    "dice_artery_mean",
    "dice_artery_std",
    "dice_artery_median",
    "dice_artery_q1",
    "dice_artery_q3",
    "dice_artery_valid_ostia_mean",
    "dice_artery_invalid_ostia_mean",
    "dice_artery_before_morphology_mean",
    "dice_artery_after_morphology_mean",
    "dice_artery_morphology_delta_mean",
    "effective_upper_threshold_hu_count",
    "effective_upper_threshold_hu_mean",
    "effective_upper_threshold_hu_min",
    "effective_upper_threshold_hu_max",
    "image_slice_count_count",
    "image_slice_count_sum",
    "image_slice_count_mean",
    "image_slice_count_std",
    "image_slice_count_median",
    "image_slice_count_min",
    "image_slice_count_max",
    "artery_voxel_count_count",
    "artery_voxel_count_mean",
    "artery_voxel_count_std",
    "artery_voxel_count_median",
    "artery_voxel_count_q1",
    "artery_voxel_count_q3",
    "artery_voxel_count_min",
    "artery_voxel_count_max",
    "artery_voxel_count_before_morphology_count",
    "artery_voxel_count_before_morphology_mean",
    "artery_voxel_count_after_morphology_count",
    "artery_voxel_count_after_morphology_mean",
    "threshold_voxel_count_count",
    "threshold_voxel_count_mean",
    "threshold_voxel_count_median",
    "threshold_voxel_count_min",
    "threshold_voxel_count_max",
    "lcc_voxel_count_count",
    "lcc_voxel_count_mean",
    "lcc_voxel_count_median",
    "lcc_voxel_count_min",
    "lcc_voxel_count_max",
    "aorta_circle_count_count",
    "aorta_circle_count_mean",
    "aorta_circle_count_median",
    "aorta_circle_count_min",
    "aorta_circle_count_max",
    "aorta_detected_circle_count_count",
    "aorta_detected_circle_count_mean",
    "aorta_interpolated_circle_count_count",
    "aorta_interpolated_circle_count_mean",
    "aorta_segmented_slice_count_count",
    "aorta_segmented_slice_count_mean",
    "aorta_segmented_slice_count_median",
    "aorta_segmented_slice_count_min",
    "aorta_segmented_slice_count_max",
    "aorta_circle_coverage_count",
    "aorta_circle_coverage_mean",
    "aorta_circle_coverage_median",
    "aorta_circle_coverage_min",
    "aorta_circle_coverage_max",
    "aorta_mask_voxel_count_count",
    "aorta_mask_voxel_count_mean",
    "aorta_mask_voxel_count_median",
    "aorta_mask_voxel_count_min",
    "aorta_mask_voxel_count_max",
    "aorta_voxels_per_segmented_slice_count",
    "aorta_voxels_per_segmented_slice_mean",
    "aorta_voxels_per_segmented_slice_median",
    "aorta_volume_fraction_count",
    "aorta_volume_fraction_mean",
    "aorta_volume_fraction_median",
    "aorta_volume_fraction_min",
    "aorta_volume_fraction_max",
)


class ResultIntegrityError(ValueError):
    """Indica que o consolidado não representa exatamente a coorte esperada."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        problems = []
        if report["duplicate_image_ids"]:
            problems.append(f"IDs duplicados: {report['duplicate_image_ids']}")
        if report["missing_image_ids"]:
            problems.append(f"IDs ausentes: {report['missing_image_ids']}")
        if report["unexpected_image_ids"]:
            problems.append(f"IDs inesperados: {report['unexpected_image_ids']}")
        super().__init__("; ".join(problems) or "Resultados inconsistentes.")


def _normalized_image_ids(values: Iterable[Any]) -> list[int]:
    return [int(value) for value in values if pd.notna(value)]


def validate_result_integrity(
    dataframe: pd.DataFrame,
    expected_image_ids: Iterable[Any],
) -> dict[str, Any]:
    """Valida unicidade e igualdade exata entre IDs persistidos e esperados."""
    expected_ids = _normalized_image_ids(expected_image_ids)
    if "IMG_ID" not in dataframe.columns:
        raise ResultIntegrityError(
            {
                "status": "incomplete",
                "expected_image_count": len(expected_ids),
                "persisted_image_count": len(dataframe),
                "missing_image_ids": expected_ids,
                "unexpected_image_ids": [],
                "duplicate_image_ids": [],
                "reason": "missing_img_id_column",
            }
        )

    persisted_ids = _normalized_image_ids(dataframe["IMG_ID"].tolist())
    persisted_series = pd.Series(persisted_ids, dtype=int)
    duplicate_ids = sorted(
        persisted_series[persisted_series.duplicated()].unique().tolist()
    )
    expected_set = set(expected_ids)
    persisted_set = set(persisted_ids)
    report = {
        "status": "complete",
        "expected_image_count": len(expected_ids),
        "persisted_image_count": len(persisted_ids),
        "missing_image_ids": sorted(expected_set - persisted_set),
        "unexpected_image_ids": sorted(persisted_set - expected_set),
        "duplicate_image_ids": duplicate_ids,
        "reason": None,
    }
    if (
        duplicate_ids
        or report["missing_image_ids"]
        or report["unexpected_image_ids"]
        or len(expected_ids) != len(persisted_ids)
    ):
        report["status"] = "incomplete"
        report["reason"] = "image_id_mismatch"
        raise ResultIntegrityError(report)
    return report


def effective_config_sha256(config: dict[str, Any]) -> str:
    """Calcula hash estável da configuração efetiva, independente de indentação."""
    payload = json.dumps(
        make_json_safe(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def infer_run_identity(
    run_dir: Path,
    split_name: str,
    resolution: str | None = None,
) -> dict[str, Any]:
    """Extrai identificadores portáveis da hierarquia do diretório do run."""
    parts = run_dir.parts
    inferred_resolution = resolution
    resolution_index = None
    for index, part in enumerate(parts):
        if part in {"mid_res", "high_res"}:
            resolution_index = index
            inferred_resolution = part.removesuffix("_res")

    run_group = None
    if resolution_index is not None:
        split_indices = [
            index
            for index in range(resolution_index + 1, len(parts) - 1)
            if parts[index] == split_name
        ]
        group_end = split_indices[-1] if split_indices else len(parts) - 1
        run_group = "/".join(parts[resolution_index + 1 : group_end]) or None

    return {
        "run_id": run_dir.name,
        "run_group": run_group,
        "split": split_name,
        "resolution": inferred_resolution,
    }


def build_run_summary_row(
    dataframe: pd.DataFrame,
    config: dict[str, Any] | None = None,
    *,
    run_dir: Path,
    split_name: str,
    expected_image_count: int,
    resolution: str | None = None,
    config_source: str | None = None,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    """Constrói o resumo v2 somente com identidade e resultados agregados.

    ``config`` e os argumentos ``config_*`` são aceitos temporariamente para
    compatibilidade de chamada, mas nunca são serializados no CSV.
    """
    del config, config_source, config_sha256
    metrics = summarize_results_df(add_internal_result_aliases(dataframe.copy()))
    feedback = metrics.pop("aorta_segmentation_feedback_counts")
    total_processed = int(metrics.pop("total_processed"))
    row: dict[str, Any] = {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        **infer_run_identity(run_dir, split_name, resolution),
        "run_status": "complete",
        "expected_image_count": int(expected_image_count),
        "processed_image_count": total_processed,
    }
    renamed_metrics = {
        "ostia_found": "ostia_found_count",
        "ostia_status_not_found": "ostia_not_found_count",
        "ostia_status_not_found_percent": "ostia_not_found_percent",
        "both_correct": "both_ostia_correct_count",
        "both_correct_percent": "both_ostia_correct_percent",
        "both_tolerable": "both_ostia_tolerable_count",
        "both_tolerable_percent": "both_ostia_tolerable_percent",
        "total_success": "ostia_success_count",
        "total_success_percent": "ostia_success_percent",
        "segmentation_attempted": "segmentation_attempted_count",
        "proceeded_with_bad_ostia": "proceeded_with_bad_ostia_count",
        "left_correct": "left_ostium_correct_count",
        "right_correct": "right_ostium_correct_count",
        "error_not_null": "pipeline_error_count",
        "ostia_error_not_null": "ostia_detection_error_count",
    }
    outcome_keys = (
        "ostia_found",
        "ostia_found_percent",
        "ostia_status_not_found",
        "ostia_status_not_found_percent",
        "both_correct",
        "both_correct_percent",
        "both_tolerable",
        "both_tolerable_percent",
        "total_success",
        "total_success_percent",
        "segmentation_attempted",
        "segmentation_attempted_percent",
        "proceeded_with_bad_ostia",
        "proceeded_with_bad_ostia_percent",
        "left_correct",
        "right_correct",
        "error_not_null",
        "ostia_error_not_null",
    )
    row.update(
        {
            renamed_metrics.get(key, key): metrics[key]
            for key in outcome_keys
            if key in metrics
        }
    )
    row.update({key: metrics.get(key) for key in SUMMARY_SCIENTIFIC_METRICS})
    row.update(
        {
            "aorta_feedback_adequate_count": feedback.get("adequate", 0),
            "aorta_feedback_undersegmentation_count": feedback.get(
                "suspected_undersegmentation", 0
            ),
            "aorta_feedback_oversegmentation_count": feedback.get(
                "suspected_oversegmentation", 0
            ),
            "aorta_feedback_insufficient_data_count": feedback.get(
                "insufficient_data", 0
            ),
        }
    )
    return row


__all__ = [
    "ResultIntegrityError",
    "SUMMARY_SCHEMA_VERSION",
    "SUMMARY_SCIENTIFIC_METRICS",
    "build_run_summary_row",
    "effective_config_sha256",
    "infer_run_identity",
    "validate_result_integrity",
]
