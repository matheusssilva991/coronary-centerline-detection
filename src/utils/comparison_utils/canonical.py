"""Consolidação de referências canônicas a partir de resultados já salvos."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from utils.comparison_utils.io import load_experiment_variant_results
from utils.project.config import scale_config_to_resolution
from utils.project.results.columns import READABLE_COLUMN_NAMES
from utils.project.results.io import save_dataframe_atomic
from utils.project.results.schema import make_readable_results_dataframe
from utils.utils.json_io import save_json_file


def consolidate_experiment_reference(
    repo_root: Path,
    source_run: Path,
    destination: Path,
    *,
    variant: str,
    split: str,
    expected_count: int,
) -> dict[str, Any]:
    """Extrai uma variante sem reexecutar o pipeline ou fabricar tempos.

    A configuração é reconstruída do snapshot e das sobrescritas registradas,
    nunca do config atual do projeto. Os arquivos científicos de origem não
    são modificados. O destino deve ser uma pasta nova ou de consolidação.
    """
    from utils.experiments.fuzzy_pipeline_comparison import split_overrides
    from utils.experiments.sweep_common import apply_overrides

    repo_root = repo_root.resolve()
    source_run = source_run.resolve()
    source_file = source_run / "results/image_results.csv"
    config_file = source_run / "run_config.json"
    frame = load_experiment_variant_results(source_run, variant=variant, split=split)
    if len(frame) != expected_count:
        raise ValueError(f"Coorte incompleta: {len(frame)} != {expected_count}")
    source_config = json.loads(config_file.read_text(encoding="utf-8"))
    specification = next(
        item for item in source_config["variants"] if item["name"] == variant
    )
    overrides, experiment = split_overrides(specification["overrides"])
    config = scale_config_to_resolution(
        apply_overrides(source_config["effective_base_config"], overrides)
    )
    config = apply_overrides(config, specification.get("post_scale_overrides", {}))
    if specification.get("disabled_scaling_groups"):
        raise ValueError(
            "Esta consolidação não aceita variantes com escalonamento parcial."
        )

    # Evita colunas duplicadas ao converter aliases para o schema persistido.
    redundant = [
        internal
        for internal, readable in READABLE_COLUMN_NAMES.items()
        if internal != readable and internal in frame and readable in frame
    ]
    normalized = make_readable_results_dataframe(frame.drop(columns=redundant))
    if normalized.columns.duplicated().any():
        raise ValueError("Colunas duplicadas na referência consolidada.")
    normalized["source_run"] = source_run.relative_to(repo_root).as_posix()
    normalized["variant"] = variant
    manifest = {
        "artifact_kind": "consolidation_of_existing_results",
        "source_run": source_run.relative_to(repo_root).as_posix(),
        "source_csv": source_file.relative_to(repo_root).as_posix(),
        "source_csv_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "source_config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "variant": variant,
        "split": split,
        "exam_count": len(normalized),
        "exam_ids": sorted(int(value) for value in normalized["IMG_ID"]),
        "recorded_variant": specification,
        "timing_note": "Sem manifest de batches equivalente; tempos não consolidados.",
    }
    # As métricas permanecem por exame; nenhum batch ou execução é sintetizado.
    save_dataframe_atomic(normalized, destination / "numeric" / f"results_{split}.csv")
    save_json_file(manifest, str(destination / "source_manifest.json"))
    save_json_file(config, str(destination / "config/effective_pipeline_config.json"))
    save_json_file(experiment, str(destination / "config/experiment_settings.json"))
    return manifest
