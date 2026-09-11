#!/usr/bin/env python3
"""Migra runs completos para o contrato científico de artefatos v2."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.project.result_paths import (  # noqa: E402
    batch_result_number,
    batch_results_filename,
    batch_timings_filename,
    metadata_candidates,
    metadata_filename,
    results_filename,
    summary_filename,
)
from utils.project.results_columns import RESULT_CONFIGURATION_COLUMNS  # noqa: E402
from utils.project.results_metadata import (  # noqa: E402
    build_metadata,
    effective_config_sha256,
    make_json_safe,
)
from utils.project.results_schema import select_per_image_result_columns  # noqa: E402
from utils.project.run_summary import (  # noqa: E402
    ResultIntegrityError,
    validate_result_integrity,
)


RESULT_PATTERN = re.compile(
    r"(?:results_|ostios_)(train|val|test|full)(?:_results)?\.csv"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "output/segmentation/runs",
        help="Raiz dos runs que serão inspecionados.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Aplica a migração; sem esta flag executa apenas a inspeção.",
    )
    return parser


def _run_dir(numeric_dir: Path) -> Path:
    return numeric_dir.parent if numeric_dir.name == "numeric" else numeric_dir


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata_path(numeric_dir: Path, split: str) -> Path | None:
    return next(
        (path for path in metadata_candidates(numeric_dir, split) if path.is_file()),
        None,
    )


def _expected_ids(run_dir: Path, numeric_dir: Path, split: str) -> list[int]:
    split_payload = _load_json(run_dir / "config/split_ids.json") or {}
    split_ids = split_payload.get("splits", {}).get(split)
    if split_ids is not None:
        return [int(image_id) for image_id in split_ids]

    metadata_path = _metadata_path(numeric_dir, split)
    metadata = _load_json(metadata_path) if metadata_path else None
    metadata_ids = (metadata or {}).get("execution_info", {}).get("image_ids")
    if metadata_ids is None:
        raise ValueError("run sem split_ids.json e sem image_ids no metadata")
    return [int(image_id) for image_id in metadata_ids]


def _resolution(run_dir: Path) -> str | None:
    for part in run_dir.parts:
        if part in {"mid_res", "high_res"}:
            return part.removesuffix("_res")
    return None


def _is_partial_layout(numeric_dir: Path) -> bool:
    return any(numeric_dir.glob("ostios_*_integrity.json")) or any(
        numeric_dir.glob("ostios_*_partial_results.csv")
    )


def _source_results(path: Path, split: str) -> Path | None:
    numeric_dir = path.parent
    candidates = (
        numeric_dir / results_filename(split),
        numeric_dir / f"ostios_{split}_results.csv",
        numeric_dir / f"ostios_{split}_summary.csv",
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if "IMG_ID" in pd.read_csv(candidate, nrows=0).columns:
            return candidate
    return None


def _legacy_config_values(dataframe: pd.DataFrame) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for column in RESULT_CONFIGURATION_COLUMNS:
        if column not in dataframe.columns or column == "max_threshold_hu":
            continue
        present = dataframe[column].dropna()
        if present.empty:
            continue
        unique = list(
            dict.fromkeys(make_json_safe(value) for value in present.tolist())
        )
        values[column] = unique[0] if len(unique) == 1 else unique
    return values


def _configuration_identity(
    run_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    config_path = run_dir / "config/effective_pipeline_config.json"
    config = _load_json(config_path)
    if config is not None:
        return {
            "source": "effective_pipeline_config",
            "sha256": effective_config_sha256(config),
        }
    previous = metadata.get("configuration") or metadata.get(
        "configuration_identity", {}
    )
    return {
        "source": previous.get("source") or "legacy_metadata",
        "sha256": previous.get("sha256") or "",
    }


def _metadata_payload(
    source: dict[str, Any],
    *,
    run_dir: Path,
    split: str,
    legacy_config: dict[str, Any],
    results: pd.DataFrame,
    batch_timings: list[dict[str, Any]],
    expected_batches: list[int],
) -> dict[str, Any]:
    effective_config = _load_json(run_dir / "config/effective_pipeline_config.json")
    identity = _configuration_identity(run_dir, source)
    legacy_values = _compact_legacy_config(source)
    legacy_values.update(legacy_config)
    return build_metadata(
        split,
        effective_config or {},
        resolution=_resolution(run_dir),
        config_source=identity["source"],
        config_sha256=identity["sha256"],
        legacy_result_config=legacy_values,
        results=results,
        batch_timings=batch_timings,
        expected_batches=expected_batches,
    )


def _compact_legacy_config(source: dict[str, Any]) -> dict[str, Any]:
    """Recupera somente os rótulos essenciais de metadados históricos."""
    compact = dict(source.get("configuration") or {})
    runtime = source.get("runtime_config") or {}
    preprocessing = source.get("preprocessing_config") or {}
    thresholding = source.get("thresholding_config") or preprocessing.get(
        "thresholding", {}
    )
    artery = source.get("artery_segmentation_config") or {}
    legacy = source.get("legacy_result_config") or {}
    candidates = {
        "use_gpu": runtime.get("use_gpu"),
        "downscale_method": preprocessing.get("downscale_method"),
        "downscale_factors": preprocessing.get("downscale_factors"),
        "min_threshold_hu": preprocessing.get("min_threshold"),
        "max_threshold_percentile": preprocessing.get("max_threshold_percentile"),
        "threshold_method": (
            thresholding.get("method") if isinstance(thresholding, dict) else None
        ),
        "artery_segmentation_method": (
            artery.get("method") if isinstance(artery, dict) else None
        )
        or runtime.get("artery_segmentation_method"),
    }
    for key, value in candidates.items():
        if value is not None and compact.get(key) is None:
            compact[key] = value
    for key, value in legacy.items():
        if value is not None and compact.get(key) is None:
            compact[key] = value
    return compact


def _write_csv_temp(path: Path, dataframe: pd.DataFrame) -> Path:
    temporary = path.with_name(f".{path.name}.migration.tmp")
    dataframe.to_csv(temporary, index=False)
    return temporary


def _write_json_temp(path: Path, payload: dict[str, Any]) -> Path:
    temporary = path.with_name(f".{path.name}.migration.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return temporary


def _batch_sources(numeric_dir: Path, split: str) -> list[tuple[int, Path]]:
    batches: dict[int, list[Path]] = {}
    for path in numeric_dir.glob("*.csv"):
        number = batch_result_number(path, split)
        if number is not None:
            batches.setdefault(number, []).append(path)
    duplicates = {number: paths for number, paths in batches.items() if len(paths) > 1}
    if duplicates:
        detail = "; ".join(
            f"lote {number}: {', '.join(path.name for path in paths)}"
            for number, paths in sorted(duplicates.items())
        )
        raise ValueError(f"lotes duplicados entre formatos: {detail}")
    return [(number, paths[0]) for number, paths in sorted(batches.items())]


def _timing_dataframe(
    numeric_dir: Path, split: str
) -> tuple[pd.DataFrame | None, Path | None]:
    candidates = (
        numeric_dir / batch_timings_filename(split),
        numeric_dir / f"ostios_{split}_batch_timings.csv",
    )
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        return None, None
    timing = pd.read_csv(source)
    if "batch_number" in timing.columns and "result_file" in timing.columns:
        timing["result_file"] = timing["batch_number"].map(
            lambda number: batch_results_filename(split, int(number))
        )
    return timing, source


def migrate_run(path: Path, *, apply: bool) -> str:
    """Inspeciona ou migra um único diretório numérico."""
    path = Path(path)
    match = RESULT_PATTERN.fullmatch(path.name)
    if match is None:
        legacy_match = re.fullmatch(
            r"ostios_(train|val|test|full)_summary\.csv", path.name
        )
        if legacy_match is None:
            return "ignored"
        split = legacy_match.group(1)
    else:
        split = match.group(1)

    numeric_dir = path.parent
    if _is_partial_layout(numeric_dir):
        return "partial_preserved"
    source_results = _source_results(path, split)
    if source_results is None:
        return "aggregate_without_results"

    run_dir = _run_dir(numeric_dir)
    original = pd.read_csv(source_results)
    expected_ids = _expected_ids(run_dir, numeric_dir, split)
    try:
        validate_result_integrity(original, expected_ids)
    except ResultIntegrityError:
        return "incomplete_preserved"

    projected = select_per_image_result_columns(original)
    validate_result_integrity(projected, expected_ids)
    batch_sources = _batch_sources(numeric_dir, split)
    timing, timing_source = _timing_dataframe(numeric_dir, split)
    legacy_config = _legacy_config_values(original)
    metadata_source_path = _metadata_path(numeric_dir, split)
    metadata_source = _load_json(metadata_source_path) if metadata_source_path else {}
    metadata_payload = _metadata_payload(
        metadata_source or {},
        run_dir=run_dir,
        split=split,
        legacy_config=legacy_config,
        results=projected,
        batch_timings=[] if timing is None else timing.to_dict("records"),
        expected_batches=[number for number, _ in batch_sources],
    )

    target_results = numeric_dir / results_filename(split)
    already_current = source_results == target_results
    target_metadata = numeric_dir / metadata_filename(split)
    target_split_ids = run_dir / "config/split_ids.json"
    current_metadata = _load_json(target_metadata)
    needs_cleanup = (
        current_metadata != metadata_payload
        or (numeric_dir / summary_filename(split)).is_file()
        or not target_split_ids.is_file()
    )
    if not apply:
        if not already_current:
            return "ready"
        return "ready_cleanup" if needs_cleanup else "already_migrated"

    staged: list[tuple[Path, Path]] = [
        (_write_json_temp(target_metadata, metadata_payload), target_metadata),
    ]
    if not already_current:
        staged.insert(0, (_write_csv_temp(target_results, projected), target_results))
    if not target_split_ids.is_file():
        target_split_ids.parent.mkdir(parents=True, exist_ok=True)
        split_payload = {
            "source": "legacy_metadata_migration",
            "splits": {split: expected_ids},
        }
        staged.append(
            (_write_json_temp(target_split_ids, split_payload), target_split_ids)
        )
    migrated_batch_sources: list[Path] = []
    for batch_number, batch_source in batch_sources:
        target = numeric_dir / batch_results_filename(split, batch_number)
        if batch_source != target:
            batch_projected = select_per_image_result_columns(pd.read_csv(batch_source))
            staged.append((_write_csv_temp(target, batch_projected), target))
            migrated_batch_sources.append(batch_source)
    if timing is not None and timing_source != numeric_dir / batch_timings_filename(
        split
    ):
        target = numeric_dir / batch_timings_filename(split)
        staged.append((_write_csv_temp(target, timing), target))

    for temporary, target in staged:
        temporary.replace(target)

    obsolete = {
        source_results if source_results != target_results else None,
        numeric_dir / summary_filename(split),
        numeric_dir / f"ostios_{split}_summary.csv",
        metadata_source_path
        if metadata_source_path != numeric_dir / metadata_filename(split)
        else None,
        timing_source
        if timing_source != numeric_dir / batch_timings_filename(split)
        else None,
        *migrated_batch_sources,
    }
    for obsolete_path in obsolete:
        if obsolete_path is not None and obsolete_path.exists():
            obsolete_path.unlink()
    return "refreshed" if already_current else "migrated"


def _candidate_results(root: Path) -> list[Path]:
    candidates: dict[Path, Path] = {}
    patterns = (
        "results_*.csv",
        "ostios_*_results.csv",
        "ostios_*_summary.csv",
    )
    for pattern in patterns:
        for path in root.rglob(pattern):
            if "_lote_" in path.name or "_partial_" in path.name:
                continue
            numeric_dir = path.parent
            candidates.setdefault(numeric_dir, path)
            if path.name.startswith("results_"):
                candidates[numeric_dir] = path
            elif "_results.csv" in path.name and not candidates[
                numeric_dir
            ].name.startswith("results_"):
                candidates[numeric_dir] = path
    return sorted(candidates.values())


def main() -> None:
    args = build_parser().parse_args()
    counts: dict[str, int] = {}
    for result_path in _candidate_results(args.root):
        status = migrate_run(result_path, apply=args.apply)
        counts[status] = counts.get(status, 0) + 1
        print(f"{status:>22}  {result_path}")
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
