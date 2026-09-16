"""Normalize legacy run statuses without changing scientific result values."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

SRC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.project.results_columns import (  # noqa: E402
    OSTIA_STATUS_PORTUGUESE_LABELS,
    STATUS_PORTUGUESE_LABELS,
)
from utils.project.results_schema import (  # noqa: E402
    normalize_ostia_status,
    normalize_result_status,
)


CURRENT_RESULT_PATTERN = re.compile(
    r"^results_(train|val|test|full)(?:_lote_\d+)?\.csv$"
)
LEGACY_RESULT_PATTERN = re.compile(
    r"^ostios_(train|val|test|full)"
    r"(?:_results|_partial_results|_summary|_lote_\d+_summary)\.csv$"
)
STATUS_NORMALIZERS: dict[str, Callable[[str], str | None]] = {
    "status": normalize_result_status,
    "ostia_detection_status": normalize_ostia_status,
    "ostia_status": normalize_ostia_status,
}
VALID_RESULT_STATUSES = frozenset(STATUS_PORTUGUESE_LABELS)
VALID_OSTIA_STATUSES = frozenset(OSTIA_STATUS_PORTUGUESE_LABELS)


@dataclass(frozen=True)
class FileNormalization:
    """Auditable description of one result CSV normalization."""

    path: str
    row_count: int
    changed_cell_count: int
    changed_columns: list[str]
    sha256_before: str
    sha256_after: str


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _is_result_csv(path: Path) -> bool:
    if "provenance" in path.parts:
        return False
    return bool(
        CURRENT_RESULT_PATTERN.fullmatch(path.name)
        or LEGACY_RESULT_PATTERN.fullmatch(path.name)
    )


def discover_result_csvs(root: Path) -> list[Path]:
    """Return physical result artifacts, excluding immutable provenance copies."""
    return sorted(path for path in root.rglob("*.csv") if _is_result_csv(path))


def _validate_status(column: str, value: str, path: Path, row_number: int) -> None:
    valid_values = VALID_RESULT_STATUSES if column == "status" else VALID_OSTIA_STATUSES
    if value not in valid_values:
        raise ValueError(
            f"Unknown normalized status {value!r} in {path}, "
            f"row {row_number}, column {column!r}"
        )


def normalize_csv_bytes(path: Path) -> tuple[bytes, FileNormalization]:
    """Return normalized CSV bytes plus an audit record, without writing."""
    original = path.read_bytes()
    text = original.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows:
        record = FileNormalization(
            path=str(path),
            row_count=0,
            changed_cell_count=0,
            changed_columns=[],
            sha256_before=_sha256(original),
            sha256_after=_sha256(original),
        )
        return original, record

    header = rows[0]
    status_indexes = {
        index: column
        for index, column in enumerate(header)
        if column in STATUS_NORMALIZERS
    }
    changed_cell_count = 0
    changed_columns: set[str] = set()
    normalized_rows = [header.copy()]

    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise ValueError(
                f"Malformed CSV row in {path}: row {row_number} has {len(row)} "
                f"fields, expected {len(header)}"
            )
        normalized_row = row.copy()
        for index, column in status_indexes.items():
            raw_value = row[index]
            if not raw_value.strip():
                continue
            normalized = STATUS_NORMALIZERS[column](raw_value)
            if normalized is None:
                continue
            _validate_status(column, normalized, path, row_number)
            if normalized != raw_value:
                normalized_row[index] = normalized
                changed_cell_count += 1
                changed_columns.add(column)
        normalized_rows.append(normalized_row)

    if changed_cell_count == 0:
        normalized = original
    else:
        output = io.StringIO(newline="")
        csv.writer(output, lineterminator="\n").writerows(normalized_rows)
        normalized = output.getvalue().encode("utf-8")

    # Protect all scientific values, identifiers, row order and column order.
    normalized_check = list(
        csv.reader(io.StringIO(normalized.decode("utf-8-sig"), newline=""))
    )
    if len(rows) != len(normalized_check) or rows[0] != normalized_check[0]:
        raise AssertionError(f"CSV structure changed unexpectedly in {path}")
    for row_number, (before, after) in enumerate(
        zip(rows[1:], normalized_check[1:], strict=True), start=2
    ):
        for index, (before_value, after_value) in enumerate(
            zip(before, after, strict=True)
        ):
            if index not in status_indexes and before_value != after_value:
                raise AssertionError(
                    f"Non-status value changed in {path}, row {row_number}, "
                    f"column {header[index]!r}"
                )

    record = FileNormalization(
        path=str(path),
        row_count=max(len(rows) - 1, 0),
        changed_cell_count=changed_cell_count,
        changed_columns=sorted(changed_columns),
        sha256_before=_sha256(original),
        sha256_after=_sha256(normalized),
    )
    return normalized, record


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def normalize_runs(
    root: Path,
    *,
    apply: bool,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    """Inspect or apply status normalization to all recognized result CSVs."""
    try:
        report_root = str(root.resolve().relative_to(REPO_ROOT))
    except ValueError:
        report_root = str(root)
    files = discover_result_csvs(root)
    records: list[FileNormalization] = []
    changed_payloads: list[tuple[Path, bytes]] = []

    for path in files:
        normalized, record = normalize_csv_bytes(path)
        relative_record = FileNormalization(
            path=str(path.relative_to(root)),
            row_count=record.row_count,
            changed_cell_count=record.changed_cell_count,
            changed_columns=record.changed_columns,
            sha256_before=record.sha256_before,
            sha256_after=record.sha256_after,
        )
        records.append(relative_record)
        if record.changed_cell_count:
            changed_payloads.append((path, normalized))

    if apply:
        for path, normalized in changed_payloads:
            _atomic_write(path, normalized)

    changed_records = [record for record in records if record.changed_cell_count]
    report: dict[str, object] = {
        "schema_version": 1,
        "migration": "normalize_run_statuses",
        "mode": "apply" if apply else "inspect",
        "root": report_root,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "inspected_file_count": len(records),
            "changed_file_count": len(changed_records),
            "unchanged_file_count": len(records) - len(changed_records),
            "inspected_row_count": sum(record.row_count for record in records),
            "changed_cell_count": sum(
                record.changed_cell_count for record in changed_records
            ),
        },
        "changed_files": [asdict(record) for record in changed_records],
    }
    if apply and manifest_path is not None:
        _atomic_write(
            manifest_path,
            (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode(),
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize legacy result statuses to stable English codes."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "output/segmentation/runs",
        help="Run tree to inspect (default: output/segmentation/runs).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist changes atomically. Without this flag, only inspect.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "output/segmentation/status_normalization_manifest.json",
        help="Audit manifest written in apply mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = normalize_runs(
        args.root,
        apply=args.apply,
        manifest_path=args.manifest if args.apply else None,
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    if not args.apply:
        print("Inspection only; rerun with --apply to persist the normalization.")
    else:
        print(f"Audit manifest: {args.manifest}")


if __name__ == "__main__":
    main()
