"""Compara dois rótulos do treino ImageCAS usando uma predição por exame."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.rebuild_canonical_metrics import build_plan, fingerprint
from utils.experiments.fuzzy_pipeline_comparison import (
    evaluate_prepared_image,
    load_downsampled_case,
    make_image_result_row,
    prepare_image_context,
)
from utils.processing.gpu_utils import use_gpu
from utils.processing.preprocessing import downscale_label
from utils.project.config import IMAGE_GRID_LABEL_PROTOCOL, LEGACY_LABEL_PROTOCOL
from utils.project.results.io import save_dataframe_atomic
from utils.project.runtime.paths import find_repository_root
from utils.utils.json_io import save_json_atomic
from utils.utils.metrics import dice_score
from utils.utils.nifti_io import load_raw_img_and_label


def compare_case(img_id: int, base_path: Path, config: dict, experiment: dict) -> dict:
    """Avalia a mesma máscara arterial final contra os dois rótulos binários."""
    case = load_downsampled_case(img_id, base_path, config)
    context = prepare_image_context(case, config, experiment)
    del case
    row = make_image_result_row(img_id, "baseline", "train", experiment, config=config)
    evaluate_prepared_image(context, row, config, experiment, include_artery_mask=True)
    prediction = row.pop("artery_mask")
    del context
    source, _ = load_raw_img_and_label(str(base_path / f"{img_id}.label.nii.gz"))
    raw_label = np.asarray(source.dataobj, dtype=np.uint8)
    legacy_label = downscale_label(
        raw_label,
        {**config, "LABEL_RESAMPLING": LEGACY_LABEL_PROTOCOL},
        allow_gpu_zoom=True,
    )
    old = float(dice_score(prediction, legacy_label == 1))
    corrected = float(row["dice_artery"])
    return {"dice_legacy": old, "dice_corrected": corrected, "delta": corrected - old}


def main() -> None:
    """Salva os pares, a cobertura e os maiores efeitos, com checkpoint por exame."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-path", type=Path, default=os.environ.get("IMAGECAS_BASE_PATH")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.base_path is None:
        parser.error("Informe --base-path ou IMAGECAS_BASE_PATH.")
    root = find_repository_root(__file__)
    plan = build_plan(root, "mid", "train")
    config = {**plan["effective_config"], "LABEL_RESAMPLING": IMAGE_GRID_LABEL_PROTOCOL}
    ids = plan["ids"]
    canonical = pd.read_csv(root / plan["source_csv"]).set_index("IMG_ID")[
        "artery_dice"
    ]
    for img_id in ids:
        for kind in ("img", "label"):
            path = args.base_path / f"{img_id}.{kind}.nii.gz"
            if not path.is_file():
                raise FileNotFoundError(path)
    manifest = {
        "ids": ids,
        "split": "train",
        "resolution": "mid",
        "source_csv": plan["source_csv"],
        "source_csv_sha256": plan["source_csv_sha256"],
        "source_config_sha256": plan["source_config_sha256"],
        "effective_config": config,
        "experiment": plan["experiment"],
        "comparison_label_protocol": LEGACY_LABEL_PROTOCOL,
        "gpu_available": use_gpu(),
        "implementation_sha256": {
            **plan["implementation_sha256"],
            **{
                path: fingerprint(root / path)
                for path in (
                    "src/experiments/compare_label_resampling.py",
                    "src/utils/processing/preprocessing.py",
                )
            },
        },
    }
    dest = args.output_dir
    manifest_path = dest / "manifest.json"
    pairs_path = dest / "dice_pairs.csv"
    if args.resume:
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(
                "A origem, o código ou a configuração do checkpoint mudou."
            )
    else:
        dest.mkdir(parents=True, exist_ok=False)
        save_json_atomic(manifest, manifest_path)
    rows = {}
    if args.resume and pairs_path.exists():
        for record in pd.read_csv(pairs_path).to_dict("records"):
            img_id = int(record["IMG_ID"])
            if img_id not in ids or img_id in rows:
                raise ValueError("IDs inválidos no checkpoint.")
            rows[img_id] = record
    for index, img_id in enumerate(ids, 1):
        if img_id in rows and rows[img_id]["status"] == "ok":
            continue
        record = {"IMG_ID": img_id, "dice_canonical": float(canonical.loc[img_id])}
        try:
            record.update(
                compare_case(img_id, args.base_path, config, plan["experiment"])
            )
            record["legacy_minus_canonical"] = (
                record["dice_legacy"] - record["dice_canonical"]
            )
            record.update(status="ok", error=None)
        except Exception as error:
            record.update(status="error", error=f"{type(error).__name__}: {error}")
        rows[img_id] = record
        save_dataframe_atomic(pd.DataFrame(rows.values()), pairs_path)
        print(f"[{index}/{len(ids)}] {record}", flush=True)
    frame = pd.DataFrame(rows.values())
    valid = frame.loc[frame["status"].eq("ok")].copy()
    summary: dict[str, Any] = {
        "total": len(ids),
        "valid_pairs": len(valid),
        "failed": len(ids) - len(valid),
    }
    if not valid.empty:
        delta = valid["delta"]
        summary.update(
            mean_legacy=float(valid["dice_legacy"].mean()),
            mean_corrected=float(valid["dice_corrected"].mean()),
            mean_delta=float(delta.mean()),
            median_delta=float(delta.median()),
            increases=int(delta.gt(0).sum()),
            decreases=int(delta.lt(0).sum()),
            ties=int(delta.eq(0).sum()),
            canonical_mismatches=int(
                valid["legacy_minus_canonical"].abs().gt(1e-10).sum()
            ),
            max_abs_canonical_difference=float(
                valid["legacy_minus_canonical"].abs().max()
            ),
            largest_increases=valid.loc[delta.gt(0)]
            .nlargest(3, "delta")[["IMG_ID", "delta"]]
            .to_dict("records"),
            largest_decreases=valid.loc[delta.lt(0)]
            .nsmallest(3, "delta")[["IMG_ID", "delta"]]
            .to_dict("records"),
        )
    save_json_atomic(summary, dest / "summary.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
