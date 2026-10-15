"""Executa ablações controladas do pipeline refinado, sem reajustar parâmetros."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Permite executar o driver diretamente, a partir de qualquer diretório.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.experiments.fuzzy_pipeline_comparison import run_image  # noqa: E402
from utils.experiments.sweep_common import csv_safe, write_json  # noqa: E402
from utils.processing import gpu_utils  # noqa: E402
from utils.project.runtime.paths import find_repository_root  # noqa: E402

REPO_ROOT = find_repository_root(Path(__file__))
DEFAULT_CONFIG = REPO_ROOT / "config/article_cbeb_ablation.json"
DEFAULT_SPLITS = REPO_ROOT / "config/imagecas_splits.json"
DEFAULT_OUTPUT = (
    REPO_ROOT / "output/segmentation/analysis/pipeline_component_ablation/runs"
)
VARIANTS = {
    "full_pipeline": {
        "threshold_mode": "normal",
        "artery_method": "region_growing",
        "aorta_segmentation_mode": "level_set",
        "vesselness_mode": "dual",
    },
    "no_level_set": {
        "threshold_mode": "normal",
        "artery_method": "region_growing",
        "aorta_segmentation_mode": "circle_mask",
        "aorta_circle_radius_factor": 1.0,
        "vesselness_mode": "dual",
    },
    "single_vesselness": {
        "threshold_mode": "normal",
        "artery_method": "region_growing",
        "aorta_segmentation_mode": "level_set",
        "vesselness_mode": "single_artery",
    },
}


def build_parser() -> argparse.ArgumentParser:
    """Cria as opções do experimento; o padrão cobre todo o teste fixo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--ids", help="IDs separados por vírgula, pertencentes ao split."
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-path", type=Path, default=None)
    gpu = parser.add_mutually_exclusive_group()
    gpu.add_argument("--gpu", dest="use_gpu", action="store_true", default=None)
    gpu.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_path(path: Path) -> Path:
    """Resolve caminhos relativos a partir do repositório."""
    path = path.expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def select_image_ids(args: argparse.Namespace) -> list[int]:
    """Seleciona a coorte fixa sem carregar imagens, inclusive no modo seco."""
    splits = json.loads(resolve_path(args.split_config).read_text())["splits"]
    all_ids = [int(value) for key in ("train", "val", "test") for value in splits[key]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("O arquivo de splits contém IDs duplicados.")
    cohort = [int(value) for value in splits[args.split]]
    if args.start_index < 0 or (args.sample_size is not None and args.sample_size <= 0):
        raise ValueError(
            "Amostra deve ser positiva; índice inicial não pode ser negativo."
        )
    if args.ids is not None:
        if args.start_index != 0 or args.sample_size is not None:
            raise ValueError("--ids não pode ser combinado com amostragem.")
        selected = [
            int(value.strip()) for value in args.ids.split(",") if value.strip()
        ]
        if not set(selected).issubset(cohort):
            raise ValueError("Todos os IDs devem pertencer ao split selecionado.")
    else:
        stop = None if args.sample_size is None else args.start_index + args.sample_size
        selected = cohort[args.start_index : stop]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("A seleção precisa conter IDs únicos e não pode ser vazia.")
    return selected


def prepare_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Congela configuração, modos experimentais e coorte para validar retomadas."""
    names = [name.strip() for name in args.variants.split(",") if name.strip()]
    if not names or len(names) != len(set(names)) or set(names) - VARIANTS.keys():
        raise ValueError(f"Selecione variantes únicas entre {list(VARIANTS)}.")
    config = json.loads(resolve_path(args.config_path).read_text())
    if args.use_gpu is not None:
        config["USE_GPU"] = bool(args.use_gpu)
    if config["DOWNSCALE_FACTORS"] != [2, 2, 1]:
        raise ValueError("Este estudo usa a resolução reduzida [2, 2, 1].")
    if (
        config.get("THRESHOLDING", {}).get("method") != "normal"
        or config.get("ARTERY_SEGMENTATION", {}).get("method") != "region_growing"
    ):
        raise ValueError("A referência deve usar threshold normal e Region Growing.")
    dataset = args.base_path or os.environ.get("IMAGECAS_BASE_PATH")
    base_path = str(resolve_path(Path(dataset))) if dataset else None
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return {
        "schema_version": 1,
        "split": args.split,
        "resolution": "mid",
        "ids": select_image_ids(args),
        "base_path": base_path,
        "config_sha256": digest,
        "effective_config": config,
        "variants": {name: VARIANTS[name] for name in names},
        "timing_mode": "independent_end_to_end_per_volume",
        "peak_ram_measured": False,
        "peak_vram_measured": False,
    }


def validate_resume(existing: dict[str, Any], requested: dict[str, Any]) -> None:
    """Rejeita mudanças de configuração, coorte, dataset ou variantes."""
    for key, value in requested.items():
        if existing.get(key) != value:
            raise ValueError(f"Retomada incompatível: {key} mudou.")


def _truth(value: Any) -> bool:
    """Lê flags atuais e serializadas sem converter NaN em sucesso."""
    return str(value).strip().lower() in {"true", "yes", "sim", "1"}


def summarize_results(
    rows: list[dict[str, Any]], manifest: dict[str, Any]
) -> pd.DataFrame:
    """Calcula médias operacionais sem descartar falhas nem contar casos pendentes."""
    summaries = []
    for name in manifest["variants"]:
        frame = pd.DataFrame([row for row in rows if row["variant"] == name])
        if frame.empty:
            continue
        dice = pd.to_numeric(frame["dice_artery"], errors="raise")
        if not dice.dropna().between(0, 1).all():
            raise ValueError("Dice deve estar no intervalo [0, 1].")
        errors = frame["error"].notna() & frame["error"].astype(str).str.strip().ne("")
        if (dice.isna() & ~errors).any():
            raise ValueError(
                "Dice ausente sem erro explícito; resumo não pode atribuir zero."
            )
        operational = dice.fillna(0)
        success = frame["ostia_success"].map(_truth)
        elapsed = pd.to_numeric(frame["elapsed_seconds"], errors="raise")
        summaries.append(
            {
                "variant": name,
                "images": len(frame),
                "planned_images": len(manifest["ids"]),
                "complete": len(frame) == len(manifest["ids"]),
                "ostia_success_n": int(success.sum()),
                "ostia_success_percent": float(success.mean() * 100),
                "mean_dice_end_to_end": float(operational.mean()),
                "mean_dice_conditional": float(operational[success].mean())
                if success.any()
                else None,
                "valid_dice_n": int(dice.notna().sum()),
                "error_count": int(errors.sum()),
                "runtime_seconds": float(elapsed.sum()),
                "mean_seconds_per_volume": float(elapsed.mean()),
                "peak_ram_bytes": None,
                "peak_vram_bytes": None,
            }
        )
    return pd.DataFrame(summaries)


def load_completed_rows(
    run_dir: Path, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    """Valida os registros existentes, incluindo erros já concluídos."""
    path = run_dir / "results/image_results.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    required = {
        "variant",
        "IMG_ID",
        "dice_artery",
        "ostia_success",
        "error",
        "elapsed_seconds",
    }
    if not required.issubset(frame.columns):
        raise ValueError("CSV de retomada sem as colunas obrigatórias.")
    if frame.duplicated(["variant", "IMG_ID"]).any():
        raise ValueError("Retomada contém pares variante/ID duplicados.")
    if (
        frame.IMG_ID.isna().any()
        or not frame.IMG_ID.isin(manifest["ids"]).all()
        or not frame.variant.isin(manifest["variants"]).all()
    ):
        raise ValueError("CSV contém registros fora da coorte/variantes.")
    times = pd.to_numeric(frame.elapsed_seconds, errors="raise")
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("CSV contém tempos inválidos.")
    rows = frame.to_dict("records")
    summarize_results(rows, manifest)
    return rows


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    """Substitui cada CSV somente após completar sua gravação."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    csv_safe(frame).to_csv(temporary, index=False)
    temporary.replace(path)


def save_checkpoint(
    run_dir: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]
) -> None:
    """Persiste resultados por exame e a tabela resumida, inclusive durante o lote."""
    summary = summarize_results(rows, manifest)
    _atomic_csv(pd.DataFrame(rows), run_dir / "results/image_results.csv")
    _atomic_csv(summary, run_dir / "summary/ablation_summary.csv")
    lines = [
        "| Configuração | N | Sucesso óstios | Dice end-to-end | Dice condicionado | s/volume |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary.to_dict("records"):
        conditional = row["mean_dice_conditional"]
        conditional_text = f"{conditional:.4f}" if pd.notna(conditional) else "—"
        lines.append(
            f"| {row['variant']} | {row['images']}/{row['planned_images']} "
            f"| {row['ostia_success_percent']:.2f}% | {row['mean_dice_end_to_end']:.4f} "
            f"| {conditional_text} | {row['mean_seconds_per_volume']:.2f} |"
        )
    lines.append(
        "\nGlobal inclui erros sem saída como zero; registros originais são preservados. "
        "N mostra concluídos/planejados; médias parciais não incluem exames pendentes. "
        "Tempo inclui carregamento e todas as etapas, sem compartilhar intermediários. "
        "Picos de RAM e VRAM não foram medidos."
    )
    target = run_dir / "summary/ablation_table.md"
    temporary = target.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(target)


def synchronize_gpu(config: dict[str, Any]) -> None:
    """Espera operações CUDA para medir o processamento completo do volume."""
    cupy = gpu_utils.cp
    if config.get("USE_GPU", False) and gpu_utils.GPU_AVAILABLE and cupy is not None:
        cupy.cuda.runtime.deviceSynchronize()


def execute(args: argparse.Namespace) -> Path | None:
    """Executa ou inspeciona o estudo, sem alterar resultados históricos."""
    manifest = prepare_manifest(args)
    resume = resolve_path(args.resume_dir) if args.resume_dir else None
    if resume:
        if args.run_name is not None:
            raise ValueError("--run-name não pode ser combinado com --resume-dir.")
        saved = json.loads((resume / "run_config.json").read_text())
        validate_resume(saved, manifest)
        rows = load_completed_rows(resume, manifest)
    else:
        rows = []
    print(f"Split: {manifest['split']}; exames: {len(manifest['ids'])}")
    print(
        f"Variantes: {list(manifest['variants'])}; configuração: {manifest['config_sha256']}"
    )
    if args.dry_run:
        print(
            f"Modo seco: {len(rows)} pares concluídos; nenhuma imagem carregada ou arquivo criado."
        )
        return resume
    if manifest["base_path"] is None:
        raise ValueError("Informe --base-path ou IMAGECAS_BASE_PATH.")
    base_path = Path(manifest["base_path"])
    for image_id in manifest["ids"]:
        for kind in ("img", "label"):
            if not (base_path / f"{image_id}.{kind}.nii.gz").is_file():
                raise FileNotFoundError(f"Arquivo ausente: {image_id}.{kind}.nii.gz")
    if resume:
        run_dir = resume
    else:
        name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        if (
            name in {".", ".."}
            or Path(name).name != name
            or "/" in name
            or "\\" in name
        ):
            raise ValueError("--run-name precisa ser um nome simples de pasta.")
        run_dir = resolve_path(args.output_root) / name
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "run_config.json", manifest)
        write_json(
            run_dir / "config/split_ids.json", {manifest["split"]: manifest["ids"]}
        )
        for name, experiment in manifest["variants"].items():
            write_json(
                run_dir / f"config/{name}.json",
                {"pipeline": manifest["effective_config"], "experiment": experiment},
            )
        write_json(
            run_dir / "runtime_environment.json",
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "hostname": platform.node(),
                "gpu_requested": manifest["effective_config"].get("USE_GPU", False),
                "gpu_available": bool(gpu_utils.GPU_AVAILABLE),
                "peak_ram_measured": False,
                "peak_vram_measured": False,
            },
        )
    done = {(row["variant"], int(row["IMG_ID"])) for row in rows}
    config = manifest["effective_config"]
    for name, experiment in manifest["variants"].items():
        for image_id in manifest["ids"]:
            if (name, image_id) in done:
                continue
            synchronize_gpu(config)
            started = time.perf_counter()
            row = run_image(
                image_id, name, manifest["split"], base_path, config, experiment
            )
            synchronize_gpu(config)
            row["elapsed_seconds"] = time.perf_counter() - started
            rows.append(row)
            done.add((name, image_id))
            save_checkpoint(run_dir, rows, manifest)
            print(
                f"{name}: IMG_ID={image_id}; Dice={row['dice_artery']}; erro={row['error']}"
            )
    if rows:
        save_checkpoint(run_dir, rows, manifest)
    print(f"Resultados: {run_dir}")
    return run_dir


def main() -> None:
    """Valida os argumentos e inicia somente a execução solicitada."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        execute(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
