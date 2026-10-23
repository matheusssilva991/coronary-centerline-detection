"""Mede RAM e VRAM de um exame em processos independentes para Mid e High."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# O monitor fica livre de imports do projeto: project.__init__ inicializa CUDA.
# Somente o worker deve criar contexto CUDA, após selecionar a GPU por UUID.
ROOT = SRC_DIR.parent
DEFAULT_OUTPUT = ROOT / "output/segmentation/analysis/pipeline_resource_benchmark/runs"
DEFAULT_CONFIG = ROOT / "config/article_cbeb_ablation.json"


def write_json(path: Path, data: Any) -> None:
    """Grava metadados e resultados sem sobrescrever parcialmente o JSON."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def query_smi(fields: str, gpu_index: int, *, processes: bool = False) -> str:
    """Consulta a GPU física sem usar a memória global como memória do worker."""
    query = "--query-compute-apps" if processes else "--query-gpu"
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            f"{query}={fields}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    return completed.stdout.strip()


def process_vram_bytes(output: str, pid: int) -> int:
    """Lê somente a VRAM atribuída ao PID, convertendo MiB em bytes."""
    total = 0
    for row in csv.reader(output.splitlines()):
        if len(row) != 2 or row[0].strip() != str(pid):
            continue
        memory = row[1].strip()
        if memory in {"N/A", "[N/A]", "Not Supported", "[Not Supported]"}:
            raise ValueError("A GPU não informa VRAM por processo.")
        total += int(float(memory) * 1024**2)
    return total


def build_parser() -> argparse.ArgumentParser:
    """Cria as opções de um benchmark por exame, sem disparar lotes científicos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", type=int, default=642)
    parser.add_argument("--base-path", type=Path, default=None)
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resolutions", default="mid,high")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--worker-resolution", choices=["mid", "high"], help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    return parser


def run_worker(args: argparse.Namespace) -> None:
    """Executa o pipeline real; o high-water RSS inclui o ciclo de vida do worker."""
    from utils.processing import gpu_utils
    from utils.project.config import scale_config_to_resolution
    from utils.project.results import make_json_safe
    from utils.segmentation.pipeline.orchestration import process_image
    from utils.utils.nifti_io import load_raw_img_and_label

    if not gpu_utils.GPU_AVAILABLE or gpu_utils.cp is None:
        raise RuntimeError(
            "GPU indisponível; benchmark não admite fallback silencioso para CPU."
        )
    config = json.loads(args.config_file.read_text())
    config["USE_GPU"] = True
    config["DOWNSCALE_FACTORS"] = (
        [1, 1, 1] if args.worker_resolution == "high" else [2, 2, 1]
    )
    config = scale_config_to_resolution(config)
    # O cabeçalho não carrega os dados completos; a execução abaixo mede o I/O real.
    image, _ = load_raw_img_and_label(
        str(args.base_path / f"{args.image_id}.img.nii.gz")
    )
    shape = list(image.shape)
    del image
    gpu_utils.cp.cuda.runtime.deviceSynchronize()
    started = time.perf_counter()
    result = process_image(args.image_id, config, args.base_path)
    gpu_utils.cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - started
    report = {
        "image_id": args.image_id,
        "resolution": args.worker_resolution,
        "pid": os.getpid(),
        "input_shape": shape,
        "elapsed_seconds": elapsed,
        "peak_ram_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "ram_metric": "Linux ru_maxrss: RSS máximo do worker, incluindo imports e contexto CUDA",
        "gpu_available": True,
        "effective_config": make_json_safe(config),
        "result": make_json_safe(result),
    }
    write_json(args.worker_result, report)
    if result.get("error") or result.get("ostia_error"):
        raise RuntimeError(
            f"Segmentação falhou; métricas diagnósticas preservadas: {result}"
        )


def benchmark(args: argparse.Namespace) -> Path | None:
    """Monitora workers sequenciais, com logs e amostras de VRAM por PID."""
    names = [name.strip() for name in args.resolutions.split(",") if name.strip()]
    if not names or len(set(names)) != len(names) or set(names) - {"mid", "high"}:
        raise ValueError("Selecione resoluções únicas entre mid e high.")
    if args.interval < 0.05:
        raise ValueError("O intervalo de amostragem deve ser >= 0,05 s.")
    dataset = args.base_path or os.environ.get("IMAGECAS_BASE_PATH")
    if not dataset:
        raise ValueError("Informe --base-path ou IMAGECAS_BASE_PATH.")
    args.base_path = Path(dataset).expanduser().resolve()
    args.config_file = args.config_file.expanduser().resolve()
    for suffix in ["img", "label"]:
        if not (args.base_path / f"{args.image_id}.{suffix}.nii.gz").is_file():
            raise FileNotFoundError(f"Arquivo ausente: {args.image_id}.{suffix}.nii.gz")
    if args.dry_run:
        print(
            f"Exame {args.image_id}; resoluções {names}; processos separados; nenhuma segmentação iniciada."
        )
        return None
    gpu_output = query_smi("uuid,name,driver_version,memory.total", args.gpu_index)
    gpu_row = next(csv.reader(gpu_output.splitlines()))
    if len(gpu_row) != 4:
        raise ValueError("Resposta inesperada de nvidia-smi para a GPU selecionada.")
    gpu_uuid, gpu_name, driver, total_mib = [x.strip() for x in gpu_row]
    run = args.output_root.expanduser().resolve() / datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S_%f"
    )
    run.mkdir(parents=True, exist_ok=False)
    write_json(
        run / "run_config.json",
        {
            "image_id": args.image_id,
            "base_path": str(args.base_path),
            "resolutions": names,
            "config_file": str(args.config_file),
            "config_sha256": hashlib.sha256(args.config_file.read_bytes()).hexdigest(),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gpu_uuid": gpu_uuid,
            "gpu_name": gpu_name,
            "driver_version": driver,
            "gpu_total_mib": float(total_mib),
            "vram_sampling_interval_seconds": args.interval,
            "vram_metric": "máximo observado de VRAM por PID via nvidia-smi; não memória total da GPU",
            "time_boundary": "process_image: I/O, pré-processamento, aorta, óstios, artérias e métricas; sem imports",
        },
    )
    print(f"Resultados: {run}", flush=True)
    reports = []
    for name in names:
        target = run / f"{name}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--image-id",
            str(args.image_id),
            "--base-path",
            str(args.base_path),
            "--config-file",
            str(args.config_file),
            "--worker-resolution",
            name,
            "--worker-result",
            str(target),
        ]
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": gpu_uuid,
            "PYTHONUNBUFFERED": "1",
        }
        peak = 0
        samples = []
        errors = []
        print(f"Iniciando {name}: exame {args.image_id}, GPU {gpu_name}", flush=True)
        with (run / f"{name}.log").open("w") as log:
            worker = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=environment
            )
            started = time.perf_counter()
            while worker.poll() is None:
                tick = time.perf_counter()
                try:
                    value = process_vram_bytes(
                        query_smi(
                            "pid,used_gpu_memory", args.gpu_index, processes=True
                        ),
                        worker.pid,
                    )
                    peak = max(peak, value)
                    samples.append({"seconds": tick - started, "vram_bytes": value})
                except (subprocess.SubprocessError, ValueError) as exc:
                    errors.append(str(exc))
                time.sleep(max(0, args.interval - (time.perf_counter() - tick)))
        with (run / f"{name}_vram_samples.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["seconds", "vram_bytes"])
            writer.writeheader()
            writer.writerows(samples)
        report = (
            json.loads(target.read_text()) if target.exists() else {"resolution": name}
        )
        report.update(
            {
                "exit_code": worker.returncode,
                "peak_vram_observed_bytes": peak if peak else None,
                "vram_samples": len(samples),
                "vram_sampling_errors": errors,
            }
        )
        write_json(target, report)
        reports.append(report)
        write_json(run / "results.json", reports)
        row_result = report.get("result", {})
        print(
            f"{name}: saída={worker.returncode}; Dice={row_result.get('dice_artery')}; "
            f"RAM={report.get('peak_ram_bytes')}; VRAM={report['peak_vram_observed_bytes']} bytes",
            flush=True,
        )
        if worker.returncode != 0 or not peak:
            raise RuntimeError(
                f"Benchmark {name} incompleto. Consulte {run / (name + '.log')}."
            )
    rows = []
    for report in reports:
        result = report["result"]
        rows.append(
            {
                "image_id": args.image_id,
                "resolution": report["resolution"],
                "slices": report["input_shape"][2],
                "dice": result.get("dice_artery"),
                "ostia_status": result.get("ostia_status"),
                "elapsed_seconds": report["elapsed_seconds"],
                "peak_ram_bytes": report["peak_ram_bytes"],
                "peak_ram_gib": report["peak_ram_bytes"] / 1024**3,
                "peak_vram_observed_bytes": report["peak_vram_observed_bytes"],
                "peak_vram_observed_gib": report["peak_vram_observed_bytes"] / 1024**3,
            }
        )
    with (run / "resource_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return run


def main() -> None:
    """Executa somente o exame explicitamente escolhido para o benchmark."""
    args = build_parser().parse_args()
    if args.worker_resolution:
        run_worker(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    main()
