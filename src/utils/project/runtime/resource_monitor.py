"""Amostra memória do PID atual sem interferir nas etapas científicas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from utils.utils.json_io import save_json_atomic

LOGGER = logging.getLogger(__name__)
RESOURCE_COLUMNS = (
    "ram_peak_observed_bytes",
    "vram_peak_observed_bytes",
    "ram_sample_count",
    "vram_sample_count",
    "resource_monitor_status",
)
DEFAULT_INTERVAL = 0.5
QUERY_TIMEOUT = 1.0


def positive_interval(value: str | float) -> float:
    """Valida um intervalo finito e estritamente positivo."""
    interval = float(value)
    if not math.isfinite(interval) or interval <= 0:
        raise argparse.ArgumentTypeError("o intervalo deve ser finito e maior que zero")
    return interval


def add_resource_monitoring_arguments(parser: argparse.ArgumentParser) -> None:
    """Acrescenta as mesmas opções operacionais às duas CLIs."""
    parser.add_argument(
        "--monitor-resources",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Amostra RSS e VRAM por PID durante cada exame (opcional).",
    )
    parser.add_argument(
        "--resource-sampling-interval",
        type=positive_interval,
        default=None,
        help="Intervalo solicitado entre amostras, em segundos (padrão: 0.5).",
    )


def apply_resource_monitoring_options(config: dict, args: argparse.Namespace) -> None:
    """Preserva opções do snapshot quando a CLI não as sobrescreve."""
    if (
        "RESOURCE_MONITORING" not in config
        and getattr(args, "monitor_resources", None) is None
        and getattr(args, "resource_sampling_interval", None) is None
    ):
        return
    options = dict(config.get("RESOURCE_MONITORING") or {})
    if getattr(args, "monitor_resources", None) is not None:
        options["enabled"] = bool(args.monitor_resources)
    if getattr(args, "resource_sampling_interval", None) is not None:
        options["sampling_interval_seconds"] = args.resource_sampling_interval
    options.setdefault("enabled", False)
    options["sampling_interval_seconds"] = positive_interval(
        options.get("sampling_interval_seconds", DEFAULT_INTERVAL)
    )
    config["RESOURCE_MONITORING"] = options


def read_process_rss(pid: int) -> int:
    """Lê RSS corrente no Linux; não usa o máximo acumulado ru_maxrss."""
    resident_pages = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def parse_process_vram(output: str, pid: int) -> tuple[int, list[str]]:
    """Soma somente alocações do PID nas GPUs, convertendo MiB em bytes."""
    total, devices = 0, set()
    for row in csv.reader(output.splitlines()):
        if not row or row[0].strip() != str(pid):
            continue
        if len(row) != 3:
            raise ValueError("Formato inesperado de VRAM por processo")
        memory = float(row[2].strip())
        if not math.isfinite(memory) or memory < 0:
            raise ValueError("VRAM por processo indisponível")
        total += int(memory * 1024**2)
        devices.add(row[1].strip())
    # Consulta válida sem linha para o PID significa nenhuma alocação reportada.
    return total, sorted(devices)


def query_process_vram(pid: int) -> tuple[int, list[str]]:
    """Consulta todas as GPUs sem interpretar índices de CUDA_VISIBLE_DEVICES."""
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=QUERY_TIMEOUT,
    )
    return parse_process_vram(query.stdout, pid)


def computational_environment() -> dict[str, Any]:
    """Registra hardware básico sem inicializar CUDA nem consultar memória global da GPU."""
    environment: dict[str, Any] = {}
    try:
        environment.update(
            {
                "host": platform.node(),
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "machine": platform.machine(),
                "cpu_count": os.cpu_count(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
        )
        environment["physical_ram_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf(
            "SC_PAGE_SIZE"
        )
        # O modelo facilita distinguir máquinas com a mesma arquitetura.
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() == "model name":
                environment["cpu_model"] = value.strip()
                break
    except Exception as error:
        environment["collection_error"] = f"{type(error).__name__}: {error}"
    return environment


class ResourceMonitor:
    """Mantém máximos observados e contadores independentes para um exame."""

    def __init__(self, config: dict[str, Any]):
        options = config.get("RESOURCE_MONITORING") or {}
        self.enabled = bool(options.get("enabled", False))
        self.interval = positive_interval(
            options.get("sampling_interval_seconds", DEFAULT_INTERVAL)
        )
        self.use_gpu = bool(config.get("USE_GPU", False))
        self.pid = os.getpid()
        self.ram_peak: int | None = None
        self.vram_peak: int | None = None
        self.ram_samples = self.vram_samples = 0
        self.errors: dict[str, int] = {}
        self.devices: set[str] = set()
        self._gpu_query_available = self.use_gpu
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._finished = False

    def _record_error(self, metric: str, error: Exception) -> None:
        message = f"{metric}: {type(error).__name__}: {error}"
        self.errors[message] = self.errors.get(message, 0) + 1

    def _sample(self) -> None:
        try:
            value = read_process_rss(self.pid)
            self.ram_peak = max(self.ram_peak or 0, value)
            self.ram_samples += 1
        except Exception as error:
            self._record_error("ram", error)
        if self._gpu_query_available:
            try:
                value, devices = query_process_vram(self.pid)
                self.vram_peak = max(self.vram_peak or 0, value)
                self.vram_samples += 1
                self.devices.update(devices)
            except Exception as error:
                self._record_error("vram", error)
                # Ausência permanente não deve criar subprocessos a cada tick.
                if isinstance(error, FileNotFoundError):
                    self._gpu_query_available = False

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval):
            self._sample()

    def start(self) -> None:
        """Obtém amostra inicial e inicia uma única thread, somente se habilitado."""
        if not self.enabled:
            return
        self._sample()
        try:
            self._thread = threading.Thread(
                target=self._run, name="resource-monitor", daemon=True
            )
            self._thread.start()
        except Exception as error:
            self._record_error("monitor", error)
            self._thread = None

    def finish(self, result: dict[str, Any]) -> None:
        """Encerra consultas, obtém amostra final e anexa somente diagnósticos."""
        if not self._finished:
            self._stop_event.set()
            if self._thread is not None:
                # Toda consulta tem timeout; join impede thread vazando ao próximo exame.
                self._thread.join()
            if self.enabled:
                self._sample()
            self._finished = True
        status = "disabled"
        if self.enabled:
            if not self.ram_samples and (not self.use_gpu or not self.vram_samples):
                status = "unavailable"
            elif (
                self.errors
                or not self.ram_samples
                or (self.use_gpu and not self.vram_samples)
            ):
                status = "partial"
            else:
                status = "ok"
        result.update(
            {
                "ram_peak_observed_bytes": self.ram_peak,
                "vram_peak_observed_bytes": self.vram_peak,
                "ram_sample_count": self.ram_samples if self.enabled else None,
                "vram_sample_count": self.vram_samples
                if self.enabled and self.use_gpu
                else None,
                "resource_monitor_status": status,
            }
        )
        if self.enabled:
            result["_resource_monitor_details"] = {
                "pid": self.pid,
                "sampling_interval_seconds": self.interval,
                "gpu_device_uuids": sorted(self.devices),
                "errors": self.errors,
                "environment": computational_environment(),
            }


def read_resource_diagnostics(path: Path) -> dict[str, Any]:
    """Carrega diagnósticos existentes sem transformar erro de coleta em falha científica."""
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Diagnósticos de recursos devem ser um objeto JSON")
        return payload
    except (OSError, ValueError) as error:
        LOGGER.warning("Não foi possível ler metadados de recursos: %s", error)
        return {"metadata_error": str(error)}


def persist_resource_diagnostics(result: dict, path: Path, exam_key: str) -> None:
    """Atualiza somente o exame processado, preservando sessões anteriores."""
    details = result.pop("_resource_monitor_details", None)
    if details is None:
        return
    try:
        payload = read_resource_diagnostics(path)
        if "metadata_error" in payload:
            return  # Nunca sobrescreve um arquivo ilegível silenciosamente.
        details = dict(details)
        environment = details.pop("environment")
        identifier = hashlib.sha256(
            json.dumps(environment, sort_keys=True).encode()
        ).hexdigest()[:16]
        payload.setdefault("environments", {})[identifier] = environment
        payload.setdefault("exams", {})[exam_key] = {
            **details,
            "environment_id": identifier,
        }
        save_json_atomic(payload, path)
    except Exception as error:
        LOGGER.warning(
            "Não foi possível salvar metadados de recursos de %s: %s", exam_key, error
        )


def resource_metadata(config: dict, diagnostics: dict | None = None) -> dict[str, Any]:
    """Descreve protocolo, ambiente e erros, sem estimar picos ausentes."""
    diagnostics = diagnostics or {}
    return {
        **(
            config.get("RESOURCE_MONITORING")
            or {"enabled": False, "sampling_interval_seconds": DEFAULT_INTERVAL}
        ),
        "scope": "current_process_pid_per_exam_including_enabled_visuals",
        "ram_method": "observed_resident_set_size_from_linux_proc_statm",
        "vram_method": "observed_pid_allocations_nvidia_smi_all_devices",
        "units": "bytes",
        "query_timeout_seconds": QUERY_TIMEOUT,
        "limitations": "Amostragem pode perder picos; inclui memória retida entre exames, não consumo incremental.",
        "environments": diagnostics.get("environments", {}),
        "gpu_device_uuids": sorted(
            {
                uuid
                for value in diagnostics.get("exams", {}).values()
                for uuid in value.get("gpu_device_uuids", [])
            }
        ),
        "errors_by_exam": {
            key: value["errors"]
            for key, value in diagnostics.get("exams", {}).items()
            if value.get("errors")
        },
        "metadata_error": diagnostics.get("metadata_error"),
    }
