"""Emite avisos opcionais ao término dos pipelines em uma sessão gráfica."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path


LOGGER = logging.getLogger(__name__)

_STATUS_LABELS = {
    "complete": ("concluído", "complete", "dialog-information"),
    "complete_with_warnings": (
        "concluído com aviso",
        "dialog-warning",
        "dialog-warning",
    ),
    "complete_with_errors": ("concluído com erros", "dialog-warning", "dialog-warning"),
    "incomplete": ("incompleto", "dialog-warning", "dialog-warning"),
    "failed": ("falhou", "dialog-error", "dialog-error"),
    "interrupted": ("interrompido", "dialog-error", "dialog-error"),
}


def _run_notification_command(command: list[str]) -> None:
    """Executa um aviso sem permitir que sua falha afete o pipeline."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception as error:
        LOGGER.warning("Aviso indisponível (%s): %s", command[0], error)
        return
    if completed.returncode != 0:
        LOGGER.warning(
            "Aviso indisponível (%s, código %s): %s",
            command[0],
            completed.returncode,
            completed.stderr.strip(),
        )


def notify_run_completion(
    *,
    pipeline: str,
    split: str,
    resolution: str,
    run_dir: Path | None,
    status: str,
    details: str | None = None,
) -> None:
    """Mostra notificação e toca um som de acordo com o estado final do run."""
    label, sound_id, icon = _STATUS_LABELS.get(status, _STATUS_LABELS["incomplete"])
    title = f"{pipeline}: {label}"
    location = str(run_dir) if run_dir is not None else "run não criado"
    body = f"Split: {split} | Resolução: {resolution}\nSaída: {location}"
    if details:
        body = f"{body}\n{details}"

    _run_notification_command(
        [
            "notify-send",
            "--app-name=Coronary Pipeline",
            f"--icon={icon}",
            title,
            body,
        ]
    )
    _run_notification_command(["canberra-gtk-play", f"--id={sound_id}"])


__all__ = ["notify_run_completion"]
