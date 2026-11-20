"""Configura o log das execuções e emite avisos opcionais de conclusão."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Callable

LOGGER = logging.getLogger("utils.project.runtime.run_notification")

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


def add_run_file_handler(
    path: str | Path,
    *,
    formatter: logging.Formatter,
    level: int = logging.NOTSET,
) -> logging.FileHandler:
    """Adiciona um handler ao logger raiz sem repetir o mesmo arquivo.

    Args:
        path (str | Path): Caminho do arquivo de log a associar ao logger raiz.
        formatter (logging.Formatter): Formatador aplicado às linhas gravadas no log.
        level (int): Nível mínimo de logging aceito pelo handler.

    Returns:
        logging.FileHandler: Handler existente reutilizado ou novo handler anexado.
    """
    destination = Path(path).resolve()
    root = logging.getLogger()
    # Reaproveita handler existente para evitar linhas duplicadas no mesmo log.
    for handler in root.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == destination
        ):
            handler.setLevel(level)
            handler.setFormatter(formatter)
            return handler
    handler = logging.FileHandler(destination, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    return handler


def _run_notification_command(command: list[str]) -> None:
    """Executa um aviso sem permitir que sua falha afete o pipeline.

    Args:
        command (list[str]): Comando e argumentos do utilitário de notificação.
    """
    # Executa o utilitário com timeout curto e degrada falhas para aviso em log.
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
    # Registra aviso caso o utilitário retorne código diferente de zero.
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
    """Mostra notificação e toca um som de acordo com o estado final do run.

    Args:
        pipeline (str): Nome legível do pipeline executado.
        split (str): Nome do split ou subconjunto processado.
        resolution (str): Perfil de resolução usado na execução.
        run_dir (Path | None): Diretório raiz da execução, ou ``None`` se ausente.
        status (str): Código de estado final mapeado para rótulo, som e ícone.
        details (str | None): Texto complementar opcional exibido na notificação.
    """
    # Monta título e corpo de acordo com o status e aciona avisos visual e sonoro.
    label, sound_id, icon = _STATUS_LABELS.get(status, _STATUS_LABELS["incomplete"])
    title = f"{pipeline}: {label}"
    location = str(run_dir) if run_dir is not None else "run não criado"
    body = f"Split: {split} | Resolução: {resolution}\nSaída: {location}"
    if details:
        body = f"{body}\n{details}"

    # Dispara a notificação visual na área de trabalho e o alerta sonoro.
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


def create_standard_run_dirs(
    run_dir: str | Path,
    *,
    create_visual: bool = False,
) -> dict[str, Path]:
    """Cria a estrutura padrão de diretórios de uma execução e devolve seus caminhos.

    Args:
        run_dir (str | Path): Diretório raiz onde as subpastas da execução
            serão criadas.
        create_visual (bool): Se ``True``, cria também o subdiretório ``visual/``.

    Returns:
        dict[str, Path]: Dicionário com caminhos de ``run_dir``, ``numeric_dir``,
            ``config_dir``, ``visual_dir`` e ``logs_dir``.
    """
    root = Path(run_dir)
    dirs = {
        "run_dir": root,
        "numeric_dir": root / "numeric",
        "config_dir": root / "config",
        "visual_dir": root / "visual",
        "logs_dir": root / "logs",
    }
    for key in ("run_dir", "numeric_dir", "config_dir", "logs_dir"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    if create_visual:
        dirs["visual_dir"].mkdir(parents=True, exist_ok=True)
    return dirs


def notify_run_exception(
    error: BaseException,
    *,
    pipeline: str,
    split: str,
    resolution: str,
    run_dir: Path | None,
    notifier: Callable[..., None] = notify_run_completion,
) -> None:
    """Emite o aviso correspondente a uma interrupção ou falha de execução.

    Args:
        error (BaseException): Exceção ou sinal de interrupção capturado na execução.
        pipeline (str): Nome legível do pipeline executado.
        split (str): Nome do split ou subconjunto em execução.
        resolution (str): Perfil de resolução usado na execução.
        run_dir (Path | None): Diretório raiz da execução, ou ``None`` se ausente.
        notifier (Callable[..., None]): Função notificadora invocada com o status
            de falha ou interrupção.
    """
    notifier(
        pipeline=pipeline,
        split=split,
        resolution=resolution,
        run_dir=run_dir,
        status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
        details="Consulte o terminal e os logs para detalhes.",
    )


__all__ = [
    "add_run_file_handler",
    "create_standard_run_dirs",
    "notify_run_completion",
    "notify_run_exception",
]
