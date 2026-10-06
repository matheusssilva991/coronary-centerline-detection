"""Compartilha a criação dos arquivos de log das execuções."""

import logging
from pathlib import Path


def add_run_file_handler(
    path: str | Path,
    *,
    formatter: logging.Formatter,
    level: int = logging.NOTSET,
) -> logging.FileHandler:
    """Adiciona um handler ao logger raiz sem repetir o mesmo arquivo."""
    destination = Path(path).resolve()
    root = logging.getLogger()
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
