"""Localiza a raiz do projeto independentemente da profundidade do módulo."""

from pathlib import Path


def find_repository_root(start: str | Path) -> Path:
    """Procura pyproject.toml e src nos ancestrais de um arquivo ou pasta.

    Args:
        start (str | Path): Caminho inicial de arquivo ou diretório no repositório.

    Returns:
        Path: Diretório raiz do repositório contendo ``pyproject.toml`` e ``src/``.
    """
    path = Path(start).expanduser().resolve()
    directory = path.parent if path.is_file() else path
    for candidate in (directory, *directory.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError(f"Raiz do projeto não encontrada a partir de: {path}")
