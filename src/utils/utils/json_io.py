"""Funções auxiliares de entrada/saída JSON com validação e erros seguros."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping


def load_json_file(path: str | Path) -> Dict[str, Any]:
    """Carrega um objeto JSON e rejeita conteúdo inválido ou de outro tipo."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo JSON não encontrado: {path}")
    if os.path.isdir(path):
        raise IsADirectoryError(
            f"Caminho aponta para diretório, não arquivo JSON: {path}"
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON inválido em {path} (linha {exc.lineno}, coluna {exc.colno})"
        ) from exc
    except OSError as exc:
        raise OSError(f"Erro ao ler arquivo JSON em {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Conteúdo JSON deve ser um objeto (dict), mas recebeu {type(data).__name__} em {path}"
        )

    return data


def make_json_safe(value: Any) -> Any:
    """Converte valores comuns de pandas/numpy/pathlib para JSON nativo."""
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "as_posix"):
        return value.as_posix()
    if hasattr(value, "tolist"):
        return make_json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def save_json_atomic(payload: Mapping[str, Any], path: str | Path) -> None:
    """Salva JSON por substituição atômica e remove temporários após falhas."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                make_json_safe(dict(payload)), stream, indent=2, ensure_ascii=False
            )
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_json_file(
    data: Dict[str, Any], path: str, indent: int = 2, ensure_ascii: bool = False
) -> None:
    """Salva dados de dicionário em JSON, criando diretórios quando necessário."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            f.write("\n")
    except TypeError as exc:
        raise TypeError(f"Dados não serializáveis para JSON em {path}: {exc}") from exc
    except OSError as exc:
        raise OSError(f"Erro ao salvar JSON em {path}: {exc}") from exc


__all__ = [
    "load_json_file",
    "make_json_safe",
    "save_json_atomic",
    "save_json_file",
]
