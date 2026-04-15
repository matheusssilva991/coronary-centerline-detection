"""Funções auxiliares de entrada/saída JSON com validação e erros seguros."""

import json
import os


def load_json_file(path: str) -> dict:
    """Carrega um arquivo JSON com validação e mensagens de erro claras."""
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


def save_json_file(
    data: dict, path: str, indent: int = 2, ensure_ascii: bool = False
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
    "save_json_file",
]
