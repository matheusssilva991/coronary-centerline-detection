"""Funções auxiliares de entrada/saída JSON com validação e erros seguros."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping


def load_json_file(path: str | Path) -> Dict[str, Any]:
    """Carrega um objeto JSON e rejeita conteúdo inválido ou de outro tipo.

    Args:
        path (str | Path): Caminho do arquivo JSON contendo um objeto raiz.

    Returns:
        Dict[str, Any]: Dicionário desserializado do arquivo JSON.
    """
    # Valida existência e garante que o caminho não aponta para um diretório.
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo JSON não encontrado: {path}")
    if os.path.isdir(path):
        raise IsADirectoryError(
            f"Caminho aponta para diretório, não arquivo JSON: {path}"
        )

    # Lê e decodifica o JSON reportando linha e coluna em caso de sintaxe inválida.
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON inválido em {path} (linha {exc.lineno}, coluna {exc.colno})"
        ) from exc
    except OSError as exc:
        raise OSError(f"Erro ao ler arquivo JSON em {path}: {exc}") from exc

    # Rejeita JSON cuja raiz não seja um objeto (dict) para manter o contrato.
    if not isinstance(data, dict):
        raise ValueError(
            f"Conteúdo JSON deve ser um objeto (dict), mas recebeu {type(data).__name__} em {path}"
        )

    return data


def make_json_safe(value: Any) -> Any:
    """Converte valores comuns de pandas/numpy/pathlib para JSON nativo.

    Args:
        value (Any): Objeto ou estrutura aninhada (dict, lista, Path, ndarray ou
            escalar NumPy).

    Returns:
        Any: Estrutura equivalente composta por tipos nativos serializáveis em JSON.
    """
    # Percorre coleções recursivamente normalizando chaves de dicionário para str.
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    # Adapta objetos Path, ndarrays e escalares NumPy/pandas para tipos Python nativos.
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
    """Salva JSON no destino por arquivo temporário e substituição atômica.

    Args:
        payload (Mapping[str, Any]): Mapeamento a serializar (convertido previamente
            via ``make_json_safe``).
        path (str | Path): Caminho do arquivo JSON de destino.
    """
    # Garante o diretório pai antes de criar o arquivo temporário no mesmo filesystem.
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        # Escreve no arquivo temporário e substitui o destino de forma atômica.
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
        # Limpa o temporário residual caso ocorra falha antes do replace.
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_json_file(
    data: Dict[str, Any], path: str, indent: int = 2, ensure_ascii: bool = False
) -> None:
    """Serializa e grava um dicionário em arquivo JSON, criando diretórios faltantes.

    Args:
        data (Dict[str, Any]): Dicionário de dados serializável em JSON.
        path (str): Caminho do arquivo JSON de saída.
        indent (int): Número de espaços de indentação na saída formatada.
        ensure_ascii (bool): Se ``True``, escapa caracteres não ASCII na saída.
    """
    # Garante o diretório de saída antes de abrir o arquivo de destino.
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Serializa os dados com quebra de linha final e propaga erros com contexto.
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
