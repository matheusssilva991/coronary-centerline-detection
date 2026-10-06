"""Carrega helpers de apresentação dos notebooks sem executar análises."""

import ast
from pathlib import Path
from typing import Any

import nbformat


def load_presentation_helpers(notebook_name: str) -> dict[str, Any]:
    """Executa imports e células marcadas, sem carregar datasets ou runs."""
    root = Path(__file__).resolve().parents[1] / "src" / "eda"
    matches = sorted(root.rglob(f"{notebook_name}.ipynb"))
    if not matches:
        raise FileNotFoundError(f"Notebook não encontrado: {notebook_name}")
    if len(matches) != 1:
        raise ValueError(f"Nome de notebook duplicado: {notebook_name}: {matches}")
    path = matches[0]
    notebook = nbformat.read(path, as_version=4)
    namespace: dict[str, Any] = {"__name__": f"notebook_helpers.{notebook_name}"}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        imports: list[ast.stmt] = [
            node
            for node in ast.parse(cell.source).body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        exec(
            compile(ast.Module(body=imports, type_ignores=[]), str(path), "exec"),
            namespace,
        )
    found = False
    for cell in notebook.cells:
        if cell.cell_type == "code" and "presentation-helpers" in cell.metadata.get(
            "tags", []
        ):
            exec(compile(cell.source, str(path), "exec"), namespace)
            found = True
    if not found:
        raise ValueError(f"Notebook sem célula de apresentação: {path}")
    return namespace
