"""Visualizações das etapas de pré-processamento."""

from typing import Any, Iterable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def _resolve_stage_image(
    volume: NDArray,
    center_slice: int,
    mode: Literal["slice", "mip"],
) -> tuple[NDArray, str]:
    """Extrai a imagem 2D e o rótulo descritivo conforme o modo de visualização.

    Args:
        volume (NDArray[Any]): Volume 3D ``(Y, X, Z)`` de uma etapa de
            pré-processamento.
        center_slice (int): Índice da fatia axial central usado no modo
            ``"slice"``.
        mode (Literal["slice", "mip"]): Modo de projeção: ``"slice"`` para fatia
            única ou ``"mip"`` para projeção máxima axial.

    Returns:
        tuple[NDArray[Any], str]: Imagem 2D resultante e texto descritivo do
        modo.
    """
    if mode == "slice":
        return volume[:, :, center_slice], f"fatia {center_slice}"
    return np.max(volume, axis=2), "MIP axial"


def plot_stage(
    preprocessed: dict[int, dict[str, Any]],
    stage_key: str,
    stage_title: str,
    img_id: int,
    mode: Literal["slice", "mip"] = "slice",
    show_title: bool = True,
    show_subtitle: bool = True,
    cmap: str = "gray",
    dpi: int = 100,
) -> None:
    """Plota uma etapa de pré-processamento para um único caso.

    Args:
        preprocessed (dict[int, dict[str, Any]]): Dicionário indexado por
            ``img_id`` contendo os volumes intermediários e ``center_slice``.
        stage_key (str): Chave do volume de etapa dentro de
            ``preprocessed[img_id]``.
        stage_title (str): Nome legível da etapa usado no título principal.
        img_id (int): Identificador do exame em ``preprocessed``.
        mode (Literal["slice", "mip"]): Modo de exibição (``"slice"`` ou
            ``"mip"``).
        show_title (bool): Se ``True``, adiciona o título geral na figura.
        show_subtitle (bool): Se ``True``, adiciona subtítulo com ID e modo.
        cmap (str): Nome do mapa de cores do Matplotlib.
        dpi (int): Resolução da figura em pontos por polegada.
    """
    # Seleciona o volume da etapa e extrai a fatia central ou MIP axial.
    volume = preprocessed[img_id][stage_key]
    center_slice = preprocessed[img_id]["center_slice"]
    image_to_show, mode_text = _resolve_stage_image(volume, center_slice, mode)

    # Plota a imagem 2D e ajusta títulos e margens antes de exibir.
    plt.figure(figsize=(6, 5), dpi=dpi)
    plt.imshow(image_to_show, cmap=cmap, origin="upper")

    if show_subtitle:
        plt.title(f"ID {img_id} - {mode_text}")

    plt.axis("off")

    if show_title:
        plt.suptitle(f"{stage_title} - {mode_text}", fontsize=13)
        plt.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        plt.tight_layout()

    plt.show()
    plt.close()


def plot_pipeline_preprocessing_stages(
    image_data: dict[str, Any],
    slice_idx: int,
    *,
    dpi: int = 110,
) -> None:
    """Plota lado a lado as quatro etapas de pré-processamento do pipeline.

    Args:
        image_data (dict[str, Any]): Dicionário de dados do caso contendo as
            chaves ``image``, ``down_image``, ``threshold_mask`` e
            ``lcc_image``.
        slice_idx (int): Índice da fatia axial exibida nas etapas reamostradas.
        dpi (int): Resolução da figura em pontos por polegada.
    """
    # Recupera os volumes de cada etapa e limita o índice à profundidade bruta.
    image = image_data["image"]
    down_image = image_data["down_image"]
    threshold_mask = image_data["threshold_mask"]
    lcc_image = image_data["lcc_image"]
    raw_slice_idx = min(slice_idx, image.shape[2] - 1)

    # Renderiza as quatro etapas lado a lado para inspeção visual.
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=dpi)
    axes[0].imshow(image[:, :, raw_slice_idx], cmap="gray")
    axes[0].set_title("Imagem original")
    axes[1].imshow(down_image[:, :, slice_idx], cmap="gray")
    axes[1].set_title("Downsampling")
    axes[2].imshow(threshold_mask[:, :, slice_idx], cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Máscara de threshold")
    axes[3].imshow(lcc_image[:, :, slice_idx], cmap="gray")
    axes[3].set_title("Imagem após LCC")
    for axis in axes:
        axis.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_preprocessing_grid(
    preprocessed: dict[int, dict[str, Any]],
    ids_to_plot: Optional[Iterable[int]] = None,
    mode: Literal["slice", "mip", "both"] = "slice",
    show_title: bool = True,
    show_subtitle: bool = True,
    cmap: str = "gray",
    dpi: int = 100,
) -> None:
    """Plota grade comparativa das etapas de pré-processamento por caso.

    Args:
        preprocessed (dict[int, dict[str, Any]]): Dicionário indexado por ID com
            ``down_image``, ``thresh_image``, ``lcc_image`` e ``center_slice``.
        ids_to_plot (Iterable[int] | None): Sequência de IDs a incluir; ``None``
            seleciona todos os IDs ordenados.
        mode (Literal["slice", "mip", "both"]): Modo de visualização por coluna
            (``"slice"``, ``"mip"`` ou ``"both"``).
        show_title (bool): Se ``True``, adiciona título geral acima da grade.
        show_subtitle (bool): Se ``True``, adiciona subtítulo com ID, etapa e
            modo.
        cmap (str): Nome do mapa de cores do Matplotlib.
        dpi (int): Resolução da figura em pontos por polegada.
    """
    # Define a lista de IDs e os modos de exibição por coluna.
    if ids_to_plot is None:
        ids_to_plot = sorted(preprocessed.keys())
    else:
        ids_to_plot = list(ids_to_plot)
    if len(ids_to_plot) == 0:
        raise ValueError("ids_to_plot não pode ser vazio.")

    stages = [
        ("down_image", "Imagem reduzida"),
        ("thresh_image", "Imagem limiarizada"),
        ("lcc_image", "Imagem LCC"),
    ]

    display_modes: list[Literal["slice", "mip"]]
    if mode == "both":
        display_modes = ["slice", "mip"]
    else:
        display_modes = [mode]

    # Dimensiona a grade de subplots e normaliza o arranjo 2D de eixos.
    n_rows = len(stages)
    n_cols = len(ids_to_plot) * len(display_modes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=dpi)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    # Preenche linhas por etapa e colunas por caso/modo de projeção.
    for row, (stage_key, stage_title) in enumerate(stages):
        col = 0
        for img_id in ids_to_plot:
            volume = preprocessed[img_id][stage_key]
            center_slice = preprocessed[img_id]["center_slice"]

            for current_mode in display_modes:
                image_to_show, mode_text = _resolve_stage_image(
                    volume, center_slice, current_mode
                )
                ax = axes[row, col]
                ax.imshow(image_to_show, cmap=cmap, origin="upper")

                if show_subtitle:
                    ax.set_title(f"ID {img_id} - {stage_title} - {mode_text}")

                ax.axis("off")
                col += 1

    # Configura título principal conforme o modo e exibe a grade.
    if show_title:
        if mode == "slice":
            header = "Grid das etapas de pre-processamento (fatia central)"
        elif mode == "mip":
            header = "Grid das etapas de pre-processamento (MIP axial)"
        else:
            header = "Grid das etapas de pre-processamento (fatia central + MIP axial)"
        plt.suptitle(header, fontsize=15)
        plt.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        plt.tight_layout()

    plt.show()
    plt.close(fig)
