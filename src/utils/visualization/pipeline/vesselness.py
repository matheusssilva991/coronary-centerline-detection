"""Visualizações e cálculo auxiliar de mapas de vesselness."""

from typing import Any, Iterable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from utils.visualization.images.image_slices import plot_mip_projection


def display_vesselness_summary(
    vesselness: NDArray,
    *,
    label: str,
    title: str,
    cmap: str = "gray",
) -> None:
    """Imprime estatísticas descritivas e plota a MIP de um mapa de vesselness.

    Args:
        vesselness (NDArray[Any]): Volume 3D com escores de vesselness.
        label (str): Rótulo impresso antes das estatísticas no terminal.
        title (str): Título exibido na figura de projeções MIP.
        cmap (str): Nome do mapa de cores passado para ``plot_mip_projection``.
    """
    print(
        f"{label}: "
        f"min={vesselness.min():.4f}, "
        f"max={vesselness.max():.4f}, "
        f"média={vesselness.mean():.4f}"
    )
    plot_mip_projection(vesselness, title=title, cmap=cmap)


def compute_vesselness_maps(
    preprocessed: dict[int, dict[str, Any]],
    ids_to_plot: Optional[Iterable[int]] = None,
    ostia_config: Optional[dict[str, Any]] = None,
    artery_config: Optional[dict[str, Any]] = None,
) -> dict[int, dict[str, NDArray]]:
    """Computa mapas de vesselness para óstios e artérias a partir da imagem LCC.

    Args:
        preprocessed (dict[int, dict[str, Any]]): Dicionário indexado por ID
            contendo ao menos a chave ``lcc_image``.
        ids_to_plot (Iterable[int] | None): IDs dos casos a processar; ``None``
            processa todos os IDs disponíveis em ordem crescente.
        ostia_config (dict[str, Any] | None): Hiperparâmetros do filtro de
            Frangi para óstios (``sigmas``, ``alpha``, ``beta``, ``gamma`` e
            ``normalization``); ``None`` usa valores padrão.
        artery_config (dict[str, Any] | None): Hiperparâmetros do filtro de
            Frangi para artérias (``sigmas``, ``alpha``, ``beta``, ``gamma`` e
            ``normalization``); ``None`` usa valores padrão.

    Returns:
        dict[int, dict[str, NDArray[Any]]]: Dicionário por ID com os volumes 3D
        nas chaves ``"vesselness_ostia"`` e ``"vesselness_artery"``.
    """
    from utils.processing.frangi import get_vesselness

    # Define IDs alvo e parâmetros padrão de Frangi para óstios e artérias.
    if ids_to_plot is None:
        ids_to_plot = sorted(preprocessed.keys())

    if ostia_config is None:
        ostia_config = {
            "sigmas": [2.5],
            "alpha": 0.5,
            "beta": 1.0,
            "gamma": 30,
            "normalization": "none",
        }

    if artery_config is None:
        artery_config = {
            "sigmas": [1.5, 2.0, 2.5, 3.0],
            "alpha": 0.5,
            "beta": 0.5,
            "gamma": 55,
            "normalization": "none",
        }

    # Calcula os dois mapas de vesselness a partir da imagem LCC de cada caso.
    vessel_maps: dict[int, dict[str, NDArray]] = {}
    for img_id in ids_to_plot:
        lcc_image = preprocessed[img_id]["lcc_image"]

        vesselness_ostia = get_vesselness(
            lcc_image,
            sigmas=ostia_config["sigmas"],
            alpha=ostia_config["alpha"],
            beta=ostia_config["beta"],
            gamma=ostia_config["gamma"],
            normalization=ostia_config["normalization"],
        )

        vesselness_artery = get_vesselness(
            lcc_image,
            sigmas=artery_config["sigmas"],
            alpha=artery_config["alpha"],
            beta=artery_config["beta"],
            gamma=artery_config["gamma"],
            normalization=artery_config["normalization"],
        )

        vessel_maps[img_id] = {
            "vesselness_ostia": vesselness_ostia,
            "vesselness_artery": vesselness_artery,
        }

    return vessel_maps


def plot_vesselness_mip(
    vessel_maps: dict[int, dict[str, NDArray]],
    img_id: int,
    map_key: Literal["vesselness_ostia", "vesselness_artery"] = "vesselness_artery",
    title: str = "Mapa de vasos (MIP axial)",
    cmap: str = "gray",
    show_title: bool = True,
    show_subtitle: bool = True,
    show_colorbar: bool = True,
    dpi: int = 100,
) -> None:
    """Plota a MIP axial de um mapa de vesselness para um único caso.

    Args:
        vessel_maps (dict[int, dict[str, NDArray[Any]]]): Mapas de vesselness
            indexados por ID, no formato retornado por
            ``compute_vesselness_maps``.
        img_id (int): Identificador do exame a visualizar.
        map_key (Literal["vesselness_ostia", "vesselness_artery"]): Chave do
            mapa selecionado (``"vesselness_ostia"`` ou
            ``"vesselness_artery"``).
        title (str): Título principal exibido quando ``show_title`` é ``True``.
        cmap (str): Nome do mapa de cores do Matplotlib.
        show_title (bool): Se ``True``, adiciona título geral acima do gráfico.
        show_subtitle (bool): Se ``True``, exibe o ID do exame no subtítulo.
        show_colorbar (bool): Se ``True``, inclui barra de escala lateral.
        dpi (int): Resolução da figura em pontos por polegada.
    """
    # Projeta o máximo ao longo do eixo axial e renderiza a imagem 2D.
    mip = np.max(vessel_maps[img_id][map_key], axis=2)

    plt.figure(figsize=(7, 5), dpi=dpi)
    im = plt.imshow(mip, cmap=cmap, origin="upper")

    if show_subtitle:
        plt.title(f"ID {img_id}")

    plt.axis("off")

    # Adiciona barra de cores, ajusta layout e exibe o gráfico.
    if show_colorbar:
        plt.colorbar(im, fraction=0.046, pad=0.04)

    if show_title:
        plt.suptitle(title, fontsize=13)
        plt.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        plt.tight_layout()

    plt.show()
    plt.close()
