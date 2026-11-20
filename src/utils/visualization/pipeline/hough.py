"""Visualizações de diagnóstico da detecção por Hough."""

from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def _resolve_cmap(cmap: str, invert_cmap: bool):
    """Retorna o nome do colormap ou sua versão invertida no Matplotlib.

    Args:
        cmap (str): Nome do mapa de cores do Matplotlib.
        invert_cmap (bool): Se ``True``, retorna o objeto ``Colormap``
            invertido.

    Returns:
        str | Any: Nome original do mapa de cores ou instância invertida do
        Matplotlib.
    """
    return plt.get_cmap(cmap).reversed() if invert_cmap else cmap


def plot_hough_initial_circle(
    img_slice: NDArray,
    diagnostics: dict[str, Any],
    title: str = "Transformada de Hough - círculo inicial",
    cmap: str = "gray",
    invert_cmap: bool = False,
    show_title: bool = True,
    show_subtitle: bool = True,
    dpi: int = 100,
) -> None:
    """Plota o círculo inicial detectado pela transformada de Hough sobre a fatia.

    Args:
        img_slice (NDArray[Any]): Fatia axial 2D exibida como imagem de fundo.
        diagnostics (dict[str, Any]): Dicionário de diagnósticos da localização
            da aorta contendo opcionalmente a chave ``initial_circle`` (com
            ``center_x``, ``center_y`` e ``radius`` em pixels).
        title (str): Título principal exibido quando ``show_title`` é ``True``.
        cmap (str): Nome do mapa de cores do Matplotlib.
        invert_cmap (bool): Se ``True``, inverte o mapa de cores da imagem.
        show_title (bool): Se ``True``, adiciona o título geral na figura.
        show_subtitle (bool): Se ``True``, adiciona subtítulo e legenda no eixo.
        dpi (int): Resolução da figura em pontos por polegada.
    """
    # Extrai o círculo inicial dos diagnósticos e renderiza a fatia base.
    initial_circle = diagnostics.get("initial_circle")

    cmap_to_use = _resolve_cmap(cmap, invert_cmap)
    plt.figure(figsize=(6, 6), dpi=dpi)
    plt.imshow(img_slice, cmap=cmap_to_use, origin="upper")

    # Desenha centro e contorno do círculo inicial quando encontrado.
    if initial_circle is not None:
        plt.scatter(
            [initial_circle["center_x"]],
            [initial_circle["center_y"]],
            c="lime",
            s=90,
            marker="x",
            label="círculo inicial",
        )
        plt.gca().add_patch(
            patches.Circle(
                (initial_circle["center_x"], initial_circle["center_y"]),
                initial_circle["radius"],
                fill=False,
                edgecolor="lime",
                linewidth=2,
            )
        )

    # Configura rótulos, legenda, layout e exibe a figura.
    plt.axis("off")
    if show_subtitle:
        plt.title("Círculo inicial detectado")
    if initial_circle is not None and show_subtitle:
        plt.legend(loc="lower right")

    if show_title:
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        plt.tight_layout()

    plt.show()
    plt.close()


def plot_hough_refinement_candidates(
    img_slice: NDArray,
    diagnostics: dict[str, Any],
    title: str = "Transformada de Hough - candidatos para refinamento",
    cmap: str = "gray",
    invert_cmap: bool = False,
    show_title: bool = True,
    show_subtitle: bool = True,
    dpi: int = 100,
) -> None:
    """Plota os círculos vizinhos avaliados no refinamento da transformada de Hough.

    Args:
        img_slice (NDArray[Any]): Fatia axial 2D exibida como imagem de fundo.
        diagnostics (dict[str, Any]): Dicionário de diagnósticos da localização
            da aorta com a lista ``refinement_candidates`` (cada item contendo
            ``center_x``, ``center_y`` e ``radius`` em pixels).
        title (str): Título principal exibido quando ``show_title`` é ``True``.
        cmap (str): Nome do mapa de cores do Matplotlib.
        invert_cmap (bool): Se ``True``, inverte o mapa de cores da imagem.
        show_title (bool): Se ``True``, exibe o título geral da figura.
        show_subtitle (bool): Se ``True``, exibe o subtítulo do painel.
        dpi (int): Resolução da figura em pontos por polegada.
    """
    # Obtém os círculos candidatos e prepara a fatia de fundo.
    refinement_candidates = diagnostics.get("refinement_candidates", [])

    cmap_to_use = _resolve_cmap(cmap, invert_cmap)
    plt.figure(figsize=(6, 6), dpi=dpi)
    plt.imshow(img_slice, cmap=cmap_to_use, origin="upper")

    # Sobrepõe cada candidato de refinamento com seu centro e contorno.
    for candidate in refinement_candidates:
        plt.gca().add_patch(
            patches.Circle(
                (candidate["center_x"], candidate["center_y"]),
                candidate["radius"],
                fill=False,
                edgecolor="orange",
                linewidth=1.8,
                alpha=0.9,
            )
        )
        plt.scatter(
            [candidate["center_x"]],
            [candidate["center_y"]],
            c="orange",
            s=30,
            marker="o",
            alpha=0.85,
        )

    # Ajusta títulos, layout da figura e finaliza a exibição.
    plt.axis("off")
    if show_subtitle:
        plt.title("Círculos vizinhos usados no refinamento")

    if show_title:
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        plt.tight_layout()

    plt.show()
    plt.close()


def plot_hough_refined_circle(
    img_slice: NDArray,
    diagnostics: dict[str, Any],
    title: str = "Transformada de Hough - círculo refinado",
    cmap: str = "gray",
    invert_cmap: bool = False,
    show_title: bool = True,
    show_subtitle: bool = True,
    dpi: int = 100,
) -> None:
    """Plota o círculo final refinado da aorta sobre a fatia axial.

    Args:
        img_slice (NDArray[Any]): Fatia axial 2D exibida como imagem de fundo.
        diagnostics (dict[str, Any]): Dicionário de diagnósticos da localização
            da aorta contendo opcionalmente ``refined_circle`` (com
            ``center_x``, ``center_y`` e ``radius`` em pixels).
        title (str): Título principal exibido quando ``show_title`` é ``True``.
        cmap (str): Nome do mapa de cores do Matplotlib.
        invert_cmap (bool): Se ``True``, inverte o mapa de cores da imagem.
        show_title (bool): Se ``True``, adiciona o título geral na figura.
        show_subtitle (bool): Se ``True``, adiciona subtítulo e legenda no eixo.
        dpi (int): Resolução da figura em pontos por polegada.
    """
    # Recupera o círculo refinado e renderiza a imagem base.
    refined_circle = diagnostics.get("refined_circle")

    cmap_to_use = _resolve_cmap(cmap, invert_cmap)
    plt.figure(figsize=(6, 6), dpi=dpi)
    plt.imshow(img_slice, cmap=cmap_to_use, origin="upper")

    # Destaca contorno e centro do círculo final refinado, se existir.
    if refined_circle is not None:
        plt.gca().add_patch(
            patches.Circle(
                (refined_circle["center_x"], refined_circle["center_y"]),
                refined_circle["radius"],
                fill=False,
                edgecolor="red",
                linewidth=2.5,
            )
        )
        plt.scatter(
            [refined_circle["center_x"]],
            [refined_circle["center_y"]],
            c="red",
            s=90,
            marker="x",
            label="círculo refinado",
        )

    # Configura subtítulo, legenda, layout e exibe o resultado.
    plt.axis("off")
    if show_subtitle:
        plt.title("Círculo final refinado")
    if refined_circle is not None and show_subtitle:
        plt.legend(loc="lower right")

    if show_title:
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        plt.tight_layout()

    plt.show()
    plt.close()
