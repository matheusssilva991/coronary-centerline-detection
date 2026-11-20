"""Gera artefatos visuais sem interface para segmentações em lote."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from utils.visualization.images.volume import visualize_aorta_ostia_artery

logger = logging.getLogger(__name__)

STAGE_VIEW_FILENAMES = {
    "mip": "mip_axial.png",
    "first": "first_slice.png",
    "middle": "middle_slice.png",
    "last": "last_slice.png",
}


def _finite_display_limits(volume: NDArray[Any]) -> tuple[float, float]:
    """Calcula limites inferior e superior robustos para exibição do volume.

    Args:
        volume (NDArray[Any]): Array numérico cujo intervalo finito de
            intensidades define a escala de exibição.

    Returns:
        tuple[float, float]: Par ``(vmin, vmax)`` padronizado em ``(0.0, 1.0)``
        para máscaras binárias ou vazias, ou percentis 1,0% e 99,5% para
        volumes contínuos.
    """
    # Trata volumes vazios ou binários e usa percentis robustos para volumes contínuos.
    finite = np.asarray(volume)[np.isfinite(volume)]
    if finite.size == 0:
        return 0.0, 1.0
    unique = np.unique(finite)
    if unique.size <= 2 and set(unique.tolist()).issubset({0, 1}):
        return 0.0, 1.0
    lower, upper = np.percentile(finite, (1.0, 99.5))
    if lower == upper:
        return float(lower - 0.5), float(upper + 0.5)
    return float(lower), float(upper)


def save_stage_views(
    volume: NDArray[Any],
    output_dir: str | Path,
    *,
    title: str,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 140,
) -> dict[str, Path]:
    """Salva MIP e fatias axiais inicial, central e final de uma etapa em PNG.

    Args:
        volume (NDArray[Any]): Volume 3D ``(Y, X, Z)`` da etapa a documentar.
        output_dir (str | Path): Diretório onde as imagens PNG serão salvas.
        title (str): Prefixo do título exibido em cada imagem gerada.
        cmap (str): Nome do mapa de cores do Matplotlib.
        vmin (float | None): Limite inferior da escala de intensidades; ``None``
            estima automaticamente a partir de ``volume``.
        vmax (float | None): Limite superior da escala de intensidades; ``None``
            estima automaticamente a partir de ``volume``.
        dpi (int): Resolução das imagens exportadas em pontos por polegada.

    Returns:
        dict[str, Path]: Mapeamento das chaves ``"mip"``, ``"first"``,
        ``"middle"`` e ``"last"`` para os respectivos arquivos salvos.
    """
    # Valida que o volume é 3D com eixo axial não vazio.
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError("volume deve ser tridimensional.")
    if array.shape[2] == 0:
        raise ValueError("volume não pode possuir eixo axial vazio.")

    # Garante o diretório de saída e resolve limites automáticos de intensidade.
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    if vmin is None or vmax is None:
        automatic_min, automatic_max = _finite_display_limits(array)
        vmin = automatic_min if vmin is None else vmin
        vmax = automatic_max if vmax is None else vmax

    # Prepara a MIP axial e as fatias inicial, central e final.
    middle_index = array.shape[2] // 2
    views = {
        "mip": (np.max(array, axis=2), "MIP axial"),
        "first": (array[:, :, 0], "Primeira fatia (0)"),
        "middle": (array[:, :, middle_index], f"Fatia média ({middle_index})"),
        "last": (array[:, :, -1], f"Última fatia ({array.shape[2] - 1})"),
    }
    # Renderiza e exporta cada vista como PNG sem bloquear em interface.
    saved: dict[str, Path] = {}
    for key, (image, subtitle) in views.items():
        figure, axis = plt.subplots(figsize=(6, 6), dpi=dpi)
        axis.imshow(
            image,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            interpolation="none",
        )
        axis.set_title(f"{title}\n{subtitle}")
        axis.axis("off")
        figure.tight_layout()
        output_path = target / STAGE_VIEW_FILENAMES[key]
        figure.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
        plt.close(figure)
        saved[key] = output_path
    return saved


def save_detected_circles_figure(
    image: NDArray[Any],
    detected_circles: Sequence[dict[str, Any]],
    output_path: str | Path,
    *,
    num_samples: int = 6,
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 140,
) -> Path:
    """Salva em arquivo uma grade com fatias representativas e círculos da aorta.

    Args:
        image (NDArray[Any]): Volume 3D ``(Y, X, Z)`` usado como fundo das
            fatias.
        detected_circles (Sequence[dict[str, Any]]): Sequência de dicionários
            com ``slice_index``, ``center_x``, ``center_y`` e ``radius`` em
            pixels.
        output_path (str | Path): Caminho do arquivo de imagem a ser gerado.
        num_samples (int): Número máximo de fatias amostradas na grade.
        vmin (float | None): Limite inferior da escala de intensidades; ``None``
            estima automaticamente.
        vmax (float | None): Limite superior da escala de intensidades; ``None``
            estima automaticamente.
        dpi (int): Resolução da figura salva em pontos por polegada.

    Returns:
        Path: Caminho resolvido do arquivo de imagem salvo.
    """
    # Valida entradas e determina limites automáticos de contraste quando omitidos.
    if not detected_circles:
        raise ValueError("detected_circles não pode ser vazio.")
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError("image deve ser tridimensional.")
    if vmin is None or vmax is None:
        automatic_min, automatic_max = _finite_display_limits(array)
        vmin = automatic_min if vmin is None else vmin
        vmax = automatic_max if vmax is None else vmax

    # Amostra uniformemente índices de fatias que contêm círculos detectados.
    slice_indices = sorted(
        {
            int(circle["slice_index"])
            for circle in detected_circles
            if circle.get("slice_index") is not None
        }
    )
    positions = (
        np.linspace(
            0,
            len(slice_indices) - 1,
            num=min(num_samples, len(slice_indices)),
        )
        .round()
        .astype(int)
    )
    selected_slices = [slice_indices[position] for position in positions]

    # Cria grade de subplots e desenha cada fatia com seus círculos sobrepostos.
    columns = min(3, len(selected_slices))
    rows = int(np.ceil(len(selected_slices) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows), dpi=dpi)
    axes_array = np.atleast_1d(axes).ravel()
    for axis, slice_index in zip(axes_array, selected_slices):
        axis.imshow(
            array[:, :, slice_index],
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            origin="upper",
        )
        for circle in detected_circles:
            if int(circle.get("slice_index", -1)) != slice_index:
                continue
            circle_patch = patches.Circle(
                (float(circle["center_x"]), float(circle["center_y"])),
                float(circle["radius"]),
                fill=False,
                edgecolor="red",
                linewidth=1.8,
            )
            axis.add_patch(circle_patch)
            axis.plot(
                float(circle["center_x"]),
                float(circle["center_y"]),
                "r+",
                markersize=8,
            )
        axis.set_title(f"Fatia {slice_index}")
        axis.axis("off")
    # Oculta painéis vazios, grava a figura final e fecha o canvas.
    for axis in axes_array[len(selected_slices) :]:
        axis.axis("off")

    figure.suptitle("Círculos da aorta após filtro de trajetória")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)
    return destination


def save_segmentation_visual_to_path(
    output_path: str | Path,
    *,
    plot_name: str,
    aorta_mask: Any,
    ostia_left: Sequence[float] | None,
    ostia_right: Sequence[float] | None,
    artery_mask: Any | None,
    spacing: Sequence[float],
    label_artery: Any | None = None,
) -> Path:
    """Salva uma visualização 3D interativa no caminho especificado.

    Args:
        output_path (str | Path): Caminho de destino do arquivo HTML.
        plot_name (str): Título identificador da cena K3D.
        aorta_mask (Any): Máscara binária 3D ``(Y, X, Z)`` da aorta.
        ostia_left (Sequence[float] | None): Coordenada ``(y, x, z)`` do óstio
            esquerdo em voxels, ou ``None`` se ausente.
        ostia_right (Sequence[float] | None): Coordenada ``(y, x, z)`` do óstio
            direito em voxels, ou ``None`` se ausente.
        artery_mask (Any | None): Máscara 3D predita das artérias; ``None``
            omite a camada predita.
        spacing (Sequence[float]): Espaçamento dos voxels ``(dy, dx, dz)`` em
            mm.
        label_artery (Any | None): Máscara 3D de referência das artérias;
            ``None`` omite a camada de referência.

    Returns:
        Path: Caminho do arquivo HTML gerado.
    """
    # Garante o diretório pai e gera o HTML 3D sem exibição interativa.
    html_path = Path(output_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    visualize_aorta_ostia_artery(
        aorta_mask,
        ostia_left,
        ostia_right,
        artery_mask=artery_mask,
        label_artery=label_artery,
        spacing=spacing,
        use_physical_coords=True,
        save_html_path=str(html_path),
        display_plot=False,
        plot_name=plot_name,
    )
    return html_path


def save_segmentation_visual(
    output_dir: str | Path,
    img_id: int | str,
    *,
    aorta_mask: Any,
    ostia_left: Sequence[float] | None,
    ostia_right: Sequence[float] | None,
    artery_mask: Any | None,
    label_artery: Any,
    spacing: Sequence[float],
) -> Path | None:
    """Salva aorta, óstios, predição e referência em um HTML 3D interativo.

    A visualização é um artefato auxiliar; qualquer erro de geração é
    registrado no log e não invalida os resultados numéricos da imagem.

    Args:
        output_dir (str | Path): Diretório onde o arquivo HTML será criado.
        img_id (int | str): Identificador numérico da imagem usado no nome do
            arquivo e no título da cena.
        aorta_mask (Any): Máscara binária 3D ``(Y, X, Z)`` da aorta.
        ostia_left (Sequence[float] | None): Coordenada ``(y, x, z)`` do óstio
            esquerdo em voxels, ou ``None`` se não localizado.
        ostia_right (Sequence[float] | None): Coordenada ``(y, x, z)`` do óstio
            direito em voxels, ou ``None`` se não localizado.
        artery_mask (Any | None): Máscara 3D predita das artérias; ``None``
            omite a predição na cena.
        label_artery (Any): Máscara 3D de referência das artérias (ou ``None``
            quando indisponível).
        spacing (Sequence[float]): Espaçamento dos voxels ``(dy, dx, dz)`` em
            mm.

    Returns:
        Path | None: Caminho do arquivo HTML gerado, ou ``None`` se ocorrer
        falha durante a geração da cena 3D.
    """
    # Monta o caminho padrão da imagem e delega a geração isolando falhas.
    html_path = Path(output_dir) / f"img_{int(img_id)}_aorta_ostia_artery.html"

    try:
        save_segmentation_visual_to_path(
            html_path,
            plot_name=f"IMG {int(img_id)} | Aorta, óstios e artérias",
            aorta_mask=aorta_mask,
            ostia_left=ostia_left,
            ostia_right=ostia_right,
            artery_mask=artery_mask,
            label_artery=label_artery,
            spacing=spacing,
        )
    except Exception:
        logger.exception(
            "Não foi possível salvar a visualização 3D da imagem %s.",
            img_id,
        )
        return None

    # Registra conclusão apenas quando a exportação HTML é bem-sucedida.
    logger.info("Visualização 3D salva em: %s", html_path)
    return html_path


__all__ = [
    "STAGE_VIEW_FILENAMES",
    "save_detected_circles_figure",
    "save_segmentation_visual",
    "save_segmentation_visual_to_path",
    "save_stage_views",
]
