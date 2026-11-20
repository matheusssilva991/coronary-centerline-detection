"""Auxiliares de visualização para comparação de limiares HU da aorta entre MM-WHS e ImageCAS."""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.figure import Figure

from utils.project.analysis.aorta_hu_analysis import (
    HU_REGION_NAMES,
    HuReferenceVolume,
    hu_interval_mask,
)
from utils.project.dataframe import require_series_column
from utils.visualization.images.hu_threshold import (
    hu_axial_mip,
    reference_axial_slices,
)

__all__ = [
    "plot_hu_threshold_examples",
    "show_retention",
]


def plot_hu_threshold_examples(
    volume: HuReferenceVolume,
    intervals: Mapping[str, tuple[float, float]],
    *,
    hu_window: tuple[float, float] = (-200, 1000),
) -> Figure:
    """Compara o volume CT original e os intervalos de limiar HU em MIP e três fatias RAS.

    Args:
        volume (HuReferenceVolume): Volume de referência com imagem em HU e máscaras anatômicas.
        intervals (Mapping[str, tuple[float, float]]): Intervalos HU nomeados por região de origem.
        hu_window (tuple[float, float]): Janela de intensidade `(min_hu, max_hu)` para exibição.

    Returns:
        Figure: Figura Matplotlib com a grade de projeções MIP e fatias axiais.
    """
    # Valida a janela de visualização e a compatibilidade entre imagem e máscara de referência.
    if not np.isfinite(hu_window).all() or hu_window[0] >= hu_window[1]:
        raise ValueError("A janela HU deve ter limites finitos e crescentes.")
    reference = volume.regions[
        "mmwhs_heart" if volume.dataset == "MM-WHS" else "imagecas_aorta"
    ]
    if reference.shape != volume.image.shape:
        raise ValueError("Imagem e referência possuem shapes incompatíveis.")

    # Seleciona as três fatias axiais representativas e prepara as colunas de comparação.
    indices = reference_axial_slices(reference)
    columns: list[tuple[str, tuple[float, float] | None]] = [("CT original", None)]
    columns.extend(
        (f"{HU_REGION_NAMES[key]}\n{low:.1f}–{high:.1f} HU", (low, high))
        for key, (low, high) in intervals.items()
    )
    fig = plt.figure(figsize=(4.1 * len(columns), 12), layout="constrained")
    extent = (
        -volume.spacing[0] / 2,
        (volume.image.shape[0] - 0.5) * volume.spacing[0],
        -volume.spacing[1] / 2,
        (volume.image.shape[1] - 0.5) * volume.spacing[1],
    )

    # Renderiza o MIP axial na primeira linha e as três fatias nas linhas seguintes.
    for column, (name, interval) in enumerate(columns):
        mip = hu_axial_mip(volume.image, interval)
        for row in range(4):
            ax = fig.add_subplot(4, len(columns), row * len(columns) + column + 1)
            if row == 0:
                values = mip
            else:
                values = volume.image[:, :, indices[row - 1]]
                selected = (
                    np.isfinite(values)
                    if interval is None
                    else hu_interval_mask(values, interval)
                )
                values = np.where(selected, values, np.nan)
            ax.set_facecolor("black")
            ax.imshow(
                np.ma.masked_invalid(values.T),
                cmap="gray",
                origin="lower",
                vmin=hu_window[0],
                vmax=hu_window[1],
                extent=extent,
                aspect="equal",
                interpolation="nearest",
            )
            if row == 0:
                ax.set_title(name, fontsize=10)
            if column == 0:
                ax.set_ylabel("MIP axial" if row == 0 else f"Fatia {indices[row - 1]}")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"{volume.dataset} · {volume.exam_id} · HU nativos · RAS\n"
        "Intervalos no volume completo; sem LCC ou segmentação adicional",
        fontsize=12,
    )
    return fig


def show_retention(frame: pd.DataFrame) -> None:
    """Exibe tabela resumindo o volume global selecionado e a preservação da região anatômica.

    Args:
        frame (pd.DataFrame): DataFrame com métricas de retenção por intervalo e região.
    """
    # Seleciona as colunas relevantes e traduz identificadores técnicos para nomes legíveis.
    table = frame.loc[
        :,
        [
            "dataset",
            "exam_id",
            "interval",
            "region",
            "selected_volume_percent",
            "retained_region_percent",
        ],
    ].copy()
    table["interval"] = require_series_column(table, "interval").map(
        lambda key: HU_REGION_NAMES[str(key)]
    )
    table["region"] = require_series_column(table, "region").map(
        lambda key: HU_REGION_NAMES[str(key)]
    )
    # Exibe o resumo arredondado com cabeçalhos em português.
    display(
        table.rename(
            columns={
                "dataset": "Banco",
                "exam_id": "Exame",
                "interval": "Intervalo",
                "region": "Região avaliada",
                "selected_volume_percent": "Volume selecionado (%)",
                "retained_region_percent": "Região preservada (%)",
            }
        ).round(2)
    )
