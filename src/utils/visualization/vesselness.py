"""Visualizações e cálculo auxiliar de mapas de vesselness."""

from typing import Any, Literal, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def compute_vesselness_maps(
    preprocessed: dict[int, dict[str, Any]],
    ids_to_plot: Optional[Sequence[int]] = None,
    ostia_config: Optional[dict[str, Any]] = None,
    artery_config: Optional[dict[str, Any]] = None,
) -> dict[int, dict[str, NDArray]]:
    """Computa mapas de vesselness para ostios e arterias a partir da imagem LCC."""
    from ..processing.frangi import get_vesselness

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


def plot_vesselness_mip_grid(
    vessel_maps: dict[int, dict[str, NDArray]],
    ids_to_plot: Optional[Sequence[int]] = None,
    map_key: Literal["vesselness_ostia", "vesselness_artery"] = "vesselness_artery",
    title: str = "Mapa de vasos (MIP axial)",
    cmap: str = "gray",
    dpi: int = 100,
) -> None:
    """Plota MIP axial dos mapas de vesselness para uma lista de IDs."""
    if ids_to_plot is None:
        ids_to_plot = sorted(vessel_maps.keys())
    else:
        ids_to_plot = list(ids_to_plot)
    if len(ids_to_plot) == 0:
        raise ValueError("ids_to_plot não pode ser vazio.")

    fig, axes = plt.subplots(
        1, len(ids_to_plot), figsize=(7 * len(ids_to_plot), 5), dpi=dpi
    )
    if len(ids_to_plot) == 1:
        axes = [axes]

    for idx, img_id in enumerate(ids_to_plot):
        mip = np.max(vessel_maps[img_id][map_key], axis=2)
        ax = axes[idx]
        im = ax.imshow(mip, cmap=cmap, origin="upper")
        ax.set_title(f"ID {img_id}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.close(fig)


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
    """Plota MIP axial de um mapa de vesselness para um unico ID."""
    mip = np.max(vessel_maps[img_id][map_key], axis=2)

    plt.figure(figsize=(7, 5), dpi=dpi)
    im = plt.imshow(mip, cmap=cmap, origin="upper")

    if show_subtitle:
        plt.title(f"ID {img_id}")

    plt.axis("off")

    if show_colorbar:
        plt.colorbar(im, fraction=0.046, pad=0.04)

    if show_title:
        plt.suptitle(title, fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()

    plt.show()
    plt.close()
