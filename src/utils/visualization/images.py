from typing import Any, Literal, Optional, Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_mip_projection(
    image_volume: NDArray,
    title: str = "Maximum Intensity Projections (MIP)",
    show_title: bool = True,
    cmap: str = "gray",
    invert_cmap: bool = False,
    views: Sequence[Literal["axial", "coronal", "sagittal"]] = (
        "axial",
        "coronal",
        "sagittal",
    ),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    projection: Literal["max", "min", "mean"] = "max",
    window_level: Optional[float] = None,
    window_width: Optional[float] = None,
    return_fig: bool = False,
    show_labels: bool = True,
    dpi: int = 100,
) -> Any:
    """Plota projeções MIP (ou min/mean) em vistas ortogonais de um volume 3D."""
    # Garante entrada 3D antes da projeção.
    if image_volume.ndim != 3:
        raise ValueError("A imagem deve ser 3D.")

    if window_level is not None and window_width is not None:
        # Aplica windowing de intensidade no volume.
        lower = window_level - window_width // 2
        upper = window_level + window_width // 2
        image_volume = np.clip(image_volume, lower, upper)

    # Define eixo colapsado para cada vista.
    axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}
    proj_func = {"max": np.max, "min": np.min, "mean": np.mean}[projection]

    n_views = len(views)
    fig, axes = plt.subplots(1, n_views, figsize=(6 * n_views, 6), dpi=dpi)
    if n_views == 1:
        axes = [axes]

    cmap_to_use = plt.get_cmap(cmap).reversed() if invert_cmap else cmap
    imshow_kwargs = {"cmap": cmap_to_use}
    if vmin is not None:
        imshow_kwargs["vmin"] = vmin
    if vmax is not None:
        imshow_kwargs["vmax"] = vmax

    for i, view in enumerate(views):
        if view not in axis_map:
            raise ValueError(
                f"View inválida: {view}. Use 'axial', 'coronal' ou 'sagittal'."
            )
        # Gera projeção da vista atual.
        mip = proj_func(image_volume, axis=axis_map[view])
        axes[i].imshow(mip, **imshow_kwargs)
        if show_labels:
            axes[i].set_title(
                f"MIP {view.capitalize()} ({projection}) - shape {mip.shape}"
            )
        axes[i].axis("off")

    if show_title:
        plt.suptitle(title, fontsize=16, y=0.96)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        plt.tight_layout()

    if return_fig:
        return fig, axes
    plt.show()
    plt.close(fig)


def plot_slices(
    img: NDArray,
    slices_indices: Sequence[int],
    cmap: str = "gray",
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Exibe múltiplas fatias 2D selecionadas de um volume."""
    # Calcula grade de subplots para as fatias solicitadas.
    n_slices = len(slices_indices)
    n_cols = min(5, n_slices)
    n_rows = (n_slices + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = np.array([axes]).reshape(n_rows, n_cols)

    for i, slice_idx in enumerate(slices_indices):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        # Renderiza a fatia no eixo correspondente.
        ax.imshow(img[:, :, slice_idx], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Fatia {slice_idx}")
        ax.axis("off")

    for i in range(n_slices, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        fig.delaxes(axes[row, col])

    if title:
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.subplots_adjust(top=0.90)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def visualize_circles_on_slices(
    image: NDArray,
    detected_circles: Sequence[dict],
    num_samples: int = 6,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Sobrepõe círculos detectados em fatias amostradas do volume."""
    # Seleciona fatias de amostra onde há círculos detectados.
    slice_indices = sorted(set([c["slice_index"] for c in detected_circles]))
    step = max(1, len(slice_indices) // num_samples)
    selected_slices = slice_indices[::step][:num_samples]

    # Painel fixo para comparação visual rápida.
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, slice_idx in enumerate(selected_slices):
        if i >= len(axes):
            break

        circles_in_slice = [
            c for c in detected_circles if c["slice_index"] == slice_idx
        ]
        # Desenha a fatia de fundo.
        axes[i].imshow(image[:, :, slice_idx], cmap="gray", vmin=vmin, vmax=vmax)

        for circle in circles_in_slice:
            circle_patch = patches.Circle(
                (circle["center_x"], circle["center_y"]),
                circle["radius"],
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
            axes[i].add_patch(circle_patch)
            axes[i].plot(circle["center_x"], circle["center_y"], "r+", markersize=10)

        axes[i].set_title(f"Fatia {slice_idx}")
        axes[i].axis("off")

    for i in range(len(selected_slices), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def _resolve_stage_image(
    volume: NDArray,
    center_slice: int,
    mode: Literal["slice", "mip"],
) -> tuple[NDArray, str]:
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
    """Plota uma etapa de pre-processamento para um unico caso."""
    volume = preprocessed[img_id][stage_key]
    center_slice = preprocessed[img_id]["center_slice"]
    image_to_show, mode_text = _resolve_stage_image(volume, center_slice, mode)

    plt.figure(figsize=(6, 5), dpi=dpi)
    plt.imshow(image_to_show, cmap=cmap)

    if show_subtitle:
        plt.title(f"ID {img_id} - {mode_text}")

    plt.axis("off")

    if show_title:
        plt.suptitle(f"{stage_title} - {mode_text}", fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()

    plt.show()
    plt.close()


def plot_preprocessing_grid(
    preprocessed: dict[int, dict[str, Any]],
    ids_to_plot: Optional[Sequence[int]] = None,
    mode: Literal["slice", "mip", "both"] = "slice",
    show_title: bool = True,
    show_subtitle: bool = True,
    cmap: str = "gray",
    dpi: int = 100,
) -> None:
    """Plota grid das etapas de pre-processamento em fatia, MIP ou ambos."""
    if ids_to_plot is None:
        ids_to_plot = sorted(preprocessed.keys())

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

    n_rows = len(stages)
    n_cols = len(ids_to_plot) * len(display_modes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=dpi)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

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
                ax.imshow(image_to_show, cmap=cmap)

                if show_subtitle:
                    ax.set_title(f"ID {img_id} - {stage_title} - {mode_text}")

                ax.axis("off")
                col += 1

    if show_title:
        if mode == "slice":
            header = "Grid das etapas de pre-processamento (fatia central)"
        elif mode == "mip":
            header = "Grid das etapas de pre-processamento (MIP axial)"
        else:
            header = "Grid das etapas de pre-processamento (fatia central + MIP axial)"
        plt.suptitle(header, fontsize=15)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()

    plt.show()
    plt.close()


def compute_vesselness_maps(
    preprocessed: dict[int, dict[str, Any]],
    ids_to_plot: Optional[Sequence[int]] = None,
    ostia_config: Optional[dict[str, Any]] = None,
    artery_config: Optional[dict[str, Any]] = None,
) -> dict[int, dict[str, NDArray]]:
    """Computa mapas de vesselness para ostios e arterias a partir da imagem LCC."""
    from utils import get_vesselness

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

    fig, axes = plt.subplots(
        1, len(ids_to_plot), figsize=(7 * len(ids_to_plot), 5), dpi=dpi
    )
    if len(ids_to_plot) == 1:
        axes = [axes]

    for idx, img_id in enumerate(ids_to_plot):
        mip = np.max(vessel_maps[img_id][map_key], axis=2)
        ax = axes[idx]
        im = ax.imshow(mip, cmap=cmap, origin="lower")
        ax.set_title(f"ID {img_id}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.close()


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
    im = plt.imshow(mip, cmap=cmap, origin="lower")

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


def plot_hough_initial_diagnostics(
    img_slice: NDArray,
    diagnostics: dict[str, Any],
    title: str = "Transformada de Hough - círculo inicial",
    cmap: str = "gray",
    invert_cmap: bool = False,
    show_title: bool = True,
    show_subtitle: bool = True,
    dpi: int = 100,
):
    """Plota o círculo inicial, os candidatos de refinamento e o círculo refinado."""
    initial_circle = diagnostics.get("initial_circle")
    refined_circle = diagnostics.get("refined_circle")
    candidates = diagnostics.get("candidates", [])
    refinement_candidates = diagnostics.get("refinement_candidates", [])

    cmap_to_use = plt.get_cmap(cmap).reversed() if invert_cmap else cmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)

    axes[0].imshow(img_slice, cmap=cmap_to_use)
    if initial_circle is not None:
        axes[0].scatter(
            [initial_circle["center_x"]],
            [initial_circle["center_y"]],
            c="lime",
            s=90,
            marker="x",
            label="círculo inicial",
        )
        axes[0].add_patch(
            patches.Circle(
                (initial_circle["center_x"], initial_circle["center_y"]),
                initial_circle["radius"],
                fill=False,
                edgecolor="lime",
                linewidth=2,
            )
        )
    axes[0].set_axis_off()
    if show_subtitle:
        axes[0].set_title("Círculo inicial detectado")
    if initial_circle is not None and show_subtitle:
        axes[0].legend(loc="lower right")

    axes[1].imshow(img_slice, cmap=cmap_to_use)
    for candidate in candidates:
        axes[1].add_patch(
            patches.Circle(
                (candidate["center_x"], candidate["center_y"]),
                candidate["radius"],
                fill=False,
                edgecolor="steelblue",
                linewidth=1,
                alpha=0.35,
            )
        )
    for candidate in refinement_candidates:
        axes[1].add_patch(
            patches.Circle(
                (candidate["center_x"], candidate["center_y"]),
                candidate["radius"],
                fill=False,
                edgecolor="orange",
                linewidth=1.5,
                alpha=0.8,
            )
        )
    if refined_circle is not None:
        axes[1].add_patch(
            patches.Circle(
                (refined_circle["center_x"], refined_circle["center_y"]),
                refined_circle["radius"],
                fill=False,
                edgecolor="red",
                linewidth=2.5,
            )
        )
        axes[1].scatter(
            [refined_circle["center_x"]],
            [refined_circle["center_y"]],
            c="red",
            s=90,
            marker="x",
            label="círculo refinado",
        )
    axes[1].set_axis_off()
    if show_subtitle:
        axes[1].set_title("Candidatos e refinamento")
    if refined_circle is not None:
        axes[1].legend(loc="lower right")

    if show_title:
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        plt.tight_layout()

    plt.show()
    plt.close()


def plot_hough_initial_circle(
    img_slice: NDArray,
    diagnostics: dict[str, Any],
    title: str = "Transformada de Hough - círculo inicial",
    cmap: str = "gray",
    invert_cmap: bool = False,
    show_title: bool = True,
    show_subtitle: bool = True,
    dpi: int = 100,
):
    """Plota apenas o círculo inicial detectado."""
    initial_circle = diagnostics.get("initial_circle")

    cmap_to_use = plt.get_cmap(cmap).reversed() if invert_cmap else cmap
    plt.figure(figsize=(6, 6), dpi=dpi)
    plt.imshow(img_slice, cmap=cmap_to_use)

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

    plt.axis("off")
    if show_subtitle:
        plt.title("Círculo inicial detectado")
    if initial_circle is not None and show_subtitle:
        plt.legend(loc="lower right")

    if show_title:
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        plt.tight_layout()

    plt.show()


def plot_hough_refinement_candidates(
    img_slice: NDArray,
    diagnostics: dict[str, Any],
    title: str = "Transformada de Hough - candidatos para refinamento",
    cmap: str = "gray",
    invert_cmap: bool = False,
    show_title: bool = True,
    show_subtitle: bool = True,
    dpi: int = 100,
):
    """Plota apenas os círculos vizinhos usados no refinamento."""
    refinement_candidates = diagnostics.get("refinement_candidates", [])

    cmap_to_use = plt.get_cmap(cmap).reversed() if invert_cmap else cmap
    plt.figure(figsize=(6, 6), dpi=dpi)
    plt.imshow(img_slice, cmap=cmap_to_use)

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

    plt.axis("off")
    if show_subtitle:
        plt.title("Círculos vizinhos usados no refinamento")

    if show_title:
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        plt.tight_layout()

    plt.show()


def plot_hough_refined_circle(
    img_slice: NDArray,
    diagnostics: dict[str, Any],
    title: str = "Transformada de Hough - círculo refinado",
    cmap: str = "gray",
    invert_cmap: bool = False,
    show_title: bool = True,
    show_subtitle: bool = True,
    dpi: int = 100,
):
    """Plota apenas o círculo final refinado."""
    refined_circle = diagnostics.get("refined_circle")

    cmap_to_use = plt.get_cmap(cmap).reversed() if invert_cmap else cmap
    plt.figure(figsize=(6, 6), dpi=dpi)
    plt.imshow(img_slice, cmap=cmap_to_use)

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

    plt.axis("off")
    if show_subtitle:
        plt.title("Círculo final refinado")
    if refined_circle is not None and show_subtitle:
        plt.legend(loc="lower right")

    if show_title:
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        plt.tight_layout()

    plt.show()


def plot_spaced_detected_circles(
    image_volume: NDArray,
    detected_circles: list[dict[str, Any]],
    sample_count: int = 4,
    title: str = "Círculos da Hough em fatias espaçadas",
    cmap: str = "gray",
    invert_cmap: bool = False,
    show_title: bool = True,
    show_subtitle: bool = True,
    dpi: int = 100,
):
    """Plota círculos detectados em fatias espaçadas ao longo do volume."""
    if not detected_circles:
        raise ValueError("detected_circles não pode ser vazio.")

    sample_count = max(1, min(sample_count, len(detected_circles)))
    sample_indices = np.linspace(0, len(detected_circles) - 1, sample_count)
    sample_indices = np.unique(np.round(sample_indices).astype(int))

    n_cols = len(sample_indices)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6), dpi=dpi)
    if n_cols == 1:
        axes = [axes]

    cmap_to_use = plt.get_cmap(cmap).reversed() if invert_cmap else cmap

    for ax, circle_idx in zip(axes, sample_indices, strict=False):
        circle = detected_circles[circle_idx]
        slice_idx = int(circle["slice_index"])
        ax.imshow(image_volume[:, :, slice_idx], cmap=cmap_to_use)
        ax.add_patch(
            patches.Circle(
                (circle["center_x"], circle["center_y"]),
                circle["radius"],
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
        )
        ax.scatter([circle["center_x"]], [circle["center_y"]], c="red", s=40)
        if show_subtitle:
            ax.set_title(
                f"Círculo {circle_idx + 1}/{len(detected_circles)} - fatia {slice_idx}"
            )
        ax.set_axis_off()

    if show_title:
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        plt.tight_layout()

    plt.show()
