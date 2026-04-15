from typing import Literal, Optional, Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_mip_projection(
    image_volume: NDArray,
    title: str = "Maximum Intensity Projections (MIP)",
    cmap: str = "gray",
    views: Sequence[Literal["axial", "coronal", "sagital"]] = (
        "axial",
        "coronal",
        "sagital",
    ),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    projection: Literal["max", "min", "mean"] = "max",
    window_level: Optional[float] = None,
    window_width: Optional[float] = None,
    return_fig: bool = False,
):
    """Plota projeções MIP (ou min/mean) em vistas ortogonais de um volume 3D."""
    if image_volume.ndim != 3:
        raise ValueError("A imagem deve ser 3D.")

    if window_level is not None and window_width is not None:
        lower = window_level - window_width // 2
        upper = window_level + window_width // 2
        image_volume = np.clip(image_volume, lower, upper)

    axis_map = {"axial": 2, "coronal": 1, "sagital": 0}
    proj_func = {"max": np.max, "min": np.min, "mean": np.mean}[projection]

    n_views = len(views)
    fig, axes = plt.subplots(1, n_views, figsize=(6 * n_views, 6))
    if n_views == 1:
        axes = [axes]

    imshow_kwargs = {"cmap": cmap}
    if vmin is not None:
        imshow_kwargs["vmin"] = vmin
    if vmax is not None:
        imshow_kwargs["vmax"] = vmax

    for i, view in enumerate(views):
        if view not in axis_map:
            raise ValueError(
                f"View inválida: {view}. Use 'axial', 'coronal' ou 'sagital'."
            )
        mip = proj_func(image_volume, axis=axis_map[view])
        axes[i].imshow(mip, **imshow_kwargs)
        axes[i].set_title(f"MIP {view.capitalize()} ({projection}) - shape {mip.shape}")
        axes[i].axis("off")

    plt.suptitle(title, fontsize=16, y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if return_fig:
        return fig, axes
    plt.show()


def plot_slices(img, slices_indices, cmap="gray", title=None, vmin=None, vmax=None):
    """Exibe múltiplas fatias 2D selecionadas de um volume."""
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


def visualize_circles_on_slices(
    image, detected_circles, num_samples=6, vmin=None, vmax=None
):
    """Sobrepõe círculos detectados em fatias amostradas do volume."""
    slice_indices = sorted(set([c["slice_index"] for c in detected_circles]))
    step = max(1, len(slice_indices) // num_samples)
    selected_slices = slice_indices[::step][:num_samples]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, slice_idx in enumerate(selected_slices):
        if i >= len(axes):
            break

        circles_in_slice = [
            c for c in detected_circles if c["slice_index"] == slice_idx
        ]
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
