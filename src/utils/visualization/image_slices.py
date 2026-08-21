"""Visualizações básicas de volumes, fatias e círculos detectados."""

from pathlib import Path

from typing import Any, Literal, Optional, Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def _create_volume_slice_figure(
    image_volume: NDArray,
    slice_index: int,
    *,
    axis: int = 2,
    cmap: str = "gray",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    origin: Literal["upper", "lower"] = "upper",
    dpi: int = 300,
    figure_width: float = 7.0,
    padding_fraction: float = 0.0,
    output_size_px: tuple[int, int] | None = None,
) -> tuple[Any, Any]:
    """Create a complete 2D slice figure shared by display and export."""
    if image_volume.ndim != 3:
        raise ValueError("image_volume deve ser 3D.")
    if axis not in (0, 1, 2):
        raise ValueError("axis deve ser 0, 1 ou 2.")
    if padding_fraction < 0:
        raise ValueError("padding_fraction deve ser >= 0.")
    if output_size_px is not None and any(size <= 0 for size in output_size_px):
        raise ValueError("output_size_px deve conter largura e altura positivas.")

    axis_size = image_volume.shape[axis]
    if not -axis_size <= slice_index < axis_size:
        raise IndexError(
            f"slice_index={slice_index} fora do eixo {axis}, que possui "
            f"{axis_size} fatias."
        )
    # Índices negativos são convertidos para a posição positiva equivalente.
    normalized_index = slice_index % axis_size
    image_slice = np.take(image_volume, normalized_index, axis=axis)
    height, width = image_slice.shape
    padding = max(height, width) * padding_fraction
    canvas_height = height + 2 * padding
    canvas_width = width + 2 * padding
    # Mantém a proporção real ou respeita exatamente o tamanho solicitado em pixels.
    if output_size_px is None:
        figure_height = figure_width * canvas_height / canvas_width
    else:
        output_width, output_height = output_size_px
        figure_width = output_width / dpi
        figure_height = output_height / dpi

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=dpi)
    ax.imshow(
        image_slice,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin=origin,
        interpolation="none",
        aspect="equal",
    )
    # Limites explícitos impedem que o Matplotlib recorte pixels das bordas.
    ax.set_xlim(-0.5 - padding, width - 0.5 + padding)
    if origin == "upper":
        ax.set_ylim(height - 0.5 + padding, -0.5 - padding)
    else:
        ax.set_ylim(-0.5 - padding, height - 0.5 + padding)
    ax.set_facecolor("black")
    ax.margins(0)
    ax.axis("off")
    ax.set_position((0, 0, 1, 1))
    return fig, ax


def plot_volume_slice(
    image_volume: NDArray,
    slice_index: int,
    *,
    axis: int = 2,
    cmap: str = "gray",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    origin: Literal["upper", "lower"] = "upper",
    dpi: int = 100,
    figure_width: float = 4.0,
    padding_fraction: float = 0.0,
    output_size_px: tuple[int, int] | None = None,
    show: bool = True,
) -> tuple[Any, Any]:
    """Display one complete volume slice without clipping border pixels."""
    fig, ax = _create_volume_slice_figure(
        image_volume,
        slice_index,
        axis=axis,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin=origin,
        dpi=dpi,
        figure_width=figure_width,
        padding_fraction=padding_fraction,
        output_size_px=output_size_px,
    )
    if show:
        plt.show()
    return fig, ax


def save_volume_slice_figure(
    image_volume: NDArray,
    slice_index: int,
    output_path: str | Path,
    *,
    axis: int = 2,
    cmap: str = "gray",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    origin: Literal["upper", "lower"] = "upper",
    dpi: int = 300,
    figure_width: float = 7.0,
    padding_fraction: float = 0.0,
    output_size_px: tuple[int, int] | None = None,
) -> Path:
    """Salva uma fatia 2D completa, sem recorte automático do Matplotlib.

    O tamanho da figura acompanha a proporção real da fatia e os limites dos
    eixos incluem explicitamente todos os pixels. Isso evita que o backend do
    notebook ou ``bbox_inches='tight'`` remova parte da imagem.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = _create_volume_slice_figure(
        image_volume,
        slice_index,
        axis=axis,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin=origin,
        dpi=dpi,
        figure_width=figure_width,
        padding_fraction=padding_fraction,
        output_size_px=output_size_px,
    )
    fig.savefig(output, dpi=dpi, pad_inches=0, facecolor="black")
    plt.close(fig)
    return output


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
    if image_volume.ndim != 3:
        raise ValueError("A imagem deve ser 3D.")

    if window_level is not None and window_width is not None:
        lower = window_level - window_width // 2
        upper = window_level + window_width // 2
        image_volume = np.clip(image_volume, lower, upper)

    # Traduz cada vista anatômica para o eixo reduzido pela projeção.
    axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}
    proj_func = {"max": np.max, "min": np.min, "mean": np.mean}[projection]

    n_views = len(views)
    fig, axes = plt.subplots(1, n_views, figsize=(6 * n_views, 6), dpi=dpi)
    if n_views == 1:
        axes = [axes]

    cmap_to_use = plt.get_cmap(cmap).reversed() if invert_cmap else cmap
    imshow_kwargs: dict[str, Any] = {"cmap": cmap_to_use}
    if vmin is not None:
        imshow_kwargs["vmin"] = vmin
    if vmax is not None:
        imshow_kwargs["vmax"] = vmax

    for i, view in enumerate(views):
        if view not in axis_map:
            raise ValueError(
                f"View inválida: {view}. Use 'axial', 'coronal' ou 'sagittal'."
            )
        mip = proj_func(image_volume, axis=axis_map[view])
        axes[i].imshow(mip, **imshow_kwargs)
        if show_labels:
            axes[i].set_title(
                f"MIP {view.capitalize()} ({projection}) - shape {mip.shape}"
            )
        axes[i].axis("off")

    if show_title:
        plt.suptitle(title, fontsize=16, y=0.96)
        plt.tight_layout(rect=(0, 0, 1, 0.94))
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
    n_slices = len(slices_indices)
    if n_slices == 0:
        raise ValueError("slices_indices não pode ser vazio.")

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
    plt.close(fig)


def visualize_circles_on_slices(
    image: NDArray,
    detected_circles: Sequence[dict],
    num_samples: int = 6,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = 100,
) -> None:
    """Sobrepõe círculos detectados em fatias amostradas do volume."""
    if not detected_circles:
        raise ValueError("detected_circles não pode ser vazio.")

    # Amostra uniformemente somente entre fatias que possuem círculos detectados.
    slice_indices = sorted(set([c["slice_index"] for c in detected_circles]))
    step = max(1, len(slice_indices) // num_samples)
    selected_slices = slice_indices[::step][:num_samples]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6), dpi=dpi)
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
    plt.close(fig)
