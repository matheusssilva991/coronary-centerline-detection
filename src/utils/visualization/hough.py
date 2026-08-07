"""Visualizações de diagnóstico da detecção por Hough."""

from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def _resolve_cmap(cmap: str, invert_cmap: bool):
    return plt.get_cmap(cmap).reversed() if invert_cmap else cmap


def plot_hough_initial_diagnostics(
    img_slice: NDArray,
    diagnostics: dict[str, Any],
    title: str = "Transformada de Hough - círculo inicial",
    cmap: str = "gray",
    invert_cmap: bool = False,
    show_title: bool = True,
    show_subtitle: bool = True,
    dpi: int = 100,
) -> None:
    """Plota o círculo inicial, os candidatos de refinamento e o círculo refinado."""
    initial_circle = diagnostics.get("initial_circle")
    refined_circle = diagnostics.get("refined_circle")
    candidates = diagnostics.get("candidates", [])
    refinement_candidates = diagnostics.get("refinement_candidates", [])

    cmap_to_use = _resolve_cmap(cmap, invert_cmap)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)

    axes[0].imshow(img_slice, cmap=cmap_to_use, origin="upper")
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

    axes[1].imshow(img_slice, cmap=cmap_to_use, origin="upper")
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
        plt.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        plt.tight_layout()

    plt.show()
    plt.close(fig)


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
    """Plota apenas o círculo inicial detectado."""
    initial_circle = diagnostics.get("initial_circle")

    cmap_to_use = _resolve_cmap(cmap, invert_cmap)
    plt.figure(figsize=(6, 6), dpi=dpi)
    plt.imshow(img_slice, cmap=cmap_to_use, origin="upper")

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
    """Plota apenas os círculos vizinhos usados no refinamento."""
    refinement_candidates = diagnostics.get("refinement_candidates", [])

    cmap_to_use = _resolve_cmap(cmap, invert_cmap)
    plt.figure(figsize=(6, 6), dpi=dpi)
    plt.imshow(img_slice, cmap=cmap_to_use, origin="upper")

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
    """Plota apenas o círculo final refinado."""
    refined_circle = diagnostics.get("refined_circle")

    cmap_to_use = _resolve_cmap(cmap, invert_cmap)
    plt.figure(figsize=(6, 6), dpi=dpi)
    plt.imshow(img_slice, cmap=cmap_to_use, origin="upper")

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
        plt.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        plt.tight_layout()

    plt.show()
    plt.close()


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
) -> None:
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

    cmap_to_use = _resolve_cmap(cmap, invert_cmap)

    for ax, circle_idx in zip(axes, sample_indices, strict=False):
        circle = detected_circles[circle_idx]
        slice_idx = int(circle["slice_index"])
        ax.imshow(image_volume[:, :, slice_idx], cmap=cmap_to_use, origin="upper")
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
        plt.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        plt.tight_layout()

    plt.show()
    plt.close(fig)
