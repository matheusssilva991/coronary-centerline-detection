"""Headless visual artifacts for batch segmentation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


STAGE_VIEW_FILENAMES = {
    "mip": "mip_axial.png",
    "first": "first_slice.png",
    "middle": "middle_slice.png",
    "last": "last_slice.png",
}


def _finite_display_limits(volume: NDArray[Any]) -> tuple[float, float]:
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
    """Save axial MIP and first, middle and last axial slices for one stage."""
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError("volume deve ser tridimensional.")
    if array.shape[2] == 0:
        raise ValueError("volume não pode possuir eixo axial vazio.")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    if vmin is None or vmax is None:
        automatic_min, automatic_max = _finite_display_limits(array)
        vmin = automatic_min if vmin is None else vmin
        vmax = automatic_max if vmax is None else vmax

    middle_index = array.shape[2] // 2
    views = {
        "mip": (np.max(array, axis=2), "MIP axial"),
        "first": (array[:, :, 0], "Primeira fatia (0)"),
        "middle": (array[:, :, middle_index], f"Fatia média ({middle_index})"),
        "last": (array[:, :, -1], f"Última fatia ({array.shape[2] - 1})"),
    }
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
    """Save representative axial slices with detected aorta circles."""
    if not detected_circles:
        raise ValueError("detected_circles não pode ser vazio.")
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError("image deve ser tridimensional.")
    if vmin is None or vmax is None:
        automatic_min, automatic_max = _finite_display_limits(array)
        vmin = automatic_min if vmin is None else vmin
        vmax = automatic_max if vmax is None else vmax

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
    for axis in axes_array[len(selected_slices) :]:
        axis.axis("off")

    figure.suptitle("Círculos da aorta após filtro de trajetória")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)
    return destination


__all__ = [
    "STAGE_VIEW_FILENAMES",
    "save_detected_circles_figure",
    "save_stage_views",
]
