"""Compatibilidade para visualizações de imagens e diagnósticos."""

from .hough import (
    plot_hough_initial_circle,
    plot_hough_initial_diagnostics,
    plot_hough_refined_circle,
    plot_hough_refinement_candidates,
    plot_spaced_detected_circles,
)
from .image_slices import (
    plot_mip_projection,
    plot_slices,
    visualize_circles_on_slices,
)
from .preprocessing_views import plot_preprocessing_grid, plot_stage
from .vesselness import (
    compute_vesselness_maps,
    plot_vesselness_mip,
    plot_vesselness_mip_grid,
)

__all__ = [
    "compute_vesselness_maps",
    "plot_hough_initial_circle",
    "plot_hough_initial_diagnostics",
    "plot_hough_refined_circle",
    "plot_hough_refinement_candidates",
    "plot_mip_projection",
    "plot_preprocessing_grid",
    "plot_slices",
    "plot_spaced_detected_circles",
    "plot_stage",
    "plot_vesselness_mip",
    "plot_vesselness_mip_grid",
    "visualize_circles_on_slices",
]
