"""
Pacote utils para segmentação de artérias coronárias.

Módulos disponíveis:
- artery_segmentation: Segmentação de artérias coronárias
- aorta_segmentation: Segmentação da aorta
- aorta_localization: Localização e detecção de círculos na aorta
- frangi: Filtro de Frangi para detecção de vasos
- preprocessing: Pré-processamento de imagens
- plots: Visualização de imagens e máscaras 3D
- utils: Funções utilitárias diversas
"""

# Localização dos ostios
from .ostia_detection import (
    find_aorta_surface,
    calculate_robust_diameter,
    check_ostium_intersection,
    find_ostia,
)

# Segmentação de artérias
from .artery_segmentation import (
    region_growing_segmentation,
    region_growing_article,
)

# Segmentação da aorta
from .aorta_segmentation import (
    level_set_segmentation,
    remove_leaks_morphology,
)

# Localização da aorta
from .aorta_localization import (
    detect_initial_circle,
    refine_circle_with_neighbors,
    detect_aorta_circles,
)

# Filtro de Frangi
from .frangi import (
    get_vesselness,
    get_vesselness_optimized,
    save_vesselness_cache,
    load_vesselness_cache,
)

# Pré-processamento
from .preprocessing import (
    normalize_image,
    downscale_image_ndi,
    threshold_image,
    threshold_image_with_offset,
    largest_connected_component,
    keep_largest_component,
    run_core_preprocessing_pipeline,
)

# Visualização
from .plots import (
    plot_mip_projection,
    plot_slices,
    visualize_circles_on_slices,
    visualize_3d_k3d,
    visualize_aorta_with_ostia,
)

# Utilitários gerais
from .utils import (
    load_img_and_label,
    load_raw_img_and_label,
    save_nii_image,
    extract_square_region,
    extract_circular_region,
    segment_by_hu,
    dice_score,
    save_npy_array,
)

__all__ = [
    # Ostia detection
    "find_aorta_surface",
    "calculate_robust_diameter",
    "check_ostium_intersection",
    "find_ostia",
    # Artery segmentation
    "region_growing_segmentation",
    "region_growing_article",
    # Aorta segmentation
    "level_set_segmentation",
    "remove_leaks_morphology",
    # Aorta localization
    "detect_initial_circle",
    "refine_circle_with_neighbors",
    "detect_aorta_circles",
    # Frangi filter
    "get_vesselness",
    "get_vesselness_optimized",
    "save_vesselness_cache",
    "load_vesselness_cache",
    # Preprocessing
    "normalize_image",
    "downscale_image",
    "threshold_image",
    "threshold_image_with_offset",
    "largest_connected_component",
    "keep_largest_component",
    "run_core_preprocessing_pipeline",
    # Plots
    "plot_mip_projection",
    "plot_slices",
    "visualize_circles_on_slices",
    "visualize_3d_k3d",
    "visualize_aorta_with_ostia",
    # Utils
    "load_img_and_label",
    "load_raw_img_and_label",
    "save_nii_image",
    "extract_square_region",
    "extract_circular_region",
    "segment_by_hu",
    "dice_score",
    "save_npy_array",
]
