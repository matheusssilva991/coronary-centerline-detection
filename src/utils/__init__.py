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

# GPU Utils
from .gpu_utils import (
    use_gpu,
    to_gpu,
    to_cpu,
    get_array_module,
)

# Configuração externa (JSON)
from .config_utils import (
    deep_update_dict,
    normalize_runtime_config,
    serialize_config_for_json,
    load_config_json,
    save_config_json,
    scale_config_to_resolution,
)

# Dataset utilities
from .dataset_utils import get_data_splits

# Result/report utilities
from .results_utils import (
    create_timestamped_output_dir,
    make_result_dataframe,
    save_results,
    save_metadata,
)

# Pipeline steps
from .pipeline_steps import (
    load_and_preprocess_image,
    get_or_compute_vesselness,
    get_or_detect_aorta_circles,
    get_or_segment_aorta,
    detect_and_evaluate_ostia,
    segment_arteries_from_ostia,
)

# Binary Operations (operações morfológicas + componentes conectados com GPU)
from .binary_operations import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    label,
    keep_largest_component,
)

# Pré-processamento
from .preprocessing import (
    downscale_image_ndi,
    downscale_image_opencv,
    downscale_image,
    threshold_image,
    threshold_image_with_offset,
    largest_connected_component,
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
    normalize_image,
    robust_normalize,
    load_json_file,
    save_json_file,
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
    # GPU Utils
    "use_gpu",
    "to_gpu",
    "to_cpu",
    "get_array_module",
    # Config
    "deep_update_dict",
    "normalize_runtime_config",
    "serialize_config_for_json",
    "load_config_json",
    "save_config_json",
    "scale_config_to_resolution",
    # Dataset
    "get_data_splits",
    # Results
    "create_timestamped_output_dir",
    "make_result_dataframe",
    "save_results",
    "save_metadata",
    # Pipeline steps
    "load_and_preprocess_image",
    "get_or_compute_vesselness",
    "get_or_detect_aorta_circles",
    "get_or_segment_aorta",
    "detect_and_evaluate_ostia",
    "segment_arteries_from_ostia",
    # Morphology
    "binary_closing",
    "binary_dilation",
    "binary_erosion",
    "binary_opening",
    "label",
    "keep_largest_component",
    # Preprocessing
    "downscale_image_ndi",
    "downscale_image_opencv",
    "downscale_image",
    "threshold_image",
    "threshold_image_with_offset",
    "largest_connected_component",
    "run_core_preprocessing_pipeline",
    # Plots
    "plot_mip_projection",
    "plot_slices",
    "visualize_circles_on_slices",
    "visualize_3d_k3d",
    "visualize_aorta_with_ostia",
    # Utils
    "normalize_image",
    "robust_normalize",
    "load_json_file",
    "save_json_file",
    "load_img_and_label",
    "load_raw_img_and_label",
    "save_nii_image",
    "extract_square_region",
    "extract_circular_region",
    "segment_by_hu",
    "dice_score",
    "save_npy_array",
]
