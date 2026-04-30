"""Grouped exports for dataset/config/results/general I/O utilities - lazy loaded."""

from importlib import import_module

_SYMBOL_TO_MODULE = {
    # config_utils
    "deep_update_dict": "config_utils",
    "load_config_json": "config_utils",
    "normalize_runtime_config": "config_utils",
    "save_config_json": "config_utils",
    "scale_config_to_resolution": "config_utils",
    "serialize_config_for_json": "config_utils",
    # dataset_utils
    "get_data_splits": "dataset_utils",
    # results_utils
    "create_timestamped_output_dir": "results_utils",
    "make_result_dataframe": "results_utils",
    "save_metadata": "results_utils",
    "save_results": "results_utils",
    # utils
    "dice_score": "utils",
    "extract_circular_region": "utils",
    "extract_square_region": "utils",
    "load_img_and_label": "utils",
    "load_json_file": "utils",
    "load_raw_img_and_label": "utils",
    "normalize_image": "utils",
    "robust_normalize": "utils",
    "save_json_file": "utils",
    "save_nii_image": "utils",
    "save_npy_array": "utils",
    "segment_by_hu": "utils",
}


def __getattr__(name):
    if name in _SYMBOL_TO_MODULE:
        submodule = _SYMBOL_TO_MODULE[name]
        module = import_module(f"utils.{submodule}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
