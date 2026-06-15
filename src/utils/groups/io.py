"""Exports agrupados para config, dataset, resultados e I/O geral."""

from importlib import import_module

_SYMBOL_TO_MODULE = {
    # project.config
    "deep_update_dict": "project.config",
    "load_config_json": "project.config",
    "normalize_runtime_config": "project.config",
    "save_config_json": "project.config",
    "scale_config_to_resolution": "project.config",
    "serialize_config_for_json": "project.config",
    # project.dataset
    "get_data_splits": "project.dataset",
    # project.results
    "create_timestamped_output_dir": "project.results",
    "get_batch_result_file": "project.results",
    "list_batch_result_files": "project.results",
    "make_result_dataframe": "project.results",
    "merge_batch_results": "project.results",
    "save_metadata": "project.results",
    "save_results": "project.results",
    # utils
    "dice_score": "utils",
    "extract_circular_region": "utils",
    "extract_square_region": "utils",
    "load_img_and_label": "utils",
    "load_json_file": "utils",
    "load_raw_img_and_label": "utils",
    "save_json_file": "utils",
    "save_nii_image": "utils",
    "save_npy_array": "utils",
    "segment_by_hu": "utils",
}


def __getattr__(name):
    if name in _SYMBOL_TO_MODULE:
        submodule = _SYMBOL_TO_MODULE[name]
        # Import relative to the parent `utils` package to avoid circular
        # absolute imports while the top-level package is still initializing.
        parent_pkg = __package__.rpartition(".")[0] or __package__
        module = import_module(f".{submodule}", parent_pkg)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
