import os
import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import nbformat

from tests.notebook_helpers import load_presentation_helpers
from utils.project.datasets.imagecas import DEFAULT_SPLIT_CONFIG_PATH
from utils.project.runtime.notebook_env import configure_notebook_environment
from utils.project.runtime.paths import find_repository_root


class ProjectLayoutTest(unittest.TestCase):
    def test_all_segmentation_visualization_and_utils_exports_resolve(self):
        for package_name, mapping_name in (
            ("utils.segmentation", "_SYMBOL_TO_MODULE"),
            ("utils.visualization", "_SYMBOL_TO_MODULE"),
            ("utils.comparison_utils", "_SYMBOL_TO_MODULE"),
            ("utils", "_LAZY_EXPORTS"),
        ):
            package = import_module(package_name)
            for symbol, module_name in getattr(package, mapping_name).items():
                with self.subTest(package=package_name, symbol=symbol):
                    destination = (
                        import_module(module_name)
                        if module_name.startswith("utils.")
                        else import_module(
                            module_name
                            if module_name.startswith(".")
                            else "." + module_name,
                            package_name,
                        )
                    )
                    self.assertIs(
                        getattr(package, symbol), getattr(destination, symbol)
                    )

    def test_grouped_modules_have_no_legacy_redirect_files(self):
        root = find_repository_root(Path(__file__))
        for name in (
            "aorta_localization",
            "aorta_segmentation",
            "aorta_diagnostics",
            "fuzzy_connectedness",
            "fuzzy_threshold",
        ):
            with self.subTest(module=name):
                self.assertFalse(
                    (root / "src/utils/segmentation" / f"{name}.py").exists()
                )
        for name in (
            "image_slices",
            "volume",
            "intensity",
            "hu_threshold",
            "preprocessing_views",
            "pipeline_artifacts",
            "hough",
            "vesselness",
            "comparison",
            "segmentation_eda",
            "variant_comparison",
        ):
            with self.subTest(module=name):
                self.assertFalse(
                    (root / "src/utils/visualization" / f"{name}.py").exists()
                )

    def test_root_and_split_config_do_not_depend_on_module_depth(self):
        root = find_repository_root(Path(__file__))
        self.assertEqual(
            DEFAULT_SPLIT_CONFIG_PATH, root / "config/imagecas_splits.json"
        )
        self.assertEqual(configure_notebook_environment(chdir_to_src=False), root)
        self.assertEqual(find_repository_root(root / "src/eda/results"), root)

    def test_root_search_requires_project_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "src" / "analysis" / "nested"
            nested.mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                find_repository_root(nested)
            (root / "pyproject.toml").write_text("[project]\nname='synthetic'\n")
            self.assertEqual(find_repository_root(nested), root)

    def test_notebook_lookup_rejects_absence_and_duplicates(self):
        with patch("tests.notebook_helpers.Path.rglob", return_value=[]):
            with self.assertRaisesRegex(FileNotFoundError, "não encontrado"):
                load_presentation_helpers("missing")
        with patch(
            "tests.notebook_helpers.Path.rglob",
            return_value=[Path("a/test.ipynb"), Path("b/test.ipynb")],
        ):
            with self.assertRaisesRegex(ValueError, "duplicado"):
                load_presentation_helpers("test")

    def test_all_notebooks_initialize_from_root_and_their_own_folder(self):
        root = find_repository_root(Path(__file__))
        notebooks = sorted((root / "src/eda").rglob("*.ipynb"))
        self.assertEqual(len(notebooks), 18)
        original_cwd = Path.cwd()
        original_sys_path = list(sys.path)
        try:
            for path in notebooks:
                notebook = nbformat.read(path, as_version=4)
                first = next(
                    cell for cell in notebook.cells if cell.cell_type == "code"
                )
                for cwd in (root, path.parent):
                    with self.subTest(notebook=path.name, cwd=cwd):
                        os.chdir(cwd)
                        # Obriga o bootstrap a adicionar src à busca de imports.
                        sys.path[:] = [
                            entry
                            for entry in original_sys_path
                            if entry != str(root / "src")
                        ]
                        namespace = {"__name__": "notebook_bootstrap"}
                        with patch(
                            "pandas.read_csv",
                            side_effect=AssertionError("leitura de run"),
                        ):
                            exec(compile(first.source, str(path), "exec"), namespace)
                        self.assertIn(str(root / "src"), sys.path)
                        self.assertEqual(namespace["REPO_ROOT"], root)
        finally:
            os.chdir(original_cwd)
            sys.path[:] = original_sys_path

    def test_results_facade_and_top_level_exports_remain_available(self):
        import utils
        import utils.project as project
        import utils.segmentation as segmentation
        from utils.project import results
        from utils.project.results.io import save_dataframe_atomic
        from utils.segmentation.pipeline.preprocessing import preprocess_ccta_volume

        self.assertIs(results.save_dataframe_atomic, save_dataframe_atomic)
        self.assertIs(segmentation.preprocess_ccta_volume, preprocess_ccta_volume)
        self.assertIs(utils.preprocess_ccta_volume, preprocess_ccta_volume)
        self.assertIs(
            project.configure_notebook_environment, configure_notebook_environment
        )
