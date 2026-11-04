"""Testes sintéticos do volume arterial e da retomada sem reler máscaras."""

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
from nibabel.loadsave import save as save_nifti
from nibabel.nifti1 import Nifti1Image

from utils.project.analysis import artery_volume as volume


class ArteryVolumeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "dataset"
        self.root.mkdir()
        self.csv = Path(self.temporary.name) / "analysis" / "artery_volumes.csv"

    def _save_label(
        self, img_id, data=None, spacing=(0.5, 0.5, 1.0), unit="mm", folder=""
    ):
        path = self.root / folder / f"{img_id}.label.nii.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        label = np.ones((2, 3, 4), dtype=np.uint8) if data is None else data
        image = Nifti1Image(label, np.diag([*spacing, 1]))
        image.header.set_xyzt_units(unit)
        save_nifti(image, path)
        return path

    def _load(self, **kwargs):
        return volume.load_or_compute_artery_volumes(
            self.csv, self.root, show_progress=False, **kwargs
        )

    def test_anisotropic_volume_selects_only_coronary_class(self):
        label = np.array([0, 1, 1, 2, 3, 0], dtype=np.uint8).reshape(2, 3, 1)
        before = label.copy()
        result = volume.calculate_artery_volume(label, (0.5, 0.8, 2.0))
        self.assertEqual(result["voxel_count"], 2)
        self.assertEqual(result["label_value"], 1)
        self.assertAlmostEqual(result["voxel_volume_mm3"], 0.8)
        self.assertAlmostEqual(result["volume_ml"], 0.0016)
        np.testing.assert_array_equal(label, before)

    def test_empty_mask_is_real_zero_and_invalid_inputs_raise(self):
        label = np.zeros((2, 2, 2), dtype=np.uint8)
        self.assertEqual(
            volume.calculate_artery_volume(label, (1, 1, 1))["volume_ml"], 0
        )
        for spacing in ((1, 1), (1, 0, 1), (-1, 1, 1), (1, np.nan, 1), (1, np.inf, 1)):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                volume.calculate_artery_volume(label, spacing)
        with self.assertRaises(ValueError):
            volume.calculate_artery_volume(np.zeros((2, 2)), (1, 1, 1))
        with self.assertRaises(ValueError):
            volume.calculate_artery_volume(np.full((1, 1, 1), np.nan), (1, 1, 1))

    def test_iqr_and_std_outliers_use_only_valid_volumes(self):
        values = list(range(9)) + [100]
        frame = pd.DataFrame(
            {
                "IMG_ID": range(1, 12),
                "volume_ml": [*values, None],
                "status": [*["ok"] * 10, "error"],
            }
        )
        for method in ("iqr", "std"):
            with self.subTest(method=method):
                classified, limits = volume.classify_artery_volume_outliers(
                    frame, method=method
                )
                self.assertEqual(limits["exam_count"], 10)
                self.assertEqual(
                    classified.loc[classified.is_volume_outlier, "IMG_ID"].tolist(),
                    [10],
                )
                self.assertEqual(classified.iloc[-1]["volume_outlier_group"], "high")
                self.assertAlmostEqual(limits["std_ml"], np.std(values, ddof=1))
        self.assertNotIn("is_volume_outlier", frame.columns)

    def test_outlier_bounds_are_exclusive_and_constant_volumes_have_zero_zscore(self):
        frame = pd.DataFrame(
            {
                "IMG_ID": range(9),
                "volume_ml": [1, 1, 1, 1, 2, 3, 3, 3, 3],
                "status": "ok",
            }
        )
        classified, limits = volume.classify_artery_volume_outliers(
            frame, iqr_multiplier=0
        )
        self.assertEqual((limits["lower_limit_ml"], limits["upper_limit_ml"]), (1, 3))
        self.assertFalse(classified.is_volume_outlier.any())
        frame["volume_ml"] = 2.0
        classified, _ = volume.classify_artery_volume_outliers(frame, method="std")
        self.assertFalse(classified.is_volume_outlier.any())
        self.assertTrue(classified.volume_z_score.eq(0).all())
        empty, limits = volume.classify_artery_volume_outliers(frame.iloc[:0])
        self.assertTrue(empty.empty)
        self.assertEqual(limits["exam_count"], 0)

    def test_outlier_parameters_and_duplicate_ids_are_rejected(self):
        frame = pd.DataFrame({"IMG_ID": [1, 1], "volume_ml": [2, 3], "status": "ok"})
        with self.assertRaises(ValueError):
            volume.classify_artery_volume_outliers(frame)
        invalid_options: tuple[dict[str, Any], ...] = (
            {"method": "unknown"},
            {"iqr_multiplier": -1},
            {"std_multiplier": np.inf},
        )
        for kwargs in invalid_options:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                volume.classify_artery_volume_outliers(frame.iloc[:1], **kwargs)

    def test_csv_roundtrip_does_not_load_completed_labels(self):
        self._save_label(10, folder="subset")
        first = self._load()
        self.assertEqual(first.loc[0, "label_file"], "subset/10.label.nii.gz")
        self.assertAlmostEqual(first.loc[0, "volume_ml"], 24 * 0.25 / 1000)
        with patch.object(volume, "load_nifti") as loader:
            second = self._load()
        loader.assert_not_called()
        np.testing.assert_allclose(first["volume_ml"], second["volume_ml"])
        self.assertEqual(second["status"].tolist(), ["ok"])
        self.assertFalse(any(self.csv.parent.glob("*.tmp")))

    def test_interruption_preserves_each_completed_exam(self):
        for img_id in (1, 2, 3):
            self._save_label(img_id)
        original_loader = volume.load_nifti

        def interrupt_second(path):
            if Path(path).name.startswith("2."):
                raise KeyboardInterrupt
            return original_loader(path)

        with (
            patch.object(volume, "load_nifti", side_effect=interrupt_second),
            self.assertRaises(KeyboardInterrupt),
        ):
            self._load()
        saved = pd.read_csv(self.csv)
        self.assertEqual(saved["IMG_ID"].tolist(), [1])
        with patch.object(volume, "load_nifti", wraps=original_loader) as loader:
            resumed = self._load()
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(resumed["IMG_ID"].tolist(), [1, 2, 3])

    def test_new_and_failed_files_are_processed_without_reloading_success(self):
        self._save_label(1)
        damaged = self.root / "2.label.nii.gz"
        damaged.write_text("not a nifti", encoding="utf-8")
        first = self._load().set_index("IMG_ID")
        self.assertEqual(first.loc[2, "status"], "error")
        self.assertTrue(pd.isna(first.loc[2, "volume_ml"]))
        self.assertTrue(first.loc[2, "error"])
        self._save_label(2)
        self._save_label(3)
        with patch.object(volume, "load_nifti", wraps=volume.load_nifti) as loader:
            second = self._load()
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(second["status"].tolist(), ["ok", "ok", "ok"])
        self.assertTrue(second["error"].isna().all())

    def test_cached_csv_remains_readable_when_dataset_is_unmounted(self):
        self._save_label(1)
        self._load()
        with (
            patch.object(volume, "load_nifti") as loader,
            self.assertWarnsRegex(UserWarning, "sem verificar"),
        ):
            cached = volume.load_or_compute_artery_volumes(
                self.csv, self.root / "unmounted", show_progress=False
            )
        loader.assert_not_called()
        self.assertEqual(cached["IMG_ID"].tolist(), [1])
        with (
            patch.object(
                volume, "resolve_imagecas_base_path", side_effect=FileNotFoundError
            ),
            self.assertWarns(UserWarning),
        ):
            self.assertEqual(len(volume.load_or_compute_artery_volumes(self.csv)), 1)

    def test_force_recompute_reads_changed_label_and_requires_dataset(self):
        self._save_label(1)
        first = self._load()
        self._save_label(1, data=np.zeros((2, 2, 2), dtype=np.uint8))
        self.assertEqual(self._load().loc[0, "volume_ml"], first.loc[0, "volume_ml"])
        with patch.object(volume, "load_nifti", wraps=volume.load_nifti) as loader:
            recalculated = self._load(force_recompute=True)
        loader.assert_called_once()
        self.assertEqual(recalculated.loc[0, "volume_ml"], 0)
        previous = self.csv.read_bytes()
        with self.assertRaises(FileNotFoundError):
            volume.load_or_compute_artery_volumes(
                self.csv, self.root / "missing", force_recompute=True
            )
        self.assertEqual(previous, self.csv.read_bytes())

    def test_nifti_meter_units_are_converted_to_mm(self):
        self._save_label(1, spacing=(0.001, 0.001, 0.001), unit="meter")
        row = self._load().iloc[0]
        self.assertAlmostEqual(row["spacing_x_mm"], 1, places=6)
        self.assertAlmostEqual(row["volume_ml"], 0.024, places=6)

    def test_missing_dataset_and_empty_inventory_do_not_create_csv(self):
        with self.assertRaises(FileNotFoundError):
            self._load()
        self.assertFalse(self.csv.exists())
        with self.assertRaises(FileNotFoundError):
            volume.load_or_compute_artery_volumes(self.csv, self.root / "missing")

    def test_duplicate_ids_are_rejected_before_overwriting_csv(self):
        self._save_label(1)
        self._load()
        previous = self.csv.read_bytes()
        self._save_label(1, folder="duplicate")
        with self.assertRaisesRegex(ValueError, "duplicado"):
            self._load()
        self.assertEqual(previous, self.csv.read_bytes())

    def test_invalid_cache_is_not_silently_reused_or_overwritten(self):
        self._save_label(1)
        self._load()
        dataframe = pd.read_csv(self.csv)
        dataframe.loc[0, "volume_ml"] = -1
        dataframe.to_csv(self.csv, index=False)
        previous = self.csv.read_bytes()
        with self.assertRaises(ValueError):
            self._load()
        self.assertEqual(previous, self.csv.read_bytes())

    def test_notebook_cells_run_with_synthetic_data_and_all_error_results(self):
        import matplotlib
        import nbformat

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        notebook_path = (
            Path(__file__).resolve().parents[2]
            / "src/eda/images/artery_volume_analysis.ipynb"
        )
        notebook = nbformat.read(notebook_path, as_version=4)
        nbformat.validate(notebook)
        self._save_label(1)
        self._save_label(2, data=np.zeros((2, 2, 2), dtype=np.uint8))
        for fail in (False, True):
            namespace = {}
            original_loader = volume.load_nifti
            with (
                patch.object(
                    volume, "resolve_imagecas_base_path", return_value=self.root
                ),
                patch.object(
                    volume,
                    "load_nifti",
                    side_effect=OSError("synthetic") if fail else original_loader,
                ),
                patch.object(plt, "show", side_effect=lambda: plt.close("all")),
                patch("plotly.basedatatypes.BaseFigure.show"),
                redirect_stdout(StringIO()),
            ):
                for cell in notebook.cells:
                    if cell.cell_type != "code":
                        continue
                    self.assertIsNone(cell.execution_count)
                    self.assertEqual(cell.outputs, [])
                    exec(compile(cell.source, str(notebook_path), "exec"), namespace)
                    if cell.id == "configuration":
                        namespace.update(
                            REPO_ROOT=Path(self.temporary.name),
                            CSV_PATH=self.csv,
                            FORCE_RECOMPUTE=True,
                            PLOT_SPLIT="train",
                            display=lambda *_args, **_kwargs: None,
                            get_default_split_paths=lambda root: {
                                resolution: {"train": root / resolution / "train"}
                                for resolution in ("mid_res", "high_res")
                            },
                            load_split_results=lambda _paths, _resolution, split: (
                                pd.DataFrame(
                                    {
                                        "IMG_ID": [1, 2],
                                        "dice_artery": [0.8, 0.2],
                                        "status": ["both_correct", "one_correct"],
                                    }
                                )
                                if split == "train"
                                else None
                            ),
                        )
            self.assertEqual(len(namespace["results_df"]), 2)
            self.assertEqual(len(namespace["valid_df"]), 0 if fail else 2)
            self.assertEqual(len(namespace["comparison_df"]), 0 if fail else 4)
            self.assertEqual(len(namespace["volume_dice_pairs"]), 0 if fail else 2)
            figure = namespace["volume_dice_figure"]
            if fail:
                self.assertIsNone(figure)
            else:
                selected = figure.layout.updatemenus[0].buttons[1].args[0]["visible"]
                self.assertEqual(
                    sum(
                        len(trace.x)
                        for trace, visible in zip(figure.data, selected)
                        if visible
                    ),
                    2,
                )
