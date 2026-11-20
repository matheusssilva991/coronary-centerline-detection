"""Confere pareamento CTI/referência sem reamostrar os rótulos de calcificação."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat
import numpy as np

from eda.images.view.orcascore_calcification_views import show_calcification_view
from utils.project.datasets.ccta import discover_orcascore_acquisitions
from utils.project.datasets.orcascore import (
    load_orcascore_calcification_reference,
    orcascore_calcification_reference_path,
)


class OrcascoreCalcificationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.labels = np.zeros((4, 5, 6), dtype=np.int16)
        self.labels[1, 2, 3] = 1
        self.labels[2, 1, 4] = 2
        self.labels[3, 4, 5] = 3
        self.image_path = self.write_volume("Images/CASECTI", self.labels * 150)
        self.reference_path = self.write_volume("Reference_standard/CASER", self.labels)
        self.record = discover_orcascore_acquisitions(self.root).iloc[0]

    def write_volume(self, relative, volume, **overrides):
        path = self.root / "Training_set" / f"{relative}.mhd"
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = path.with_suffix(".raw")
        raw.write_bytes(volume.transpose(2, 1, 0).astype("<i2").tobytes())
        fields = {
            "ObjectType": "Image",
            "NDims": "3",
            "BinaryDataByteOrderMSB": "False",
            "CompressedData": "False",
            "DimSize": " ".join(map(str, volume.shape)),
            "ElementSpacing": "0.4 0.5 2.5",
            "ElementType": "MET_SHORT",
            "TransformMatrix": "1 0 0 0 1 0 0 0 1",
            "Offset": "0 0 0",
            "AnatomicalOrientation": "RAI",
            "ElementDataFile": raw.name,
            **overrides,
        }
        path.write_text("\n".join(f"{key} = {value}" for key, value in fields.items()))
        return path

    def test_reference_matches_native_axes_and_all_three_classes(self):
        result = load_orcascore_calcification_reference(self.record)
        np.testing.assert_array_equal(result, self.labels)
        self.assertEqual(
            orcascore_calcification_reference_path(self.record), self.reference_path
        )

    def test_reference_cannot_be_attached_to_contrast_scan(self):
        record = self.record.copy()
        record["acquisition"] = "contrast"
        with self.assertRaisesRegex(ValueError, "sem contraste"):
            load_orcascore_calcification_reference(record)

    def test_mismatch_in_exam_filename_is_rejected(self):
        record = self.record.copy()
        record["exam_id"] = "OTHER"
        with self.assertRaisesRegex(ValueError, "ID do exame"):
            load_orcascore_calcification_reference(record)

    def test_absent_reference_differs_from_negative_reference(self):
        self.write_volume("Reference_standard/CASER", np.zeros_like(self.labels))
        result = load_orcascore_calcification_reference(self.record)
        self.assertIsNotNone(result)
        self.assertFalse(np.any(result))
        self.reference_path.unlink()
        self.assertIsNone(load_orcascore_calcification_reference(self.record))

    def test_incompatible_geometry_is_rejected_without_resampling(self):
        for field, value in {
            "DimSize": "5 4 6",
            "ElementSpacing": "0.4 0.5 1.25",
            "Offset": "1 0 0",
            "TransformMatrix": "-1 0 0 0 1 0 0 0 1",
            "CenterOfRotation": "1 0 0",
            "AnatomicalOrientation": "LAS",
        }.items():
            with self.subTest(field=field):
                self.write_volume(
                    "Reference_standard/CASER", self.labels, **{field: value}
                )
                with self.assertRaisesRegex(ValueError, "difere da CTI"):
                    load_orcascore_calcification_reference(self.record)

    def test_unknown_labels_are_rejected(self):
        labels = self.labels.copy()
        labels[0, 0, 0] = 4
        self.write_volume("Reference_standard/CASER", labels)
        with self.assertRaisesRegex(
            ValueError, "Classes de calcificação desconhecidas"
        ):
            load_orcascore_calcification_reference(self.record)

    def test_permanent_notebook_compiles_and_has_no_outputs(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "src/eda/images/orcascore_calcification_visualization.ipynb"
        )
        book = nbformat.read(path, as_version=4)
        nbformat.validate(book)
        for cell in book.cells:
            if cell.cell_type == "code":
                compile(cell.source, str(path), "exec")
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)

    def test_overlay_preserves_native_axes_and_reference_classes(self):
        ct_slice = np.arange(20).reshape(4, 5)
        label_slice = self.labels[:, :, 3]
        exported = []
        with patch("matplotlib.pyplot.show"):
            show_calcification_view(
                "comparison",
                calcification_available=True,
                calcification_ct_slice=ct_slice,
                calcification_label_slice=label_slice,
                calcification_label=self.labels,
                calcification_crop=tuple(slice(0, size) for size in self.labels.shape),
                calcification_z=3,
                spacing_xyz=(0.4, 0.5, 2.5),
                figure_width_mm=90,
                preview_dpi=72,
                export_dpi=600,
                window_level_hu=200,
                window_width_hu=800,
                calcification_alpha=0.5,
                calcification_margin_mm=10,
                calcification_slice_index=None,
                calcification_reference_path=self.reference_path,
                exam_id="CASE",
                subset="train",
                pair={"noncontrast": self.record},
                save_figure=lambda fig, _stem, metadata: exported.append(
                    (fig, metadata)
                ),
            )
        figure, metadata = exported[0]
        np.testing.assert_array_equal(figure.axes[0].images[0].get_array(), ct_slice)
        np.testing.assert_array_equal(figure.axes[2].images[0].get_array(), ct_slice)
        np.testing.assert_array_equal(
            figure.axes[2].images[1].get_array().filled(0), label_slice
        )
        self.assertEqual(metadata["acquisition"], "noncontrast")
        self.assertEqual(metadata["flipped_axes"], [])
        self.assertEqual(metadata["rotation_clockwise_degrees"], 0)


if __name__ == "__main__":
    unittest.main()
