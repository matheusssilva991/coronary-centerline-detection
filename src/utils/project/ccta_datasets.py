"""Discovery and loading helpers for the OrCaScore, MM-WHS and ImageCAS CCTA."""

from __future__ import annotations

import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from numpy.typing import NDArray


_MHD_DTYPES = {
    "MET_CHAR": np.dtype("i1"),
    "MET_UCHAR": np.dtype("u1"),
    "MET_SHORT": np.dtype("i2"),
    "MET_USHORT": np.dtype("u2"),
    "MET_INT": np.dtype("i4"),
    "MET_UINT": np.dtype("u4"),
    "MET_FLOAT": np.dtype("f4"),
    "MET_DOUBLE": np.dtype("f8"),
}


def read_mhd_header(path: str | Path) -> dict[str, str]:
    """Read scalar fields from a MetaImage ``.mhd`` header."""
    header_path = Path(path)
    fields: dict[str, str] = {}
    for raw_line in header_path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", maxsplit=1)
        fields[key.strip()] = value.strip()
    return fields


def _mhd_geometry(
    header: Mapping[str, str],
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    shape_xyz = tuple(int(value) for value in header["DimSize"].split())
    spacing_xyz = tuple(float(value) for value in header["ElementSpacing"].split())
    if len(shape_xyz) != 3 or len(spacing_xyz) != 3:
        raise ValueError("A análise suporta apenas volumes MHD tridimensionais.")
    return shape_xyz, spacing_xyz


def load_mhd_volume(path: str | Path) -> NDArray[np.generic]:
    """Load a 3-D MetaImage volume and return it in ``(z, y, x)`` order."""
    header_path = Path(path)
    header = read_mhd_header(header_path)
    shape_xyz, _ = _mhd_geometry(header)
    element_type = header.get("ElementType")
    if element_type not in _MHD_DTYPES:
        raise ValueError(f"ElementType MHD não suportado: {element_type!r}")

    dtype = _MHD_DTYPES[element_type]
    byte_order_msb = header.get("BinaryDataByteOrderMSB", "False").lower() == "true"
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder(">" if byte_order_msb else "<")

    data_file = header.get("ElementDataFile")
    if not data_file or data_file.upper() == "LOCAL":
        raise ValueError("ElementDataFile ausente ou LOCAL não é suportado.")
    raw = (header_path.parent / data_file).read_bytes()
    if header.get("CompressedData", "False").lower() == "true":
        raw = zlib.decompress(raw)

    expected_voxels = int(np.prod(shape_xyz))
    volume = np.frombuffer(raw, dtype=dtype)
    if volume.size != expected_voxels:
        raise ValueError(
            f"Volume MHD possui {volume.size} voxels; esperado {expected_voxels}."
        )
    return volume.reshape(tuple(reversed(shape_xyz)))


def load_nifti_volume_zyx(path: str | Path) -> NDArray[np.generic]:
    """Load a NIfTI image, orient it canonically and return ``(z, y, x)``."""
    image = nib.as_closest_canonical(nib.load(str(path)))
    volume_xyz = np.asanyarray(image.dataobj)
    if volume_xyz.dtype == np.float64:
        volume_xyz = volume_xyz.astype(np.float32)
    if volume_xyz.ndim != 3:
        raise ValueError("A análise suporta apenas volumes NIfTI tridimensionais.")
    return np.transpose(volume_xyz, (2, 1, 0))


def load_nifti_volume_xyz(path: str | Path) -> NDArray[np.generic]:
    """Load a NIfTI image in its native ``(x, y, z)`` voxel layout."""
    image = nib.load(str(path))
    volume_xyz = np.asanyarray(image.dataobj)
    if volume_xyz.dtype == np.float64:
        volume_xyz = volume_xyz.astype(np.float32)
    if volume_xyz.ndim != 3:
        raise ValueError("A análise suporta apenas volumes NIfTI tridimensionais.")
    return volume_xyz


def load_mhd_volume_xyz(path: str | Path) -> NDArray[np.generic]:
    """Load a MetaImage volume in native ``(x, y, z)`` voxel layout."""
    return np.transpose(load_mhd_volume(path), (2, 1, 0))


def _orcascore_records(base_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for subset_dir, subset in (("Training_set", "train"), ("Test_set", "test")):
        for path in sorted((base_path / subset_dir / "Images").glob("*CTAI.mhd")):
            header = read_mhd_header(path)
            shape_xyz, spacing_xyz = _mhd_geometry(header)
            records.append(
                _geometry_record(
                    dataset="OrCaScore",
                    subset=subset,
                    exam_id=path.stem.removesuffix("CTAI"),
                    path=path,
                    file_format="MHD/ZRAW",
                    shape_xyz=shape_xyz,
                    spacing_xyz=spacing_xyz,
                    dtype=header.get("ElementType", "unknown"),
                    orientation=header.get("AnatomicalOrientation", "unknown"),
                )
            )
    return records


def _mmwhs_records(base_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for subset_dir, subset in (("ct_train", "train"), ("ct_test", "test")):
        for path in sorted((base_path / subset_dir).glob("ct_*_image.nii.gz")):
            image = nib.load(str(path))
            shape_xyz = tuple(int(value) for value in image.shape)
            spacing_xyz = tuple(float(value) for value in image.header.get_zooms()[:3])
            exam_id = path.name.removesuffix("_image.nii.gz")
            records.append(
                _geometry_record(
                    dataset="MM-WHS",
                    subset=subset,
                    exam_id=exam_id,
                    path=path,
                    file_format="NIfTI",
                    shape_xyz=shape_xyz,
                    spacing_xyz=spacing_xyz,
                    dtype=str(image.get_data_dtype()),
                    orientation="".join(nib.aff2axcodes(image.affine)),
                )
            )
    return records


def _imagecas_records(base_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    paths = sorted(
        base_path.glob("*.img.nii.gz"),
        key=lambda path: int(path.name.removesuffix(".img.nii.gz")),
    )
    for path in paths:
        image = nib.load(str(path))
        shape_xyz = tuple(int(value) for value in image.shape)
        spacing_xyz = tuple(float(value) for value in image.header.get_zooms()[:3])
        records.append(
            _geometry_record(
                dataset="ImageCAS",
                subset="full",
                exam_id=path.name.removesuffix(".img.nii.gz"),
                path=path,
                file_format="NIfTI",
                shape_xyz=shape_xyz,
                spacing_xyz=spacing_xyz,
                dtype=str(image.get_data_dtype()),
                orientation="".join(nib.aff2axcodes(image.affine)),
            )
        )
    return records


def _geometry_record(
    *,
    dataset: str,
    subset: str,
    exam_id: str,
    path: Path,
    file_format: str,
    shape_xyz: tuple[int, ...],
    spacing_xyz: tuple[float, ...],
    dtype: str,
    orientation: str,
) -> dict[str, Any]:
    size_x, size_y, size_z = shape_xyz
    spacing_x, spacing_y, spacing_z = spacing_xyz
    return {
        "dataset": dataset,
        "subset": subset,
        "exam_id": exam_id,
        "path": path,
        "file_format": file_format,
        "dtype": dtype,
        "reported_orientation": orientation,
        "size_x": size_x,
        "size_y": size_y,
        "slice_count": size_z,
        "spacing_x_mm": spacing_x,
        "spacing_y_mm": spacing_y,
        "spacing_z_mm": spacing_z,
        "fov_x_mm": size_x * spacing_x,
        "fov_y_mm": size_y * spacing_y,
        "coverage_z_mm": size_z * spacing_z,
        "voxel_volume_mm3": spacing_x * spacing_y * spacing_z,
    }


def discover_ccta_volumes(
    orcascore_path: str | Path,
    mmwhs_path: str | Path,
    imagecas_path: str | Path | None = None,
) -> pd.DataFrame:
    """Inventory only contrast-enhanced CCTA images from the datasets.

    OrCaScore ``*CTI.mhd`` non-contrast scans, reference masks, MM-WHS MRI
    volumes and MM-WHS labels are intentionally excluded.
    """
    records: list[dict[str, Any]] = [
        *_orcascore_records(Path(orcascore_path)),
        *_mmwhs_records(Path(mmwhs_path)),
    ]
    if imagecas_path is not None:
        records.extend(_imagecas_records(Path(imagecas_path)))
    if not records:
        raise FileNotFoundError("Nenhum volume CCTA foi encontrado nos bancos.")
    return pd.DataFrame.from_records(records)


def discover_ccta_dataset(
    dataset: str,
    base_path: str | Path,
) -> pd.DataFrame:
    """Inventory one supported external CCTA dataset."""
    dataset_key = dataset.strip().lower().replace("_", "-")
    if dataset_key in {"orcascore", "orca-score", "orca"}:
        records = _orcascore_records(Path(base_path))
    elif dataset_key in {"mm-whs", "mmwhs", "whs"}:
        records = _mmwhs_records(Path(base_path))
    else:
        raise ValueError("dataset deve ser 'orcascore' ou 'mmwhs'.")

    if not records:
        raise FileNotFoundError(
            f"Nenhum volume CCTA de {dataset!r} foi encontrado em {base_path}."
        )
    return pd.DataFrame.from_records(records)


def load_ccta_volume(record: Mapping[str, Any] | pd.Series) -> NDArray[np.generic]:
    """Load one inventory record in native ``(x, y, z)`` voxel layout."""
    path = Path(record["path"])
    if record["file_format"] == "MHD/ZRAW":
        return load_mhd_volume_xyz(path)
    if record["file_format"] == "NIfTI":
        return load_nifti_volume_xyz(path)
    raise ValueError(f"Formato não suportado: {record['file_format']!r}")


def align_ccta_volume_to_imagecas_view(
    volume: NDArray[np.generic],
    dataset: str,
) -> tuple[NDArray[np.generic], tuple[int, ...]]:
    """Align an external CCTA volume with the ImageCAS visual convention.

    OrCaScore requires a flip of axis 1 to remove the horizontal mirroring
    observed against ImageCAS. The transform preserves the axial slice order,
    voxel values and spacing. MM-WHS and ImageCAS are returned unchanged.
    """
    dataset_key = dataset.strip().lower().replace("_", "-")
    if dataset_key in {"orcascore", "orca-score", "orca"}:
        return volume[:, ::-1, :], (1,)
    return volume, ()


def select_representative_exams(
    inventory: pd.DataFrame,
    *,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> pd.DataFrame:
    """Select exams nearest to slice-count quantiles within each dataset."""
    selected_indices: list[int] = []
    for _, group in inventory.groupby("dataset", sort=False):
        available = group.copy()
        for quantile in quantiles:
            if available.empty:
                break
            target = float(group["slice_count"].quantile(quantile))
            index = (available["slice_count"] - target).abs().idxmin()
            selected_indices.append(int(index))
            available = available.drop(index=index)
    return inventory.loc[selected_indices].reset_index(drop=True)


def summarize_ccta_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    """Return compact geometry statistics for each dataset."""
    rows: list[dict[str, Any]] = []
    for dataset, group in inventory.groupby("dataset", sort=False):
        rows.append(
            {
                "dataset": dataset,
                "exams": len(group),
                "train": int(group["subset"].eq("train").sum()),
                "val": int(group["subset"].eq("val").sum()),
                "test": int(group["subset"].eq("test").sum()),
                "full": int(group["subset"].eq("full").sum()),
                "reported_orientation": ", ".join(
                    sorted(group["reported_orientation"].unique())
                ),
                "matrix_xy": ", ".join(
                    sorted({f"{x}x{y}" for x, y in zip(group.size_x, group.size_y)})
                ),
                "slices_median_min_max": _median_min_max(group["slice_count"]),
                "spacing_xy_mm_median_min_max": _median_min_max(
                    (group["spacing_x_mm"] + group["spacing_y_mm"]) / 2
                ),
                "spacing_z_mm_median_min_max": _median_min_max(group["spacing_z_mm"]),
                "coverage_z_mm_median_min_max": _median_min_max(group["coverage_z_mm"]),
            }
        )
    return pd.DataFrame(rows)


def _median_min_max(values: pd.Series) -> str:
    return f"{values.median():.3g} [{values.min():.3g}; {values.max():.3g}]"


__all__ = [
    "align_ccta_volume_to_imagecas_view",
    "discover_ccta_dataset",
    "discover_ccta_volumes",
    "load_ccta_volume",
    "load_mhd_volume",
    "load_mhd_volume_xyz",
    "load_nifti_volume_xyz",
    "load_nifti_volume_zyx",
    "read_mhd_header",
    "select_representative_exams",
    "summarize_ccta_inventory",
]
