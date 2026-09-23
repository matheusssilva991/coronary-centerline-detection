"""Descobre e carrega CCTA dos bancos OrCaScore, MM-WHS e ImageCAS."""

from __future__ import annotations

import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nibabel.loadsave import load as load_nifti
from nibabel.orientations import aff2axcodes
from nibabel.spatialimages import SpatialImage
from numpy.typing import NDArray

from .dataframe import require_series_column


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
MMWHS_AORTA_LABEL_VALUE = 820


def read_mhd_header(path: str | Path) -> dict[str, str]:
    """Lê campos escalares do cabeçalho MetaImage ``.mhd``."""
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
    """Carrega um volume MetaImage 3D na ordem ``(z, y, x)``."""
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


def load_nifti_volume_xyz(path: str | Path) -> NDArray[np.generic]:
    """Carrega uma imagem NIfTI na disposição nativa ``(x, y, z)``."""
    image = _load_spatial_image(path)
    volume_xyz = np.asanyarray(image.dataobj)
    if volume_xyz.dtype == np.float64:
        volume_xyz = volume_xyz.astype(np.float32)
    if volume_xyz.ndim != 3:
        raise ValueError("A análise suporta apenas volumes NIfTI tridimensionais.")
    return volume_xyz


def load_mhd_volume_xyz(path: str | Path) -> NDArray[np.generic]:
    """Carrega um volume MetaImage na disposição nativa ``(x, y, z)``."""
    return np.transpose(load_mhd_volume(path), (2, 1, 0))


def _load_spatial_image(path: str | Path) -> SpatialImage:
    image = load_nifti(str(path))
    if not isinstance(image, SpatialImage):
        raise ValueError(f"O arquivo não contém uma imagem espacial NIfTI: {path}")
    return image


def _orcascore_acquisition_records(
    base_path: Path,
    acquisitions: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    """Monta registros OrCaScore para os tipos de aquisição solicitados."""
    records: list[dict[str, Any]] = []
    for subset_dir, subset in (("Training_set", "train"), ("Test_set", "test")):
        image_dir = base_path / subset_dir / "Images"
        for suffix, acquisition in acquisitions:
            for path in sorted(image_dir.glob(f"*{suffix}.mhd")):
                header = read_mhd_header(path)
                shape_xyz, spacing_xyz = _mhd_geometry(header)
                record = _geometry_record(
                    dataset="OrCaScore",
                    subset=subset,
                    exam_id=path.stem.removesuffix(suffix),
                    path=path,
                    file_format="MHD/ZRAW",
                    shape_xyz=shape_xyz,
                    spacing_xyz=spacing_xyz,
                    dtype=header.get("ElementType", "unknown"),
                    orientation=header.get("AnatomicalOrientation", "unknown"),
                )
                record["acquisition"] = acquisition
                records.append(record)
    return records


def _orcascore_records(base_path: Path) -> list[dict[str, Any]]:
    return _orcascore_acquisition_records(base_path, (("CTAI", "contrast"),))


def discover_orcascore_acquisitions(base_path: str | Path) -> pd.DataFrame:
    """Inventaria aquisições OrCaScore pareadas com e sem contraste."""
    records = _orcascore_acquisition_records(
        Path(base_path),
        (("CTAI", "contrast"), ("CTI", "noncontrast")),
    )
    if not records:
        raise FileNotFoundError("Nenhum volume CTI/CTAI foi encontrado no OrCaScore.")
    return pd.DataFrame.from_records(records)


def _mmwhs_records(base_path: Path) -> list[dict[str, Any]]:
    """Monta registros NIfTI dos subsets de CT do MM-WHS."""
    records: list[dict[str, Any]] = []
    for subset_dir, subset in (("ct_train", "train"), ("ct_test", "test")):
        for path in sorted((base_path / subset_dir).glob("ct_*_image.nii.gz")):
            image = _load_spatial_image(path)
            shape_xyz = tuple(int(value) for value in image.shape)
            spacing_xyz = tuple(float(value) for value in image.header.get_zooms()[:3])
            exam_id = path.name.removesuffix("_image.nii.gz")
            label_candidate = path.with_name(
                path.name.replace("_image.nii.gz", "_label.nii.gz")
            )
            label_path = label_candidate if label_candidate.is_file() else None
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
                    orientation="".join(aff2axcodes(image.affine)),
                    label_path=label_path,
                    aorta_label_value=(
                        MMWHS_AORTA_LABEL_VALUE if label_path is not None else None
                    ),
                )
            )
    return records


def _imagecas_records(base_path: Path) -> list[dict[str, Any]]:
    """Monta registros NIfTI do ImageCAS em ordem numérica."""
    records: list[dict[str, Any]] = []
    paths = sorted(
        base_path.glob("*.img.nii.gz"),
        key=lambda path: int(path.name.removesuffix(".img.nii.gz")),
    )
    for path in paths:
        image = _load_spatial_image(path)
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
                orientation="".join(aff2axcodes(image.affine)),
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
    label_path: Path | None = None,
    aorta_label_value: int | None = None,
) -> dict[str, Any]:
    """Monta um registro tabular com geometria e espaçamento do volume."""
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
        "label_path": label_path,
        "aorta_label_value": aorta_label_value,
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
    """Inventaria apenas imagens CCTA com contraste dos bancos.

    Exclui aquisições OrCaScore ``*CTI.mhd`` sem contraste, máscaras de
    referência, volumes de ressonância e rótulos do MM-WHS.
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
    """Inventaria um banco CCTA externo suportado."""
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
    """Carrega um registro do inventário na disposição nativa ``(x, y, z)``."""
    path_value = record.get("path")
    file_format = record.get("file_format")
    if not isinstance(path_value, (str, Path)):
        raise TypeError("O registro CCTA deve conter um caminho válido em 'path'.")
    if not isinstance(file_format, str):
        raise TypeError("O registro CCTA deve conter texto em 'file_format'.")

    path = Path(path_value)
    if file_format == "MHD/ZRAW":
        return load_mhd_volume_xyz(path)
    if file_format == "NIfTI":
        return load_nifti_volume_xyz(path)
    raise ValueError(f"Formato não suportado: {file_format!r}")


def load_ccta_aorta_ground_truth(
    record: Mapping[str, Any] | pd.Series,
) -> NDArray[np.bool_] | None:
    """Carrega a máscara binária da aorta quando o inventário possui referência."""
    label_path_value = record.get("label_path")
    label_value = record.get("aorta_label_value")
    if not isinstance(label_path_value, (str, Path)):
        return None
    if not isinstance(label_value, (int, float, np.integer, np.floating)):
        return None
    if not np.isfinite(float(label_value)):
        return None

    label_path = Path(label_path_value)
    if not label_path.is_file():
        return None
    label = load_nifti_volume_xyz(label_path)
    expected_dimensions: list[int] = []
    for column in ("size_x", "size_y", "slice_count"):
        dimension = record.get(column)
        if not isinstance(dimension, (int, float, np.integer, np.floating)):
            raise ValueError(f"Dimensão inválida no inventário: {column}.")
        if not np.isfinite(float(dimension)) or int(dimension) <= 0:
            raise ValueError(f"Dimensão inválida no inventário: {column}.")
        expected_dimensions.append(int(dimension))
    expected_shape = (
        expected_dimensions[0],
        expected_dimensions[1],
        expected_dimensions[2],
    )
    if label.shape != expected_shape:
        raise ValueError(
            "O label da aorta possui shape diferente da imagem inventariada: "
            f"{label.shape} != {expected_shape}."
        )

    aorta_mask = np.asarray(label == int(label_value), dtype=bool)
    if not np.any(aorta_mask):
        raise ValueError(f"O label não contém a classe de aorta {int(label_value)}.")
    return aorta_mask


def align_ccta_volume_to_imagecas_view(
    volume: NDArray[np.generic],
    dataset: str,
) -> tuple[NDArray[np.generic], tuple[int, ...]]:
    """Alinha um volume CCTA externo à convenção visual do ImageCAS.

    O OrCaScore exige inversão do eixo 1 para remover o espelhamento horizontal.
    A transformação preserva fatias, valores e espaçamento; os demais bancos
    permanecem inalterados.
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
    """Seleciona exames próximos aos quantis de fatias de cada banco."""
    selected_indices: list[int] = []
    for _, group in inventory.groupby("dataset", sort=False):
        available = group.copy()
        slice_counts = require_series_column(group, "slice_count")
        for quantile in quantiles:
            if available.empty:
                break
            target = float(np.asarray(slice_counts.quantile(quantile)).item())
            available_slice_counts = require_series_column(available, "slice_count")
            index = (available_slice_counts - target).abs().idxmin()
            selected_indices.append(int(index))
            available = available.drop(index=index)
    return inventory.loc[selected_indices].reset_index(drop=True)


def summarize_ccta_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    """Resume estatísticas geométricas de cada banco."""
    rows: list[dict[str, Any]] = []
    for dataset, group in inventory.groupby("dataset", sort=False):
        subset = require_series_column(group, "subset")
        orientation = require_series_column(group, "reported_orientation")
        size_x = require_series_column(group, "size_x")
        size_y = require_series_column(group, "size_y")
        slice_count = require_series_column(group, "slice_count")
        spacing_x = require_series_column(group, "spacing_x_mm")
        spacing_y = require_series_column(group, "spacing_y_mm")
        spacing_z = require_series_column(group, "spacing_z_mm")
        coverage_z = require_series_column(group, "coverage_z_mm")
        spacing_xy = pd.Series(
            (spacing_x.to_numpy(dtype=float) + spacing_y.to_numpy(dtype=float)) / 2
        )
        rows.append(
            {
                "dataset": dataset,
                "exams": len(group),
                "train": _count_equal(subset, "train"),
                "val": _count_equal(subset, "val"),
                "test": _count_equal(subset, "test"),
                "full": _count_equal(subset, "full"),
                "reported_orientation": ", ".join(sorted(orientation.unique())),
                "matrix_xy": ", ".join(
                    sorted(
                        {
                            f"{x}x{y}"
                            for x, y in zip(
                                size_x.to_numpy(), size_y.to_numpy(), strict=True
                            )
                        }
                    )
                ),
                "slices_median_min_max": _median_min_max(slice_count),
                "spacing_xy_mm_median_min_max": _median_min_max(spacing_xy),
                "spacing_z_mm_median_min_max": _median_min_max(spacing_z),
                "coverage_z_mm_median_min_max": _median_min_max(coverage_z),
            }
        )
    return pd.DataFrame(rows)


def _count_equal(values: pd.Series, expected: str) -> int:
    return int(np.count_nonzero(values.eq(expected).to_numpy()))


def _median_min_max(values: pd.Series) -> str:
    return f"{values.median():.3g} [{values.min():.3g}; {values.max():.3g}]"


__all__ = [
    "align_ccta_volume_to_imagecas_view",
    "discover_ccta_dataset",
    "discover_ccta_volumes",
    "discover_orcascore_acquisitions",
    "load_ccta_aorta_ground_truth",
    "load_ccta_volume",
    "load_mhd_volume",
    "load_mhd_volume_xyz",
    "load_nifti_volume_xyz",
    "read_mhd_header",
    "select_representative_exams",
    "summarize_ccta_inventory",
]
