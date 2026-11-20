"""Descobre e carrega CCTA dos bancos OrCaScore, MM-WHS e ImageCAS."""

from __future__ import annotations

import zlib
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nibabel.loadsave import load as load_nifti
from nibabel.orientations import aff2axcodes
from nibabel.spatialimages import SpatialImage
from numpy.typing import NDArray

from utils.project.dataframe import require_series_column

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
    """Lê campos escalares do cabeçalho MetaImage ``.mhd``.

    Args:
        path (str | Path): Caminho do arquivo de cabeçalho ``.mhd``.

    Returns:
        dict[str, str]: Mapeamento de chaves para valores textuais sem espaços.
    """
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
    """Extrai dimensões e espaçamentos 3D de um cabeçalho MetaImage.

    Args:
        header (Mapping[str, str]): Campos lidos de um arquivo ``.mhd``.

    Returns:
        tuple[tuple[int, ...], tuple[float, ...]]: Tupla ``(shape_xyz, spacing_xyz)``
            para volumes tridimensionais.
    """
    shape_xyz = tuple(int(value) for value in header["DimSize"].split())
    spacing_xyz = tuple(float(value) for value in header["ElementSpacing"].split())
    if len(shape_xyz) != 3 or len(spacing_xyz) != 3:
        raise ValueError("A análise suporta apenas volumes MHD tridimensionais.")
    return shape_xyz, spacing_xyz


def load_mhd_volume(path: str | Path) -> NDArray[np.generic]:
    """Carrega um volume MetaImage 3D na ordem ``(z, y, x)``.

    Args:
        path (str | Path): Caminho do arquivo ``.mhd`` de entrada.

    Returns:
        NDArray[np.generic]: Array NumPy 3D na convenção ``(z, y, x)``.
    """
    # Lê metadados geométricos e valida o tipo de voxel declarado.
    header_path = Path(path)
    header = read_mhd_header(header_path)
    shape_xyz, _ = _mhd_geometry(header)
    element_type = header.get("ElementType")
    if element_type not in _MHD_DTYPES:
        raise ValueError(f"ElementType MHD não suportado: {element_type!r}")

    # Configura ordem de bytes conforme BinaryDataByteOrderMSB para tipos multibyte.
    dtype = _MHD_DTYPES[element_type]
    byte_order_msb = header.get("BinaryDataByteOrderMSB", "False").lower() == "true"
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder(">" if byte_order_msb else "<")

    # Lê o arquivo binário externo e descomprime caso CompressedData esteja ativo.
    data_file = header.get("ElementDataFile")
    if not data_file or data_file.upper() == "LOCAL":
        raise ValueError("ElementDataFile ausente ou LOCAL não é suportado.")
    raw = (header_path.parent / data_file).read_bytes()
    if header.get("CompressedData", "False").lower() == "true":
        raw = zlib.decompress(raw)

    # Confere o total de voxels e reorganiza o buffer plano para (z, y, x).
    expected_voxels = int(np.prod(shape_xyz))
    volume = np.frombuffer(raw, dtype=dtype)
    if volume.size != expected_voxels:
        raise ValueError(
            f"Volume MHD possui {volume.size} voxels; esperado {expected_voxels}."
        )
    return volume.reshape(tuple(reversed(shape_xyz)))


def load_nifti_volume_xyz(path: str | Path) -> NDArray[np.generic]:
    """Carrega uma imagem NIfTI na disposição nativa ``(x, y, z)``.

    Args:
        path (str | Path): Caminho do arquivo NIfTI tridimensional.

    Returns:
        NDArray[np.generic]: Volume 3D em ``(x, y, z)``, convertido para ``float32``
            quando armazenado como ``float64``.
    """
    image = _load_spatial_image(path)
    volume_xyz = np.asanyarray(image.dataobj)
    if volume_xyz.dtype == np.float64:
        volume_xyz = volume_xyz.astype(np.float32)
    if volume_xyz.ndim != 3:
        raise ValueError("A análise suporta apenas volumes NIfTI tridimensionais.")
    return volume_xyz


def load_mhd_volume_xyz(path: str | Path) -> NDArray[np.generic]:
    """Carrega um volume MetaImage na disposição nativa ``(x, y, z)``.

    Args:
        path (str | Path): Caminho do arquivo ``.mhd`` de entrada.

    Returns:
        NDArray[np.generic]: Volume 3D transposto para ``(x, y, z)``.
    """
    return np.transpose(load_mhd_volume(path), (2, 1, 0))


def _load_spatial_image(path: str | Path) -> SpatialImage:
    """Carrega e valida que o arquivo NIfTI representa uma ``SpatialImage``.

    Args:
        path (str | Path): Caminho do arquivo NIfTI a abrir.

    Returns:
        SpatialImage: Objeto espacial NiBabel carregado.
    """
    image = load_nifti(str(path))
    if not isinstance(image, SpatialImage):
        raise ValueError(f"O arquivo não contém uma imagem espacial NIfTI: {path}")
    return image


def _orcascore_acquisition_records(
    base_path: Path,
    acquisitions: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    """Monta registros OrCaScore para os tipos de aquisição solicitados.

    Args:
        base_path (Path): Diretório raiz do banco OrCaScore.
        acquisitions (tuple[tuple[str, str], ...]): Pares ``(sufixo, aquisicao)`` a
            descobrir nos diretórios de treino e teste.

    Returns:
        list[dict[str, Any]]: Registros geométricos com o campo ``acquisition``.
    """
    records: list[dict[str, Any]] = []
    # Percorre os subconjuntos oficiais procurando os sufixos de aquisição pedidos.
    for subset_dir, subset in (("Training_set", "train"), ("Test_set", "test")):
        image_dir = base_path / subset_dir / "Images"
        for suffix, acquisition in acquisitions:
            for path in sorted(image_dir.glob(f"*{suffix}.mhd")):
                # Extrai metadados do cabeçalho .mhd sem carregar os dados binários.
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
    """Lista apenas os registros CTAI com contraste do OrCaScore.

    Args:
        base_path (Path): Diretório raiz do banco OrCaScore.

    Returns:
        list[dict[str, Any]]: Registros geométricos das aquisições com contraste.
    """
    return _orcascore_acquisition_records(base_path, (("CTAI", "contrast"),))


def discover_orcascore_acquisitions(base_path: str | Path) -> pd.DataFrame:
    """Inventaria aquisições OrCaScore pareadas com e sem contraste.

    Args:
        base_path (str | Path): Diretório raiz do banco OrCaScore.

    Returns:
        pd.DataFrame: Inventário tabular das aquisições CTAI e CTI encontradas.
    """
    records = _orcascore_acquisition_records(
        Path(base_path),
        (("CTAI", "contrast"), ("CTI", "noncontrast")),
    )
    if not records:
        raise FileNotFoundError("Nenhum volume CTI/CTAI foi encontrado no OrCaScore.")
    return pd.DataFrame.from_records(records)


def _mmwhs_records(base_path: Path) -> list[dict[str, Any]]:
    """Monta registros NIfTI dos subsets de CT do MM-WHS.

    Args:
        base_path (Path): Diretório raiz do banco MM-WHS.

    Returns:
        list[dict[str, Any]]: Lista de registros geométricos das CTs encontradas.
    """
    records: list[dict[str, Any]] = []
    # Itera pelos subconjuntos de tomografia ignorando volumes de ressonância.
    for subset_dir, subset in (("ct_train", "train"), ("ct_test", "test")):
        for path in sorted((base_path / subset_dir).glob("ct_*_image.nii.gz")):
            image = _load_spatial_image(path)
            shape_xyz = tuple(int(value) for value in image.shape)
            spacing_xyz = tuple(float(value) for value in image.header.get_zooms()[:3])
            exam_id = path.name.removesuffix("_image.nii.gz")
            # Verifica existência da máscara de referência pareada para este exame.
            label_candidate = path.with_name(
                path.name.replace("_image.nii.gz", "_label.nii.gz")
            )
            label_path = label_candidate if label_candidate.is_file() else None
            # Registra geometria e orientação obtidas apenas do cabeçalho NIfTI.
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
    """Monta registros NIfTI do ImageCAS em ordem numérica.

    Args:
        base_path (Path): Diretório contendo os arquivos ``*.img.nii.gz`` do ImageCAS.

    Returns:
        list[dict[str, Any]]: Lista ordenada de metadados geométricos dos exames.
    """
    records: list[dict[str, Any]] = []
    # Ordena os volumes numericamente pelo identificador antes de ler os cabeçalhos.
    paths = sorted(
        base_path.glob("*.img.nii.gz"),
        key=lambda path: int(path.name.removesuffix(".img.nii.gz")),
    )
    # Extrai dimensões, espaçamentos e códigos de orientação de cada exame.
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
    """Monta um registro tabular com geometria e espaçamento do volume.

    Args:
        dataset (str): Nome do banco de origem.
        subset (str): Subconjunto ou split do exame.
        exam_id (str): Identificador do exame no banco.
        path (Path): Caminho do arquivo de imagem.
        file_format (str): Formato de armazenamento (``MHD/ZRAW`` ou ``NIfTI``).
        shape_xyz (tuple[int, ...]): Dimensões em voxels na ordem ``(x, y, z)``.
        spacing_xyz (tuple[float, ...]): Espaçamentos físicos em mm em ``(x, y, z)``.
        dtype (str): Tipo escalar dos voxels declarado no cabeçalho.
        orientation (str): Orientação anatômica reportada pelo arquivo.
        label_path (Path | None): Caminho opcional da máscara de referência.
        aorta_label_value (int | None): Código inteiro da aorta no rótulo, se existir.

    Returns:
        dict[str, Any]: Registro padronizado com métricas físicas e caminhos do exame.
    """
    # Desempacota dimensões em voxels e espaçamentos físicos por eixo.
    size_x, size_y, size_z = shape_xyz
    spacing_x, spacing_y, spacing_z = spacing_xyz
    # Calcula campo de visão (FOV), cobertura axial e volume unitário de voxel.
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

    Args:
        orcascore_path (str | Path): Diretório raiz do banco OrCaScore.
        mmwhs_path (str | Path): Diretório raiz do banco MM-WHS.
        imagecas_path (str | Path | None): Diretório opcional do ImageCAS a incluir.

    Returns:
        pd.DataFrame: Inventário consolidado dos exames CCTA descobertos.
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
    """Inventaria um banco CCTA externo suportado.

    Args:
        dataset (str): Nome do banco externo (``"orcascore"`` ou ``"mmwhs"``).
        base_path (str | Path): Diretório raiz do banco informado.

    Returns:
        pd.DataFrame: Inventário tabular dos volumes CCTA encontrados para o banco.
    """
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
    """Carrega um registro do inventário na disposição nativa ``(x, y, z)``.

    Args:
        record (Mapping[str, Any] | pd.Series): Registro contendo ``path`` e
            ``file_format`` do exame.

    Returns:
        NDArray[np.generic]: Array 3D do volume carregado em ``(x, y, z)``.
    """
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
    """Carrega a máscara binária da aorta quando o inventário possui referência.

    Args:
        record (Mapping[str, Any] | pd.Series): Registro do inventário com caminho de
            rótulo, código da aorta e dimensões esperadas.

    Returns:
        NDArray[np.bool_] | None: Máscara booleana 3D da aorta em ``(x, y, z)`` ou
            ``None`` quando não há referência aplicável.
    """
    # Verifica se o registro declara caminho de rótulo e valor numérico da aorta.
    label_path_value = record.get("label_path")
    label_value = record.get("aorta_label_value")
    if not isinstance(label_path_value, (str, Path)):
        return None
    if not isinstance(label_value, (int, float, np.integer, np.floating)):
        return None
    if not np.isfinite(float(label_value)):
        return None

    # Carrega o volume de rótulos quando o arquivo existe no disco.
    label_path = Path(label_path_value)
    if not label_path.is_file():
        return None
    label = load_nifti_volume_xyz(label_path)

    # Valida que o rótulo possui exatamente as dimensões registradas para a imagem.
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

    # Extrai a classe binária da aorta e exige que ela esteja representada no rótulo.
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

    Args:
        volume (NDArray[np.generic]): Volume 3D na disposição nativa ``(x, y, z)``.
        dataset (str): Nome do banco de origem do volume.

    Returns:
        tuple[NDArray[np.generic], tuple[int, ...]]: Volume alinhado e tupla dos eixos
            invertidos.
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
    """Seleciona exames próximos aos quantis de fatias de cada banco.

    Args:
        inventory (pd.DataFrame): Inventário consolidado contendo ``dataset`` e
            ``slice_count``.
        quantiles (tuple[float, ...]): Quantis alvo na distribuição de contagem de
            fatias por banco.

    Returns:
        pd.DataFrame: Subconjunto sem exames repetidos representativo de cada banco.
    """
    selected_indices: list[Hashable] = []
    for _, group in inventory.groupby("dataset", sort=False):
        available = group.copy()
        slice_counts = require_series_column(group, "slice_count")
        for quantile in quantiles:
            if available.empty:
                break
            # Escolhe o exame ainda disponível mais próximo do quantil sem repetir IDs.
            target = float(np.asarray(slice_counts.quantile(quantile)).item())
            available_slice_counts = require_series_column(available, "slice_count")
            index = (available_slice_counts - target).abs().idxmin()
            selected_indices.append(index)
            available = available.drop(index=index)
    return inventory.loc[selected_indices].reset_index(drop=True)


def summarize_ccta_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    """Resume estatísticas geométricas de cada banco.

    Args:
        inventory (pd.DataFrame): Inventário consolidado dos exames CCTA.

    Returns:
        pd.DataFrame: Tabela sumária com contagens de splits, matrizes XY e medianas
            físicas por banco.
    """
    rows: list[dict[str, Any]] = []
    for dataset, group in inventory.groupby("dataset", sort=False):
        # Extrai séries geométricas e calcula o espaçamento médio no plano axial XY.
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
        # Consolida contagens por split, formatos XY e resumos mediana [mín; máx].
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
    """Conta ocorrências exatamente iguais ao valor informado em uma Series.

    Args:
        values (pd.Series): Série tabular a avaliar.
        expected (str): Valor textual esperado.

    Returns:
        int: Número de elementos iguais a ``expected``.
    """
    return int(np.count_nonzero(values.eq(expected).to_numpy()))


def _median_min_max(values: pd.Series) -> str:
    """Formata mediana, mínimo e máximo de uma Series numérica.

    Args:
        values (pd.Series): Série numérica a resumir.

    Returns:
        str: Texto formatado como ``mediana [mínimo; máximo]``.
    """
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
