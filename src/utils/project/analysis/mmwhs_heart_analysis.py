"""Analisa intensidades HU dos rótulos cardíacos de referência do MM-WHS."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nibabel.funcs import as_closest_canonical
from numpy.typing import NDArray

from utils.project.dataframe import numeric_series, require_series_column
from utils.utils.nifti_io import load_raw_img_and_label
from utils.utils.roi import mask_bounding_box_slices

MMWHS_HEART_LABEL_SPECS: dict[int, tuple[str, int]] = {
    500: ("Ventrículo Esquerdo", 0xE53935),
    600: ("Ventrículo Direito", 0x1E88E5),
    420: ("Átrio Esquerdo", 0x8E24AA),
    550: ("Átrio Direito", 0x00ACC1),
    205: ("Miocárdio", 0xFDD835),
    820: ("Aorta Ascendente", 0xFB8C00),
    850: ("Artéria Pulmonar", 0x43A047),
}
COMBINED_REGION_NAMES = {
    "heart": "Coração completo",
    "background_full": "Fundo completo",
    "background_local": "Fundo próximo ao coração",
}


def select_mmwhs_heart_exams(
    inventory: pd.DataFrame,
    *,
    n_exams: int | None = 10,
    random_seed: int = 42,
    exam_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Seleciona pares CT/label de treino por IDs ou sorteio reproduzível."""
    train = inventory.loc[
        require_series_column(inventory, "dataset").eq("MM-WHS")
        & require_series_column(inventory, "subset").eq("train")
    ].copy()
    paths = require_series_column(train, "label_path")
    available = paths.map(
        lambda path: isinstance(path, (str, Path)) and Path(path).is_file()
    )
    train = train.loc[available].sort_values("exam_id").reset_index(drop=True)
    ids = require_series_column(train, "exam_id")
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("O inventário contém IDs de treino ausentes ou duplicados.")
    if train.empty:
        raise ValueError("Nenhum par CT/label de treino MM-WHS está disponível.")
    if exam_ids is not None:
        requested = list(exam_ids)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("EXAM_IDS deve conter IDs distintos e não pode ser vazio.")
        unknown = sorted(set(requested).difference(ids.tolist()))
        if unknown:
            raise ValueError(f"Exames sem par CT/label de treino: {unknown}")
        return train.set_index("exam_id").loc[requested].reset_index()
    if n_exams is not None:
        if n_exams <= 0:
            raise ValueError("N_EXAMS deve ser positivo ou None.")
        rng = np.random.default_rng(random_seed)
        indices = np.sort(
            rng.choice(len(train), size=min(n_exams, len(train)), replace=False)
        )
        train = train.iloc[indices]
    return train.reset_index(drop=True)


def load_mmwhs_heart_volume(
    image_path: str | Path,
    label_path: str | Path,
) -> tuple[NDArray[np.float32], NDArray[Any], tuple[float, float, float]]:
    """Carrega HU escalonados e labels canônicos na mesma grade física.

    A canonicalização apenas reordena/inverte eixos. Não reamostra volumes.
    Pares com shapes ou affines incompatíveis são rejeitados.
    """
    raw_image, raw_label = load_raw_img_and_label(str(image_path), str(label_path))
    if raw_label is None:
        raise ValueError(f"Label MM-WHS ausente: {label_path}")
    image_nii = as_closest_canonical(raw_image)
    label_nii = as_closest_canonical(raw_label)
    if len(image_nii.shape) != 3 or len(label_nii.shape) != 3:
        raise ValueError(f"Imagem e label devem ser 3D: {image_path}")
    if image_nii.shape != label_nii.shape or not np.allclose(
        image_nii.affine, label_nii.affine, rtol=0, atol=1e-5
    ):
        raise ValueError(f"Imagem e label possuem geometria incompatível: {image_path}")
    # dataobj aplica slope/intercept do NIfTI, incluindo offsets como −1024 HU.
    image = np.asarray(image_nii.dataobj, dtype=np.float32)
    label = np.asanyarray(label_nii.dataobj)
    zooms = image_nii.header.get_zooms()
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return image, label, spacing


def _region_masks(
    label: NDArray[Any],
    spacing: Sequence[float],
    margin_mm: float,
) -> Iterator[tuple[str, str, int | None, tuple[slice, ...], NDArray[np.bool_]]]:
    """Produz máscaras por região sem armazenar todas as máscaras do volume."""
    present = np.unique(label)
    unknown = present[~np.isin(present, (0, *MMWHS_HEART_LABEL_SPECS))]
    if unknown.size:
        raise ValueError(f"Valores de label MM-WHS desconhecidos: {unknown.tolist()}")
    heart = np.isin(label, tuple(MMWHS_HEART_LABEL_SPECS))
    crop = mask_bounding_box_slices(heart, spacing, margin_mm)
    full = (slice(None),) * 3
    for value, (name, _) in MMWHS_HEART_LABEL_SPECS.items():
        yield f"label_{value}", name, value, full, label == value
    yield "heart", COMBINED_REGION_NAMES["heart"], None, full, heart
    del heart
    yield (
        "background_full",
        COMBINED_REGION_NAMES["background_full"],
        0,
        full,
        label == 0,
    )
    yield (
        "background_local",
        COMBINED_REGION_NAMES["background_local"],
        0,
        crop,
        label[crop] == 0,
    )


def summarize_hu_region(
    image: NDArray[Any],
    mask: NDArray[Any],
    spacing: Sequence[float],
    *,
    exam_id: str,
    region: str,
    region_name: str,
    bin_edges: NDArray[Any] | None = None,
    percentiles: tuple[float, float] | None = None,
    label_value: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calcula HU e densidade de uma região sem alterar o volume original.

    Valores não finitos ficam fora das estatísticas; os finitos fora dos bins
    continuam no denominador da densidade. Regiões vazias não recebem HU zero.
    """
    if image.ndim != 3 or image.shape != mask.shape:
        raise ValueError("Imagem e máscara devem ser 3D e possuir o mesmo shape.")
    spacing_array = np.asarray(spacing, dtype=float)
    if (
        spacing_array.shape != (3,)
        or not np.isfinite(spacing_array).all()
        or np.any(spacing_array <= 0)
    ):
        raise ValueError("O espaçamento deve conter três valores finitos e positivos.")
    edges = (
        np.arange(-1500.0, 3001.0, 10.0)
        if bin_edges is None
        else np.asarray(bin_edges, dtype=float)
    )
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.isfinite(edges).all()
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("Os bins HU devem ser finitos e estritamente crescentes.")
    if percentiles is not None and not (
        np.isfinite(percentiles).all() and 0 <= percentiles[0] < percentiles[1] <= 100
    ):
        raise ValueError(
            "Os percentis devem satisfazer 0 <= inferior < superior <= 100."
        )
    values = image[np.asarray(mask, dtype=bool)]
    voxel_count = int(values.size)
    finite = np.isfinite(values)
    if not finite.all():
        values = values[finite]
    valid_count = int(values.size)
    below = int(np.count_nonzero(values < edges[0]))
    above = int(np.count_nonzero(values > edges[-1]))
    row: dict[str, Any] = {
        "exam_id": exam_id,
        "region": region,
        "region_name": region_name,
        "label_value": label_value,
        "voxel_count": voxel_count,
        "valid_voxel_count": valid_count,
        "nonfinite_voxel_count": voxel_count - valid_count,
        "volume_ml": voxel_count * (float(np.prod(spacing_array)) / 1000.0),
        "below_range_count": below,
        "above_range_count": above,
        "outside_range_percent": 100.0 * (below + above) / valid_count
        if valid_count
        else None,
    }
    row.update(
        dict.fromkeys(
            (
                "mean_hu",
                "std_hu",
                "median_hu",
                "q1_hu",
                "q3_hu",
                "p5_hu",
                "p95_hu",
                "min_hu",
                "max_hu",
            )
        )
    )
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    if percentiles is not None:
        row.update(lower_percentile_hu=None, upper_percentile_hu=None)
        quantiles.extend(value / 100 for value in percentiles)
    if valid_count:
        row["mean_hu"] = float(np.mean(values, dtype=np.float64))
        row["std_hu"] = float(np.std(values, dtype=np.float64, ddof=0))
        # A indexação criou uma cópia: ordenar não modifica os HU do CT.
        qs = np.quantile(values, quantiles, overwrite_input=True)
        row.update(
            p5_hu=float(qs[0]),
            q1_hu=float(qs[1]),
            median_hu=float(qs[2]),
            q3_hu=float(qs[3]),
            p95_hu=float(qs[4]),
            min_hu=float(values.min()),
            max_hu=float(values.max()),
        )
        if percentiles is not None:
            row.update(
                lower_percentile_hu=float(qs[5]), upper_percentile_hu=float(qs[6])
            )
    counts, _ = np.histogram(values, bins=edges)
    histogram = pd.DataFrame(
        {
            "exam_id": exam_id,
            "region": region,
            "region_name": region_name,
            "bin_center_hu": (edges[:-1] + edges[1:]) / 2,
            "count": counts,
            "density": counts / (valid_count * np.diff(edges))
            if valid_count
            else np.full(counts.size, np.nan),
        }
    )
    return row, histogram


def summarize_mmwhs_heart_hu(
    image: NDArray[Any],
    label: NDArray[Any],
    spacing: Sequence[float],
    *,
    exam_id: str,
    margin_mm: float = 10.0,
    bin_edges: NDArray[Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula estatísticas exatas e histogramas de todas as regiões de um exame.

    Valores não finitos são contados e excluídos das estatísticas. A densidade
    considera todos os voxels finitos, inclusive os que ficam fora dos bins.
    Labels ausentes recebem estatísticas vazias e não contribuem ao agregado.
    """
    if image.ndim != 3 or image.shape != label.shape:
        raise ValueError("Imagem e label devem ser 3D e possuir o mesmo shape.")
    rows: list[dict[str, Any]] = []
    histograms: list[pd.DataFrame] = []
    for region, name, value, slices, mask in _region_masks(label, spacing, margin_mm):
        row, histogram = summarize_hu_region(
            image[slices],
            mask,
            spacing,
            exam_id=exam_id,
            region=region,
            region_name=name,
            bin_edges=bin_edges,
            label_value=value,
        )
        histograms.append(histogram)
        rows.append(row)
        del mask
    return pd.DataFrame(rows), pd.concat(histograms, ignore_index=True)


def aggregate_mmwhs_heart_hu(
    statistics: pd.DataFrame,
    histograms: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combina estatísticas e densidades com o mesmo peso por exame válido."""
    if statistics.duplicated(["exam_id", "region"]).any():
        raise ValueError("Estatísticas contêm regiões duplicadas para um exame.")
    if histograms.duplicated(["exam_id", "region", "bin_center_hu"]).any():
        raise ValueError("Histogramas contêm bins duplicados para um exame/região.")
    rows: list[dict[str, Any]] = []
    for _, group in statistics.groupby("region", sort=False):
        valid = group.loc[numeric_series(group, "valid_voxel_count").gt(0)]
        means = numeric_series(valid, "mean_hu")
        rows.append(
            {
                "region": str(group.iloc[0]["region"]),
                "region_name": str(group.iloc[0]["region_name"]),
                "selected_exam_count": len(group),
                "exam_count": len(valid),
                "mean_of_means_hu": float(means.mean()) if len(valid) else None,
                "std_between_exam_means_hu": float(
                    np.std(means.to_numpy(dtype=np.float64), ddof=1)
                )
                if len(valid) > 1
                else None,
                "median_of_medians_hu": float(
                    numeric_series(valid, "median_hu").median()
                )
                if len(valid)
                else None,
                "mean_outside_range_percent": float(
                    numeric_series(valid, "outside_range_percent").mean()
                )
                if len(valid)
                else None,
                "nonfinite_voxel_count": int(
                    numeric_series(group, "nonfinite_voxel_count").sum()
                ),
            }
        )
    densities = histograms.groupby(
        ["region", "region_name", "bin_center_hu"], sort=False, as_index=False
    ).agg(density=("density", "mean"))
    if not isinstance(densities, pd.DataFrame):
        raise TypeError("A agregação de densidades não retornou um DataFrame.")
    return pd.DataFrame(rows), densities
