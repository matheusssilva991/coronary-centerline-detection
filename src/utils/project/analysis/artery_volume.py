"""Volume físico das coronárias de referência do ImageCAS, com retomada em CSV."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from nibabel.loadsave import load as load_nifti
from nibabel.nifti1 import Nifti1Header
from nibabel.spatialimages import SpatialImage
from numpy.typing import NDArray
from tqdm.auto import tqdm

from utils.project.dataframe import require_series_column, to_numeric_series
from utils.project.results.io import save_dataframe_atomic
from utils.project.runtime.notebook_env import resolve_imagecas_base_path

ARTERY_LABEL = 1
VOLUME_COLUMNS = [
    "IMG_ID",
    "label_file",
    "label_value",
    "voxel_count",
    "spacing_x_mm",
    "spacing_y_mm",
    "spacing_z_mm",
    "voxel_volume_mm3",
    "volume_ml",
    "status",
    "error",
]
NUMERIC_COLUMNS = VOLUME_COLUMNS[3:9]


def calculate_artery_volume(
    label: NDArray[Any], spacing_mm: Sequence[float] | NDArray[Any]
) -> dict[str, Any]:
    """Conta somente label == 1 e converte mm³ para mL na geometria nativa."""
    if label.ndim != 3:
        raise ValueError("O rótulo deve ser um volume 3D.")
    spacing = np.asarray(spacing_mm, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("O spacing deve conter três valores finitos e positivos.")
    if not np.isfinite(label).all():
        raise ValueError("O rótulo contém valores não finitos.")
    voxel_count = int(np.count_nonzero(label == ARTERY_LABEL))
    voxel_volume = float(np.prod(spacing))
    if not np.isfinite(voxel_volume) or voxel_volume <= 0:
        raise ValueError("O volume de um voxel deve ser finito e positivo.")
    return {
        "label_value": ARTERY_LABEL,
        "voxel_count": voxel_count,
        "spacing_x_mm": float(spacing[0]),
        "spacing_y_mm": float(spacing[1]),
        "spacing_z_mm": float(spacing[2]),
        "voxel_volume_mm3": voxel_volume,
        "volume_ml": voxel_count * voxel_volume / 1000.0,
    }


def classify_artery_volume_outliers(
    volumes: pd.DataFrame,
    *,
    method: str = "iqr",
    iqr_multiplier: float = 1.5,
    std_multiplier: float = 2.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Classifica volumes válidos por IQR ou desvios-padrão, sem alterar o CSV.

    Limites descritivos usam todos os volumes válidos fornecidos; valores
    exatamente no limite permanecem na faixa. O z-score usa DP amostral.
    """
    if method not in {"iqr", "std"}:
        raise ValueError("method deve ser 'iqr' ou 'std'.")
    if any(
        not np.isfinite(value) or value < 0
        for value in (iqr_multiplier, std_multiplier)
    ):
        raise ValueError("Os multiplicadores devem ser finitos e não negativos.")
    selected = volumes.loc[volumes["status"].eq("ok")].copy()
    if selected["IMG_ID"].isna().any() or selected["IMG_ID"].duplicated().any():
        raise ValueError("IDs dos volumes devem ser únicos e não ausentes.")
    values = to_numeric_series(
        require_series_column(selected, "volume_ml"), errors="raise"
    )
    if not np.isfinite(values.to_numpy(dtype=float)).all() or values.lt(0).any():
        raise ValueError("Volumes concluídos devem ser finitos e não negativos.")
    selected["volume_ml"] = values
    mean = float(values.mean())
    std = (
        float(np.std(values.to_numpy(dtype=float), ddof=1))
        if len(values) > 1
        else float("nan")
    )
    quantiles = (
        np.quantile(values.to_numpy(dtype=float), [0.25, 0.75])
        if len(values)
        else [np.nan, np.nan]
    )
    q1, q3 = float(quantiles[0]), float(quantiles[1])
    lower, upper = (
        (q1 - iqr_multiplier * (q3 - q1), q3 + iqr_multiplier * (q3 - q1))
        if method == "iqr"
        else (mean - std_multiplier * std, mean + std_multiplier * std)
    )
    selected["volume_z_score"] = (values - mean) / std if std > 0 else np.nan
    if std == 0:
        selected["volume_z_score"] = 0.0
    selected["volume_outlier_group"] = "in_range"
    selected.loc[values.lt(lower), "volume_outlier_group"] = "low"
    selected.loc[values.gt(upper), "volume_outlier_group"] = "high"
    selected["is_volume_outlier"] = selected["volume_outlier_group"].ne("in_range")
    return selected, {
        "method": method,
        "exam_count": len(selected),
        "mean_ml": mean,
        "std_ml": std,
        "q1_ml": q1,
        "q3_ml": q3,
        "iqr_ml": q3 - q1,
        "lower_limit_ml": lower,
        "upper_limit_ml": upper,
        "iqr_multiplier": iqr_multiplier,
        "std_multiplier": std_multiplier,
    }


def _read_cached_volumes(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    missing = set(VOLUME_COLUMNS) - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"CSV de volumes incompatível; colunas ausentes: {sorted(missing)}"
        )
    dataframe = dataframe.loc[:, VOLUME_COLUMNS].copy()
    ids = to_numeric_series(require_series_column(dataframe, "IMG_ID"), errors="raise")
    if (
        ids.isna().any()
        or (ids <= 0).any()
        or ids.mod(1).ne(0).any()
        or ids.duplicated().any()
    ):
        raise ValueError("O CSV deve conter IDs positivos, inteiros e únicos.")
    dataframe["IMG_ID"] = ids.astype(int)
    if not dataframe["status"].isin(["ok", "error"]).all():
        raise ValueError("Status inválido no CSV de volumes.")
    if (
        not to_numeric_series(
            require_series_column(dataframe, "label_value"), errors="raise"
        )
        .eq(ARTERY_LABEL)
        .all()
    ):
        raise ValueError("O CSV deve conter somente a classe coronariana 1.")
    for relative in dataframe["label_file"]:
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("O CSV deve registrar caminhos relativos de rótulos.")
    for column in NUMERIC_COLUMNS:
        dataframe[column] = to_numeric_series(
            require_series_column(dataframe, column), errors="raise"
        )
    valid = dataframe.loc[dataframe["status"].eq("ok"), NUMERIC_COLUMNS]
    if not np.isfinite(valid.to_numpy(dtype=float)).all():
        raise ValueError(
            "Métricas ausentes ou não finitas em registros concluídos do CSV."
        )
    if (valid[["voxel_count", "volume_ml"]] < 0).any().any() or (
        valid[NUMERIC_COLUMNS[1:5]] <= 0
    ).any().any():
        raise ValueError("Métricas inválidas em registros concluídos do CSV.")
    expected = valid["voxel_count"] * valid["voxel_volume_mm3"] / 1000.0
    expected_voxel = valid[["spacing_x_mm", "spacing_y_mm", "spacing_z_mm"]].prod(
        axis=1
    )
    if (
        (valid["voxel_count"] % 1 != 0).any()
        or not np.allclose(
            expected_voxel, valid["voxel_volume_mm3"], rtol=1e-9, atol=1e-12
        )
        or not np.allclose(expected, valid["volume_ml"], rtol=1e-9, atol=1e-12)
    ):
        raise ValueError("Volumes do CSV não correspondem às contagens e ao spacing.")
    return dataframe.sort_values("IMG_ID").reset_index(drop=True)


def _calculate_label_file(path: Path) -> dict[str, Any]:
    image = cast(SpatialImage, load_nifti(str(path)))
    # O ImageCAS usa mm; unidades NIfTI explícitas são convertidas quando necessário.
    header = cast(Nifti1Header, image.header)
    unit = str(header.get_xyzt_units()[0])
    factors = {"mm": 1.0, "unknown": 1.0, "meter": 1000.0, "micron": 0.001}
    if unit not in factors:
        raise ValueError(f"Unidade espacial NIfTI não reconhecida: {unit}")
    spacing_mm = np.asarray(image.header.get_zooms()[:3], dtype=float) * factors[unit]
    return calculate_artery_volume(np.asanyarray(image.dataobj), spacing_mm)


def load_or_compute_artery_volumes(
    csv_path: str | Path,
    base_path: str | Path | None = None,
    *,
    force_recompute: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Carrega o CSV e calcula só máscaras pendentes ou com erro.

    Cada exame é salvo atomicamente. Interrupções preservam os anteriores;
    ``force_recompute`` descarta o cache apenas após localizar as máscaras.
    Sem dataset montado, o CSV continua consultável com um aviso. Arquivos
    concluídos modificados no disco exigem recálculo explícito.
    """
    csv_path = Path(csv_path)
    cached = (
        _read_cached_volumes(csv_path)
        if csv_path.is_file() and not force_recompute
        else pd.DataFrame(columns=VOLUME_COLUMNS)
    )
    try:
        root = (
            Path(base_path) if base_path is not None else resolve_imagecas_base_path()
        )
        if not root.is_dir():
            raise FileNotFoundError(f"Diretório ImageCAS indisponível: {root}")
        labels = sorted(root.rglob("*.label.nii.gz"))
        if not labels:
            raise FileNotFoundError(
                "Nenhuma máscara *.label.nii.gz encontrada no ImageCAS."
            )
    except FileNotFoundError:
        if cached.empty or force_recompute:
            raise
        warnings.warn(
            "Dataset indisponível: CSV carregado sem verificar novos arquivos ou retomar erros.",
            UserWarning,
            stacklevel=2,
        )
        return cached

    # Descobre todos os IDs antes de gravar, evitando sobrescrita por duplicatas.
    inventory: dict[int, Path] = {}
    for path in labels:
        identifier = path.name.removesuffix(".label.nii.gz")
        if not identifier.isdecimal() or int(identifier) <= 0:
            raise ValueError(f"Nome de rótulo ImageCAS inválido: {path.name}")
        img_id = int(identifier)
        if img_id in inventory:
            raise ValueError(f"ID ImageCAS duplicado: {img_id}")
        inventory[img_id] = path
    rows: dict[int, dict[str, Any]] = {
        int(row["IMG_ID"]): {str(key): value for key, value in row.items()}
        for row in cached.to_dict(orient="records")
    }
    absent = set(rows) - set(inventory)
    if absent:
        warnings.warn(
            f"{len(absent)} exames do CSV não foram localizados; seus resultados foram preservados.",
            UserWarning,
            stacklevel=2,
        )
    pending = [
        (img_id, path)
        for img_id, path in sorted(inventory.items())
        if img_id not in rows
        or rows[img_id]["status"] != "ok"
        or rows[img_id]["label_file"] != path.relative_to(root).as_posix()
    ]
    for img_id, path in tqdm(
        pending, desc="Volume arterial ImageCAS", disable=not show_progress
    ):
        row: dict[str, Any] = {
            "IMG_ID": img_id,
            "label_file": path.relative_to(root).as_posix(),
            "label_value": ARTERY_LABEL,
            **dict.fromkeys(NUMERIC_COLUMNS),
            "status": "error",
            "error": None,
        }
        try:
            row.update(_calculate_label_file(path))
            row["status"] = "ok"
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        rows[img_id] = row
        # O próximo exame só começa depois de persistir este resultado.
        dataframe = pd.DataFrame(list(rows.values()), columns=VOLUME_COLUMNS)
        save_dataframe_atomic(dataframe.sort_values("IMG_ID"), csv_path)
    return (
        pd.DataFrame(list(rows.values()), columns=VOLUME_COLUMNS)
        .sort_values("IMG_ID")
        .reset_index(drop=True)
    )
