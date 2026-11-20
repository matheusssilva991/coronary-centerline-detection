"""Carrega calcificações de referência na grade nativa CTI do OrCaScore.

Classes oficiais: https://orcascore.grand-challenge.org/Data/
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from utils.project.datasets.ccta import load_mhd_volume_xyz, read_mhd_header

ORCASCORE_CALCIFICATION_LABEL_SPECS = {
    1: ("LAD (inclui tronco esquerdo)", 0xE69F00),
    2: ("LCX", 0x56B4E9),
    3: ("RCA", 0xCC79A7),
}


def orcascore_calcification_reference_path(
    record: Mapping[str, Any] | pd.Series,
) -> Path:
    """Resolve a referência somente para a aquisição sem contraste do exame.

    Args:
        record (Mapping[str, Any] | pd.Series): Registro do inventário OrCaScore
            referente à aquisição CTI sem contraste.

    Returns:
        Path: Caminho esperado do arquivo ``*R.mhd`` em ``Reference_standard``.
    """
    if (
        record.get("dataset") != "OrCaScore"
        or record.get("acquisition") != "noncontrast"
    ):
        raise ValueError(
            "A referência de calcificação exige OrCaScore sem contraste (CTI)."
        )
    image_path = Path(record["path"])
    if image_path.name != f"{record['exam_id']}CTI.mhd":
        raise ValueError("Nome do arquivo CTI incompatível com o ID do exame.")
    return image_path.parent.parent / "Reference_standard" / f"{record['exam_id']}R.mhd"


def load_orcascore_calcification_reference(
    record: Mapping[str, Any] | pd.Series,
) -> NDArray[np.generic] | None:
    """Carrega labels 0–3 sem reamostrar, distinguindo ausência de máscara vazia.

    Args:
        record (Mapping[str, Any] | pd.Series): Registro do inventário para a
            aquisição CTI sem contraste.

    Returns:
        NDArray[np.generic] | None: Volume 3D de rótulos ``0`` a ``3`` na ordem
            ``(x, y, z)``, ou ``None`` se o arquivo de referência não existir.
    """
    # Resolve o caminho da referência e distingue ausência de arquivo de máscara vazia.
    path = orcascore_calcification_reference_path(record)
    if not path.is_file():
        return None

    # Garante correspondência exata de grade, transformação e orientação com a CTI.
    image_header = read_mhd_header(record["path"])
    label_header = read_mhd_header(path)
    for field, size, default in (
        ("DimSize", 3, ""),
        ("ElementSpacing", 3, ""),
        ("TransformMatrix", 9, "1 0 0 0 1 0 0 0 1"),
        ("Offset", 3, "0 0 0"),
        ("CenterOfRotation", 3, "0 0 0"),
    ):
        image_value = np.fromstring(image_header.get(field, default), sep=" ")
        label_value = np.fromstring(label_header.get(field, default), sep=" ")
        if (
            image_value.size != size
            or label_value.size != size
            or not np.isfinite(image_value).all()
            or not np.isfinite(label_value).all()
            or not np.allclose(image_value, label_value, rtol=1e-6, atol=1e-6)
        ):
            raise ValueError(f"Grade da referência difere da CTI: {field}.")
    if image_header.get("AnatomicalOrientation") != label_header.get(
        "AnatomicalOrientation"
    ):
        raise ValueError("Orientação da referência difere da CTI.")

    # Carrega os rótulos nativos sem reamostragem e valida o conjunto de classes 0–3.
    label = load_mhd_volume_xyz(path)
    unknown = set(np.unique(label)) - {0, *ORCASCORE_CALCIFICATION_LABEL_SPECS}
    if unknown:
        raise ValueError(f"Classes de calcificação desconhecidas: {sorted(unknown)}")
    return label
