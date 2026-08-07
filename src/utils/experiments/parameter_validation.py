"""Definições e seleção de casos para validação de parâmetros do pipeline."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _as_boolean(series: pd.Series) -> pd.Series:
    """Converte colunas booleanas lidas de CSV sem tratar ``"False"`` como true."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "sim"})


def parameter_validation_variants() -> list[dict[str, Any]]:
    """Retorna as variantes OFAT da análise de sensibilidade do artigo.

    A referência usa os valores centrais declarados no estudo. Cada variante
    altera apenas um parâmetro, mantendo todos os demais na referência.
    """
    reference = {
        "threshold_mode": "normal",
        "artery_method": "region_growing",
        "MAX_THRESHOLD_PERCENTILE": 99.7,
        "OSTIA_DETECTION.max_z_diff_mm": 40.0,
        "REGION_GROWING.threshold_divisor": 7.0,
        "REGION_GROWING.min_vesselness_fraction": 0.078,
        "REGION_GROWING.relaxed_floor_factor": 0.98,
    }

    def variant(
        name: str,
        group: str,
        description: str,
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "name": name,
            "parameter_group": group,
            "description": description,
            "overrides": {**reference, **overrides},
        }

    return [
        variant(
            "baseline",
            "baseline",
            "Referência normal_rg de 2026-06-29: P99.7, z=40 mm, D=7 e piso 7.8%.",
            {},
        ),
        *[
            variant(
                f"upper_p{str(value).replace('.', '')}",
                "upper_percentile",
                f"Percentil superior do threshold = {value}.",
                {"MAX_THRESHOLD_PERCENTILE": value},
            )
            for value in (99.5, 99.9)
        ],
        *[
            variant(
                f"ostia_z{int(value)}",
                "ostia_z_limit",
                f"Diferença máxima em z entre óstios = {value:.0f} mm.",
                {"OSTIA_DETECTION.max_z_diff_mm": value},
            )
            for value in (30.0, 50.0)
        ],
        *[
            variant(
                f"ostia_lower_{int(value * 100)}",
                "ostia_search_fraction",
                f"Fração inferior da aorta usada na busca dos óstios = {value:.0%}.",
                {"OSTIA_DETECTION.lower_fraction": value},
            )
            for value in (0.70, 1.00)
        ],
        *[
            variant(
                f"rg_divisor_{int(value)}",
                "rg_threshold_divisor",
                f"Divisor do limiar global do RG = {value:.0f}.",
                {"REGION_GROWING.threshold_divisor": value},
            )
            for value in (5.0, 9.0)
        ],
        *[
            variant(
                f"rg_vessel_{int(initial * 100):02d}",
                "rg_vesselness_floor",
                (f"Fração mínima de vesselness do RG = {initial:.0%}."),
                {
                    "REGION_GROWING.min_vesselness_fraction": initial,
                },
            )
            for initial in (0.05, 0.09)
        ],
    ]


def validate_parameter_validation_append(
    existing: dict[str, Any],
    *,
    split: str,
    image_ids: list[int],
    resolution: str,
    aorta_ostia_method: str,
    config_path: Path,
    use_gpu: bool,
) -> None:
    """Impede combinar partes produzidas com configurações incompatíveis."""
    expected = {
        "split": split,
        "ids": image_ids,
        "resolution": resolution,
        "aorta_ostia_method": aorta_ostia_method,
        "config_path": str(config_path),
        "use_gpu": use_gpu,
    }
    mismatches = [key for key, value in expected.items() if existing.get(key) != value]
    if mismatches:
        details = ", ".join(
            f"{key}: existente={existing.get(key)!r}, atual={expected[key]!r}"
            for key in mismatches
        )
        raise ValueError(f"Run incompatível com --append ({details}).")


def select_parameter_validation_cases(
    image_results: pd.DataFrame,
    variant: str,
    *,
    target_dice: float = 0.58,
) -> pd.DataFrame:
    """Seleciona casos 3D de desempenho e falha para uma variante.

    Vazamentos são candidatos quantitativos: volume de aorta muito alto ou
    razão predição/ground truth arterial alta. A confirmação permanece visual.
    """
    df = image_results.loc[image_results["variant"] == variant].copy()
    if df.empty:
        raise ValueError(f"Variante sem resultados: {variant}")

    numeric_columns = [
        "IMG_ID",
        "dice_artery",
        "aorta_voxels",
        "aorta_volume_fraction",
        "artery_volume_ratio",
        "left_dist_mm",
        "right_dist_mm",
    ]
    for column in numeric_columns:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ("ostia_success", "ostia_found"):
        if column in df:
            df[column] = _as_boolean(df[column])

    selected_ids: set[int] = set()
    rows: list[dict[str, Any]] = []

    def add_case(case_type: str, candidates: pd.DataFrame, reason: str) -> None:
        available = candidates.loc[~candidates["IMG_ID"].isin(selected_ids)]
        if available.empty:
            available = candidates
        if available.empty:
            return
        row = available.iloc[0].to_dict()
        image_id = int(row["IMG_ID"])
        selected_ids.add(image_id)
        row.update({"case_type": case_type, "selection_reason": reason})
        rows.append(row)

    valid_dice = df.dropna(subset=["dice_artery"])
    add_case(
        "high_dice",
        valid_dice.sort_values("dice_artery", ascending=False),
        "Maior Dice da variante selecionada.",
    )
    near_mean = valid_dice.assign(
        target_distance=(valid_dice["dice_artery"] - target_dice).abs()
    ).sort_values("target_distance")
    add_case(
        "near_target_mean",
        near_mean,
        f"Dice mais próximo do alvo {target_dice:.2f}.",
    )

    ostia_failure = df.loc[~df["ostia_success"]].copy()
    ostia_failure["ostia_failure_priority"] = np.where(
        ostia_failure["ostia_found"],
        0,
        1,
    )
    ostia_failure = ostia_failure.sort_values(
        ["ostia_failure_priority", "dice_artery"],
        ascending=[True, True],
        na_position="first",
    )
    add_case(
        "ostia_failure",
        ostia_failure,
        "Falha de localização/validação dos óstios.",
    )

    aorta_candidates = df.dropna(subset=["aorta_volume_fraction"]).sort_values(
        "aorta_volume_fraction",
        ascending=False,
    )
    add_case(
        "suspected_aorta_leak",
        aorta_candidates,
        "Maior fração do volume ocupada pela aorta; candidato a vazamento visual.",
    )

    segmentation_failure = df.loc[df["ostia_success"]].dropna(subset=["dice_artery"])
    if "artery_volume_ratio" in segmentation_failure:
        segmentation_failure = segmentation_failure.sort_values(
            ["dice_artery", "artery_volume_ratio"],
            ascending=[True, False],
        )
    else:
        segmentation_failure = segmentation_failure.sort_values("dice_artery")
    add_case(
        "segmentation_failure",
        segmentation_failure,
        "Baixo Dice apesar de óstios aceitáveis; falha/vazamento arterial provável.",
    )

    return pd.DataFrame(rows)


def variant_by_name(name: str) -> dict[str, Any]:
    """Retorna uma cópia da definição de uma variante pelo nome."""
    variants = {item["name"]: item for item in parameter_validation_variants()}
    if name not in variants:
        raise ValueError(f"Variante desconhecida: {name}")
    return copy.deepcopy(variants[name])


__all__ = [
    "parameter_validation_variants",
    "select_parameter_validation_cases",
    "validate_parameter_validation_append",
    "variant_by_name",
]
