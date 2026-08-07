"""Classificação de falhas em comparações de variantes do pipeline."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


DEFAULT_VARIANTS = {
    "normal_rg": "normal_rg",
    "fuzzy_rg": "th_fuzzy_rg",
    "normal_fc": "normal_fc",
    "fuzzy_fc": "th_fuzzy_fc",
}

SUCCESS_STATUSES = {
    "both correct",
    "both tolerable",
    "both ostia correct",
    "both ostia tolerable",
}

_SOURCE_COLUMNS = {
    "artery_dice": "dice",
    "artery_voxel_count": "predicted_voxels",
    "ostia_detection_status": "ostia_status",
    "left_ostium": "left_ostium",
    "right_ostium": "right_ostium",
}

FOCUSED_COHORT_COLUMNS = ["IMG_ID", "cohort_kind", "cohort_roles"]

_CATEGORY_COLUMNS = [
    "fuzzy_threshold_rescue",
    "fuzzy_threshold_regression",
    "shared_ostia_failure",
    "shared_low_dice_with_good_ostia",
    "fc_prevents_rg_leakage_normal",
    "fc_recovers_rg_undersegmentation_normal",
    "fc_undersegments_normal",
    "fc_undersegments_fuzzy",
    "threshold_changes_segmentation_with_same_ostia",
]

_FOCUSED_CATEGORY_SPECS = [
    ("fuzzy_threshold_rescue", "fuzzy_rg_delta_vs_normal_rg", False),
    ("fuzzy_threshold_regression", "fuzzy_rg_delta_vs_normal_rg", True),
    ("fc_prevents_rg_leakage_normal", "normal_fc_delta_vs_rg", False),
    ("fc_recovers_rg_undersegmentation_normal", "normal_fc_delta_vs_rg", False),
    ("fc_undersegments_normal", "normal_fc_delta_vs_rg", True),
    ("fc_undersegments_fuzzy", "fuzzy_fc_delta_vs_rg", True),
    ("shared_low_dice_with_good_ostia", "normal_rg_dice", True),
    (
        "threshold_changes_segmentation_with_same_ostia",
        "fuzzy_rg_delta_vs_normal_rg",
        False,
    ),
]


def _success_status(series: pd.Series) -> pd.Series:
    """Converte os rótulos de óstios em sucesso binário."""
    return series.astype(str).str.strip().str.lower().isin(SUCCESS_STATUSES)


def _normalized_coordinate(series: pd.Series) -> pd.Series:
    """Normaliza coordenadas serializadas para comparação entre runs."""
    return series.fillna("").astype(str).str.replace(" ", "", regex=False)


def _variant_table(
    results_df: pd.DataFrame,
    variant: str,
    prefix: str,
) -> pd.DataFrame:
    """Seleciona e prefixa as métricas necessárias de uma variante."""
    selected = results_df.loc[results_df["folder_variant"].eq(variant)].copy()
    if selected.empty:
        raise ValueError(f"Variante ausente nos resultados: {variant}")
    if selected["IMG_ID"].duplicated().any():
        duplicated = selected.loc[selected["IMG_ID"].duplicated(), "IMG_ID"].tolist()
        raise ValueError(
            f"A variante {variant} possui IDs duplicados: {duplicated[:5]}"
        )

    table = selected[["IMG_ID"]].copy()
    for source, target in _SOURCE_COLUMNS.items():
        table[f"{prefix}_{target}"] = (
            selected[source] if source in selected.columns else np.nan
        )
    table[f"{prefix}_dice"] = pd.to_numeric(table[f"{prefix}_dice"], errors="coerce")
    table[f"{prefix}_predicted_voxels"] = pd.to_numeric(
        table[f"{prefix}_predicted_voxels"], errors="coerce"
    )
    table[f"{prefix}_ostia_success"] = _success_status(table[f"{prefix}_ostia_status"])
    return table.set_index("IMG_ID")


def _same_ostia(catalog: pd.DataFrame, left: str, right: str) -> pd.Series:
    """Compara o par de óstios de duas variantes."""
    return (
        _normalized_coordinate(catalog[f"{left}_left_ostium"])
        == _normalized_coordinate(catalog[f"{right}_left_ostium"])
    ) & (
        _normalized_coordinate(catalog[f"{left}_right_ostium"])
        == _normalized_coordinate(catalog[f"{right}_right_ostium"])
    )


def build_failure_case_catalog(
    results_df: pd.DataFrame,
    *,
    variants: Mapping[str, str] | None = None,
    low_dice_threshold: float = 0.4,
    meaningful_delta: float = 0.05,
    volume_ratio: float = 1.5,
) -> pd.DataFrame:
    """Cria um catálogo por exame com padrões prováveis de falha."""
    required = {"IMG_ID", "folder_variant", "artery_dice"}
    missing = required - set(results_df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")
    if low_dice_threshold <= 0 or meaningful_delta <= 0 or volume_ratio <= 1:
        raise ValueError("Os limiares devem ser positivos e volume_ratio > 1.")

    names = dict(DEFAULT_VARIANTS if variants is None else variants)
    missing_roles = set(DEFAULT_VARIANTS) - set(names)
    if missing_roles:
        raise ValueError(f"Papéis de variantes ausentes: {sorted(missing_roles)}")

    # Alinha as quatro abordagens pelo mesmo IMG_ID antes de comparar resultados.
    tables = [
        _variant_table(results_df, names[role], role) for role in DEFAULT_VARIANTS
    ]
    catalog = pd.concat(tables, axis=1, join="inner").reset_index()

    # Calcula deltas que separam efeito do threshold e efeito do segmentador.
    catalog["same_ostia_between_thresholds"] = _same_ostia(
        catalog, "normal_rg", "fuzzy_rg"
    )
    catalog["normal_fc_delta_vs_rg"] = (
        catalog["normal_fc_dice"] - catalog["normal_rg_dice"]
    )
    catalog["fuzzy_fc_delta_vs_rg"] = (
        catalog["fuzzy_fc_dice"] - catalog["fuzzy_rg_dice"]
    )
    catalog["fuzzy_rg_delta_vs_normal_rg"] = (
        catalog["fuzzy_rg_dice"] - catalog["normal_rg_dice"]
    )

    normal_success = catalog["normal_rg_ostia_success"]
    fuzzy_success = catalog["fuzzy_rg_ostia_success"]
    normal_fc_delta = catalog["normal_fc_delta_vs_rg"]
    fuzzy_fc_delta = catalog["fuzzy_fc_delta_vs_rg"]
    rg_volume = catalog["normal_rg_predicted_voxels"]
    fc_volume = catalog["normal_fc_predicted_voxels"]

    # Cada flag representa uma hipótese de falha útil para inspeção qualitativa.
    catalog["fuzzy_threshold_rescue"] = ~normal_success & fuzzy_success
    catalog["fuzzy_threshold_regression"] = normal_success & ~fuzzy_success
    catalog["shared_ostia_failure"] = ~normal_success & ~fuzzy_success
    catalog["shared_low_dice_with_good_ostia"] = (
        normal_success
        & fuzzy_success
        & catalog[
            ["normal_rg_dice", "fuzzy_rg_dice", "normal_fc_dice", "fuzzy_fc_dice"]
        ]
        .fillna(0)
        .lt(low_dice_threshold)
        .all(axis=1)
    )
    catalog["fc_prevents_rg_leakage_normal"] = (
        normal_success
        & (normal_fc_delta > meaningful_delta)
        & (rg_volume > fc_volume * volume_ratio)
    )
    catalog["fc_recovers_rg_undersegmentation_normal"] = (
        normal_success
        & (normal_fc_delta > meaningful_delta)
        & (fc_volume > rg_volume * volume_ratio)
    )
    catalog["fc_undersegments_normal"] = (
        normal_success
        & (normal_fc_delta < -meaningful_delta)
        & (fc_volume * volume_ratio < rg_volume)
    )
    catalog["fc_undersegments_fuzzy"] = (
        fuzzy_success
        & (fuzzy_fc_delta < -meaningful_delta)
        & (
            catalog["fuzzy_fc_predicted_voxels"] * volume_ratio
            < catalog["fuzzy_rg_predicted_voxels"]
        )
    )
    catalog["threshold_changes_segmentation_with_same_ostia"] = catalog[
        "same_ostia_between_thresholds"
    ] & (catalog["fuzzy_rg_delta_vs_normal_rg"].abs() > meaningful_delta)

    # Uma imagem pode pertencer a mais de uma categoria simultaneamente.
    catalog["failure_categories"] = catalog.apply(
        lambda row: (
            ";".join(category for category in _CATEGORY_COLUMNS if bool(row[category]))
            or "no_flag"
        ),
        axis=1,
    )
    return catalog.sort_values("IMG_ID").reset_index(drop=True)


def summarize_failure_categories(catalog: pd.DataFrame) -> pd.DataFrame:
    """Resume a quantidade e o percentual de exames em cada categoria."""
    total = len(catalog)
    rows = []
    for category in _CATEGORY_COLUMNS:
        if category not in catalog.columns:
            continue
        count = int(catalog[category].fillna(False).astype(bool).sum())
        rows.append(
            {
                "category": category,
                "count": count,
                "percent": 100.0 * count / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def _evenly_spaced_rows(df: pd.DataFrame, count: int) -> pd.DataFrame:
    """Seleciona linhas distribuídas pelo conjunto ordenado."""
    if count <= 0 or df.empty:
        return df.iloc[0:0]
    if len(df) <= count:
        return df
    positions = np.linspace(0, len(df) - 1, num=count, dtype=int)
    return df.iloc[np.unique(positions)]


def select_focused_failure_cohort(
    catalog: pd.DataFrame,
    *,
    max_per_category: int = 6,
    shared_ostia_cases: int = 6,
    stable_controls: int = 10,
) -> pd.DataFrame:
    """Seleciona falhas representativas e controles para ajustar parâmetros."""
    if min(max_per_category, shared_ostia_cases, stable_controls) < 0:
        raise ValueError("Os limites da coorte devem ser >= 0.")

    roles_by_id: dict[int, set[str]] = {}

    def add_rows(rows: pd.DataFrame, role: str) -> None:
        for image_id in rows["IMG_ID"].astype(int):
            roles_by_id.setdefault(image_id, set()).add(role)

    # Retém os casos mais representativos de cada mecanismo de falha.
    for flag, metric, ascending in _FOCUSED_CATEGORY_SPECS:
        if flag not in catalog or metric not in catalog:
            continue
        candidates = catalog.loc[catalog[flag].fillna(False)].sort_values(
            [metric, "IMG_ID"],
            ascending=[ascending, True],
            na_position="last",
        )
        add_rows(candidates.head(max_per_category), flag)

    shared_candidates = catalog.loc[
        catalog.get("shared_ostia_failure", False)
    ].sort_values("IMG_ID")
    add_rows(
        _evenly_spaced_rows(shared_candidates, shared_ostia_cases),
        "shared_ostia_failure",
    )

    dice_columns = [
        "normal_rg_dice",
        "fuzzy_rg_dice",
        "normal_fc_dice",
        "fuzzy_fc_dice",
    ]
    dice = catalog[dice_columns].apply(pd.to_numeric, errors="coerce")
    # Controles estáveis ajudam a detectar regressões durante novos ajustes.
    stable_mask = (
        catalog["normal_rg_ostia_success"].fillna(False)
        & catalog["fuzzy_rg_ostia_success"].fillna(False)
        & dice.min(axis=1).ge(0.6)
        & dice.max(axis=1).sub(dice.min(axis=1)).le(0.08)
    )
    stable_candidates = catalog.loc[stable_mask].sort_values("IMG_ID")
    add_rows(_evenly_spaced_rows(stable_candidates, stable_controls), "stable_control")

    selected_ids = sorted(roles_by_id)
    cohort = catalog.loc[catalog["IMG_ID"].isin(selected_ids)].copy()
    cohort.insert(
        1,
        "cohort_kind",
        cohort["IMG_ID"].map(
            lambda image_id: (
                "control"
                if roles_by_id[int(image_id)] == {"stable_control"}
                else "failure"
            )
        ),
    )
    cohort.insert(
        2,
        "cohort_roles",
        cohort["IMG_ID"].map(
            lambda image_id: ";".join(sorted(roles_by_id[int(image_id)]))
        ),
    )
    return cohort.sort_values(["cohort_kind", "IMG_ID"]).reset_index(drop=True)


def compact_focused_failure_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    """Mantém somente as colunas usadas pelos experimentos subsequentes."""
    missing = set(FOCUSED_COHORT_COLUMNS) - set(cohort.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias da coorte ausentes: {sorted(missing)}")
    return cohort[FOCUSED_COHORT_COLUMNS].copy()


__all__ = [
    "DEFAULT_VARIANTS",
    "FOCUSED_COHORT_COLUMNS",
    "build_failure_case_catalog",
    "compact_focused_failure_cohort",
    "select_focused_failure_cohort",
    "summarize_failure_categories",
]
