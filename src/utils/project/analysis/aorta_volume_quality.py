"""Cálculos tabulares de volume e qualidade visual da aorta sem dependências gráficas."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_overview_row(cohort_df: pd.DataFrame, cohort_name: str) -> dict[str, Any]:
    """Resume a escala geral das máscaras da aorta em uma coorte.

    Args:
        cohort_df (pd.DataFrame): DataFrame com métricas volumétricas da aorta por exame.
        cohort_name (str): Nome legível da coorte (ex.: ``treino`` ou ``validação``).

    Returns:
        dict[str, Any]: Dicionário com médias e medianas de voxels, fatias e ocupação.
    """
    return {
        "coorte": cohort_name,
        "imagens": len(cohort_df),
        "voxels_aorta_media": cohort_df["aorta_mask_voxel_count"].mean(),
        "voxels_aorta_mediana": cohort_df["aorta_mask_voxel_count"].median(),
        "fatias_segmentadas_media": cohort_df["aorta_segmented_slice_count"].mean(),
        "voxels_por_fatia_media": cohort_df["aorta_voxels_per_segmented_slice"].mean(),
        "volume_percentual_medio": cohort_df["aorta_volume_percentage"].mean(),
        "volume_percentual_mediano": cohort_df["aorta_volume_percentage"].median(),
    }


def build_ostia_summary(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa as métricas da aorta pelo desfecho da localização dos óstios.

    Args:
        cohort_df (pd.DataFrame): DataFrame da coorte contendo ``ostia_outcome``.

    Returns:
        pd.DataFrame: Tabela agregada por desfecho dos óstios.
    """
    return (
        cohort_df.groupby("ostia_outcome", observed=True)
        .agg(
            imagens=("IMG_ID", "size"),
            voxels_aorta_media=("aorta_mask_voxel_count", "mean"),
            voxels_aorta_mediana=("aorta_mask_voxel_count", "median"),
            fatias_segmentadas_media=("aorta_segmented_slice_count", "mean"),
            voxels_por_fatia_media=("aorta_voxels_per_segmented_slice", "mean"),
            volume_percentual_medio=("aorta_volume_percentage", "mean"),
            distancia_maxima_media_mm=("max_ostium_distance_mm", "mean"),
            dice_arterial_medio=("artery_dice", "mean"),
        )
        .reset_index()
    )


def build_quality_crosstab(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Cruza a classificação visual da aorta com o desfecho dos óstios.

    Args:
        cohort_df (pd.DataFrame): DataFrame com ``visual_aorta_quality`` e ``ostia_outcome``.

    Returns:
        pd.DataFrame: Tabela de contingência com totais marginais.
    """
    table = pd.crosstab(
        cohort_df["visual_aorta_quality"],
        cohort_df["ostia_outcome"],
        margins=True,
    )
    table.index.name = "qualidade_visual_aorta"
    return table


def build_visual_summary(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Resume volume, área por fatia e Dice arterial por qualidade visual da aorta.

    Args:
        cohort_df (pd.DataFrame): DataFrame da coorte revisada visualmente.

    Returns:
        pd.DataFrame: Estatísticas agregadas para aortas boas e ruins.
    """
    return (
        cohort_df.groupby("visual_aorta_quality", observed=True)
        .agg(
            imagens=("IMG_ID", "size"),
            voxels_aorta_media=("aorta_mask_voxel_count", "mean"),
            voxels_aorta_mediana=("aorta_mask_voxel_count", "median"),
            fatias_segmentadas_media=("aorta_segmented_slice_count", "mean"),
            voxels_por_fatia_media=("aorta_voxels_per_segmented_slice", "mean"),
            voxels_por_fatia_mediana=(
                "aorta_voxels_per_segmented_slice",
                "median",
            ),
            volume_percentual_medio=("aorta_volume_percentage", "mean"),
            volume_percentual_mediano=("aorta_volume_percentage", "median"),
            dice_arterial_medio=("artery_dice", "mean"),
        )
        .reset_index()
    )


def find_volume_threshold(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona no conjunto de treino o corte percentual com maior acurácia balanceada.

    Args:
        cohort_df (pd.DataFrame): Coorte de treino com ``aorta_volume_percentage`` e
            ``visual_aorta_quality``.

    Returns:
        pd.DataFrame: Candidatos de limiar ordenados por acurácia balanceada e acurácia.
    """
    # Gera limiares candidatos nos pontos médios entre valores adjacentes observados.
    sorted_values = np.sort(cohort_df["aorta_volume_percentage"].dropna().unique())
    candidate_thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2.0
    is_bad = cohort_df["visual_aorta_quality"].eq("ruim")

    # Avalia sensibilidade, especificidade e acurácia balanceada para cada corte.
    rows = []
    for threshold in candidate_thresholds:
        predicted_bad = cohort_df["aorta_volume_percentage"].gt(threshold)
        sensitivity = predicted_bad.loc[is_bad].mean()
        specificity = (~predicted_bad.loc[~is_bad]).mean()
        rows.append(
            {
                "threshold_percentual": threshold,
                "acuracia": predicted_bad.eq(is_bad).mean(),
                "sensibilidade_aorta_ruim": sensitivity,
                "especificidade_aorta_boa": specificity,
                "acuracia_balanceada": (sensitivity + specificity) / 2.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["acuracia_balanceada", "acuracia"],
        ascending=False,
    )


def evaluate_volume_threshold(
    cohort_df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Avalia em outra coorte um limiar percentual de volume definido no treino.

    Args:
        cohort_df (pd.DataFrame): Coorte avaliada (ex.: validação).
        threshold (float): Limiar percentual de volume da aorta aprendido no treino.

    Returns:
        pd.DataFrame: DataFrame de uma linha com contagens e métricas de classificação.
    """
    is_bad = cohort_df["visual_aorta_quality"].eq("ruim")
    predicted_bad = cohort_df["aorta_volume_percentage"].gt(threshold)
    sensitivity = predicted_bad.loc[is_bad].mean()
    specificity = (~predicted_bad.loc[~is_bad]).mean()
    return pd.DataFrame(
        {
            "corte_do_treino_percentual": [threshold],
            "imagens": [len(cohort_df)],
            "aortas_ruins_reais": [int(is_bad.sum())],
            "aortas_sinalizadas": [int(predicted_bad.sum())],
            "acuracia": [predicted_bad.eq(is_bad).mean()],
            "sensibilidade_aorta_ruim": [sensitivity],
            "especificidade_aorta_boa": [specificity],
            "acuracia_balanceada": [(sensitivity + specificity) / 2.0],
        }
    )
