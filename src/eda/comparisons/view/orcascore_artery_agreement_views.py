"""Apresenta o histograma e as cenas arteriais 3D do OrCaScore."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.figure import Figure
from nibabel.affines import voxel_sizes

from utils.project.analysis.artery_mask_agreement import (
    align_artery_prediction,
    binary_artery_mask,
    load_artery_prediction,
)
from utils.visualization.images.volume import visualize_aorta_ostia_artery


def plot_dice_histogram(histogram: pd.DataFrame) -> Figure:
    """Exibe o histograma geral de Dice com a contribuição de treino e teste."""
    figure, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    width = histogram["Fim"] - histogram["Início"]
    axis.bar(
        histogram["Início"],
        histogram["Treino"],
        width,
        align="edge",
        color="#0072b2",
        edgecolor="white",
        label="Treino",
    )
    axis.bar(
        histogram["Início"],
        histogram["Teste"],
        width,
        bottom=histogram["Treino"],
        align="edge",
        color="#e69f00",
        edgecolor="white",
        label="Teste",
    )
    axis.set(xlabel="Dice de concordância", ylabel="Número de exames", xlim=(0, 1))
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    return figure


def plot_orcascore_artery_agreement_3d(
    case: pd.Series, *, aorta_path: Path | None = None
) -> Any:
    """Monta a cena K3D de um par na grade do pipeline, em mm, com óstios do CSV.

    A aorta opcional é uma máscara binária na mesma grade processada. Os
    marcadores seguem a ordem de índices ``(y, x, z)`` registrada pelo pipeline.
    """
    target = load_artery_prediction(case["pipeline_path"])
    prediction = load_artery_prediction(case["ai_path"])
    pipeline_mask = binary_artery_mask(target)
    ai_mask, _, _ = align_artery_prediction(prediction, target)
    aorta_mask = None
    if aorta_path is not None:
        aorta = load_artery_prediction(aorta_path)
        if aorta.shape != target.shape or not np.allclose(
            aorta.affine, target.affine, rtol=0, atol=1e-4
        ):
            raise ValueError(
                "A máscara da aorta deve estar na grade processada do exame."
            )
        aorta_mask = binary_artery_mask(aorta)

    ostia = []
    for side in ("left", "right"):
        point = pd.to_numeric(
            [case.get(f"ostia_{side}_{axis}") for axis in "yxz"], errors="coerce"
        )
        if (
            np.isfinite(point).all()
            and np.all(point >= 0)
            and np.all(point < target.shape)
        ):
            ostia.append(tuple(int(index) for index in point))
        else:
            ostia.append(None)
    dice = f"{case['dice']:.3f}" if pd.notna(case["dice"]) else "indisponível"
    return visualize_aorta_ostia_artery(
        aorta_mask,
        ostia[0],
        ostia[1],
        artery_mask=pipeline_mask,
        label_artery=ai_mask,
        spacing=tuple(voxel_sizes(np.asarray(target.affine))),
        artery_name="Artérias · Pipeline",
        label_artery_name="Artérias · IA",
        plot_name=f"{case['exam_id']} · {case['variant']} · Dice {dice}",
        display_plot=False,
    )


def show_orcascore_dice_case(
    case: pd.Series | None, aorta_paths: Mapping[str, str | Path]
) -> None:
    """Exibe a cena de um caso selecionado e inclui a aorta quando fornecida."""
    if case is None:
        print("Nenhum exame da seleção tem Dice disponível para inspeção.")
        return
    exam_id = case["exam_id"]
    aorta_path = aorta_paths.get(exam_id)
    display(
        plot_orcascore_artery_agreement_3d(
            case, aorta_path=Path(aorta_path) if aorta_path else None
        )
    )
