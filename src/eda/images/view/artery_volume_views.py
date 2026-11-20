"""Auxiliares Plotly para análise de volume arterial coronariano e desempenho de segmentação."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

OSTIA_COLORS = {"Óstios válidos": "#247A58", "Falha dos óstios": "#C44747"}
SPLIT_LABELS = {
    "train": "Treino",
    "val": "Validação",
    "test": "Teste",
    "all": "Conjunto completo",
}

__all__ = [
    "OSTIA_COLORS",
    "SPLIT_LABELS",
    "plot_volume_band_performance",
    "plot_volume_dice_delta",
    "plot_volume_dice_panels",
    "plot_volume_histogram",
]


def plot_volume_dice_panels(
    data: pd.DataFrame,
    pairs: pd.DataFrame,
    title: str,
    *,
    y_column: str = "dice",
    volume_plot_xlim: tuple[float, float] | None = None,
) -> Any:
    """Plota painéis lado a lado (Mid e High) relacionando volume arterial e métrica alvo.

    Args:
        data (pd.DataFrame): Dados por exame e resolução com volume, status de óstios e métrica.
        pairs (pd.DataFrame): Pares Mid/High com indicador de sucesso conjunto dos óstios.
        title (str): Título principal do gráfico interativo.
        y_column (str): Coluna numérica exibida no eixo Y (`dice` ou `predicted_volume_ml`).
        volume_plot_xlim (tuple[float, float] | None): Limites explícitos do eixo X em mL.

    Returns:
        Any: Figura Plotly com menu de coorte ou `None` quando não há exames.
    """
    if data.empty:
        print("Nenhum exame disponível para esta comparação.")
        return None

    # Determina a coorte com óstios válidos simultaneamente em Mid e High.
    valid_ids = (
        pairs.loc[pairs["ostia_success_both"], "IMG_ID"] if not pairs.empty else []
    )
    eligible_ids = set(data.loc[data["resolution"].eq("mid_res"), "IMG_ID"]) & set(
        data.loc[data["resolution"].eq("high_res"), "IMG_ID"]
    )
    valid_ids = [img_id for img_id in valid_ids if img_id in eligible_ids]

    # Calcula limites dos eixos compartilhados conforme a métrica exibida.
    default_xlim = (
        (0.0, max(float(data["volume_ml"].max()) * 1.05, 1.0))
        if volume_plot_xlim is None
        else volume_plot_xlim
    )
    axis_limit = (
        default_xlim
        if y_column == "dice"
        else (0.0, max(float(data[["volume_ml", y_column]].max().max()) * 1.05, 1.0))
    )
    cohorts = {
        "Todos": data,
        "Óstios válidos em ambas": data.loc[data["IMG_ID"].isin(valid_ids)],
    }
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes="all" if y_column == "dice" else False,
        shared_yaxes=y_column == "dice",
        subplot_titles=("Mid", "High"),
        horizontal_spacing=0.07,
    )

    # Adiciona traces separados por coorte, resolução, sucesso dos óstios e status de outlier.
    trace_cohorts = []
    annotations_by_cohort = {}
    for cohort_index, (cohort_name, cohort) in enumerate(cohorts.items()):
        annotations = []
        legend_seen = set()
        for col, (resolution, label) in enumerate(
            (("mid_res", "Mid"), ("high_res", "High")), start=1
        ):
            group = cohort.loc[cohort["resolution"].eq(resolution)]
            annotation = fig.layout.annotations[col - 1].to_plotly_json()
            annotation["text"] = f"{label} (N={len(group)})"
            annotations.append(annotation)
            for success, status_label in (
                (True, "Óstios válidos"),
                (False, "Falha dos óstios"),
            ):
                for outlier, symbol in ((False, "circle"), (True, "x")):
                    selected = group.loc[
                        group["ostia_success"].eq(success)
                        & group["is_volume_outlier"].eq(outlier)
                    ]
                    volume_label = "Outlier" if outlier else "Dentro da faixa"
                    legend_name = f"{status_label} · {volume_label}"
                    customdata = [
                        [
                            img_id,
                            label,
                            status_label,
                            {
                                "low": "Baixo (outlier)",
                                "in_range": "Dentro da faixa",
                                "high": "Alto (outlier)",
                            }[volume_group],
                            dice_value,
                        ]
                        for img_id, volume_group, dice_value in selected[
                            ["IMG_ID", "volume_outlier_group", "dice"]
                        ].itertuples(index=False, name=None)
                    ]
                    fig.add_trace(
                        go.Scatter(
                            x=selected["volume_ml"].tolist(),
                            y=selected[y_column].tolist(),
                            mode="markers",
                            name=legend_name,
                            legendgroup=legend_name,
                            showlegend=bool(len(selected))
                            and legend_name not in legend_seen,
                            visible=cohort_index == 0,
                            customdata=customdata,
                            marker=dict(
                                color=OSTIA_COLORS[status_label],
                                symbol=symbol,
                                size=7,
                                opacity=0.65,
                            ),
                            hovertemplate=(
                                "Exame: %{customdata[0]}<br>Resolução: %{customdata[1]}"
                                "<br>Referência: %{x:.3f} mL"
                                + (
                                    "<br>Dice: %{y:.4f}"
                                    if y_column == "dice"
                                    else "<br>Predição: %{y:.3f} mL<br>Dice: %{customdata[4]:.4f}"
                                )
                                + "<br>Óstios: %{customdata[2]}<br>Volume: %{customdata[3]}"
                                "<extra></extra>"
                            ),
                        ),
                        row=1,
                        col=col,
                    )
                    trace_cohorts.append(cohort_name)
                    if len(selected):
                        legend_seen.add(legend_name)
        if cohort.empty:
            annotations.append(
                dict(
                    text="Nenhum exame elegível nesta coorte.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                )
            )
        annotations_by_cohort[cohort_name] = annotations

    # Constrói botões de filtro interativo por coorte e ajusta eixos/linhas de identidade.
    buttons = [
        dict(
            label=cohort_name,
            method="update",
            args=[
                {"visible": [key == cohort_name for key in trace_cohorts]},
                {
                    "annotations": annotations_by_cohort[cohort_name],
                    "title": {"text": f"{title} — {cohort_name}"},
                },
            ],
        )
        for cohort_name in cohorts
    ]
    fig.update_xaxes(
        title_text="Volume arterial de referência (mL)",
        range=list(axis_limit),
        autorange=False,
    )
    fig.update_yaxes(
        range=[0, 1] if y_column == "dice" else list(axis_limit), autorange=False
    )
    fig.update_yaxes(
        title_text="Dice" if y_column == "dice" else "Volume predito (mL)", row=1, col=1
    )
    if y_column != "dice":
        for col in (1, 2):
            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=axis_limit[1],
                y1=axis_limit[1],
                line=dict(color="#555555", dash="dash"),
                row=1,
                col=col,
            )
            fig.update_yaxes(
                scaleanchor="x" if col == 1 else "x2", scaleratio=1, row=1, col=col
            )
    fig.update_layout(
        title=dict(text=f"{title} — Todos"),
        annotations=annotations_by_cohort["Todos"],
        template="plotly_white",
        height=560,
        margin=dict(t=130, b=110),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                x=0,
                y=1.22,
                xanchor="left",
                yanchor="top",
                active=0,
            )
        ],
    )
    return fig


def plot_volume_histogram(
    data: pd.DataFrame,
    limits: dict[str, Any],
    bins: int,
) -> Any:
    """Plota o histograma de volumes arteriais de referência com KDE e limites de outlier.

    Args:
        data (pd.DataFrame): DataFrame contendo a coluna `volume_ml` dos exames válidos.
        limits (dict[str, Any]): Dicionário com limites inferior/superior e contagem de exames.
        bins (int): Número de intervalos do histograma.

    Returns:
        Any: Figura Plotly com o histograma ou `None` quando não há dados válidos.
    """
    if data.empty:
        print("Não há volumes válidos para o histograma.")
        return None

    # Calcula contagens e centros dos bins do histograma em mL.
    values = data["volume_ml"].to_numpy(dtype=float)
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    fig = go.Figure(
        go.Bar(
            x=centers,
            y=counts,
            width=widths,
            name="Exames",
            marker_color="#4682B4",
            customdata=np.column_stack((edges[:-1], edges[1:])),
            hovertemplate="Intervalo: %{customdata[0]:.3f}–%{customdata[1]:.3f} mL"
            "<br>Exames: %{y}<extra></extra>",
        )
    )

    # Estima e adiciona a curva de densidade suavizada (KDE) quando há variância positiva.
    peak = float(counts.max())
    if len(values) > 1 and np.ptp(values) > 0:
        try:
            xs = np.linspace(edges[0], edges[-1], 250)
            ys = gaussian_kde(values)(xs) * len(values) * widths[0]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name="Densidade suavizada",
                    line=dict(color="#365E80"),
                )
            )
            peak = max(peak, float(ys.max()))
        except (np.linalg.LinAlgError, ValueError):
            print("Curva de densidade indisponível; contagens preservadas.")

    # Adiciona linhas verticais para média, mediana e limites de outlier.
    lines = [
        (float(np.mean(values)), "Média", "#B22222", "dash"),
        (float(np.median(values)), "Mediana", "#006400", "dot"),
    ]
    if limits["exam_count"]:
        lines.extend(
            [
                (limits["lower_limit_ml"], "Limite inferior", "#D97706", "dashdot"),
                (limits["upper_limit_ml"], "Limite superior", "#7E22CE", "dashdot"),
            ]
        )
    for value, label, color, dash in lines:
        if np.isfinite(value):
            fig.add_trace(
                go.Scatter(
                    x=[value, value],
                    y=[0, max(peak * 1.08, 1)],
                    mode="lines",
                    name=f"{label}: {value:.2f} mL",
                    line=dict(color=color, dash=dash),
                    hovertemplate=f"{label}: {value:.3f} mL<extra></extra>",
                )
            )
    fig.update_layout(
        title=f"Volume coronariano de referência — ImageCAS (N={len(values)})",
        template="plotly_white",
        height=500,
        barmode="overlay",
        xaxis_title="Volume arterial (mL)",
        yaxis_title="Número de exames",
        legend=dict(orientation="h", y=-0.22),
        margin=dict(b=120),
    )
    return fig


def plot_volume_dice_delta(
    pairs: pd.DataFrame,
    *,
    split_name: str = "test",
    volume_plot_xlim: tuple[float, float] | None = None,
) -> Any:
    """Plota a diferença de Dice (High menos Mid) em função do volume arterial de referência.

    Args:
        pairs (pd.DataFrame): DataFrame com pares Mid/High e coluna `delta_dice`.
        split_name (str): Identificador do conjunto analisado (`train`, `val`, `test` ou `all`).
        volume_plot_xlim (tuple[float, float] | None): Limites explícitos do eixo X em mL.

    Returns:
        Any: Figura Plotly com a dispersão de deltas ou `None` se `pairs` estiver vazio.
    """
    if pairs.empty:
        print("Nenhum par Mid/High com volume conhecido.")
        return None

    # Separa as coortes completa e condicionada ao sucesso dos óstios nas duas resoluções.
    cohorts = {
        "Todos": pairs,
        "Óstios válidos em ambas": pairs.loc[pairs["ostia_success_both"]],
    }
    fig = go.Figure()
    trace_cohorts = []
    for cohort_index, (name, cohort) in enumerate(cohorts.items()):
        for success, color in ((True, "#247A58"), (False, "#C44747")):
            status = (
                "Óstios válidos em ambas" if success else "Falha em alguma resolução"
            )
            for outlier, symbol in ((False, "circle"), (True, "x")):
                group = cohort.loc[
                    cohort["ostia_success_both"].eq(success)
                    & cohort["is_volume_outlier"].eq(outlier)
                ]
                custom = [
                    [
                        row.IMG_ID,
                        row.dice_mid,
                        row.dice_high,
                        "Válidos" if row.ostia_success_mid else "Falha",
                        "Válidos" if row.ostia_success_high else "Falha",
                        {
                            "low": "Baixo (outlier)",
                            "in_range": "Dentro da faixa",
                            "high": "Alto (outlier)",
                        }[row.volume_outlier_group],
                    ]
                    for row in group.itertuples(index=False)
                ]
                fig.add_trace(
                    go.Scatter(
                        x=group["volume_ml"].tolist(),
                        y=group["delta_dice"].tolist(),
                        customdata=custom,
                        mode="markers",
                        visible=cohort_index == 0,
                        name=f"{status} · {'Outlier' if outlier else 'Dentro da faixa'}",
                        marker=dict(color=color, symbol=symbol, size=7, opacity=0.65),
                        showlegend=bool(len(group)),
                        hovertemplate="Exame: %{customdata[0]}<br>Volume: %{x:.3f} mL"
                        "<br>Dice Mid: %{customdata[1]:.4f}<br>Dice High: %{customdata[2]:.4f}"
                        "<br>Δ Dice: %{y:.4f}<br>Óstios Mid: %{customdata[3]}"
                        "<br>Óstios High: %{customdata[4]}<br>Volume: %{customdata[5]}<extra></extra>",
                    )
                )
                trace_cohorts.append(name)

    # Configura botões de alternância de coorte, linha horizontal zero e limites dos eixos.
    title = f"{SPLIT_LABELS.get(split_name, split_name)}: Δ Dice (High − Mid)"
    buttons = []
    for name, cohort in cohorts.items():
        buttons.append(
            dict(
                label=name,
                method="update",
                args=[
                    {"visible": [key == name for key in trace_cohorts]},
                    {
                        "title": {"text": f"{title} — {name} (N={len(cohort)})"},
                        "annotations": []
                        if len(cohort)
                        else [
                            dict(
                                text="Nenhum exame elegível nesta coorte.",
                                x=0.5,
                                y=0.5,
                                xref="paper",
                                yref="paper",
                                showarrow=False,
                            )
                        ],
                    },
                ],
            )
        )
    resolved_xlim = (
        (0.0, max(float(pairs["volume_ml"].max()) * 1.05, 1.0))
        if volume_plot_xlim is None
        else volume_plot_xlim
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#555555")
    fig.update_layout(
        title=f"{title} — Todos (N={len(pairs)})",
        template="plotly_white",
        height=550,
        xaxis=dict(
            title="Volume arterial de referência (mL)",
            range=list(resolved_xlim),
            autorange=False,
        ),
        yaxis=dict(title="Δ Dice (High − Mid)", range=[-1, 1], autorange=False),
        updatemenus=[dict(buttons=buttons, x=0, y=1.2, active=0)],
        legend=dict(orientation="h", y=-0.22),
        margin=dict(t=120, b=120),
    )
    return fig


def plot_volume_band_performance(
    summary: pd.DataFrame,
    *,
    split_name: str = "test",
) -> Any:
    """Plota Dice médio e taxa de sucesso dos óstios por faixa de volume de referência.

    Args:
        summary (pd.DataFrame): Resumo agregado por faixa de volume (`band`) e resolução.
        split_name (str): Identificador do conjunto analisado (`train`, `val`, `test` ou `all`).

    Returns:
        Any: Figura Plotly com dois painéis de barras ou `None` se `summary` estiver vazio.
    """
    if summary.empty:
        print("Sem dados para comparar faixas de volume.")
        return None

    # Cria dois painéis lado a lado para Dice médio (± DP) e percentual de sucesso dos óstios.
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Dice médio ± DP", "Sucesso dos óstios")
    )
    for resolution, label, color in (
        ("mid_res", "Mid", "#3574A5"),
        ("high_res", "High", "#B76A32"),
    ):
        group = summary.loc[summary["resolution"].eq(resolution)].sort_values("band")
        custom = [
            [
                row.lower_ml,
                row.upper_ml,
                row.exam_count,
                row.dice_mean,
                "Indisponível (N=1)"
                if pd.isna(row.dice_std)
                else f"{row.dice_std:.4f}",
                row.dice_median,
                row.ostia_success_count,
            ]
            for row in group.itertuples(index=False)
        ]
        hover = (
            "Faixa: %{customdata[0]:.3f}–%{customdata[1]:.3f} mL"
            "<br>N: %{customdata[2]}<br>Dice médio: %{customdata[3]:.4f}"
            "<br>DP: %{customdata[4]}<br>Mediana: %{customdata[5]:.4f}"
            "<br>Sucessos: %{customdata[6]}<extra>%{fullData.name}</extra>"
        )
        for col, y in ((1, "dice_mean"), (2, "ostia_success_percent")):
            fig.add_trace(
                go.Bar(
                    x=[f"F{int(band)}" for band in group["band"]],
                    y=group[y].tolist(),
                    name=label,
                    legendgroup=label,
                    showlegend=col == 1,
                    marker_color=color,
                    offsetgroup=label,
                    customdata=custom,
                    hovertemplate=hover,
                    error_y=dict(
                        type="data", array=group["dice_std"].fillna(0).tolist()
                    )
                    if col == 1
                    else None,
                ),
                row=1,
                col=col,
            )

    # Ajusta escalas dos eixos e título conforme o conjunto avaliado.
    fig.update_yaxes(title_text="Dice", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Exames (%)", range=[0, 100], row=1, col=2)
    fig.update_xaxes(title_text="Faixa de volume de referência")
    fig.update_layout(
        title=f"{SPLIT_LABELS.get(split_name, split_name)}: desempenho por faixa de volume",
        template="plotly_white",
        height=520,
        barmode="group",
    )
    return fig
