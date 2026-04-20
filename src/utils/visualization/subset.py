import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def prepare_subset_plot_df(subset_summary_df):
    """Prepara DataFrame de subconjuntos disponível para plotagem."""
    # Filtra apenas linhas marcadas como disponíveis.
    plot_subset_df = subset_summary_df[subset_summary_df["disponivel"]].copy()
    # Traduz nome do subset para rótulo do eixo X.
    plot_subset_df["conjunto"] = plot_subset_df["subset"].map(
        {"train": "Treino", "val": "Validacao", "test": "Teste"}
    )
    return plot_subset_df


def plot_subset_metric_by_resolution(
    subset_summary_df,
    metric_col,
    title,
    ylabel,
    palette,
    ylim=None,
    hline_y=None,
    hline_kwargs=None,
    bar_label_fmt="%.3f",
):
    """Plota uma métrica por subconjunto com separação por resolução."""
    # Gera base de plot com subsets válidos.
    plot_subset_df = prepare_subset_plot_df(subset_summary_df)

    # Barra agrupada por subset e resolução.
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=plot_subset_df,
        x="conjunto",
        y=metric_col,
        hue="resolucao",
        palette=palette,
    )

    if ylim is not None:
        # Aplica limite fixo quando definido.
        ax.set_ylim(*ylim)

    if hline_y is not None:
        # Linha de referência opcional.
        line_kwargs = {"color": "black", "linewidth": 1, "linestyle": "--"}
        if hline_kwargs:
            line_kwargs.update(hline_kwargs)
        ax.axhline(hline_y, **line_kwargs)

    ax.set_title(title)
    ax.set_xlabel("Conjunto")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    # Legenda identifica cada resolução.
    ax.legend(title="Resolucao")

    for container in ax.containers:
        # Rótulo numérico em cada barra.
        ax.bar_label(container, fmt=bar_label_fmt, padding=3)

    plt.tight_layout()
    plt.show()
    return ax


def plot_subset_execution_time_by_resolution(
    subset_summary_df,
    palette=None,
    ymax_factor=1.12,
):
    """Plota tempo de execução por subconjunto e resolução."""
    # Plota tempo (min) por subset usando helper comum.
    if palette is None:
        palette = ["#F58518", "#E45756"]

    plot_subset_df = prepare_subset_plot_df(subset_summary_df)
    max_time = plot_subset_df["tempo_execucao_min"].max()
    if pd.notna(max_time):
        ylim = (0, max_time * ymax_factor)
    else:
        ylim = (0, 1)

    # Ajusta o formato do label para reforçar a leitura dos minutos.
    return plot_subset_metric_by_resolution(
        subset_summary_df=subset_summary_df,
        metric_col="tempo_execucao_min",
        title="Tempo de execução por conjunto e resolução",
        ylabel="Tempo de execução (min)",
        palette=palette,
        ylim=ylim,
        bar_label_fmt="%.1f",
    )


def plot_subset_ostia_success_by_resolution(
    subset_summary_df,
    palette=None,
):
    """Plota sucesso de detecção de óstios por subconjunto e resolução."""
    # Plota sucesso total (%) por subset usando helper comum.
    if palette is None:
        palette = ["#54A24B", "#2E8B57"]

    return plot_subset_metric_by_resolution(
        subset_summary_df=subset_summary_df,
        metric_col="sucesso_total_percent",
        title="Percentual de óstios detectados com sucesso por conjunto e resolução",
        ylabel="Óstios detectados com sucesso (%)",
        palette=palette,
        ylim=(0, 100),
        bar_label_fmt="%.1f%%",
    )
