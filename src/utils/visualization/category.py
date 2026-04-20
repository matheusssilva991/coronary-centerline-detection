import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_category_metric_bar(
    df,
    x_col,
    y_col,
    title,
    xlabel,
    ylabel,
    color,
    figsize=(8, 5),
    ylim=None,
    ymax_factor=None,
    x_rotation=15,
    bar_label_fmt="%.3f",
):
    """Plota barra de uma métrica categórica com rótulos e grade."""
    # Função base para barras usadas nos relatórios do notebook.
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=df, x=x_col, y=y_col, color=color)

    if ylim is not None:
        # Usa escala fixa quando o limite já é conhecido.
        ax.set_ylim(*ylim)
    elif ymax_factor is not None:
        max_val = df[y_col].max()
        if pd.notna(max_val):
            # Escala automática com folga para os rótulos.
            ax.set_ylim(0, max_val * ymax_factor)

    # Configura rótulos e grade padrão.
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=x_rotation)
    ax.grid(axis="y", alpha=0.3)

    # Exibe valor em cada barra.
    for container in ax.containers:
        ax.bar_label(container, fmt=bar_label_fmt, padding=3)

    plt.tight_layout()
    plt.show()
    return ax


def plot_downscale_execution_time(downscale_df, color="#F58518"):
    """Plota tempo de execução por método de downscale."""
    # Compara minutos de execução por método de downscale.
    return plot_category_metric_bar(
        df=downscale_df,
        x_col="metodo",
        y_col="tempo_execucao_min",
        title="Tempo de execução por método de redução de escala",
        xlabel="Método",
        ylabel="Tempo de execução (min)",
        color=color,
        figsize=(8, 5),
        ymax_factor=1.12,
        x_rotation=15,
        bar_label_fmt="%.1f",
    )


def plot_downscale_dice(downscale_df, color="#4C78A8"):
    """Plota Dice médio por método de downscale."""
    # Compara Dice médio por método de downscale.
    return plot_category_metric_bar(
        df=downscale_df,
        x_col="metodo",
        y_col="dice_medio",
        title="Dice médio por método de redução de escala",
        xlabel="Método",
        ylabel="Dice médio",
        color=color,
        figsize=(8, 5),
        ylim=(0, 1),
        x_rotation=15,
        bar_label_fmt="%.3f",
    )


def plot_downscale_ostia_success(downscale_df, color="#54A24B"):
    """Plota sucesso de detecção de óstios por método de downscale."""
    # Compara sucesso total (%) por método de downscale.
    return plot_category_metric_bar(
        df=downscale_df,
        x_col="metodo",
        y_col="sucesso_total_percent",
        title="Sucesso na detecção dos óstios por método de redução de escala",
        xlabel="Método",
        ylabel="Sucesso total (%)",
        color=color,
        figsize=(8, 5),
        ylim=(0, 100),
        x_rotation=15,
        bar_label_fmt="%.1f%%",
    )


def plot_validation_dice(summary_df, color="#4C78A8"):
    """Plota Dice médio por conjunto de validação."""
    # Compara Dice médio entre conjuntos de validação.
    return plot_category_metric_bar(
        df=summary_df,
        x_col="dataset",
        y_col="dice_medio",
        title="Dice médio por conjunto de validação",
        xlabel="Conjunto de validação",
        ylabel="Dice médio",
        color=color,
        figsize=(7, 5),
        ylim=(0, 1),
        x_rotation=0,
        bar_label_fmt="%.3f",
    )


def plot_validation_execution_time(summary_df, color="#F58518"):
    """Plota tempo de execução por conjunto de validação."""
    # Compara tempo total (min) entre conjuntos de validação.
    return plot_category_metric_bar(
        df=summary_df,
        x_col="dataset",
        y_col="tempo_execucao_min",
        title="Tempo de execução por conjunto de validação",
        xlabel="Conjunto de validação",
        ylabel="Tempo de execução (min)",
        color=color,
        figsize=(7, 5),
        ymax_factor=1.12,
        x_rotation=0,
        bar_label_fmt="%.1f",
    )


def plot_validation_ostia_success(summary_df, color="#54A24B"):
    """Plota sucesso total de óstios por conjunto de validação."""
    # Compara sucesso total (%) entre conjuntos de validação.
    return plot_category_metric_bar(
        df=summary_df,
        x_col="dataset",
        y_col="sucesso_total_percent",
        title="Sucesso total de detecção dos óstios",
        xlabel="Conjunto de validação",
        ylabel="Sucesso total (%)",
        color=color,
        figsize=(7, 5),
        ylim=(0, 100),
        x_rotation=0,
        bar_label_fmt="%.1f%%",
    )
