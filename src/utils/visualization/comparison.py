import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..comparison_utils.ia_math import prettify_method_label


def plot_comparison_bar_by_resolution(agg, resolution):
    """Plota comparação de Dice por método para uma resolução alvo."""
    # Filtra somente a resolução solicitada.
    subset = agg[agg["target_resolution"] == resolution].copy()
    if subset.empty:
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, f"Sem dados para {resolution}", ha="center", va="center")
        plt.axis("off")
        plt.show()
        return

    subset["origin_label"] = subset["source"].map({"ia": "IA", "math": "Matematico"})
    subset["method_label"] = subset["method"].apply(prettify_method_label)
    # Ordena métodos por média de Dice.
    method_order = subset.sort_values("mean_dice", ascending=False)[
        "method_label"
    ].tolist()
    subset["method_label"] = pd.Categorical(
        subset["method_label"], categories=method_order, ordered=True
    )
    subset = subset.sort_values("method_label")

    # Plota barras separando IA e Matemático por cor.
    plt.figure(figsize=(12, 5))
    ax = sns.barplot(
        data=subset,
        x="method_label",
        y="mean_dice",
        hue="origin_label",
        palette={"IA": "#4C78A8", "Matematico": "#F58518"},
    )

    for idx, row in subset.reset_index(drop=True).iterrows():
        # Usa desvio padrão como barra de erro.
        std_val = row["std_dice"]
        if pd.notna(std_val):
            ax.errorbar(
                x=idx,
                y=row["mean_dice"],
                yerr=std_val,
                fmt="none",
                ecolor="black",
                elinewidth=1.2,
                capsize=3,
            )

    ax.set_title(f"Comparacao de Dice por metodo - {resolution}")
    ax.set_xlabel("Metodo")
    ax.set_ylabel("Dice medio")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=35)
    for tick_label in ax.get_xticklabels():
        tick_label.set_ha("right")

    for container in ax.containers:
        if hasattr(container, "patches") and container.patches:
            ax.bar_label(container, fmt="%.3f", padding=3)

    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Origem")
    plt.tight_layout()
    plt.show()


def plot_dice_distribution_by_subset(df_mid, df_high, subset_label):
    """Plota distribuição dos Dice scores para mid e high por subconjunto."""
    # Mostra histogramas de Dice para Mid e High no mesmo painel.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if df_mid is not None and not df_mid.empty:
        # Converte e remove valores inválidos de Dice.
        dice_scores_mid = pd.to_numeric(
            df_mid["dice_artery"],
            errors="coerce",
        ).dropna()
        mu_mid = dice_scores_mid.mean()
        sigma_mid = dice_scores_mid.std()
        # KDE mostra a forma da distribuição.
        sns.histplot(
            dice_scores_mid,
            bins=20,
            kde=True,
            color="skyblue",
            edgecolor="black",
            ax=axes[0],
        )
        axes[0].axvline(
            mu_mid,
            color="red",
            linestyle="-",
            linewidth=2,
            label=f"Media = {mu_mid:.3f}",
        )
        axes[0].axvline(
            mu_mid - sigma_mid,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"-1sigma = {mu_mid - sigma_mid:.3f}",
        )
        axes[0].axvline(
            mu_mid + sigma_mid,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"+1sigma = {mu_mid + sigma_mid:.3f}",
        )
        axes[0].axvspan(
            mu_mid - sigma_mid, mu_mid + sigma_mid, color="orange", alpha=0.15
        )
        # Marca média e faixa de 1 sigma.
        axes[0].set_title(f"Distribuicao dos Dice Scores - Mid Res ({subset_label})")
        axes[0].set_xlabel("Dice Score")
        axes[0].set_ylabel("Frequencia")
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].legend()
    else:
        axes[0].text(
            0.5,
            0.5,
            f"Sem dados de {subset_label.lower()} para Mid Res",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        axes[0].set_title(f"Distribuicao dos Dice Scores - Mid Res ({subset_label})")

    if df_high is not None and not df_high.empty:
        # Repete o mesmo fluxo para High.
        dice_scores_high = pd.to_numeric(
            df_high["dice_artery"], errors="coerce"
        ).dropna()
        mu_high = dice_scores_high.mean()
        sigma_high = dice_scores_high.std()
        sns.histplot(
            dice_scores_high,
            bins=20,
            kde=True,
            color="lightgreen",
            edgecolor="black",
            ax=axes[1],
        )
        axes[1].axvline(
            mu_high,
            color="red",
            linestyle="-",
            linewidth=2,
            label=f"Media = {mu_high:.3f}",
        )
        axes[1].axvline(
            mu_high - sigma_high,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"-1sigma = {mu_high - sigma_high:.3f}",
        )
        axes[1].axvline(
            mu_high + sigma_high,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"+1sigma = {mu_high + sigma_high:.3f}",
        )
        axes[1].axvspan(
            mu_high - sigma_high, mu_high + sigma_high, color="orange", alpha=0.15
        )
        axes[1].set_title(f"Distribuicao dos Dice Scores - High Res ({subset_label})")
        axes[1].set_xlabel("Dice Score")
        axes[1].set_ylabel("Frequencia")
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].legend()
    else:
        axes[1].text(
            0.5,
            0.5,
            f"Sem dados de {subset_label.lower()} para High Res\n(Em preparacao)",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            fontsize=11,
            color="gray",
        )
        axes[1].set_title(f"Distribuicao dos Dice Scores - High Res ({subset_label})")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

    plt.tight_layout()
    plt.show()
