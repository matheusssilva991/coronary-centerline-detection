import matplotlib.pyplot as plt
import pandas as pd


def _get_split_df(data_by_resolution, resolution, split_name):
    if data_by_resolution is None:
        return None
    resolution_data = data_by_resolution.get(resolution, {})
    return resolution_data.get(split_name)


def _get_split_title(split_name):
    return str(split_name).upper()


def _get_status_color(split_name):
    return {
        "train": "skyblue",
        "val": "lightgreen",
        "test": "salmon",
    }.get(split_name, "steelblue")


def plot_status_distribution_by_subset(
    data_by_resolution, split_name, status_column="status"
):
    """Plota a distribuição de status para High e Mid dentro de um subset."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    title_suffix = _get_split_title(split_name)
    color = _get_status_color(split_name)

    for idx, resolution in enumerate(["high", "mid"]):
        df_split = _get_split_df(data_by_resolution, resolution, split_name)
        ax = axes[idx]

        if df_split is None or df_split.empty or status_column not in df_split.columns:
            ax.text(
                0.5,
                0.5,
                f"Sem dados para {resolution.upper()}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(
                f"{resolution.upper()} Resolution - Distribuição dos Status ({title_suffix})"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        status_counts = df_split[status_column].value_counts()
        status_counts.plot(kind="bar", color=color, edgecolor="black", ax=ax)
        # Ajusta o limite superior para evitar que os rótulos sobreponham o título
        max_count = max(status_counts.values) if len(status_counts.values) > 0 else 0
        ax.set_ylim(0, max_count * 1.18 if max_count > 0 else 1)
        ax.set_title(
            f"{resolution.upper()} Resolution - Distribuição dos Status ({title_suffix})",
            pad=12,
        )
        ax.set_xlabel("Status")
        ax.set_ylabel("Número de Imagens")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        for i, value in enumerate(status_counts.values):
            text_offset = max_count * 0.03 if max_count > 0 else 0.1
            ax.text(
                i,
                value + text_offset,
                str(value),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.show()


def plot_success_error_by_subset(
    data_by_resolution,
    split_name,
    success_status,
    status_column="status",
    success_color="#4CAF50",
    error_color="#F44336",
):
    """Plota acertos vs erros para High e Mid dentro de um subset."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    title_suffix = _get_split_title(split_name)

    for idx, resolution in enumerate(["high", "mid"]):
        df_split = _get_split_df(data_by_resolution, resolution, split_name)
        ax = axes[idx]

        if df_split is None or df_split.empty or status_column not in df_split.columns:
            ax.text(
                0.5,
                0.5,
                f"Sem dados para {resolution.upper()}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(
                f"{resolution.upper()} Resolution - Acertos vs Erros ({title_suffix})"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        total = len(df_split)
        successful_count = int(df_split[status_column].isin(success_status).sum())
        error_count = total - successful_count

        success_pct = 100 * successful_count / total if total > 0 else 0
        error_pct = 100 * error_count / total if total > 0 else 0

        bars = ax.bar(
            ["Acertos", "Erros"],
            [successful_count, error_count],
            color=[success_color, error_color],
            alpha=0.8,
            edgecolor="black",
        )

        ax.set_ylabel("Número de Imagens")
        # Ajusta limite superior do eixo y para dar espaço aos rótulos
        max_count = max(successful_count, error_count)
        ax.set_ylim(0, max_count * 1.18 if max_count > 0 else 1)
        ax.set_title(
            f"{resolution.upper()} Resolution - Acertos vs Erros ({title_suffix})",
            pad=12,
        )
        ax.grid(axis="y", alpha=0.3)

        for i, (bar, value) in enumerate(zip(bars, [successful_count, error_count])):
            percentage = [success_pct, error_pct][i]
            height = bar.get_height()
            text_offset = max_count * 0.03 if max_count > 0 else 0.1
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + text_offset,
                f"{int(value)}\n({percentage:.1f}%)",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.show()


def plot_distance_distribution_by_subset(
    data_by_resolution,
    split_name,
    left_column="left_dist_mm",
    right_column="right_dist_mm",
    tolerance_mm=7.0,
    bins=None,
    high_color="#FF6B6B",
    mid_color="#4ECDC4",
):
    """Plota histogramas de distâncias left/right para High e Mid dentro de um subset."""
    if bins is None:
        bins = {"train": 25, "val": 15, "test": 25}.get(split_name, 25)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title_suffix = _get_split_title(split_name)

    for idx, resolution in enumerate(["high", "mid"]):
        df_split = _get_split_df(data_by_resolution, resolution, split_name)
        ax_left = axes[0, idx]
        ax_right = axes[1, idx]

        if df_split is None or df_split.empty:
            for ax in [ax_left, ax_right]:
                ax.text(
                    0.5,
                    0.5,
                    f"Sem dados para {resolution.upper()}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
            ax_left.set_title(
                f"{resolution.upper()} - Left Distance Distribution ({title_suffix})"
            )
            ax_right.set_title(
                f"{resolution.upper()} - Right Distance Distribution ({title_suffix})"
            )
            continue

        left_dist = (
            pd.to_numeric(df_split[left_column], errors="coerce").dropna()
            if left_column in df_split.columns
            else pd.Series(dtype=float)
        )
        right_dist = (
            pd.to_numeric(df_split[right_column], errors="coerce").dropna()
            if right_column in df_split.columns
            else pd.Series(dtype=float)
        )
        color = high_color if resolution == "high" else mid_color

        # Contagem de valores abaixo e acima do limiar
        left_below = int((left_dist <= tolerance_mm).sum())
        left_above = int((left_dist > tolerance_mm).sum())
        right_below = int((right_dist <= tolerance_mm).sum())
        right_above = int((right_dist > tolerance_mm).sum())

        print(f"\n{resolution.upper()} - {title_suffix}")
        print(
            f"  Left Distance: {left_below} abaixo de {tolerance_mm}mm, {left_above} acima"
        )
        print(
            f"  Right Distance: {right_below} abaixo de {tolerance_mm}mm, {right_above} acima"
        )

        ax_left.hist(left_dist, bins=bins, alpha=0.8, color=color, edgecolor="black")
        ax_left.axvline(
            tolerance_mm,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Tolerância ({tolerance_mm:g}mm)",
        )
        ax_left.set_xlabel("Left Distance (mm)")
        ax_left.set_ylabel("Frequência")
        ax_left.set_title(
            f"{resolution.upper()} - Left Distance Distribution ({title_suffix})"
        )
        ax_left.legend()
        ax_left.grid(axis="y", alpha=0.3)

        ax_right.hist(right_dist, bins=bins, alpha=0.8, color=color, edgecolor="black")
        ax_right.axvline(
            tolerance_mm,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Tolerância ({tolerance_mm:g}mm)",
        )
        ax_right.set_xlabel("Right Distance (mm)")
        ax_right.set_ylabel("Frequência")
        ax_right.set_title(
            f"{resolution.upper()} - Right Distance Distribution ({title_suffix})"
        )
        ax_right.legend()
        ax_right.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def build_dice_summary_by_subset(
    data_by_resolution, split_name, dice_column="dice_artery"
):
    """Retorna um resumo de Dice por resolução para um subset."""
    rows = []

    for resolution in ["high", "mid"]:
        df_split = _get_split_df(data_by_resolution, resolution, split_name)
        if df_split is None or df_split.empty or dice_column not in df_split.columns:
            dice_values = pd.Series(dtype=float)
        else:
            dice_values = pd.to_numeric(df_split[dice_column], errors="coerce").dropna()

        rows.append(
            {
                "resolution": resolution.upper(),
                "split": _get_split_title(split_name),
                "count": int(dice_values.shape[0]),
                "mean": float(dice_values.mean()) if not dice_values.empty else pd.NA,
                "median": float(dice_values.median())
                if not dice_values.empty
                else pd.NA,
                "std": float(dice_values.std()) if not dice_values.empty else pd.NA,
                "min": float(dice_values.min()) if not dice_values.empty else pd.NA,
                "max": float(dice_values.max()) if not dice_values.empty else pd.NA,
            }
        )

    return pd.DataFrame(rows)
