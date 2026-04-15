import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def compare_shared_bad_cases(
    df_mid,
    df_high,
    subset_label,
    dice_threshold=0.30,
    max_examples=10,
):
    """Compara casos ruins compartilhados entre resultados mid e high."""
    if df_mid is None or df_high is None or df_mid.empty or df_high.empty:
        print(f"Sem dados suficientes para comparativo em {subset_label}.")
        return None

    id_col_mid = "IMG_ID" if "IMG_ID" in df_mid.columns else "img_id"
    id_col_high = "IMG_ID" if "IMG_ID" in df_high.columns else "img_id"

    def _status_ids(df, status_label, id_col):
        if "status" not in df.columns:
            return set()
        return set(
            df.loc[df["status"].eq(status_label).fillna(False), id_col]
            .dropna()
            .astype(int)
        )

    def _low_dice_mask(df):
        return (
            pd.to_numeric(df.get("dice_artery"), errors="coerce") < dice_threshold
        ).fillna(False)

    ids_low_mid = set(
        df_mid.loc[_low_dice_mask(df_mid), id_col_mid].dropna().astype(int)
    )
    ids_low_high = set(
        df_high.loc[_low_dice_mask(df_high), id_col_high].dropna().astype(int)
    )
    ids_low_both = ids_low_mid & ids_low_high

    status_groups = [
        ("Ostios nao encontrados em ambos", "óstios não encontrados"),
        ("Um correto em ambos", "um correto"),
        ("Nenhum correto em ambos", "nenhum correto"),
        ("Erro em ambos", "erro"),
    ]

    status_intersections = {}
    for display_name, status_label in status_groups:
        ids_mid = _status_ids(df_mid, status_label, id_col_mid)
        ids_high = _status_ids(df_high, status_label, id_col_high)
        status_intersections[display_name] = ids_mid & ids_high

    print("\n" + "-" * 60)
    print(f"COMPARATIVO MID vs HIGH - {subset_label}")
    print("-" * 60)
    print(f"Mesmo caso com Dice < {dice_threshold} em ambos: {len(ids_low_both)}")
    for display_name in status_intersections:
        print(f"{display_name}: {len(status_intersections[display_name])}")

    if ids_low_both:
        exemplos_low = sorted(ids_low_both)[:max_examples]
        print(f"Exemplos (Dice baixo em ambos): {exemplos_low}")

    for display_name, ids in status_intersections.items():
        if ids:
            exemplos = sorted(ids)[:max_examples]
            print(f"Exemplos ({display_name.lower()}): {exemplos}")

    compare_rows = [
        {"tipo": f"Dice < {dice_threshold} em ambos", "quantidade": len(ids_low_both)}
    ]
    compare_rows.extend(
        {"tipo": display_name, "quantidade": len(ids)}
        for display_name, ids in status_intersections.items()
    )
    compare_df = pd.DataFrame(compare_rows)

    plt.figure(figsize=(9.5, 4.8))
    ax_cmp = sns.barplot(data=compare_df, x="tipo", y="quantidade", color="#4C78A8")
    ax_cmp.set_title(f"Intersecao Mid vs High por IMG_ID - {subset_label}")
    ax_cmp.set_xlabel("Tipo de intersecao")
    ax_cmp.set_ylabel("Quantidade de imagens")
    ax_cmp.tick_params(axis="x", rotation=15)
    ax_cmp.grid(axis="y", alpha=0.3)
    for container in ax_cmp.containers:
        ax_cmp.bar_label(container, fmt="%d", padding=3)
    plt.tight_layout()
    plt.show()

    return {"ids_low_both": ids_low_both, "status_intersections": status_intersections}


def change_status_label_for_plot(status):
    """Normaliza rótulos de status para visualização em gráfico."""
    status = str(status)
    if "erro" in status.lower():
        return "ostios nao encontrados"
    if "ambos toler" in status.lower() or "ambos corret" in status.lower():
        return "baixo dice score"
    return status.lower()


def plot_bad_cases_by_subset(df_mid, df_high, df_mid_bad, df_high_bad, subset_label):
    """Plota distribuição de status dos casos ruins por subconjunto."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if df_mid is not None and not df_mid.empty:
        mid_status = df_mid_bad["status"].fillna("sem status").value_counts()
        mid_status.index = mid_status.index.map(change_status_label_for_plot)
        mid_status = mid_status.groupby(level=0).sum()
        sns.barplot(
            x=mid_status.index, y=mid_status.values, color="#4C78A8", ax=axes[0]
        )
        axes[0].set_xlabel("Status")
        axes[0].set_ylabel("Quantidade de casos")
        axes[0].set_title(
            f"Distribuicao de status dos casos ruins - Mid Res ({subset_label})"
        )
        axes[0].tick_params(axis="x", rotation=45)
        for patch in axes[0].patches:
            height = patch.get_height()
            axes[0].text(
                patch.get_x() + patch.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )
    else:
        axes[0].text(
            0.5,
            0.5,
            f"Sem dados de {subset_label.lower()} para Mid Res",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        axes[0].set_title(
            f"Distribuicao de status dos casos ruins - Mid Res ({subset_label})"
        )

    if df_high is not None and not df_high.empty:
        high_status = df_high_bad["status"].fillna("sem status").value_counts()
        high_status.index = high_status.index.map(change_status_label_for_plot)
        high_status = high_status.groupby(level=0).sum()
        sns.barplot(
            x=high_status.index, y=high_status.values, color="#F58518", ax=axes[1]
        )
        axes[1].set_xlabel("Status")
        axes[1].set_ylabel("Quantidade de casos")
        axes[1].set_title(
            f"Distribuicao de status dos casos ruins - High Res ({subset_label})"
        )
        axes[1].tick_params(axis="x", rotation=45)
        for patch in axes[1].patches:
            height = patch.get_height()
            axes[1].text(
                patch.get_x() + patch.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )
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
        axes[1].set_title(
            f"Distribuicao de status dos casos ruins - High Res ({subset_label})"
        )
        axes[1].set_xticks([])
        axes[1].set_yticks([])

    plt.tight_layout()
    plt.show()
