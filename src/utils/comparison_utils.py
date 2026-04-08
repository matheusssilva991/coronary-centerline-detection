"""Utilities for IA vs mathematical result comparison in EDA notebooks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import load_json_file


def load_split_metadata(split_paths_by_resolution, resolution, subset_name):
    """Load metadata JSON for a given resolution/split, or None when unavailable."""
    split_paths = split_paths_by_resolution.get(resolution, {})
    if subset_name not in split_paths:
        return None

    metadata_path = (
        Path(split_paths[subset_name]) / f"ostios_{subset_name}_metadata.json"
    )
    return load_json_file(str(metadata_path))


def load_split_summary(split_paths_by_resolution, resolution, subset_name):
    """Load summary CSV for a given resolution/split, or None when unavailable."""
    split_paths = split_paths_by_resolution.get(resolution, {})
    if subset_name not in split_paths:
        return None

    summary_path = Path(split_paths[subset_name]) / f"ostios_{subset_name}_summary.csv"
    return pd.read_csv(summary_path)


def get_bad_cases(df, success_status=None, dice_threshold=0.30):
    """Return bad cases by status or Dice threshold."""
    if success_status is None:
        success_status = ["ambos toleráveis", "ambos corretos"]

    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else None)

    return df[
        (~df["status"].isin(success_status))
        | (pd.to_numeric(df["dice_artery"], errors="coerce") < dice_threshold)
    ]


def map_ia_resolution_to_target(ia_resolution):
    """Map IA resolution bucket to the target mathematical resolution."""
    return "high_res" if ia_resolution == "high" else "mid_res"


def prettify_method_label(method_name):
    """Create a more readable method label for plot axes."""
    if method_name == "pipeline_matematico":
        return "Pipeline Matematico"

    resolution_prefix = ""
    base_name = method_name
    if "::" in method_name:
        prefix, base_name = method_name.split("::", 1)
        resolution_prefix = prefix.upper()

    token_map = {
        "fcn": "FCN",
        "gcn": "GCN",
        "gru": "GRU",
        "lstm": "LSTM",
        "ag": "AG",
    }

    pretty_tokens = []
    for token in base_name.split("_"):
        pretty_tokens.append(
            token_map.get(token.lower(), token.upper() if token.isdigit() else token)
        )

    pretty_base = " ".join(pretty_tokens).replace("  ", " ").strip()
    if resolution_prefix:
        return f"{resolution_prefix} - {pretty_base}"
    return pretty_base


def load_ia_results_for_comparison(
    ia_results_base, allowed_ia_resolutions=("mid", "high")
):
    """Load IA CSV outputs from all folds and methods."""
    ia_results_base = Path(ia_results_base)
    ia_frames = []
    missing_ia_files = []
    allowed_ia_resolutions = set(allowed_ia_resolutions)

    columns = [
        "img_id",
        "dice",
        "origem",
        "ia_resolucao",
        "resolucao_alvo",
        "fold",
        "metodo",
    ]

    if not ia_results_base.exists():
        return pd.DataFrame(columns=columns), [
            f"Diretorio nao encontrado: {ia_results_base}"
        ]

    for ia_resolution_dir in sorted(ia_results_base.iterdir()):
        if not ia_resolution_dir.is_dir():
            continue

        ia_resolution = ia_resolution_dir.name
        if ia_resolution not in allowed_ia_resolutions:
            continue

        for fold_dir in sorted(ia_resolution_dir.glob("fold_*")):
            if not fold_dir.is_dir():
                continue

            fold_name = fold_dir.name
            csv_files = sorted(fold_dir.glob("result_*.csv"))
            if not csv_files:
                missing_ia_files.append(f"Sem CSVs em {fold_dir}")
                continue

            for csv_file in csv_files:
                method_name = csv_file.stem.replace("result_", "")
                df_ia = pd.read_csv(csv_file)

                if "dice" not in df_ia.columns or "ID" not in df_ia.columns:
                    missing_ia_files.append(f"Schema inesperado em {csv_file}")
                    continue

                df_ia = df_ia[["ID", "dice"]].copy()
                df_ia["dice"] = pd.to_numeric(df_ia["dice"], errors="coerce")
                df_ia = df_ia.dropna(subset=["dice"])
                if df_ia.empty:
                    continue

                df_ia = df_ia.rename(columns={"ID": "img_id"})
                df_ia["origem"] = "ia"
                df_ia["ia_resolucao"] = ia_resolution
                df_ia["resolucao_alvo"] = map_ia_resolution_to_target(ia_resolution)
                df_ia["fold"] = fold_name
                df_ia["metodo"] = ia_resolution + "::" + method_name
                ia_frames.append(df_ia)

    if ia_frames:
        return pd.concat(ia_frames, ignore_index=True), missing_ia_files

    return pd.DataFrame(columns=columns), missing_ia_files


def load_math_results_for_comparison(math_paths):
    """Load mathematical pipeline summary CSVs for the selected resolutions."""
    math_frames = []
    missing_math_files = []

    columns = [
        "img_id",
        "dice",
        "origem",
        "ia_resolucao",
        "resolucao_alvo",
        "fold",
        "metodo",
    ]

    for target_resolution, summary_path in math_paths.items():
        summary_path = Path(summary_path)
        if not summary_path.exists():
            missing_math_files.append(f"Arquivo nao encontrado: {summary_path}")
            continue

        df_math = pd.read_csv(summary_path)
        if "IMG_ID" not in df_math.columns or "dice_artery" not in df_math.columns:
            missing_math_files.append(f"Schema inesperado em {summary_path}")
            continue

        df_math = df_math[["IMG_ID", "dice_artery"]].copy()
        df_math["dice"] = pd.to_numeric(df_math["dice_artery"], errors="coerce")
        df_math = df_math.dropna(subset=["dice"])
        if df_math.empty:
            continue

        df_math = df_math.rename(columns={"IMG_ID": "img_id"})
        df_math = df_math.drop(columns=["dice_artery"])
        df_math["origem"] = "matematico"
        df_math["ia_resolucao"] = "n/a"
        df_math["resolucao_alvo"] = target_resolution
        df_math["fold"] = "global"
        df_math["metodo"] = "pipeline_matematico"
        math_frames.append(df_math)

    if math_frames:
        return pd.concat(math_frames, ignore_index=True), missing_math_files

    return pd.DataFrame(columns=columns), missing_math_files


def build_comparison_agg_df(comparison_raw):
    """Aggregate Dice metrics by resolution, source and method."""
    if comparison_raw.empty:
        return pd.DataFrame()

    agg = comparison_raw.groupby(
        ["resolucao_alvo", "origem", "metodo"], as_index=False
    )["dice"].agg(
        n_amostras="size",
        media_dice="mean",
        desvio_padrao="std",
        mediana_dice="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
    )

    fold_coverage = (
        comparison_raw[comparison_raw["origem"] == "ia"]
        .groupby(["resolucao_alvo", "metodo"], as_index=False)["fold"]
        .nunique()
        .rename(columns={"fold": "n_folds_presentes"})
    )

    expected_folds = (
        comparison_raw[comparison_raw["origem"] == "ia"]
        .groupby("resolucao_alvo")["fold"]
        .nunique()
        .to_dict()
    )

    agg = agg.merge(fold_coverage, on=["resolucao_alvo", "metodo"], how="left")
    agg["n_folds_presentes"] = agg["n_folds_presentes"].fillna(np.nan)
    agg["folds_esperados"] = agg["resolucao_alvo"].map(expected_folds)
    agg["cobertura_incompleta"] = (agg["origem"] == "ia") & (
        agg["n_folds_presentes"] < agg["folds_esperados"]
    )

    return agg.sort_values(["resolucao_alvo", "media_dice"], ascending=[True, False])


def plot_comparison_bar_by_resolution(agg, resolution):
    """Render a bar chart for one resolution with readable method labels."""
    subset = agg[agg["resolucao_alvo"] == resolution].copy()
    if subset.empty:
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, f"Sem dados para {resolution}", ha="center", va="center")
        plt.axis("off")
        plt.show()
        return

    subset["tipo"] = subset["origem"].map({"ia": "IA", "matematico": "Matematico"})
    subset["metodo_label"] = subset["metodo"].apply(prettify_method_label)
    method_order = subset.sort_values("media_dice", ascending=False)[
        "metodo_label"
    ].tolist()
    subset["metodo_label"] = pd.Categorical(
        subset["metodo_label"], categories=method_order, ordered=True
    )
    subset = subset.sort_values("metodo_label")

    plt.figure(figsize=(12, 5))
    ax = sns.barplot(
        data=subset,
        x="metodo_label",
        y="media_dice",
        hue="tipo",
        palette={"IA": "#4C78A8", "Matematico": "#F58518"},
    )

    for idx, row in subset.reset_index(drop=True).iterrows():
        std_val = row["desvio_padrao"]
        if pd.notna(std_val):
            ax.errorbar(
                x=idx,
                y=row["media_dice"],
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
