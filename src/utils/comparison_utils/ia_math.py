from pathlib import Path

import numpy as np
import pandas as pd


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
        "source",
        "ia_resolution",
        "target_resolution",
        "fold",
        "method",
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
                df_ia["source"] = "ia"
                df_ia["ia_resolution"] = ia_resolution
                df_ia["target_resolution"] = map_ia_resolution_to_target(ia_resolution)
                df_ia["fold"] = fold_name
                df_ia["method"] = ia_resolution + "::" + method_name
                ia_frames.append(df_ia)

    if ia_frames:
        return pd.concat(ia_frames, ignore_index=True), missing_ia_files

    return pd.DataFrame(columns=columns), missing_ia_files


def load_math_results_for_comparison(math_paths):
    """Load mathematical pipeline summary CSVs for selected resolutions/splits."""
    math_frames = []
    missing_math_files = []

    columns = [
        "img_id",
        "dice",
        "source",
        "ia_resolution",
        "target_resolution",
        "fold",
        "method",
    ]

    for target_resolution, resolution_paths in math_paths.items():
        if isinstance(resolution_paths, dict):
            split_items = resolution_paths.items()
        else:
            split_items = [("test", resolution_paths)]

        for split_name, summary_path in split_items:
            summary_path = Path(summary_path)
            if not summary_path.exists():
                missing_math_files.append(
                    f"Arquivo nao encontrado: {summary_path} ({target_resolution}/{split_name})"
                )
                continue

            df_math = pd.read_csv(summary_path)
            if "IMG_ID" not in df_math.columns or "dice_artery" not in df_math.columns:
                missing_math_files.append(
                    f"Schema inesperado em {summary_path} ({target_resolution}/{split_name})"
                )
                continue

            df_math = df_math[["IMG_ID", "dice_artery"]].copy()
            df_math["dice"] = pd.to_numeric(df_math["dice_artery"], errors="coerce")
            df_math = df_math.dropna(subset=["dice"])
            if df_math.empty:
                continue

            df_math = df_math.rename(columns={"IMG_ID": "img_id"})
            df_math = df_math.drop(columns=["dice_artery"])
            df_math["source"] = "math"
            df_math["ia_resolution"] = "n/a"
            df_math["target_resolution"] = target_resolution
            df_math["fold"] = split_name
            df_math["method"] = "pipeline_matematico"
            math_frames.append(df_math)

    if math_frames:
        return pd.concat(math_frames, ignore_index=True), missing_math_files

    return pd.DataFrame(columns=columns), missing_math_files


def build_comparison_agg_df(comparison_raw):
    """Aggregate Dice metrics by resolution, source and method."""
    if comparison_raw.empty:
        return pd.DataFrame()

    required_cols_for_filter = {"source", "target_resolution", "img_id"}
    if required_cols_for_filter.issubset(comparison_raw.columns):
        ia_keys = (
            comparison_raw[comparison_raw["source"] == "ia"][
                ["target_resolution", "img_id"]
            ]
            .dropna(subset=["target_resolution", "img_id"])
            .drop_duplicates()
        )

        if not ia_keys.empty:
            math_mask = comparison_raw["source"] == "math"
            df_non_math = comparison_raw[~math_mask]
            df_math = comparison_raw[math_mask]

            df_math = df_math.merge(
                ia_keys,
                on=["target_resolution", "img_id"],
                how="inner",
            )

            comparison_raw = pd.concat([df_non_math, df_math], ignore_index=True)

    agg = comparison_raw.groupby(
        ["target_resolution", "source", "method"], as_index=False
    )["dice"].agg(
        n_samples="size",
        mean_dice="mean",
        std_dice="std",
        median_dice="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
    )

    fold_coverage = (
        comparison_raw[comparison_raw["source"] == "ia"]
        .groupby(["target_resolution", "method"], as_index=False)["fold"]
        .nunique()
        .rename(columns={"fold": "n_folds_present"})
    )

    expected_folds = (
        comparison_raw[comparison_raw["source"] == "ia"]
        .groupby("target_resolution")["fold"]
        .nunique()
        .to_dict()
    )

    agg = agg.merge(fold_coverage, on=["target_resolution", "method"], how="left")
    agg["n_folds_present"] = agg["n_folds_present"].fillna(np.nan)
    agg["expected_folds"] = agg["target_resolution"].map(expected_folds)
    agg["incomplete_coverage"] = (agg["source"] == "ia") & (
        agg["n_folds_present"] < agg["expected_folds"]
    )

    return agg.sort_values(["target_resolution", "mean_dice"], ascending=[True, False])
