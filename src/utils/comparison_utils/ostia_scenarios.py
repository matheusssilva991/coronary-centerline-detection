from pathlib import Path

import pandas as pd

from ..project.results import add_internal_result_aliases
from .bad_cases import filter_correct_ostia_cases
from .ia_math import filter_to_common_ia_math_ids, get_common_ia_math_keys


def load_math_results_for_ostia_scenario(math_paths, scenario):
    """Load mathematical comparison results for a specific ostia scenario."""
    math_frames = []
    missing_math_files = []

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

            df_math = add_internal_result_aliases(pd.read_csv(summary_path))
            if scenario == "correct":
                df_math = filter_correct_ostia_cases(df_math)
            elif scenario == "incorrect":
                df_correct = filter_correct_ostia_cases(df_math)
                df_math = df_math.loc[~df_math.index.isin(df_correct.index)].copy()
            elif scenario != "full":
                raise ValueError(f"Cenario desconhecido: {scenario}")

            if df_math.empty:
                continue

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

    columns = [
        "img_id",
        "dice",
        "source",
        "ia_resolution",
        "target_resolution",
        "fold",
        "method",
    ]
    return pd.DataFrame(columns=columns), missing_math_files


def filter_ia_results_for_math_ids(ia_results_df, math_results_df):
    """Keep only IA rows whose image IDs are present in the mathematical subset."""
    if ia_results_df.empty:
        return ia_results_df.copy()

    if math_results_df.empty:
        return ia_results_df.iloc[0:0].copy()

    common_keys = get_common_ia_math_keys(
        pd.concat([ia_results_df, math_results_df], ignore_index=True)
    )
    if common_keys.empty:
        return ia_results_df.iloc[0:0].copy()

    return ia_results_df.merge(
        common_keys, on=["target_resolution", "img_id"], how="inner"
    )


def load_ostia_comparison_scenario(ia_results_df, math_paths, scenario):
    """Build the IA and mathematical subsets for a comparison scenario."""
    scenario_math_df, missing_math_files = load_math_results_for_ostia_scenario(
        math_paths,
        scenario,
    )

    scenario_raw_df = pd.concat([ia_results_df, scenario_math_df], ignore_index=True)
    scenario_common_df = filter_to_common_ia_math_ids(scenario_raw_df)
    scenario_ia_df = scenario_common_df.loc[
        scenario_common_df["source"] == "ia"
    ].copy()
    scenario_math_df = scenario_common_df.loc[
        scenario_common_df["source"] == "math"
    ].copy()

    return scenario_ia_df, scenario_math_df, missing_math_files


def build_ostia_image_comparison_df(ia_results_df, math_results_df):
    """Pair every IA method per image with the mathematical Dice.

    Keeping one row per IA method avoids selecting the best model separately
    for each examination, which would make the visual comparison optimistic.
    """
    columns = [
        "target_resolution",
        "img_id",
        "ia_dice",
        "ia_method",
        "math_dice",
        "math_method",
    ]

    if ia_results_df.empty or math_results_df.empty:
        return pd.DataFrame(columns=columns)

    required_cols = {"target_resolution", "img_id", "dice", "method"}
    if not required_cols.issubset(ia_results_df.columns):
        return pd.DataFrame(columns=columns)

    if not required_cols.issubset(math_results_df.columns):
        return pd.DataFrame(columns=columns)

    ia_subset = ia_results_df.dropna(
        subset=["target_resolution", "img_id", "dice", "method"]
    ).copy()
    ia_subset["dice"] = pd.to_numeric(ia_subset["dice"], errors="coerce")
    ia_subset = ia_subset.dropna(subset=["dice"])
    ia_subset = ia_subset.sort_values(
        ["target_resolution", "img_id", "method", "dice"],
        ascending=[True, True, True, False],
    )
    ia_by_method_df = (
        ia_subset.groupby(
            ["target_resolution", "img_id", "method"], as_index=False
        )
        .first()
        .rename(columns={"dice": "ia_dice", "method": "ia_method"})
    )

    math_subset = math_results_df.dropna(
        subset=["target_resolution", "img_id", "dice", "method"]
    ).copy()
    math_subset["dice"] = pd.to_numeric(math_subset["dice"], errors="coerce")
    math_subset = math_subset.dropna(subset=["dice"])
    math_subset = math_subset.sort_values(
        ["target_resolution", "img_id", "dice"],
        ascending=[True, True, False],
    )
    math_df = (
        math_subset.groupby(["target_resolution", "img_id"], as_index=False)
        .first()
        .rename(columns={"dice": "math_dice", "method": "math_method"})
    )

    return (
        ia_by_method_df[["target_resolution", "img_id", "ia_dice", "ia_method"]]
        .merge(
            math_df[["target_resolution", "img_id", "math_dice", "math_method"]],
            on=["target_resolution", "img_id"],
            how="inner",
        )
        .sort_values(["target_resolution", "ia_method", "img_id"])
    )
