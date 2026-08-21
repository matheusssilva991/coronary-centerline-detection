import numpy as np
import pandas as pd


def get_total_success_percent(metadata, default=np.nan):
    """Read total success percent with backward-compatible fallback."""
    # Lê a metrica principal de sucesso dos metadados.
    results_summary = metadata.get("results_summary", {})
    success_total_percent = results_summary.get("total_success_percent", default)
    if pd.isna(success_total_percent):
        # Fallback para schema antigo: correto + toleravel.
        both_correct = results_summary.get("both_correct_percent", 0)
        both_tolerable = results_summary.get("both_tolerable_percent", 0)
        success_total_percent = both_correct + both_tolerable
    return success_total_percent


def get_execution_time_seconds(metadata, default=np.nan):
    """Read execution time from metadata."""
    # Extrai tempo total de execucao (segundos).
    execution_info = metadata.get("execution_info", {})
    return execution_info.get("execution_time_seconds", default)


def get_num_images(metadata, default=np.nan):
    """Read number of images from metadata."""
    # Extrai quantidade de imagens processadas.
    execution_info = metadata.get("execution_info", {})
    return execution_info.get("num_images", default)


def build_split_resolution_summary(
    split_paths_by_resolution,
    valid_splits=("train", "val", "test"),
):
    """Build the summary table consumed by split/resolution EDA plots.

    Missing resolution/split pairs are retained with ``is_available=False`` so
    the notebook can report incomplete result collections without special-case
    loading logic.
    """
    from .bad_cases import filter_correct_ostia_cases
    from .io import load_split_metadata, load_split_summary

    rows = []
    for resolution in split_paths_by_resolution:
        for subset_name in valid_splits:
            metadata = load_split_metadata(
                split_paths_by_resolution,
                resolution,
                subset_name,
            )
            summary_df = load_split_summary(
                split_paths_by_resolution,
                resolution,
                subset_name,
            )

            if metadata is None or summary_df is None:
                rows.append(
                    {
                        "subset": subset_name,
                        "resolution": resolution,
                        "num_images": np.nan,
                        "mean_dice": np.nan,
                        "mean_dice_correct": np.nan,
                        "mean_dice_all": np.nan,
                        "execution_time_min": np.nan,
                        "total_success_percent": np.nan,
                        "total_ostia_success": np.nan,
                        "is_available": False,
                    }
                )
                continue

            # Resume Dice para todos os casos e para óstios aceitos.
            dice_all = pd.to_numeric(summary_df["dice_artery"], errors="coerce")
            dice_all = dice_all.dropna()
            correct_cases = filter_correct_ostia_cases(summary_df)
            dice_correct = pd.to_numeric(
                correct_cases["dice_artery"], errors="coerce"
            ).dropna()

            execution_time_seconds = get_execution_time_seconds(metadata)
            num_images = get_num_images(metadata)
            total_success_percent = get_total_success_percent(metadata)
            if pd.notna(num_images) and pd.notna(total_success_percent):
                total_ostia_success = (num_images * 2) * (
                    total_success_percent / 100
                )
            else:
                total_ostia_success = np.nan

            rows.append(
                {
                    "subset": subset_name,
                    "resolution": resolution,
                    "num_images": num_images,
                    "mean_dice": (
                        dice_correct.mean() if not dice_correct.empty else np.nan
                    ),
                    "mean_dice_correct": (
                        dice_correct.mean() if not dice_correct.empty else np.nan
                    ),
                    "mean_dice_all": (
                        dice_all.mean() if not dice_all.empty else np.nan
                    ),
                    "execution_time_min": (
                        execution_time_seconds / 60
                        if pd.notna(execution_time_seconds)
                        else np.nan
                    ),
                    "total_success_percent": total_success_percent,
                    "total_ostia_success": total_ostia_success,
                    "is_available": True,
                }
            )

    summary = pd.DataFrame(rows)
    summary["subset"] = pd.Categorical(
        summary["subset"],
        categories=list(valid_splits),
        ordered=True,
    )
    summary = summary.sort_values(["subset", "resolution"]).reset_index(drop=True)

    # Mantém os aliases usados pelos helpers de visualização existentes.
    aliases = {
        "resolucao": "resolution",
        "num_imagens": "num_images",
        "dice_medio": "mean_dice",
        "dice_medio_correto": "mean_dice_correct",
        "dice_medio_todos": "mean_dice_all",
        "tempo_execucao_min": "execution_time_min",
        "sucesso_total_percent": "total_success_percent",
        "ostios_sucesso_total": "total_ostia_success",
        "disponivel": "is_available",
    }
    for alias, source in aliases.items():
        summary[alias] = summary[source]
    return summary
