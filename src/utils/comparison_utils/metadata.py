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
