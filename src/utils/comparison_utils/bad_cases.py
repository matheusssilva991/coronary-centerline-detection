import json
from pathlib import Path

import numpy as np
import pandas as pd


STATUS_MAP_TO_ENGLISH = {
    "ambos corretos": "both_correct",
    "ambos toleráveis": "both_tolerable",
    "ostios não encontrados": "ostia_not_found",
    "óstios não encontrados": "ostia_not_found",
    "um correto": "one_correct",
    "nenhum correto": "none_correct",
    "erro": "error",
    "not_found": "ostia_not_found",
    "both_correct": "both_correct",
    "both_tolerable": "both_tolerable",
    "found_but_wrong": "found_but_wrong",
    "not_evaluated": "not_evaluated",
}


def _status_to_english(value):
    """Normalize status values to the canonical English format."""
    # Normaliza status para as chaves canônicas do módulo.
    if pd.isna(value):
        return None

    # Remove variações de caixa e espaços antes do lookup.
    normalized = str(value).strip().lower()
    return STATUS_MAP_TO_ENGLISH.get(normalized, normalized)


def _compute_success_mask(df, success_status):
    """Compute success mask supporting multiple summary schemas."""
    # Produz máscara booleana de sucesso de óstio.
    if {"both_correct", "both_tolerable"}.issubset(df.columns):
        # Schema novo: sucesso = correto OU tolerável.
        return df["both_correct"].fillna(False).astype(bool) | df[
            "both_tolerable"
        ].fillna(False).astype(bool)

    if "ostia_status" in df.columns:
        # Schema intermediário: status consolidado por linha.
        return df["ostia_status"].isin(["both_correct", "both_tolerable"])

    # Schema antigo: converte coluna textual para padrão interno.
    status_series = df.get("status", pd.Series(index=df.index, dtype="object"))
    status_english = status_series.map(_status_to_english)
    success_status_english = {_status_to_english(status) for status in success_status}
    return status_english.isin(success_status_english)


def _compute_bad_case_status(df, bad_mask, success_mask, low_dice_mask):
    """Build a reason label for each bad case."""
    # Rotula o motivo de cada caso ruim.
    status_series = df.get("status", pd.Series(index=df.index, dtype="object"))
    ostia_status_series = df.get(
        "ostia_status", pd.Series(index=df.index, dtype="object")
    )

    # Série final alinhada ao índice original.
    bad_case_status = pd.Series(index=df.index, dtype="object")

    # Falhas de óstio entram primeiro.
    failed_status_mask = bad_mask & (~success_mask)
    bad_case_status.loc[failed_status_mask] = status_series.loc[failed_status_mask].map(
        _status_to_english
    )

    # Fallback para coluna alternativa de status.
    missing_status_mask = failed_status_mask & bad_case_status.isna()
    bad_case_status.loc[missing_status_mask] = ostia_status_series.loc[
        missing_status_mask
    ].map(_status_to_english)

    # Óstio correto + Dice baixo vira low_dice.
    low_dice_only_mask = bad_mask & success_mask & low_dice_mask
    bad_case_status.loc[low_dice_only_mask] = "low_dice"

    # Qualquer restante sem rótulo vira unknown.
    bad_case_status.loc[bad_mask & bad_case_status.isna()] = "unknown"
    return bad_case_status


def get_bad_cases(df, success_status=None, dice_threshold=0.30):
    """Return bad cases by status or Dice threshold with `bad_case_status`."""
    # Seleciona casos ruins por falha de óstio ou Dice baixo.
    if success_status is None:
        success_status = [
            "ambos toleráveis",
            "ambos corretos",
            "both_tolerable",
            "both_correct",
        ]

    if df is None or df.empty:
        # Retorno vazio mantendo o mesmo contrato.
        return pd.DataFrame(columns=df.columns if df is not None else None)

    # Máscara de sucesso com suporte aos schemas conhecidos.
    success_mask = _compute_success_mask(df, success_status)
    # Converte Dice para número antes do threshold.
    dice_scores = pd.to_numeric(df["dice_artery"], errors="coerce")
    low_dice_mask = dice_scores < dice_threshold
    bad_mask = (~success_mask) | low_dice_mask

    # Recorte final das linhas classificadas como ruins.
    bad_df = df.loc[bad_mask].copy()
    bad_status = _compute_bad_case_status(
        df,
        bad_mask,
        success_mask,
        low_dice_mask,
    )
    # Anexa a justificativa de cada linha selecionada.
    bad_df["bad_case_status"] = bad_status.loc[bad_mask].values
    return bad_df


def filter_correct_ostia_cases(df, success_status=None):
    """Return only cases where ostia detection is considered successful."""
    # Mantém somente linhas com sucesso de óstio.
    if success_status is None:
        success_status = [
            "ambos toleráveis",
            "ambos corretos",
            "both_tolerable",
            "both_correct",
        ]

    if df is None or df.empty:
        # Retorno vazio com mesmo schema.
        return pd.DataFrame(columns=df.columns if df is not None else None)

    # Reaproveita a mesma regra de sucesso deste módulo.
    success_mask = _compute_success_mask(df, success_status)
    return df.loc[success_mask].copy()


def build_bad_cases_export_df(df_bad_cases, subset_name, resolution):
    """Create a standardized bad-cases export DataFrame with English keys."""
    # Monta tabela padrão para exportação.
    if df_bad_cases is None or df_bad_cases.empty:
        # Sem linhas: devolve apenas cabeçalho padrão.
        return pd.DataFrame(
            columns=["image_id", "bad_case_status", "subset", "resolution"]
        )

    # Aceita IMG_ID e image_id para compatibilidade.
    image_id_col = "IMG_ID" if "IMG_ID" in df_bad_cases.columns else "image_id"
    if image_id_col not in df_bad_cases.columns:
        raise KeyError("Bad cases dataframe must contain 'IMG_ID' or 'image_id'.")

    # Copia somente campos de exportação.
    export_df = pd.DataFrame(
        {
            "image_id": df_bad_cases[image_id_col],
            "bad_case_status": df_bad_cases.get(
                "bad_case_status", pd.Series(dtype="object")
            ),
            "subset": subset_name,
            "resolution": resolution,
        }
    )
    return export_df


def save_bad_cases_artifacts(df_bad_cases, output_dir, subset_name, resolution):
    """Save bad cases to CSV and JSON, separated by subset and resolution."""

    # Garante pasta de saída.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Normaliza schema antes de gravar arquivos.
    export_df = build_bad_cases_export_df(df_bad_cases, subset_name, resolution)
    stem = f"bad_cases_{subset_name}_{resolution}"

    # Caminhos de saída por formato.
    csv_path = output_path / f"{stem}.csv"
    json_path = output_path / f"{stem}.json"

    # Grava CSV e JSON do mesmo conteúdo.
    export_df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as file_handle:
        json.dump(
            export_df.to_dict(orient="records"),
            file_handle,
            indent=2,
            ensure_ascii=False,
        )

    # Retorna metadados úteis para logs/notebook.
    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "num_bad_cases": int(export_df.shape[0]),
    }


def prepare_bad_cases_for_subset(
    split_paths_by_resolution,
    split_name,
    output_dir,
    valid_splits=("train", "val", "test"),
):
    """Load, filter and export bad cases for a given subset."""
    # Pipeline completo: carregar, filtrar e exportar.
    if split_name not in valid_splits:
        raise ValueError(f"split_name must be one of {valid_splits}")

    # Import tardio evita ciclo de import.
    from .io import load_split_summary

    # Carrega summaries Mid/High desse subset.
    df_mid = load_split_summary(split_paths_by_resolution, "mid_res", split_name)
    df_high = load_split_summary(split_paths_by_resolution, "high_res", split_name)

    # Filtra bad cases para cada resolução.
    df_mid_bad = get_bad_cases(df_mid) if df_mid is not None else pd.DataFrame()
    df_high_bad = get_bad_cases(df_high) if df_high is not None else pd.DataFrame()

    # Exporta resultado Mid.
    mid_export = save_bad_cases_artifacts(
        df_bad_cases=df_mid_bad,
        output_dir=output_dir,
        subset_name=split_name,
        resolution="mid_res",
    )

    high_export = None
    if df_high is not None:
        # Exporta resultado High quando disponível.
        high_export = save_bad_cases_artifacts(
            df_bad_cases=df_high_bad,
            output_dir=output_dir,
            subset_name=split_name,
            resolution="high_res",
        )

    # Retorna DataFrames e caminhos para uso no notebook.
    return {
        "df_mid": df_mid,
        "df_high": df_high,
        "df_mid_bad": df_mid_bad,
        "df_high_bad": df_high_bad,
        "mid_export": mid_export,
        "high_export": high_export,
        "output_dir": output_dir,
    }


def summarize_bad_dice_with_threshold(df_bad, dice_threshold=0.3):
    """Summarize bad-case Dice with and without low-dice successful ostia cases."""
    # Resume Dice com e sem casos low_dice de óstio correto.
    if df_bad is None or df_bad.empty or "dice_artery" not in df_bad.columns:
        # Retorno padrão para ausência de dados.
        return {
            "mean_with_low_dice": np.nan,
            "mean_without_low_dice": np.nan,
            "n_with_low_dice": 0,
            "n_without_low_dice": 0,
            "n_low_dice_correct": 0,
        }

    # Prepara vetor numérico de Dice válido.
    dice = pd.to_numeric(df_bad["dice_artery"], errors="coerce")
    valid_dice = dice.notna()
    # Identifica linhas com sucesso de óstio.
    success_mask = _compute_success_mask(
        df_bad,
        [
            "ambos toleráveis",
            "ambos corretos",
            "both_tolerable",
            "both_correct",
        ],
    )
    # Seleciona casos corretos abaixo do limiar de Dice.
    low_dice_correct_mask = valid_dice & success_mask & (dice < dice_threshold)

    # Série completa de Dice válido.
    dice_with_low = dice[valid_dice]
    # Série sem low_dice correto.
    dice_without_low = dice[valid_dice & ~low_dice_correct_mask]

    # Retorna médias e contagens para tabela/gráfico.
    return {
        "mean_with_low_dice": dice_with_low.mean()
        if not dice_with_low.empty
        else np.nan,
        "mean_without_low_dice": (
            dice_without_low.mean() if not dice_without_low.empty else np.nan
        ),
        "n_with_low_dice": int(dice_with_low.shape[0]),
        "n_without_low_dice": int(dice_without_low.shape[0]),
        "n_low_dice_correct": int(low_dice_correct_mask.sum()),
    }
