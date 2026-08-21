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
    # Remove casos de sucesso para evitar sobreposição com low_dice.
    failed_status_mask = bad_mask & (~success_mask)

    # Tenta mapear status textual para inglês, quando disponível.
    bad_case_status.loc[failed_status_mask] = status_series.loc[failed_status_mask].map(
        _status_to_english
    )

    # Se o status original for ausente, tenta usar ostia_status.
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


def _classify_intersection_group(row):
    """Classify a bad case by ostia availability and artery intersection."""
    ostia_status = str(row.get("ostia_status", "")).strip().lower()
    if ostia_status == "not_found" or row.get("ostia_found") is False:
        return "ostia_not_found"
    if bool(row.get("left_intersects", False)) or bool(
        row.get("right_intersects", False)
    ):
        return "with_intersection"
    return "without_intersection"


def _sample_image_ids(ids, sample_size, rng):
    """Sample unique image IDs deterministically with the supplied generator."""
    unique_ids = sorted(set(int(image_id) for image_id in ids))
    if len(unique_ids) <= sample_size:
        return unique_ids
    return sorted(
        rng.choice(unique_ids, size=sample_size, replace=False).astype(int).tolist()
    )


def prepare_bad_case_qualitative_comparison(
    split_paths_by_resolution,
    bad_cases_export_dir,
    *,
    split_name="test",
    resolutions=("high", "mid"),
    samples_per_group=2,
    random_seed=42,
):
    """Select reproducible bad cases for qualitative resolution comparison.

    The selection prioritizes images with the same failure category in both
    resolutions, then fills missing slots with failures available in only one
    resolution or with a different category in the other resolution.
    """
    from .io import load_split_summary

    available_resolutions = tuple(
        resolution
        for resolution in resolutions
        if split_name
        in split_paths_by_resolution.get(f"{resolution}_res", {})
    )
    if not available_resolutions:
        raise FileNotFoundError(
            f"No consolidated results found for split '{split_name}'."
        )
    if samples_per_group <= 0:
        raise ValueError("samples_per_group must be greater than zero.")

    summaries_by_resolution = {}
    for resolution in available_resolutions:
        summary = load_split_summary(
            split_paths_by_resolution,
            f"{resolution}_res",
            split_name,
        ).copy()
        summary["image_id"] = summary["IMG_ID"].astype(int)
        summary["resolution"] = resolution
        summaries_by_resolution[resolution] = summary

    bad_cases_by_resolution = {}
    detail_columns = [
        "image_id",
        "status",
        "ostia_status",
        "ostia_found",
        "left_intersects",
        "right_intersects",
        "both_correct",
        "both_tolerable",
        "dice_artery",
    ]
    export_dir = Path(bad_cases_export_dir)
    for resolution, summary in summaries_by_resolution.items():
        export_path = export_dir / f"bad_cases_{split_name}_{resolution}_res.csv"
        bad_cases = (
            pd.read_csv(export_path).copy()
            if export_path.is_file()
            else get_bad_cases(summary).copy()
        )
        if "image_id" not in bad_cases.columns and "IMG_ID" in bad_cases.columns:
            bad_cases = bad_cases.rename(columns={"IMG_ID": "image_id"})
        bad_cases["image_id"] = bad_cases["image_id"].astype(int)
        bad_cases["resolution"] = resolution

        available_details = [
            column for column in detail_columns if column in summary.columns
        ]
        bad_cases = bad_cases.drop(
            columns=[
                column
                for column in available_details
                if column != "image_id"
            ],
            errors="ignore",
        ).merge(summary[available_details], on="image_id", how="left")
        bad_cases["intersection_group"] = bad_cases.apply(
            _classify_intersection_group,
            axis=1,
        )
        bad_cases_by_resolution[resolution] = bad_cases

    all_bad_cases = pd.concat(bad_cases_by_resolution.values(), ignore_index=True)
    bad_case_matrix = all_bad_cases.pivot_table(
        index="image_id",
        columns="resolution",
        values="bad_case_status",
        aggfunc="first",
    )

    rng = np.random.default_rng(random_seed)
    selected_records = []
    intersection_groups = (
        "with_intersection",
        "without_intersection",
        "ostia_not_found",
    )
    for error_type in sorted(all_bad_cases["bad_case_status"].dropna().unique()):
        error_rows = all_bad_cases[all_bad_cases["bad_case_status"].eq(error_type)]
        for intersection_group in intersection_groups:
            target_rows = error_rows[
                error_rows["intersection_group"].eq(intersection_group)
            ]
            target_ids = sorted(target_rows["image_id"].unique())
            if not target_ids:
                continue

            compares_mid_high = {"high", "mid"}.issubset(available_resolutions)
            same_error_ids = (
                [
                    image_id
                    for image_id in target_ids
                    if image_id in bad_case_matrix.index
                    and bad_case_matrix.loc[image_id].get("high") == error_type
                    and bad_case_matrix.loc[image_id].get("mid") == error_type
                ]
                if compares_mid_high
                else []
            )
            remaining_ids = [
                image_id
                for image_id in target_ids
                if image_id not in same_error_ids
            ]
            chosen_ids = _sample_image_ids(
                same_error_ids,
                samples_per_group,
                rng,
            )
            if len(chosen_ids) < samples_per_group:
                chosen_ids.extend(
                    _sample_image_ids(
                        remaining_ids,
                        samples_per_group - len(chosen_ids),
                        rng,
                    )
                )

            selected_records.extend(
                {
                    "image_id": int(image_id),
                    "target_bad_case_status": error_type,
                    "target_intersection_group": intersection_group,
                    "comparison_priority": (
                        "same_error_mid_high"
                        if image_id in same_error_ids
                        else "one_resolution_or_different_error"
                    ),
                }
                for image_id in chosen_ids
            )

    plan_columns = [
        "image_id",
        "target_bad_case_status",
        "target_intersection_group",
        "comparison_priority",
    ]
    selected_plan = pd.DataFrame(selected_records, columns=plan_columns)
    selected_plan = selected_plan.drop_duplicates(plan_columns[:3])
    selected_image_ids = list(dict.fromkeys(selected_plan["image_id"].tolist()))
    selection_reasons = (
        selected_plan.assign(
            selection_reason=lambda frame: frame["target_bad_case_status"]
            + "/"
            + frame["target_intersection_group"]
        )
        .groupby("image_id")["selection_reason"]
        .apply(lambda values: "; ".join(values))
        .to_dict()
    )

    selected_rows = []
    for image_id in selected_image_ids:
        for resolution in available_resolutions:
            bad_match = all_bad_cases[
                all_bad_cases["image_id"].eq(image_id)
                & all_bad_cases["resolution"].eq(resolution)
            ]
            if not bad_match.empty:
                row = bad_match.iloc[0].to_dict()
            else:
                summary_match = summaries_by_resolution[resolution][
                    summaries_by_resolution[resolution]["image_id"].eq(image_id)
                ]
                if summary_match.empty:
                    continue
                row = summary_match.iloc[0].to_dict()
                row["bad_case_status"] = "not_bad"
                row["subset"] = split_name
                row["intersection_group"] = _classify_intersection_group(row)
            row["selection_reason"] = selection_reasons.get(image_id, "")
            selected_rows.append(row)

    selected_cases = pd.DataFrame(selected_rows)
    if not selected_cases.empty:
        selected_cases = selected_cases.drop_duplicates(["image_id", "resolution"])

    return {
        "resolutions": available_resolutions,
        "summaries_by_resolution": summaries_by_resolution,
        "bad_cases_by_resolution": bad_cases_by_resolution,
        "all_bad_cases": all_bad_cases,
        "selected_image_plan": selected_plan,
        "selected_image_ids": selected_image_ids,
        "selected_cases": selected_cases,
    }
