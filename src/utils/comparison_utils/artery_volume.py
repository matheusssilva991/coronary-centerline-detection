"""Compara coortes canônicas antes e depois de excluir volumes atípicos."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from utils.comparison_utils.bad_cases import filter_correct_ostia_cases
from utils.project.dataframe import require_series_column, to_numeric_series
from utils.project.results.schema import (
    add_internal_result_aliases,
    normalize_result_status,
)


def _prepare_result_dice(original: pd.DataFrame, group: str) -> pd.DataFrame:
    """Normaliza Dice e sucesso dos óstios sem ocultar falhas documentadas.

    Args:
        original (pd.DataFrame): Tabela bruta de resultados por exame de um grupo.
        group (str): Identificador ``resolução/split`` usado em mensagens de erro.

    Returns:
        pd.DataFrame: Tabela com ``IMG_ID``, ``dice`` e ``ostia_success``.
    """
    # Padroniza aliases internos e valida unicidade dos identificadores de exame.
    frame = add_internal_result_aliases(original.copy())
    ids = require_series_column(frame, "IMG_ID")
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError(f"IDs repetidos ou ausentes em {group}.")
    # Converte Dice e atribui zero apenas a falhas documentadas sem métrica numérica.
    dice = to_numeric_series(
        require_series_column(frame, "dice_artery"), errors="raise"
    )
    status = require_series_column(frame, "status").map(normalize_result_status)
    failed = status.isin(["error", "not_found"])
    if "pipeline_error" in frame:
        failed |= frame["pipeline_error"].notna() & frame["pipeline_error"].astype(
            str
        ).str.strip().ne("")
    dice = dice.mask(dice.isna() & failed, 0.0)
    if dice.isna().any() or not dice.between(0, 1).all():
        raise ValueError(f"Dice inválido ou ausente sem falha em {group}.")
    # Identifica exames com óstios bem-sucedidos para compor a tabela normalizada.
    successful_ids = filter_correct_ostia_cases(frame)["IMG_ID"]
    return pd.DataFrame(
        {
            "IMG_ID": ids,
            "dice": dice,
            "ostia_success": ids.isin(successful_ids),
        }
    )


def prepare_artery_volume_dice_data(
    results_by_group: Mapping[tuple[str, str], pd.DataFrame],
    classified_volumes: pd.DataFrame,
    *,
    split: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retorna dados por resolução, pares Mid/High e cobertura para os gráficos.

    Ausências de volume ou resolução são reportadas, não imputadas. O delta
    pareado é High menos Mid; zero é imputado apenas em falhas documentadas.

    Args:
        results_by_group (Mapping[tuple[str, str], pd.DataFrame]): Mapa de
            ``(resolução, split)`` para DataFrames de resultados por exame.
        classified_volumes (pd.DataFrame): Tabela com volumes arteriais de
            referência e classificação de outliers por ``IMG_ID``.
        split (str): Subconjunto avaliado (``"train"``, ``"val"``, ``"test"``
            ou ``"all"``).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Dados empilhados por
        resolução, pares Mid/High por ``IMG_ID`` e resumo de cobertura.
    """
    # Valida o split solicitado e os volumes arteriais de referência por exame.
    if split not in {"train", "val", "test", "all"}:
        raise ValueError("split deve ser train, val, test ou all.")
    selected_splits = ("train", "val", "test") if split == "all" else (split,)
    columns = ["IMG_ID", "volume_ml", "volume_outlier_group"]
    volumes = classified_volumes.loc[:, columns].copy()
    ids = require_series_column(volumes, "IMG_ID")
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("IDs repetidos ou ausentes na classificação volumétrica.")
    values = to_numeric_series(volumes["volume_ml"], errors="raise")
    if not np.isfinite(values).all() or values.lt(0).any():
        raise ValueError("Volumes devem ser finitos e não negativos.")
    volumes["volume_ml"] = values
    if not volumes["volume_outlier_group"].isin(["low", "in_range", "high"]).all():
        raise ValueError("Classificação volumétrica inválida.")
    volumes["is_volume_outlier"] = volumes["volume_outlier_group"].ne("in_range")
    # Consolida os resultados normalizados de cada split por resolução sem duplicar IDs.
    frames = {}
    for resolution in ("mid_res", "high_res"):
        parts = [
            _prepare_result_dice(
                results_by_group[resolution, subset], f"{resolution}/{subset}"
            )
            for subset in selected_splits
            if (resolution, subset) in results_by_group
        ]
        frames[resolution] = (
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame(columns=["IMG_ID", "dice", "ostia_success"])
        )
        if frames[resolution]["IMG_ID"].duplicated().any():
            raise ValueError(f"Há IDs compartilhados entre splits de {resolution}.")
    # Cruza cada resolução com os volumes de referência e contabiliza ausências.
    rows, plotted = [], []
    for resolution, frame in frames.items():
        other = frames["high_res" if resolution == "mid_res" else "mid_res"]
        missing_volume = frame.loc[~frame["IMG_ID"].isin(ids), "IMG_ID"]
        missing_pair = frame.loc[~frame["IMG_ID"].isin(other["IMG_ID"]), "IMG_ID"]
        joined = frame.merge(volumes, on="IMG_ID", how="inner", validate="one_to_one")
        joined["resolution"] = resolution
        plotted.append(joined)
        rows.append(
            {
                "resolution": resolution,
                "split": split,
                "source_available": all(
                    (resolution, subset) in results_by_group
                    for subset in selected_splits
                ),
                "missing_splits": [
                    subset
                    for subset in selected_splits
                    if (resolution, subset) not in results_by_group
                ],
                "total_exams": len(frame),
                "plotted_exams": len(joined),
                "missing_volume_count": len(missing_volume),
                "missing_volume_ids": missing_volume.tolist(),
                "missing_pair_count": len(missing_pair),
                "missing_pair_ids": missing_pair.tolist(),
            }
        )
    # O pareamento é feito por ID, nunca pela posição das linhas nos CSVs.
    paired = (
        frames["mid_res"]
        .merge(
            frames["high_res"],
            on="IMG_ID",
            suffixes=("_mid", "_high"),
            validate="one_to_one",
        )
        .merge(volumes, on="IMG_ID", validate="one_to_one")
    )
    paired["delta_dice"] = paired["dice_high"] - paired["dice_mid"]
    paired["ostia_success_both"] = paired["ostia_success_mid"].astype(bool) & paired[
        "ostia_success_high"
    ].astype(bool)
    return pd.concat(plotted, ignore_index=True), paired, pd.DataFrame(rows)


def summarize_artery_volume_bands(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrupa resultados em faixas comuns calculadas com um volume por exame.

    Quantis repetidos são colapsados; volumes constantes formam uma faixa.
    O desvio-padrão é amostral e permanece ausente para grupos unitários.

    Args:
        data (pd.DataFrame): Tabela com ``resolution``, ``IMG_ID``, ``volume_ml``,
            ``dice`` e ``ostia_success``.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Estatísticas por resolução e faixa
        volumétrica e tabela com limites inferior e superior em mL.
    """
    # Valida consistência dos volumes e unicidade por resolução antes de criar faixas.
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    if data.duplicated(["resolution", "IMG_ID"]).any():
        raise ValueError("IDs repetidos por resolução.")
    if data.groupby("IMG_ID")["volume_ml"].nunique().gt(1).any():
        raise ValueError("Volumes de referência diferentes para o mesmo exame.")
    values = to_numeric_series(data["volume_ml"], errors="raise")
    if not np.isfinite(values).all() or values.lt(0).any():
        raise ValueError("Volumes devem ser finitos e não negativos.")
    # Define bordas quartílicas usando um único volume por exame para evitar viés.
    grouped = data.assign(volume_ml=values)
    unique_volumes = grouped.drop_duplicates("IMG_ID")["volume_ml"]
    edges = np.unique(np.quantile(unique_volumes, [0, 0.25, 0.5, 0.75, 1]))
    bounds = pd.DataFrame(
        {
            "band": np.arange(1, max(len(edges) - 1, 1) + 1),
            "lower_ml": edges[:-1] if len(edges) > 1 else edges,
            "upper_ml": edges[1:] if len(edges) > 1 else edges,
        }
    )
    grouped["band"] = (
        pd.cut(values, bins=edges, labels=False, include_lowest=True) + 1
        if len(edges) > 1
        else 1
    )
    # Agrega métricas de Dice e sucesso dos óstios em cada faixa volumétrica.
    summary = (
        grouped.groupby(["resolution", "band"], observed=True)
        .agg(
            exam_count=("IMG_ID", "size"),
            dice_mean=("dice", "mean"),
            dice_std=("dice", "std"),
            dice_median=("dice", "median"),
            ostia_success_count=("ostia_success", "sum"),
        )
        .reset_index()
    )
    summary["ostia_success_percent"] = (
        100 * summary["ostia_success_count"] / summary["exam_count"]
    )
    return summary.merge(bounds, on="band", validate="many_to_one"), bounds


def prepare_predicted_artery_volumes(
    results_by_group: Mapping[tuple[str, str], pd.DataFrame],
    classified_volumes: pd.DataFrame,
    configs_by_group: Mapping[tuple[str, str], Mapping[str, object]],
    *,
    split: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula mL pós-morfologia pela geometria registrada, sem imputar volumes.

    Retorna as linhas com volume de referência e um relatório de cobertura.
    Dados ausentes/inválidos recebem status explícito, não predição zero.

    Args:
        results_by_group (Mapping[tuple[str, str], pd.DataFrame]): Mapa de
            ``(resolução, split)`` para DataFrames de resultados por exame.
        classified_volumes (pd.DataFrame): Tabela com volumes de referência e
            espaçamentos físicos em mm por ``IMG_ID``.
        configs_by_group (Mapping[tuple[str, str], Mapping[str, object]]): Mapas
            de configuração por grupo contendo ``DOWNSCALE_FACTORS``.
        split (str): Subconjunto avaliado (``"train"``, ``"val"``, ``"test"``
            ou ``"all"``).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tabela por exame com volume predito e
        status explícito, além do relatório de cobertura por resolução.
    """
    if split == "all":
        # Valida IDs antes de reunir volumes calculados com configurações distintas.
        prepare_artery_volume_dice_data(
            results_by_group, classified_volumes, split="all"
        )
        parts = [
            prepare_predicted_artery_volumes(
                results_by_group,
                classified_volumes,
                configs_by_group,
                split=subset,
            )
            for subset in ("train", "val", "test")
        ]
        combined = pd.concat([data for data, _ in parts], ignore_index=True)
        coverage = pd.concat([report for _, report in parts], ignore_index=True)
        # Consolida os relatórios de cobertura dos três splits por resolução.
        rows = []
        for resolution in ("mid_res", "high_res"):
            reports = coverage.loc[coverage["resolution"].eq(resolution)]
            selected = combined.loc[combined["resolution"].eq(resolution)]
            rows.append(
                {
                    "resolution": resolution,
                    "total_exams": int(reports["total_exams"].sum()),
                    "missing_reference_count": int(
                        reports["missing_reference_count"].sum()
                    ),
                    "valid_predictions": int(reports["valid_predictions"].sum()),
                    "excluded_ids": selected.loc[
                        selected["prediction_volume_status"].ne("ok"), "IMG_ID"
                    ].tolist(),
                    "status_counts": selected["prediction_volume_status"]
                    .value_counts()
                    .to_dict(),
                }
            )
        return combined, pd.DataFrame(rows)
    # Reúne dados base por resolução e acopla os espaçamentos físicos em mm.
    data, _, coverage = prepare_artery_volume_dice_data(
        results_by_group,
        classified_volumes,
        split=split,
    )
    spacing_columns = ["spacing_x_mm", "spacing_y_mm", "spacing_z_mm"]
    geometry = classified_volumes.reindex(columns=["IMG_ID", *spacing_columns])
    data = data.merge(geometry, on="IMG_ID", how="left", validate="many_to_one")
    data["predicted_volume_ml"] = np.nan
    data["prediction_volume_status"] = "missing_count"
    rows = []
    for resolution in ("mid_res", "high_res"):
        selected = data["resolution"].eq(resolution)
        frame = data.loc[selected]
        # Valida fatores de redução espacial registrados na configuração do grupo.
        config = configs_by_group.get((resolution, split), {})
        try:
            factors = np.asarray(config.get("DOWNSCALE_FACTORS"), dtype=float)
            valid_factors = factors.shape == (3,) and bool(
                np.isfinite(factors).all() and (factors > 0).all()
            )
        except (TypeError, ValueError):
            factors, valid_factors = np.array([]), False
        original = results_by_group.get((resolution, split))
        counts = pd.Series(np.nan, index=frame.index, dtype=float)
        if original is not None:
            normalized = add_internal_result_aliases(original.copy()).set_index(
                "IMG_ID"
            )
            # Prefere a contagem explicitamente posterior à morfologia.
            column = next(
                (
                    c
                    for c in (
                        "artery_voxels_after_morphology",
                        "artery_voxels",
                    )
                    if c in normalized
                ),
                None,
            )
            if column is not None:
                counts = pd.to_numeric(
                    frame["IMG_ID"].map(normalized[column]), errors="coerce"
                )
        # Verifica validade das contagens de voxels e dos espaçamentos para conversão.
        spacing = frame[spacing_columns].apply(pd.to_numeric, errors="coerce")
        valid_counts = (
            counts.notna() & np.isfinite(counts) & counts.ge(0) & counts.mod(1).eq(0)
        )
        valid_spacing = pd.Series(
            np.isfinite(spacing.to_numpy(dtype=float)).all(axis=1)
            & spacing.gt(0).all(axis=1).to_numpy(),
            index=frame.index,
        )
        data.loc[frame.index[~valid_counts], "prediction_volume_status"] = (
            "missing_or_invalid_count"
        )
        data.loc[
            frame.index[valid_counts & ~valid_spacing], "prediction_volume_status"
        ] = "missing_or_invalid_spacing"
        # Calcula o volume predito em mL quando fatores, espaçamento e contagem são válidos.
        if not valid_factors:
            data.loc[selected, "prediction_volume_status"] = (
                "missing_or_invalid_factors"
            )
        else:
            valid = valid_counts & valid_spacing
            voxel_ml = spacing.prod(axis=1) * float(np.prod(factors)) / 1000
            predicted = counts * voxel_ml
            data.loc[
                frame.index[valid & ~np.isfinite(predicted)], "prediction_volume_status"
            ] = "invalid_physical_volume"
            valid &= np.isfinite(predicted)
            data.loc[frame.index[valid], "predicted_volume_ml"] = predicted.loc[valid]
            data.loc[frame.index[valid], "prediction_volume_status"] = "ok"
        # Registra cobertura e distribuição dos status de predição para a resolução.
        status = data.loc[selected, "prediction_volume_status"]
        source_coverage = coverage.loc[coverage["resolution"].eq(resolution)].iloc[0]
        rows.append(
            {
                "resolution": resolution,
                "total_exams": int(source_coverage["total_exams"]),
                "missing_reference_count": int(source_coverage["missing_volume_count"]),
                "valid_predictions": int(status.eq("ok").sum()),
                "excluded_ids": data.loc[
                    selected & data["prediction_volume_status"].ne("ok"), "IMG_ID"
                ].tolist(),
                "status_counts": status.value_counts().to_dict(),
            }
        )
    return data, pd.DataFrame(rows)


def compare_artery_volume_cohorts(
    results_by_group: Mapping[tuple[str, str], pd.DataFrame],
    classified_volumes: pd.DataFrame,
) -> pd.DataFrame:
    """Compara os mesmos IDs por resolução/split, sem excluir falhas restantes.

    Volumes ausentes são informados e preservados. Dice ausente vira zero
    somente em falhas documentadas; ausências sem diagnóstico são rejeitadas.
    As linhas ``all`` agregam splits disjuntos, não médias de médias.

    Args:
        results_by_group (Mapping[tuple[str, str], pd.DataFrame]): Mapa de
            ``(resolução, split)`` individuais para DataFrames de resultados.
        classified_volumes (pd.DataFrame): Tabela com ``IMG_ID`` e rótulo
            ``volume_outlier_group`` (``"low"``, ``"in_range"`` ou ``"high"``).

    Returns:
        pd.DataFrame: Comparativo de Dice e sucesso dos óstios antes e depois da
        exclusão dos outliers volumétricos por resolução e split.
    """
    # Valida unicidade dos IDs e rótulos da classificação volumétrica.
    volume_ids = require_series_column(classified_volumes, "IMG_ID")
    if volume_ids.isna().any() or volume_ids.duplicated().any():
        raise ValueError("IDs repetidos ou ausentes na classificação volumétrica.")
    volume_info = classified_volumes.loc[:, ["IMG_ID", "volume_outlier_group"]]
    if not volume_info["volume_outlier_group"].isin(["low", "in_range", "high"]).all():
        raise ValueError("Classificação volumétrica inválida.")
    # Normaliza resultados por split individual e constrói o agregado disjunto 'all'.
    groups: dict[tuple[str, str], pd.DataFrame] = {}
    by_resolution: dict[str, list[pd.DataFrame]] = {}
    for (resolution, split), original in results_by_group.items():
        if split == "all":
            raise ValueError(
                "Forneça somente os splits individuais; 'all' é calculado."
            )
        prepared = _prepare_result_dice(original, f"{resolution}/{split}")
        groups[resolution, split] = prepared
        by_resolution.setdefault(resolution, []).append(prepared)
    for resolution, frames in by_resolution.items():
        combined = pd.concat(frames, ignore_index=True)
        if combined["IMG_ID"].duplicated().any():
            raise ValueError(f"Há IDs compartilhados entre splits de {resolution}.")
        groups[resolution, "all"] = combined

    # Calcula métricas antes e depois da remoção de outliers low/high para cada coorte.
    rows = []
    for (resolution, split), frame in groups.items():
        paired = frame.merge(
            volume_info, on="IMG_ID", how="left", validate="one_to_one"
        )
        volume_groups = require_series_column(paired, "volume_outlier_group")
        low = volume_groups.eq("low")
        high = volume_groups.eq("high")
        kept = paired.loc[~(low | high)]
        before = float(require_series_column(paired, "dice").mean())
        after = float(require_series_column(kept, "dice").mean())
        success_before = require_series_column(paired, "ostia_success")
        success_after = require_series_column(kept, "ostia_success")
        before_success = float(success_before.mean() * 100)
        after_success = float(success_after.mean() * 100)
        rows.append(
            {
                "resolution": resolution,
                "split": split,
                "total_exams": len(paired),
                "missing_volume_count": int(volume_groups.isna().sum()),
                "low_outlier_count": int(low.sum()),
                "high_outlier_count": int(high.sum()),
                "excluded_exams": int((low | high).sum()),
                "remaining_exams": len(kept),
                "dice_before": before,
                "dice_after": after,
                "delta_dice": after - before,
                "ostia_success_count_before": int(success_before.sum()),
                "ostia_success_count_after": int(success_after.sum()),
                "ostia_success_percent_before": before_success,
                "ostia_success_percent_after": after_success,
                "delta_ostia_success_pp": after_success - before_success,
            }
        )
    # Ordena a tabela comparativa pela sequência canônica de splits e resolução.
    result = pd.DataFrame(rows)
    if not result.empty:
        split_order = require_series_column(result, "split").map(
            {"train": 0, "val": 1, "test": 2, "all": 3}
        )
        result = (
            result.assign(_order=split_order)
            .sort_values(["_order", "resolution"])
            .drop(columns="_order")
        )
    return result.reset_index(drop=True)
