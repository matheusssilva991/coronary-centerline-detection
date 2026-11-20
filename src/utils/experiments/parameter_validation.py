"""Definições e seleção de casos para validação de parâmetros do pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_parameter_validation_run(
    run_dir: str | Path,
    *,
    expected_split: str | None = "val",
) -> tuple[Path, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Carrega resultados, parâmetros e configuração de um run compacto.

    Args:
        run_dir (str | Path): Diretório do run contendo CSVs de resultados e
            parâmetros e o arquivo ``run_config.json``.
        expected_split (str | None): Split exclusivo exigido no run; se ``None``,
            ignora a validação de split único.

    Returns:
        tuple[Path, pd.DataFrame, pd.DataFrame, dict[str, Any]]: Caminho do run,
        resultados por imagem, parâmetros das variantes e configuração JSON.
    """
    # Lê os artefatos principais do run e valida consistência do split esperado.
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run não encontrado: {run_dir}")

    results = pd.read_csv(run_dir / "results/image_results.csv")
    parameters = pd.read_csv(run_dir / "parameters/variant_parameters.csv")
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    if expected_split is not None:
        result_splits = set(results["split"].dropna())
        if config.get("split") != expected_split or result_splits != {expected_split}:
            raise ValueError(
                f"O run {run_dir.name} não contém apenas o split {expected_split}."
            )

    return run_dir, results, parameters, config


def _as_boolean(series: pd.Series) -> pd.Series:
    """Converte colunas booleanas lidas de CSV sem tratar ``"False"`` como true.

    Args:
        series (pd.Series): Série booleana ou textual carregada de CSV.

    Returns:
        pd.Series: Série booleana normalizada.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "sim"})


def build_parameter_sensitivity_summary(
    image_results: pd.DataFrame,
    variant_parameters: pd.DataFrame,
    *,
    baseline_variant: str = "baseline",
) -> pd.DataFrame:
    """Resume Dice e sucesso dos óstios para a análise de sensibilidade.

    Args:
        image_results (pd.DataFrame): Resultados por exame e variante.
        variant_parameters (pd.DataFrame): Metadados e grupos de parâmetros por
            variante.
        baseline_variant (str): Variante de referência usada para calcular
            ``delta_dice_vs_baseline``.

    Returns:
        pd.DataFrame: Resumo por variante com taxas dos óstios, estatísticas de
        Dice, variação contra a referência e metadados ordenados.
    """
    # Normaliza Dice e óstios e agrega estatísticas descritivas por variante.
    df = image_results.copy()
    df["dice_artery"] = pd.to_numeric(df["dice_artery"], errors="coerce").fillna(0.0)
    df["ostia_success"] = _as_boolean(df["ostia_success"])

    summary = df.groupby("variant", as_index=False).agg(
        images=("IMG_ID", "nunique"),
        ostia_success_count=("ostia_success", "sum"),
        ostia_success_rate=("ostia_success", "mean"),
        mean_dice=("dice_artery", "mean"),
        std_dice=("dice_artery", "std"),
        median_dice=("dice_artery", "median"),
    )
    summary["ostia_success_percent"] = 100 * summary["ostia_success_rate"]

    # Calcula diferença de Dice médio em relação à variante de referência.
    baseline_values = summary.loc[summary["variant"].eq(baseline_variant), "mean_dice"]
    if baseline_values.empty:
        raise ValueError(f"Variante de referência ausente: {baseline_variant}")
    summary["delta_dice_vs_baseline"] = summary["mean_dice"] - baseline_values.iloc[0]

    # Anexa metadados de grupo/descrição dos parâmetros e ordena de modo estável.
    parameter_columns = [
        column
        for column in ("variant", "parameter_group", "description")
        if column in variant_parameters.columns
    ]
    if parameter_columns != ["variant"]:
        summary = summary.merge(
            variant_parameters[parameter_columns].drop_duplicates("variant"),
            on="variant",
            how="left",
        )

    sort_columns = [
        column for column in ("parameter_group", "variant") if column in summary
    ]
    return summary.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def build_parameter_pairwise_summary(
    image_results: pd.DataFrame,
    *,
    baseline_variant: str = "baseline",
) -> pd.DataFrame:
    """Compara cada variante com a referência usando os mesmos exames.

    Args:
        image_results (pd.DataFrame): Resultados por exame contendo ``variant``,
            ``IMG_ID`` e ``dice_artery``.
        baseline_variant (str): Nome da variante baseline para pareamento.

    Returns:
        pd.DataFrame: Tabela ordenada por ``mean_delta_dice`` decrescente com
        médias, medianas e contagens de melhora, empate e piora.
    """
    # Isola o conjunto de referência e garante que a variante baseline existe.
    baseline = image_results.loc[
        image_results["variant"].eq(baseline_variant), ["IMG_ID", "dice_artery"]
    ].rename(columns={"dice_artery": "dice_baseline"})
    if baseline.empty:
        raise ValueError(f"Variante de referência ausente: {baseline_variant}")

    # Pareia cada variante candidata por IMG_ID e resume ganhos e perdas de Dice.
    rows = []
    for variant_name, variant_df in image_results.groupby("variant"):
        if variant_name == baseline_variant:
            continue
        paired = baseline.merge(
            variant_df[["IMG_ID", "dice_artery"]].rename(
                columns={"dice_artery": "dice_variant"}
            ),
            on="IMG_ID",
            how="inner",
        )
        paired[["dice_baseline", "dice_variant"]] = paired[
            ["dice_baseline", "dice_variant"]
        ].apply(pd.to_numeric, errors="coerce")
        paired = paired.dropna()
        delta = paired["dice_variant"] - paired["dice_baseline"]
        rows.append(
            {
                "variant": variant_name,
                "paired_images": len(paired),
                "mean_delta_dice": delta.mean(),
                "median_delta_dice": delta.median(),
                "improved_images": int(delta.gt(0).sum()),
                "unchanged_images": int(np.isclose(delta, 0).sum()),
                "worse_images": int(delta.lt(0).sum()),
            }
        )

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("mean_delta_dice", ascending=False)
        .reset_index(drop=True)
    )


def build_threshold_performance_data(
    image_results: pd.DataFrame,
    variant_parameters: pd.DataFrame,
    *,
    variants: tuple[str, ...] = ("baseline", "upper_p997", "upper_p995"),
) -> pd.DataFrame:
    """Relaciona Dice, percentil superior e proporção preservada por exame.

    Args:
        image_results (pd.DataFrame): Resultados por exame incluindo contagens de
            voxels preservados e totais.
        variant_parameters (pd.DataFrame): Tabela de parâmetros contendo
            ``MAX_THRESHOLD_PERCENTILE`` por variante.
        variants (tuple[str, ...]): Variantes selecionadas para comparação dos
            cortes superiores.

    Returns:
        pd.DataFrame: Tabela ordenada por ``upper_percentile`` e ``IMG_ID`` com
        a coluna ``threshold_volume_fraction``.
    """
    # Valida colunas obrigatórias nos resultados e na tabela de parâmetros.
    required_results = {
        "variant",
        "IMG_ID",
        "dice_artery",
        "threshold_voxels",
        "volume_voxels",
    }
    missing = required_results.difference(image_results.columns)
    if missing:
        raise ValueError(f"Colunas de resultados ausentes: {sorted(missing)}")
    if "MAX_THRESHOLD_PERCENTILE" not in variant_parameters.columns:
        raise ValueError("MAX_THRESHOLD_PERCENTILE ausente nos parâmetros.")

    # Associa o percentil superior configurado e calcula a fração preservada.
    percentile_by_variant = (
        variant_parameters.loc[
            variant_parameters["variant"].isin(variants),
            ["variant", "MAX_THRESHOLD_PERCENTILE"],
        ]
        .drop_duplicates("variant")
        .rename(columns={"MAX_THRESHOLD_PERCENTILE": "upper_percentile"})
    )
    selected = image_results.loc[
        image_results["variant"].isin(variants),
        [
            "variant",
            "IMG_ID",
            "dice_artery",
            "threshold_voxels",
            "volume_voxels",
            "ostia_success",
        ],
    ].copy()
    selected = selected.merge(percentile_by_variant, on="variant", how="left")
    selected[["dice_artery", "threshold_voxels", "volume_voxels"]] = selected[
        ["dice_artery", "threshold_voxels", "volume_voxels"]
    ].apply(pd.to_numeric, errors="coerce")
    selected["threshold_volume_fraction"] = (
        selected["threshold_voxels"] / selected["volume_voxels"]
    )
    return selected.sort_values(["upper_percentile", "IMG_ID"]).reset_index(drop=True)


def load_downscaled_intensity_values(
    image_id: int,
    base_path: str | Path,
    config: dict[str, Any],
) -> np.ndarray:
    """Carrega um volume e retorna seus valores finitos após o downsampling.

    Args:
        image_id (int): Identificador numérico do exame NIfTI.
        base_path (str | Path): Diretório base onde residem os arquivos
            ``<id>.img.nii.gz``.
        config (dict[str, Any]): Configuração do pipeline com método, fatores de
            redução e interpolação OpenCV.

    Returns:
        np.ndarray: Vetor 1D com as intensidades HU finitas do volume reduzido.
    """
    import cv2

    from ..processing.preprocessing import downscale_image
    from ..utils.nifti_io import load_raw_img_and_label

    # Mapeia interpolação configurada e lê o volume NIfTI em float32.
    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos4": cv2.INTER_LANCZOS4,
    }
    interpolation = interpolation_map.get(
        config.get("OPENCV_INTERPOLATION", "linear"), cv2.INTER_LINEAR
    )
    image_object, _ = load_raw_img_and_label(
        str(Path(base_path) / f"{image_id}.img.nii.gz")
    )
    image = np.asarray(image_object.get_fdata(), dtype=np.float32)
    # Executa o downsampling configurado e extrai intensidades HU finitas.
    down_image = downscale_image(
        image,
        config["DOWNSCALE_FACTORS"],
        order=3,
        use_opencv=config.get("DOWNSCALE_METHOD") == "opencv",
        opencv_interpolation=interpolation,
    )
    finite_values = np.asarray(down_image)[np.isfinite(down_image)]
    if finite_values.size == 0:
        raise ValueError(f"IMG_ID={image_id} não possui intensidades HU finitas.")
    return finite_values


def summarize_intensity_histograms(
    values: np.ndarray,
    image_id: int,
    *,
    dense_min_hu: float = 300.0,
    bins: int = 256,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Resume e discretiza histogramas completo e acima de um limite HU.

    Args:
        values (np.ndarray): Vetor ou array com intensidades HU do volume.
        image_id (int): Identificador numérico do exame associado.
        dense_min_hu (float): Limiar mínimo em HU que define voxels densos.
        bins (int): Número de intervalos para discretização dos histogramas.

    Returns:
        tuple[dict[str, float | int], pd.DataFrame]: Dicionário de estatísticas
        por exame e DataFrame com os bins dos histogramas completo e denso.
    """
    # Valida quantidade de bins e separa subconjuntos completo e denso (HU >= limiar).
    if bins <= 0:
        raise ValueError("bins deve ser maior que zero.")

    finite_values = np.asarray(values, dtype=np.float32)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("O histograma requer ao menos uma intensidade finita.")

    subsets = {
        "full": finite_values,
        "dense_hu": finite_values[finite_values >= dense_min_hu],
    }
    summary: dict[str, float | int] = {
        "IMG_ID": int(image_id),
        "dense_min_hu": float(dense_min_hu),
    }
    histogram_frames = []

    # Calcula estatísticas descritivas e histogramas para cada subconjunto HU.
    for subset_name, subset_values in subsets.items():
        prefix = "full" if subset_name == "full" else "dense"
        summary[f"{prefix}_voxel_count"] = int(subset_values.size)
        if subset_values.size == 0:
            for metric in (
                "mean_hu",
                "median_hu",
                "max_hu",
                "histogram_peak_hu",
                "histogram_peak_count",
            ):
                summary[f"{prefix}_{metric}"] = np.nan
            continue

        counts, edges = np.histogram(subset_values, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        peak_index = int(np.argmax(counts))
        summary.update(
            {
                f"{prefix}_mean_hu": float(np.mean(subset_values)),
                f"{prefix}_median_hu": float(np.median(subset_values)),
                f"{prefix}_max_hu": float(np.max(subset_values)),
                f"{prefix}_histogram_peak_hu": float(centers[peak_index]),
                f"{prefix}_histogram_peak_count": int(counts[peak_index]),
            }
        )
        histogram_frames.append(
            pd.DataFrame(
                {
                    "IMG_ID": int(image_id),
                    "histogram": subset_name,
                    "bin_left_hu": edges[:-1],
                    "bin_right_hu": edges[1:],
                    "bin_center_hu": centers,
                    "count": counts,
                }
            )
        )

    # Calcula fração percentual de voxels densos e concatena as tabelas de bins.
    summary["dense_voxel_percent"] = (
        100 * int(summary["dense_voxel_count"]) / int(summary["full_voxel_count"])
    )
    histogram_df = (
        pd.concat(histogram_frames, ignore_index=True)
        if histogram_frames
        else pd.DataFrame()
    )
    return summary, histogram_df


def build_normalized_intensity_histograms(
    histogram_bins: pd.DataFrame,
    *,
    image_ids: list[int] | None = None,
    histogram_name: str = "full",
    bins: int = 256,
) -> pd.DataFrame:
    """Redistribui histogramas por imagem em uma grade HU compartilhada.

    Converte cada histograma em probabilidade acumulada e interpola os limites
    comuns, evitando comparar imagens com grades ou quantidades incompatíveis.

    Args:
        histogram_bins (pd.DataFrame): Tabela de bins por exame produzida por
            ``summarize_intensity_histograms``.
        image_ids (list[int] | None): Lista opcional de IDs para restringir os
            exames incluídos.
        histogram_name (str): Nome do histograma selecionado (``"full"`` ou
            ``"dense_hu"``).
        bins (int): Número de bins da grade HU comum de saída.

    Returns:
        pd.DataFrame: Perfis de probabilidade por bin alinhados na grade comum.
    """
    # Valida parâmetros e colunas obrigatórias, filtrando histograma e exames.
    if bins <= 0:
        raise ValueError("bins deve ser maior que zero.")

    required_columns = {
        "IMG_ID",
        "histogram",
        "bin_left_hu",
        "bin_right_hu",
        "count",
    }
    missing = required_columns.difference(histogram_bins.columns)
    if missing:
        raise ValueError(f"Colunas de histograma ausentes: {sorted(missing)}")

    selected = histogram_bins.loc[histogram_bins["histogram"].eq(histogram_name)].copy()
    if image_ids is not None:
        selected_ids = {int(image_id) for image_id in image_ids}
        selected = selected.loc[selected["IMG_ID"].astype(int).isin(selected_ids)]
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "histogram",
                "bin_left_hu",
                "bin_right_hu",
                "bin_center_hu",
                "probability",
            ]
        )

    # Constrói grade HU comum entre os limites mínimo e máximo de toda a coorte.
    common_edges = np.linspace(
        float(selected["bin_left_hu"].min()),
        float(selected["bin_right_hu"].max()),
        bins + 1,
    )
    # Interpola a CDF de cada exame para a grade comum e obtém probabilidades por bin.
    profile_frames = []
    for image_id, image_histogram in selected.groupby("IMG_ID"):
        image_histogram = image_histogram.sort_values("bin_left_hu")
        counts = pd.to_numeric(image_histogram["count"], errors="coerce").fillna(0)
        total = float(counts.sum())
        if total <= 0:
            continue

        source_edges = np.concatenate(
            [
                image_histogram["bin_left_hu"].to_numpy(dtype=float),
                [float(image_histogram["bin_right_hu"].iloc[-1])],
            ]
        )
        source_cdf = np.concatenate([[0.0], counts.cumsum().to_numpy() / total])
        common_cdf = np.interp(
            common_edges,
            source_edges,
            source_cdf,
            left=0.0,
            right=1.0,
        )
        profile_frames.append(
            pd.DataFrame(
                {
                    "IMG_ID": int(image_id),
                    "histogram": histogram_name,
                    "bin_left_hu": common_edges[:-1],
                    "bin_right_hu": common_edges[1:],
                    "bin_center_hu": (common_edges[:-1] + common_edges[1:]) / 2,
                    "probability": np.diff(common_cdf),
                }
            )
        )

    if not profile_frames:
        return pd.DataFrame()
    return pd.concat(profile_frames, ignore_index=True)


def build_mean_normalized_intensity_histogram(
    profiles: pd.DataFrame,
    *,
    image_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Resume perfis normalizados já alinhados em uma grade HU comum.

    Args:
        profiles (pd.DataFrame): Perfis normalizados retornados por
            ``build_normalized_intensity_histograms``.
        image_ids (list[int] | None): Subconjunto opcional de IDs para restringir
            a agregação.

    Returns:
        pd.DataFrame: Perfil médio com média, desvio-padrão e número de exames
        por bin da grade HU compartilhada.
    """
    # Valida colunas esperadas e aplica eventual filtro de exames.
    required_columns = {
        "IMG_ID",
        "histogram",
        "bin_left_hu",
        "bin_right_hu",
        "bin_center_hu",
        "probability",
    }
    missing = required_columns.difference(profiles.columns)
    if missing:
        raise ValueError(f"Colunas de perfis ausentes: {sorted(missing)}")

    selected = profiles
    if image_ids is not None:
        selected_ids = {int(image_id) for image_id in image_ids}
        selected = profiles.loc[profiles["IMG_ID"].astype(int).isin(selected_ids)]
    if selected.empty:
        return pd.DataFrame()

    # Agrega média, desvio-padrão das probabilidades e contagem de imagens por bin.
    return (
        selected.groupby(
            [
                "histogram",
                "bin_left_hu",
                "bin_right_hu",
                "bin_center_hu",
            ],
            as_index=False,
        )
        .agg(
            mean_probability=("probability", "mean"),
            std_probability=("probability", "std"),
            images=("IMG_ID", "nunique"),
        )
        .fillna({"std_probability": 0.0})
    )


def compute_intensity_histogram_analysis(
    image_ids: list[int],
    base_path: str | Path,
    config: dict[str, Any],
    *,
    dense_min_hu: float = 300.0,
    bins: int = 256,
    percentiles: tuple[float, ...] = (),
    progress_every: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula métricas e histogramas dos volumes reduzidos do pipeline.

    Percentis opcionais são armazenados como colunas ``p<valor>_hu`` no resumo,
    permitindo obter histogramas e thresholds efetivos com um único carregamento.

    Args:
        image_ids (list[int]): Lista de IDs de exames a processar (duplicatas são
            removidas preservando ordem).
        base_path (str | Path): Diretório base dos volumes NIfTI.
        config (dict[str, Any]): Configuração do pipeline aplicada no downsampling.
        dense_min_hu (float): Limiar mínimo em HU para o histograma de região densa.
        bins (int): Quantidade de bins de cada histograma.
        percentiles (tuple[float, ...]): Percentis opcionais registrados como
            colunas ``p<valor>_hu`` no resumo por exame.
        progress_every (int | None): Intervalo opcional para imprimir progresso.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tabela de resumos por exame e
        DataFrame concatenado com todos os bins dos histogramas.
    """
    # Valida intervalo de log e remove IDs repetidos preservando a ordem.
    if progress_every is not None and progress_every <= 0:
        raise ValueError("progress_every deve ser maior que zero.")

    unique_image_ids = list(dict.fromkeys(int(value) for value in image_ids))
    summary_rows = []
    histogram_frames = []
    # Carrega cada volume reduzido uma vez para gerar resumo, histograma e percentis.
    for position, image_id in enumerate(unique_image_ids, start=1):
        values = load_downscaled_intensity_values(image_id, base_path, config)
        summary, histogram = summarize_intensity_histograms(
            values,
            image_id,
            dense_min_hu=dense_min_hu,
            bins=bins,
        )
        for percentile, threshold_hu in zip(
            percentiles,
            np.percentile(values, percentiles) if percentiles else (),
        ):
            percentile_key = str(percentile).replace(".", "")
            summary[f"p{percentile_key}_hu"] = float(threshold_hu)
        summary_rows.append(summary)
        histogram_frames.append(histogram)
        if progress_every and (
            position % progress_every == 0 or position == len(unique_image_ids)
        ):
            print(f"Histogramas processados: {position}/{len(unique_image_ids)}")

    # Consolida os bins dos histogramas e retorna o par resumo/bins.
    histogram_df = (
        pd.concat(histogram_frames, ignore_index=True)
        if histogram_frames
        else pd.DataFrame()
    )
    return pd.DataFrame(summary_rows), histogram_df


def compute_effective_upper_thresholds(
    image_ids: list[int],
    base_path: str | Path,
    config: dict[str, Any],
    percentiles: tuple[float, ...],
) -> pd.DataFrame:
    """Calcula thresholds e limites HU após o downsampling do pipeline.

    Cada volume é carregado e reduzido apenas uma vez; todos os percentis são
    calculados sobre esse mesmo volume para manter a comparação pareada. Os
    limites mínimo e máximo retornados também pertencem ao volume reduzido que
    efetivamente entra no thresholding, e não ao NIfTI na resolução original.

    Args:
        image_ids (list[int]): Lista de IDs de exames a avaliar.
        base_path (str | Path): Diretório base onde ficam os volumes NIfTI.
        config (dict[str, Any]): Configuração do pipeline aplicada no downsampling.
        percentiles (tuple[float, ...]): Percentis em ``[0, 100]`` avaliados
            sobre o volume reduzido.

    Returns:
        pd.DataFrame: Tabela longa por ``(IMG_ID, upper_percentile)`` com
        threshold efetivo, mínimo, máximo e amplitude HU do volume reduzido.
    """
    rows: list[dict[str, float | int]] = []

    # Reduz cada volume único uma vez e avalia extremos e percentis sobre ele.
    for image_id in sorted(set(image_ids)):
        finite_values = load_downscaled_intensity_values(image_id, base_path, config)
        image_min_hu = float(finite_values.min())
        image_max_hu = float(finite_values.max())
        values = np.percentile(finite_values, percentiles)
        rows.extend(
            {
                "IMG_ID": int(image_id),
                "upper_percentile": float(percentile),
                "max_threshold_hu": float(value),
                "image_min_hu": image_min_hu,
                "image_max_hu": image_max_hu,
                "image_intensity_range_hu": image_max_hu - image_min_hu,
            }
            for percentile, value in zip(percentiles, values)
        )

    return pd.DataFrame(rows)


__all__ = [
    "build_mean_normalized_intensity_histogram",
    "build_normalized_intensity_histograms",
    "build_parameter_pairwise_summary",
    "build_parameter_sensitivity_summary",
    "build_threshold_performance_data",
    "compute_effective_upper_thresholds",
    "compute_intensity_histogram_analysis",
    "load_downscaled_intensity_values",
    "load_parameter_validation_run",
    "summarize_intensity_histograms",
]
