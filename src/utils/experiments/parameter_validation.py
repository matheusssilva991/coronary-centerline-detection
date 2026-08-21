"""Definições e seleção de casos para validação de parâmetros do pipeline."""

from __future__ import annotations

import copy
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
    """Carrega resultados, parâmetros e configuração de um run compacto."""
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


_DOWNSTREAM_CONFIG_SECTIONS = {
    "OSTIA_DETECTION",
    "OSTIA_VALIDATION",
    "ARTERY_SEGMENTATION",
    "REGION_GROWING",
    "FUZZY_CONNECTEDNESS",
    "POSTPROCESSING",
    "NUM_BATCHES",
}


def _freeze_cache_value(value: Any) -> Any:
    """Converte configurações com arrays em uma chave imutável e comparável."""
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_cache_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_value(item) for item in value)
    if hasattr(value, "tolist"):
        return _freeze_cache_value(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def image_load_cache_key(config: dict[str, Any]) -> tuple[Any, ...]:
    """Identifica variantes que compartilham carregamento e downsampling."""
    return _freeze_cache_value(
        {
            key: config.get(key)
            for key in (
                "DOWNSCALE_FACTORS",
                "DOWNSCALE_METHOD",
                "OPENCV_INTERPOLATION",
            )
        }
    )


def prepared_context_cache_key(
    config: dict[str, Any],
    experiment: dict[str, Any],
) -> tuple[Any, ...]:
    """Identifica variantes com threshold, aorta e vesselness idênticos."""
    upstream_config = {
        key: value
        for key, value in config.items()
        if key not in _DOWNSTREAM_CONFIG_SECTIONS
    }
    preprocessing_experiment = {
        key: experiment.get(key)
        for key in ("threshold_mode", "fuzzy", "ostia_only")
        if key in experiment
    }
    return _freeze_cache_value(
        {
            "config": upstream_config,
            "experiment": preprocessing_experiment,
        }
    )


def _as_boolean(series: pd.Series) -> pd.Series:
    """Converte colunas booleanas lidas de CSV sem tratar ``"False"`` como true."""
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
    """Resume Dice e sucesso dos óstios para a análise de sensibilidade."""
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

    baseline_values = summary.loc[
        summary["variant"].eq(baseline_variant), "mean_dice"
    ]
    if baseline_values.empty:
        raise ValueError(f"Variante de referência ausente: {baseline_variant}")
    summary["delta_dice_vs_baseline"] = summary["mean_dice"] - baseline_values.iloc[0]

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
    """Compara cada variante com a referência usando os mesmos exames."""
    baseline = image_results.loc[
        image_results["variant"].eq(baseline_variant), ["IMG_ID", "dice_artery"]
    ].rename(columns={"dice_artery": "dice_baseline"})
    if baseline.empty:
        raise ValueError(f"Variante de referência ausente: {baseline_variant}")

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
    return pd.DataFrame(rows).sort_values(
        "mean_delta_dice", ascending=False
    ).reset_index(drop=True)


def build_threshold_performance_data(
    image_results: pd.DataFrame,
    variant_parameters: pd.DataFrame,
    *,
    variants: tuple[str, ...] = ("baseline", "upper_p997", "upper_p995"),
) -> pd.DataFrame:
    """Relaciona Dice, percentil superior e proporção preservada por exame."""
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


def select_top_threshold_cases(
    threshold_results: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    """Seleciona os maiores Dice separadamente para cada percentil."""
    if top_n <= 0:
        raise ValueError("top_n deve ser maior que zero.")
    return (
        threshold_results.sort_values(
            ["variant", "dice_artery", "IMG_ID"],
            ascending=[True, False, True],
        )
        .groupby("variant", as_index=False, group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def load_downscaled_intensity_values(
    image_id: int,
    base_path: str | Path,
    config: dict[str, Any],
) -> np.ndarray:
    """Carrega um volume e retorna seus valores finitos após o downsampling."""
    import cv2

    from ..processing.preprocessing import downscale_image
    from ..utils.nifti_io import load_raw_img_and_label

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
    """Resume e discretiza histogramas completo e acima de um limite HU."""
    if bins <= 0:
        raise ValueError("bins deve ser maior que zero.")

    finite_values = np.asarray(values, dtype=np.float32)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("O histograma requer ao menos uma intensidade finita.")

    subsets = {
        "full": finite_values,
        "above_300_hu": finite_values[finite_values > dense_min_hu],
    }
    summary: dict[str, float | int] = {
        "IMG_ID": int(image_id),
        "dense_min_hu": float(dense_min_hu),
    }
    histogram_frames = []

    for subset_name, subset_values in subsets.items():
        prefix = "full" if subset_name == "full" else "above_300"
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

    summary["above_300_voxel_percent"] = (
        100 * int(summary["above_300_voxel_count"]) / int(summary["full_voxel_count"])
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
    """Rebin per-image histograms as probabilities on a shared HU grid.

    Each source histogram is converted to a cumulative probability curve and
    interpolated at shared bin edges. This prevents images with more voxels or
    different original HU ranges from being compared on incompatible bins.
    """
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

    selected = histogram_bins.loc[
        histogram_bins["histogram"].eq(histogram_name)
    ].copy()
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

    common_edges = np.linspace(
        float(selected["bin_left_hu"].min()),
        float(selected["bin_right_hu"].max()),
        bins + 1,
    )
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


def build_mean_intensity_histogram(
    histogram_bins: pd.DataFrame,
    *,
    image_ids: list[int] | None = None,
    histogram_name: str = "full",
    bins: int = 256,
) -> pd.DataFrame:
    """Average normalized image histograms on a shared HU grid."""
    profiles = build_normalized_intensity_histograms(
        histogram_bins,
        image_ids=image_ids,
        histogram_name=histogram_name,
        bins=bins,
    )
    if profiles.empty:
        return pd.DataFrame()

    return (
        profiles.groupby(
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
    """
    if progress_every is not None and progress_every <= 0:
        raise ValueError("progress_every deve ser maior que zero.")

    unique_image_ids = list(dict.fromkeys(int(value) for value in image_ids))
    summary_rows = []
    histogram_frames = []
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
            print(
                f"Histogramas processados: {position}/{len(unique_image_ids)}"
            )

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
    """
    rows: list[dict[str, float | int]] = []

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


def summarize_top_threshold_cases(top_cases: pd.DataFrame) -> pd.DataFrame:
    """Resume desempenho e faixa HU entre os melhores casos de cada variante."""
    required = {"variant", "upper_percentile", "dice_artery", "max_threshold_hu"}
    missing = required.difference(top_cases.columns)
    if missing:
        raise ValueError(f"Colunas da análise de threshold ausentes: {sorted(missing)}")

    summary = top_cases.groupby(
        ["variant", "upper_percentile"], as_index=False
    ).agg(
        selected_images=("IMG_ID", "nunique"),
        mean_dice=("dice_artery", "mean"),
        min_dice=("dice_artery", "min"),
        median_threshold_hu=("max_threshold_hu", "median"),
        mean_threshold_hu=("max_threshold_hu", "mean"),
        std_threshold_hu=("max_threshold_hu", "std"),
        min_threshold_hu=("max_threshold_hu", "min"),
        max_threshold_hu=("max_threshold_hu", "max"),
    )
    quartiles = top_cases.groupby(["variant", "upper_percentile"])[
        "max_threshold_hu"
    ].quantile([0.25, 0.75]).unstack()
    quartiles.columns = ["threshold_hu_q1", "threshold_hu_q3"]
    return summary.merge(
        quartiles.reset_index(), on=["variant", "upper_percentile"], how="left"
    ).sort_values("upper_percentile", ascending=False).reset_index(drop=True)


def parameter_validation_variants() -> list[dict[str, Any]]:
    """Retorna as variantes OFAT da análise de sensibilidade do artigo.

    A referência usa os valores centrais declarados no estudo. Cada variante
    altera apenas um parâmetro, mantendo todos os demais na referência.
    """
    reference = {
        "threshold_mode": "normal",
        "artery_method": "region_growing",
        "MAX_THRESHOLD_PERCENTILE": 99.9,
        "OSTIA_DETECTION.max_z_diff_mm": 40.0,
        "REGION_GROWING.threshold_divisor": 7.0,
        "REGION_GROWING.min_vesselness_fraction": 0.078,
        "REGION_GROWING.relaxed_floor_factor": 0.98,
    }

    def variant(
        name: str,
        group: str,
        description: str,
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "name": name,
            "parameter_group": group,
            "description": description,
            "overrides": {**reference, **overrides},
        }

    return [
        variant(
            "baseline",
            "baseline",
            "Referência CBEB: P99.9, z=40 mm, D=7 e piso de vesselness 7.8%.",
            {},
        ),
        *[
            variant(
                f"upper_p{str(value).replace('.', '')}",
                "upper_percentile",
                f"Percentil superior do threshold = {value}.",
                {"MAX_THRESHOLD_PERCENTILE": value},
            )
            for value in (99.5, 99.7)
        ],
        *[
            variant(
                f"ostia_z{int(value)}",
                "ostia_z_limit",
                f"Diferença máxima em z entre óstios = {value:.0f} mm.",
                {"OSTIA_DETECTION.max_z_diff_mm": value},
            )
            for value in (30.0, 50.0)
        ],
        *[
            variant(
                f"ostia_lower_{int(value * 100)}",
                "ostia_search_fraction",
                f"Fração inferior da aorta usada na busca dos óstios = {value:.0%}.",
                {"OSTIA_DETECTION.lower_fraction": value},
            )
            for value in (0.70, 1.00)
        ],
        *[
            variant(
                f"rg_divisor_{int(value)}",
                "rg_threshold_divisor",
                f"Divisor do limiar global do RG = {value:.0f}.",
                {"REGION_GROWING.threshold_divisor": value},
            )
            for value in (5.0, 9.0)
        ],
        *[
            variant(
                f"rg_vessel_{int(initial * 100):02d}",
                "rg_vesselness_floor",
                (f"Fração mínima de vesselness do RG = {initial:.0%}."),
                {
                    "REGION_GROWING.min_vesselness_fraction": initial,
                },
            )
            for initial in (0.05, 0.09)
        ],
    ]


def resolution_scaling_variants() -> list[dict[str, Any]]:
    """Retorna variantes que isolam os grupos de escala usados em high-res.

    Todos os casos preservam a configuração P99.9 do artigo. As variantes
    iniciais desativam grupos de ``scale_config_to_resolution``; as variantes
    de refinamento aplicam valores intermediários depois da escala para estudar
    os parâmetros que efetivamente alteraram a localização dos óstios.
    """
    reference = {
        "threshold_mode": "normal",
        "artery_method": "region_growing",
        "MAX_THRESHOLD_PERCENTILE": 99.9,
    }

    def variant(
        name: str,
        group: str,
        description: str,
        disabled_groups: tuple[str, ...] = (),
        post_scale_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "parameter_group": group,
            "description": description,
            "overrides": reference.copy(),
            "disabled_scaling_groups": list(disabled_groups),
            "post_scale_overrides": copy.deepcopy(post_scale_overrides or {}),
        }

    return [
        variant(
            "all_scaled",
            "reference",
            "Aplica todos os grupos de escala atuais para high resolution.",
        ),
        variant(
            "circle_geometry_unscaled",
            "circle_geometry",
            "Mantém raios, passo e offset da Hough nos valores mid-res.",
            ("circle_geometry",),
        ),
        variant(
            "circle_tracking_unscaled",
            "circle_tracking",
            "Mantém sigma do Canny, distância entre vizinhos e padding da ROI.",
            ("circle_tracking",),
        ),
        variant(
            "level_set_iterations_unscaled",
            "level_set_iterations",
            "Mantém 31 iterações do level set em vez do ajuste high-res.",
            ("level_set_iterations",),
        ),
        variant(
            "level_set_morphology_unscaled",
            "level_set_morphology",
            "Mantém o raio de remoção de vazamentos no valor mid-res.",
            ("level_set_morphology",),
        ),
        variant(
            "ostia_surface_unscaled",
            "ostia_surface",
            "Mantém o raio de erosão da superfície da aorta no valor mid-res.",
            ("ostia_surface",),
        ),
        variant(
            "ostia_candidates_unscaled",
            "ostia_candidates",
            "Mantém top_n no valor mid-res, sem multiplicação pela área XY.",
            ("ostia_candidates",),
        ),
        variant(
            "morphology_radii_unscaled",
            "combined_morphology",
            "Mantém os raios 3D do level set e da superfície nos valores mid-res.",
            ("level_set_morphology", "ostia_surface"),
        ),
        variant(
            "canny_sigma_mid",
            "circle_tracking_refinement",
            "Mantém somente o sigma do Canny no valor mid-res 3.",
            post_scale_overrides={"CIRCLE_DETECTION.canny_sigma": 3.0},
        ),
        variant(
            "canny_sigma_4",
            "circle_tracking_refinement",
            "Usa sigma 4 no Canny, entre os valores mid-res 3 e high-res 6.",
            post_scale_overrides={"CIRCLE_DETECTION.canny_sigma": 4.0},
        ),
        variant(
            "canny_sigma_5",
            "circle_tracking_refinement",
            "Usa sigma 5 no Canny, entre os valores mid-res 3 e high-res 6.",
            post_scale_overrides={"CIRCLE_DETECTION.canny_sigma": 5.0},
        ),
        variant(
            "neighbor_distance_mid",
            "circle_tracking_refinement",
            "Mantém somente a distância de vizinhança no valor mid-res 5.",
            post_scale_overrides={
                "CIRCLE_DETECTION.neighbor_distance_threshold": 5.0
            },
        ),
        variant(
            "local_roi_padding_mid",
            "circle_tracking_refinement",
            "Mantém somente o padding da ROI no valor mid-res 30.",
            post_scale_overrides={"CIRCLE_DETECTION.local_roi_padding": 30},
        ),
        variant(
            "level_set_iterations_50",
            "level_set_iterations_refinement",
            "Usa 50 iterações como ponto intermediário entre 31 e 70.",
            post_scale_overrides={"LEVEL_SET.num_iter": 50},
        ),
        variant(
            "level_set_iterations_60",
            "level_set_iterations_refinement",
            "Usa 60 iterações como ponto intermediário próximo de high-res 70.",
            post_scale_overrides={"LEVEL_SET.num_iter": 60},
        ),
        variant(
            "canny_sigma_3_level_set_50",
            "combined_scaling_refinement",
            "Combina sigma 3 do Canny com 50 iterações do level set.",
            post_scale_overrides={
                "CIRCLE_DETECTION.canny_sigma": 3.0,
                "LEVEL_SET.num_iter": 50,
            },
        ),
        variant(
            "canny_sigma_4_level_set_50",
            "combined_scaling_refinement",
            "Combina sigma 4 do Canny com 50 iterações do level set.",
            post_scale_overrides={
                "CIRCLE_DETECTION.canny_sigma": 4.0,
                "LEVEL_SET.num_iter": 50,
            },
        ),
        variant(
            "canny_sigma_5_level_set_50",
            "combined_scaling_refinement",
            "Combina sigma 5 do Canny com 50 iterações do level set.",
            post_scale_overrides={
                "CIRCLE_DETECTION.canny_sigma": 5.0,
                "LEVEL_SET.num_iter": 50,
            },
        ),
        variant(
            "roi_padding_40_level_set_50",
            "combined_scaling_refinement",
            "Combina padding local 40 com 50 iterações do level set.",
            post_scale_overrides={
                "CIRCLE_DETECTION.local_roi_padding": 40,
                "LEVEL_SET.num_iter": 50,
            },
        ),
    ]


def validate_parameter_validation_append(
    existing: dict[str, Any],
    *,
    split: str,
    image_ids: list[int],
    resolution: str,
    aorta_ostia_method: str,
    config_path: Path,
    use_gpu: bool,
) -> None:
    """Impede combinar partes produzidas com configurações incompatíveis."""
    expected = {
        "split": split,
        "ids": image_ids,
        "resolution": resolution,
        "aorta_ostia_method": aorta_ostia_method,
        "config_path": str(config_path),
        "use_gpu": use_gpu,
    }
    mismatches = [key for key, value in expected.items() if existing.get(key) != value]
    if mismatches:
        details = ", ".join(
            f"{key}: existente={existing.get(key)!r}, atual={expected[key]!r}"
            for key in mismatches
        )
        raise ValueError(f"Run incompatível com --append ({details}).")


def select_parameter_validation_cases(
    image_results: pd.DataFrame,
    variant: str,
    *,
    target_dice: float = 0.58,
    failure_dice_floor: float = 0.02,
    min_visible_artery_voxels: int = 5_000,
) -> pd.DataFrame:
    """Seleciona casos 3D de desempenho e falha para uma variante.

    Vazamentos são candidatos quantitativos: volume de aorta muito alto ou
    razão predição/ground truth arterial alta. A confirmação permanece visual.
    """
    df = image_results.loc[image_results["variant"] == variant].copy()
    if df.empty:
        raise ValueError(f"Variante sem resultados: {variant}")

    numeric_columns = [
        "IMG_ID",
        "dice_artery",
        "aorta_voxels",
        "aorta_volume_fraction",
        "artery_volume_ratio",
        "artery_voxels",
        "left_dist_mm",
        "right_dist_mm",
    ]
    for column in numeric_columns:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ("ostia_success", "ostia_found"):
        if column in df:
            df[column] = _as_boolean(df[column])

    selected_ids: set[int] = set()
    rows: list[dict[str, Any]] = []

    def add_case(case_type: str, candidates: pd.DataFrame, reason: str) -> None:
        available = candidates.loc[~candidates["IMG_ID"].isin(selected_ids)]
        if available.empty:
            available = candidates
        if available.empty:
            return
        row = available.iloc[0].to_dict()
        image_id = int(row["IMG_ID"])
        selected_ids.add(image_id)
        row.update({"case_type": case_type, "selection_reason": reason})
        rows.append(row)

    valid_dice = df.dropna(subset=["dice_artery"])
    add_case(
        "high_dice",
        valid_dice.sort_values("dice_artery", ascending=False),
        "Maior Dice da variante selecionada.",
    )
    # O exemplo intermediário deve representar a segmentação, sem ser
    # confundido por uma localização inválida dos óstios.
    near_mean = valid_dice.loc[valid_dice["ostia_success"]].assign(
        target_distance=lambda values: (values["dice_artery"] - target_dice).abs()
    ).sort_values("target_distance")
    add_case(
        "near_target_mean",
        near_mean,
        f"Dice mais próximo do alvo {target_dice:.2f}, com ambos os óstios aceitos.",
    )

    ostia_failure = df.loc[~df["ostia_success"]].copy()
    # Prefere uma localização incorreta que ainda produziu uma máscara útil
    # para inspeção, em vez de um caso completamente vazio.
    informative_ostia_failure = ostia_failure.loc[
        ostia_failure["ostia_found"]
        & ostia_failure["dice_artery"].gt(failure_dice_floor)
    ].copy()
    if not informative_ostia_failure.empty:
        failure_median = informative_ostia_failure["dice_artery"].median()
        ostia_failure = informative_ostia_failure.assign(
            failure_distance=lambda values: (
                values["dice_artery"] - failure_median
            ).abs()
        ).sort_values("failure_distance")
    else:
        ostia_failure["ostia_failure_priority"] = np.where(
            ostia_failure["ostia_found"],
            0,
            1,
        )
        ostia_failure = ostia_failure.sort_values(
            ["ostia_failure_priority", "dice_artery"],
            ascending=[True, True],
            na_position="first",
        )
    add_case(
        "ostia_failure",
        ostia_failure,
        "Falha representativa dos óstios com máscara arterial não vazia.",
    )

    aorta_candidates = df.dropna(subset=["aorta_volume_fraction"]).sort_values(
        "aorta_volume_fraction",
        ascending=False,
    )
    add_case(
        "suspected_aorta_leak",
        aorta_candidates,
        "Maior fração do volume ocupada pela aorta; candidato a vazamento visual.",
    )

    # Evita máscaras quase vazias, que desaparecem na visualização 3D. Entre
    # os casos com óstios aceitos, escolhe o menor Dice ainda inspecionável.
    visible_failure = df["ostia_success"] & df["dice_artery"].between(
        failure_dice_floor,
        target_dice,
    )
    if "artery_voxels" in df:
        visible_failure &= df["artery_voxels"].ge(min_visible_artery_voxels)
    segmentation_failure = df.loc[visible_failure].dropna(subset=["dice_artery"])
    if segmentation_failure.empty:
        segmentation_failure = df.loc[df["ostia_success"]].dropna(
            subset=["dice_artery"]
        )
    if "artery_volume_ratio" in segmentation_failure:
        segmentation_failure = segmentation_failure.sort_values(
            ["dice_artery", "artery_volume_ratio"],
            ascending=[True, False],
        )
    else:
        segmentation_failure = segmentation_failure.sort_values("dice_artery")
    add_case(
        "segmentation_failure",
        segmentation_failure,
        "Menor Dice com ambos os óstios aceitos e máscara arterial visível.",
    )

    return pd.DataFrame(rows)


def variant_by_name(name: str) -> dict[str, Any]:
    """Retorna uma cópia da definição de uma variante pelo nome."""
    variants = {item["name"]: item for item in parameter_validation_variants()}
    if name not in variants:
        raise ValueError(f"Variante desconhecida: {name}")
    return copy.deepcopy(variants[name])


__all__ = [
    "build_parameter_pairwise_summary",
    "build_parameter_sensitivity_summary",
    "image_load_cache_key",
    "parameter_validation_variants",
    "resolution_scaling_variants",
    "prepared_context_cache_key",
    "select_parameter_validation_cases",
    "validate_parameter_validation_append",
    "variant_by_name",
]
