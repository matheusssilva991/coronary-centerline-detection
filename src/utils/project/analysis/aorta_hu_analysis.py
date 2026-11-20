"""Compara HU de regiões cardíacas e aortas previstas em CCTA de treino."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nibabel.funcs import as_closest_canonical
from nibabel.nifti1 import Nifti1Image
from numpy.typing import NDArray

from utils.experiments.aorta_visual_review import (
    get_aorta_visual_review,
    load_aorta_review_cohort,
    load_aorta_visual_reviews,
)
from utils.project.analysis.mmwhs_heart_analysis import (
    MMWHS_HEART_LABEL_SPECS,
    load_mmwhs_heart_volume,
    summarize_hu_region,
)
from utils.project.config import load_config_json
from utils.project.dataframe import numeric_series, require_series_column
from utils.project.evaluation.mmwhs_official_aorta import (
    AortaPrediction,
    restore_native_mask,
)
from utils.segmentation.pipeline.detection import (
    AortaCircleTrackingResult,
    locate_and_filter_aorta_circles,
    segment_aorta_with_diagnostics,
)
from utils.segmentation.pipeline.preprocessing import preprocess_ccta_volume
from utils.utils.nifti_io import load_raw_img_and_label

HU_REGION_NAMES = {
    "mmwhs_heart": "MM-WHS · coração (referência)",
    "mmwhs_aorta": "MM-WHS · aorta (referência)",
    "imagecas_aorta": "ImageCAS · aorta (prevista)",
}


@dataclass(frozen=True)
class HuReferenceVolume:
    """Reúne HU nativos e regiões na mesma orientação canônica RAS."""

    image: NDArray[np.float32]
    regions: dict[str, NDArray[np.bool_]]
    spacing: tuple[float, float, float]
    dataset: str
    exam_id: str


def select_reviewed_imagecas_exams(
    repo_root: Path,
    *,
    variant: str = "filter_envelope_current",
    n_exams: int | None = 5,
    random_seed: int = 42,
    exam_ids: Sequence[int] | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Seleciona apenas aortas de treino boas no run da revisão escolhida.

    Args:
        repo_root (Path): Diretório raiz do repositório.
        variant (str): Chave da variante no catálogo de revisões visuais da aorta.
        n_exams (int | None): Quantidade de exames sorteados quando ``exam_ids`` é
            ``None``; ``None`` preserva todos os casos elegíveis.
        random_seed (int): Semente aleatória para amostragem sem reposição.
        exam_ids (Sequence[int] | None): Sequência explícita de IDs ImageCAS bons a
            selecionar no lugar do sorteio.

    Returns:
        tuple[Path, pd.DataFrame]: Par ``(run_dir, selected_exams)`` com o diretório
            do run revisado e a tabela de exames filtrada.
    """
    # Carrega o catálogo de revisões visuais e verifica o snapshot da configuração.
    catalog = load_aorta_visual_reviews(repo_root / "config/aorta_visual_reviews.json")
    review = get_aorta_visual_review(catalog, variant, "train")
    run_dir = repo_root / review["run_dir"]
    if not (run_dir / "config/effective_pipeline_config.json").is_file():
        raise FileNotFoundError(f"Snapshot da configuração ausente: {run_dir}")

    # Filtra apenas os exames de treino marcados com segmentação adequada da aorta.
    cohort = load_aorta_review_cohort(repo_root, review, "train")
    good = (
        cohort.loc[
            require_series_column(cohort, "IMG_ID").isin(
                tuple(review["aorta_good_ids"])
            )
        ]
        .sort_values("IMG_ID")
        .reset_index(drop=True)
    )
    if good.empty:
        raise ValueError(
            "A revisão não contém aortas de treino classificadas como boas."
        )

    # Seleciona IDs explícitos ou realiza amostragem reproduzível sem reposição.
    if exam_ids is not None:
        requested = list(exam_ids)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("Os IDs ImageCAS devem ser distintos e não vazios.")
        unknown = sorted(set(requested).difference(good["IMG_ID"].tolist()))
        if unknown:
            raise ValueError(f"IDs ImageCAS fora das aortas boas revisadas: {unknown}")
        good = good.set_index("IMG_ID").loc[requested].reset_index()
    elif n_exams is not None:
        if n_exams <= 0:
            raise ValueError("N_IMAGECAS deve ser positivo ou None.")
        indices = np.sort(
            np.random.default_rng(random_seed).choice(
                len(good), size=min(n_exams, len(good)), replace=False
            )
        )
        good = good.iloc[indices]
    return run_dir, good.reset_index(drop=True)


def load_mmwhs_hu_reference(record: pd.Series) -> HuReferenceVolume:
    """Carrega coração e aorta rotulados, sem interpolar os HU do MM-WHS.

    Args:
        record (pd.Series): Registro do inventário MM-WHS com ``path``,
            ``label_path`` e ``exam_id``.

    Returns:
        HuReferenceVolume: Volume em HU canônicos com máscaras de referência do
            coração completo e da aorta ascendente.
    """
    image, label, spacing = load_mmwhs_heart_volume(
        str(record["path"]), str(record["label_path"])
    )
    unknown = np.unique(label)
    unknown = unknown[~np.isin(unknown, (0, *MMWHS_HEART_LABEL_SPECS))]
    if unknown.size:
        raise ValueError(f"Labels MM-WHS desconhecidos: {unknown.tolist()}")
    regions = {
        "mmwhs_heart": np.isin(label, tuple(MMWHS_HEART_LABEL_SPECS)),
        "mmwhs_aorta": label == 820,
    }
    if not np.any(regions["mmwhs_heart"]):
        raise ValueError(f"Referência cardíaca vazia: {record['exam_id']}")
    return HuReferenceVolume(image, regions, spacing, "MM-WHS", str(record["exam_id"]))


def verify_imagecas_aorta_prediction(
    mask: NDArray[Any],
    tracking: AortaCircleTrackingResult,
    result: pd.Series,
) -> None:
    """Rejeita reconstruções divergentes das contagens persistidas do ImageCAS.

    A concordância das contagens é uma verificação de reprodução, não uma
    prova de igualdade voxel a voxel: os runs históricos não guardam a máscara.

    Args:
        mask (NDArray[Any]): Máscara binária 3D da aorta na grade pré-processada.
        tracking (AortaCircleTrackingResult): Resultado do rastreamento de círculos
            aórticos usado no level set.
        result (pd.Series): Linha persistida no CSV de resultados para o exame.
    """
    # Verifica unicidade das colunas e integridade da máscara e dos círculos.
    if not result.index.is_unique:
        raise ValueError("O resultado ImageCAS contém métricas duplicadas.")
    if mask.ndim != 3 or not np.any(mask):
        raise ValueError(
            "A reconstrução da aorta produziu uma máscara vazia ou inválida."
        )
    if not tracking.original_circles or not tracking.filtered_circles:
        raise ValueError("A reconstrução da aorta não contém círculos válidos.")

    # Recalcula contagens estruturais e intervalo de fatias da aorta reconstruída.
    circle_slices = [int(circle["slice_index"]) for circle in tracking.original_circles]
    actual = {
        "aorta_mask_voxel_count": int(np.count_nonzero(mask)),
        "aorta_circle_count": len(tracking.original_circles),
        "aorta_circle_used_count": len(tracking.filtered_circles),
        "aorta_circle_first_slice": min(circle_slices),
        "aorta_circle_last_slice": max(circle_slices),
        "aorta_segmented_slice_count": int(np.count_nonzero(np.any(mask, axis=(0, 1)))),
        "image_slice_count": mask.shape[2],
    }

    # Exige igualdade exata entre todas as contagens reconstruídas e persistidas.
    for column, measured in actual.items():
        value = result.get(column)
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"Métrica persistida ausente/inválida: {column}.")
        expected = float(value)
        if not np.isfinite(expected) or expected != measured:
            raise ValueError(
                f"Aorta ImageCAS {result.get('IMG_ID')} divergente em {column}: "
                f"reconstruído={measured}, persistido={value}."
            )


def load_imagecas_aorta_hu(
    base_path: Path, run_dir: Path, result: pd.Series
) -> HuReferenceVolume:
    """Reconstrói somente a aorta e restaura sua máscara sobre o CT original.

    Args:
        base_path (Path): Diretório raiz contendo os volumes do ImageCAS.
        run_dir (Path): Diretório do run contendo ``config/effective_pipeline_config.json``.
        result (pd.Series): Linha persistida do exame com ``IMG_ID`` e contagens da aorta.

    Returns:
        HuReferenceVolume: Volume nativo em HU e máscara da aorta em orientação RAS.
    """
    # Valida o ID do exame e carrega o volume CT original com espaçamento físico.
    if not result.index.is_unique:
        raise ValueError("O resultado ImageCAS contém métricas duplicadas.")
    id_value = result.get("IMG_ID")
    if not isinstance(id_value, (int, float, np.integer, np.floating)):
        raise ValueError("O resultado ImageCAS deve conter IMG_ID numérico.")
    numeric_id = float(id_value)
    if not np.isfinite(numeric_id) or numeric_id <= 0 or numeric_id != int(numeric_id):
        raise ValueError("IMG_ID deve ser um inteiro positivo.")
    exam_id = str(int(numeric_id))
    raw, _ = load_raw_img_and_label(str(base_path / f"{exam_id}.img.nii.gz"))
    if len(raw.shape) != 3 or raw.affine is None:
        raise ValueError(f"CT ImageCAS sem geometria 3D válida: {exam_id}")
    image = np.asarray(raw.dataobj, dtype=np.float32)
    zooms = raw.header.get_zooms()
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    config = load_config_json(
        str(run_dir / "config/effective_pipeline_config.json"), {}
    )

    # O snapshot já está escalado; não se aplica novamente a resolução.
    processed = preprocess_ccta_volume(image, spacing, config)
    tracking = locate_and_filter_aorta_circles(
        processed["lcc_image"],
        processed["downscale_factors"],
        processed["scaled_spacing"],
        config["CIRCLE_DETECTION"],
    )
    segmentation = segment_aorta_with_diagnostics(
        processed["lcc_image"],
        tracking.filtered_circles,
        config["LEVEL_SET"],
        use_gpu=False,
    )
    mask = np.asarray(segmentation.mask, dtype=np.uint8)
    if mask.shape != np.asarray(processed["lcc_image"]).shape:
        raise ValueError(
            f"Shape da máscara processada incompatível: ImageCAS {exam_id}"
        )

    # Confere reprodução pelas contagens salvas e restaura a máscara à grade nativa.
    verify_imagecas_aorta_prediction(mask, tracking, result)
    del processed, segmentation
    prediction = AortaPrediction(
        mask,
        (image.shape[0], image.shape[1], image.shape[2]),
        (),
        len(tracking.filtered_circles),
    )
    restored = restore_native_mask(prediction)

    # Imagem e máscara recebem a mesma reordenação; nenhuma recebe novos HU.
    canonical_image = as_closest_canonical(raw)
    canonical_mask = as_closest_canonical(Nifti1Image(restored, raw.affine))
    if canonical_image.shape != canonical_mask.shape or not np.allclose(
        canonical_image.affine, canonical_mask.affine, rtol=0, atol=1e-5
    ):
        raise ValueError(f"Geometria restaurada incompatível: ImageCAS {exam_id}")
    canonical_zooms = canonical_image.header.get_zooms()
    return HuReferenceVolume(
        np.asarray(canonical_image.dataobj, dtype=np.float32),
        {"imagecas_aorta": np.asarray(canonical_mask.dataobj, dtype=bool)},
        (
            float(canonical_zooms[0]),
            float(canonical_zooms[1]),
            float(canonical_zooms[2]),
        ),
        "ImageCAS",
        exam_id,
    )


def summarize_hu_reference(
    volume: HuReferenceVolume,
    *,
    percentiles: tuple[float, float] = (5, 95),
    bin_edges: NDArray[Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resume regiões de um exame usando apenas HU finitos da grade nativa.

    Args:
        volume (HuReferenceVolume): Volume canônico com imagem em HU e máscaras.
        percentiles (tuple[float, float]): Par ``(inferior, superior)`` de percentis
            extras calculados para cada região.
        bin_edges (NDArray[Any] | None): Bordas explícitas dos bins em HU; ``None``
            usa o intervalo padrão de ``summarize_hu_region``.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Par ``(statistics, histograms)`` com
            métricas descritivas e densidades por região.
    """
    # Resume cada máscara do volume preservando o nome do banco de origem.
    rows: list[dict[str, Any]] = []
    histograms: list[pd.DataFrame] = []
    for region, mask in volume.regions.items():
        row, histogram = summarize_hu_region(
            volume.image,
            mask,
            volume.spacing,
            exam_id=volume.exam_id,
            region=region,
            region_name=HU_REGION_NAMES[region],
            bin_edges=bin_edges,
            percentiles=percentiles,
        )
        row["dataset"] = volume.dataset
        rows.append(row)
        histograms.append(histogram)
    return pd.DataFrame(rows), pd.concat(histograms, ignore_index=True)


def derive_hu_intervals(
    statistics: pd.DataFrame,
    *,
    overrides: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, tuple[float, float]]:
    """Obtém limites pela média dos percentis por exame, com overrides em HU.

    Args:
        statistics (pd.DataFrame): Estatísticas por exame e região contendo as colunas
            de percentis inferior e superior.
        overrides (Mapping[str, tuple[float, float]] | None): Limites manuais
            opcionais ``(min_hu, max_hu)`` indexados pela chave da região.

    Returns:
        dict[str, tuple[float, float]]: Intervalo fechado ``(min_hu, max_hu)`` por
            região definida em ``HU_REGION_NAMES``.
    """
    # Impede regiões duplicadas por exame e valida chaves de override informadas.
    if statistics.duplicated(["exam_id", "region"]).any():
        raise ValueError("Estatísticas contêm regiões duplicadas por exame.")
    overrides = {} if overrides is None else overrides
    unknown = set(overrides).difference(HU_REGION_NAMES)
    if unknown:
        raise ValueError(f"Regiões de override desconhecidas: {sorted(unknown)}")

    # Calcula médias dos percentis entre exames válidos e aplica overrides explícitos.
    intervals: dict[str, tuple[float, float]] = {}
    for region in HU_REGION_NAMES:
        group = statistics.loc[require_series_column(statistics, "region").eq(region)]
        valid = group.loc[numeric_series(group, "valid_voxel_count").gt(0)]
        if valid.empty:
            raise ValueError(f"Nenhum exame com HU finitos contribui para {region}.")
        lower = numeric_series(valid, "lower_percentile_hu").to_numpy(dtype=float)
        upper = numeric_series(valid, "upper_percentile_hu").to_numpy(dtype=float)
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError(f"Percentis inválidos para {region}.")
        interval = overrides.get(region, (float(lower.mean()), float(upper.mean())))
        if not np.isfinite(interval).all() or interval[0] > interval[1]:
            raise ValueError(f"Intervalo HU inválido para {region}: {interval}")
        intervals[region] = (float(interval[0]), float(interval[1]))
    return intervals


def hu_interval_mask(
    image: NDArray[Any], interval: tuple[float, float]
) -> NDArray[np.bool_]:
    """Seleciona HU finitos dentro de um intervalo inclusivo válido.

    Args:
        image (NDArray[Any]): Volume 3D com intensidades em unidades Hounsfield.
        interval (tuple[float, float]): Limites inclusivos ``(inferior, superior)`` em HU.

    Returns:
        NDArray[np.bool_]: Máscara booleana dos voxels finitos dentro do intervalo.
    """
    lower, upper = interval
    if not np.isfinite(interval).all() or lower > upper:
        raise ValueError(f"Intervalo HU inválido: {interval}")
    return np.isfinite(image) & (image >= lower) & (image <= upper)


def summarize_hu_retention(
    volume: HuReferenceVolume, intervals: Mapping[str, tuple[float, float]]
) -> pd.DataFrame:
    """Mede seleção global e preservação de cada região, sem inferir segmentação.

    Args:
        volume (HuReferenceVolume): Volume canônico com imagem em HU e regiões de
            referência.
        intervals (Mapping[str, tuple[float, float]]): Mapeamento de nomes de
            intervalos para limites ``(min_hu, max_hu)``.

    Returns:
        pd.DataFrame: Tabela com contagens e percentuais de retenção global e por
            região para cada intervalo.
    """
    # Contabiliza voxels finitos globais antes de testar os intervalos individuais.
    finite = np.isfinite(volume.image)
    finite_count = int(np.count_nonzero(finite))
    rows: list[dict[str, Any]] = []

    # Avalia cada intervalo contra todas as regiões e libera a máscara intermediária.
    for interval_name, interval in intervals.items():
        selected = hu_interval_mask(volume.image, interval)
        global_count = int(np.count_nonzero(selected))
        for region, mask in volume.regions.items():
            reference_count = int(np.count_nonzero(mask & finite))
            retained = int(np.count_nonzero(mask & selected))
            rows.append(
                {
                    "dataset": volume.dataset,
                    "exam_id": volume.exam_id,
                    "interval": interval_name,
                    "region": region,
                    "finite_volume_voxels": finite_count,
                    "selected_volume_voxels": global_count,
                    "selected_volume_percent": 100 * global_count / finite_count
                    if finite_count
                    else None,
                    "finite_region_voxels": reference_count,
                    "retained_region_voxels": retained,
                    "retained_region_percent": 100 * retained / reference_count
                    if reference_count
                    else None,
                }
            )
        del selected
    return pd.DataFrame(rows)


def select_hu_example_records(
    frame: pd.DataFrame,
    id_column: str,
    *,
    requested: Sequence[str] | Sequence[int] | None = None,
    n_examples: int = 2,
) -> pd.DataFrame:
    """Seleciona exemplos distintos pertencentes à coorte já analisada.

    Args:
        frame (pd.DataFrame): DataFrame da coorte analisada.
        id_column (str): Nome da coluna identificadora dos exames.
        requested (Sequence[str] | Sequence[int] | None): IDs explícitos desejados ou
            ``None`` para selecionar os primeiros ``n_examples``.
        n_examples (int): Quantidade de exemplos retornada quando ``requested`` é
            ``None``.

    Returns:
        pd.DataFrame: Subconjunto da coorte na ordem solicitada ou inicial.
    """
    # Valida que os identificadores da coorte não sejam ausentes nem duplicados.
    ids = require_series_column(frame, id_column)
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("A coorte de exemplos contém IDs ausentes ou duplicados.")

    # Retorna os N primeiros registros ou preserva a ordem explícita pedida.
    if requested is None:
        if n_examples <= 0:
            raise ValueError("N_EXAMPLES deve ser positivo.")
        return frame.head(n_examples)
    wanted = list(requested)
    if not wanted or len(set(wanted)) != len(wanted):
        raise ValueError("Os IDs dos exemplos devem ser distintos e não vazios.")
    unknown = set(wanted).difference(ids)
    if unknown:
        raise ValueError(
            f"Exemplos fora da coorte analisada: {sorted(unknown, key=str)}"
        )
    return frame.set_index(id_column).loc[wanted].reset_index()
