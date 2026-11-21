"""Pareia predições do OrCaScore e resume concordância em coortes comuns."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from nibabel.filebasedimages import ImageFileError

from utils.project.analysis.artery_mask_agreement import (
    align_artery_prediction,
    artery_pair_metrics,
    binary_artery_mask,
    load_artery_prediction,
)
from utils.utils.json_io import load_json_file


def build_orcascore_pairs(
    run_dir: Path, ai_root: Path, variants: Sequence[str]
) -> pd.DataFrame:
    """Pareia as máscaras pelo ID do exame e inclui todos os IDs selecionados no run."""
    manifest = load_json_file(run_dir / "config/run_manifest.json")
    if manifest.get("dataset") != "orcascore":
        raise ValueError("O manifesto deve pertencer a um run do OrCaScore.")
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("Informe variantes de IA únicas.")
    for variant in variants:
        if not (ai_root / variant).is_dir():
            raise FileNotFoundError(f"Variante de IA ausente: {ai_root / variant}")
    cohort = pd.DataFrame(manifest["selected_exams"])[["subset", "exam_id"]]
    results = pd.read_csv(run_dir / "numeric/results_all.csv", dtype={"exam_id": str})
    fields = ["status", "error", "artery_mask_status", "artery_mask_path"]
    missing_fields = set(["subset", "exam_id", *fields]).difference(results.columns)
    if missing_fields:
        raise ValueError(
            f"Resultados sem os campos {sorted(missing_fields)}. "
            "Selecione um run com --save-artery-masks."
        )
    ostia_fields = [
        f"ostia_{side}_{axis}"
        for side in ("left", "right")
        for axis in "yxz"
        if f"ostia_{side}_{axis}" in results
    ]
    cohort = cohort.merge(
        results[["subset", "exam_id", *fields, *ostia_fields]],
        on=["subset", "exam_id"],
        how="left",
        validate="one_to_one",
    ).rename(columns={"status": "pipeline_status", "error": "pipeline_error"})
    pipeline_paths = []
    for row in cohort.itertuples(index=False):
        expected = Path(
            "masks/artery", row.subset, row.exam_id, "artery_processed.nii.gz"
        )
        if (
            row.artery_mask_status == "saved"
            and row.artery_mask_path != expected.as_posix()
        ):
            raise ValueError(
                f"Caminho de máscara incompatível com o exame {row.exam_id}."
            )
        pipeline_paths.append(
            run_dir / expected if row.artery_mask_status == "saved" else None
        )
    cohort["pipeline_path"] = pipeline_paths
    pairs = cohort.merge(pd.DataFrame({"variant": list(variants)}), how="cross")
    pairs["ai_path"] = [
        ai_root / row.variant / f"{row.exam_id}CTAI" / "pre_label.nii.gz"
        for row in pairs.itertuples(index=False)
    ]
    return pairs


def compare_orcascore_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Calcula Dice por exame e variante, registrando ausências e falhas separadamente."""
    results = []
    errors = (ValueError, ImageFileError, OSError, np.linalg.LinAlgError)
    for _, exam in pairs.groupby(["subset", "exam_id"], sort=False):
        first = exam.iloc[0]
        pipeline_path = first["pipeline_path"]
        pipeline_saved = isinstance(pipeline_path, Path) and pipeline_path.is_file()
        target = pipeline_mask = None
        target_error = None
        if pipeline_saved:
            try:
                target = load_artery_prediction(pipeline_path)
                pipeline_mask = binary_artery_mask(target)
            except errors as error:
                target_error = str(error)
        for pair in exam.to_dict(orient="records"):
            ai_path = pair["ai_path"]
            row = {
                **pair,
                "pipeline_available": pipeline_saved,
                "ai_available": ai_path.is_file(),
                "dice": np.nan,
                "pipeline_volume_ml": np.nan,
                "ai_volume_ml": np.nan,
                "ai_source_voxels": np.nan,
                "ai_aligned_voxels": np.nan,
                "resampled": None,
                "detail": None,
            }
            if not pipeline_saved:
                row["status"] = "pipeline_missing"
            elif target is None or pipeline_mask is None:
                row.update(status="invalid_pipeline_mask", detail=target_error)
            elif not row["ai_available"]:
                row["status"] = "ai_missing"
            else:
                try:
                    source = load_artery_prediction(ai_path)
                    aligned, resampled, source_voxels = align_artery_prediction(
                        source, target
                    )
                    affine = np.asarray(target.affine)
                    voxel_volume = float(abs(np.linalg.det(affine[:3, :3])))
                    row.update(
                        artery_pair_metrics(
                            pipeline_mask, aligned, source_voxels, voxel_volume
                        )
                    )
                    row["resampled"] = resampled
                except errors as error:
                    row.update(status="invalid_ai_mask_or_grid", detail=str(error))
            results.append(row)
    return pd.DataFrame.from_records(results)


def common_orcascore_dice(results: pd.DataFrame) -> pd.DataFrame:
    """Seleciona exames com Dice definido para todas as variantes configuradas."""
    variants = results["variant"].nunique()
    counts = results.groupby(["subset", "exam_id"])["dice"].transform("count")
    return results.loc[counts.eq(variants)].copy()


def summarize_orcascore_agreement(
    results: pd.DataFrame, common: pd.DataFrame
) -> pd.DataFrame:
    """Resume cobertura e Dice sobre os mesmos exames por variante e subconjunto."""
    tables = []
    for subset in ("train", "test", "Geral"):
        rows = (
            results if subset == "Geral" else results.loc[results["subset"].eq(subset)]
        )
        shared = (
            common if subset == "Geral" else common.loc[common["subset"].eq(subset)]
        )
        if rows.empty:
            continue
        coverage = rows.groupby("variant", sort=False).agg(
            selected=("exam_id", "size"),
            pipeline_masks=("pipeline_available", "sum"),
            ai_masks=("ai_available", "sum"),
            dice_available=("dice", "count"),
        )
        stats = shared.groupby("variant")["dice"].agg(
            common_exams="count", dice_mean="mean", dice_median="median", dice_std="std"
        )
        table = coverage.join(stats)
        table["common_exams"] = table["common_exams"].fillna(0).astype(int)
        tables.append(table.assign(subset=subset).reset_index())
    return pd.concat(tables, ignore_index=True)


def summarize_orcascore_dice_histogram(
    results: pd.DataFrame, bins: int = 10
) -> pd.DataFrame:
    """Conta Dice válidos em intervalos uniformes de zero a um por subconjunto."""
    if bins < 1:
        raise ValueError("O histograma exige pelo menos um intervalo.")
    edges = np.arange(bins + 1) / bins
    counts = {
        label: np.histogram(values.dropna().to_numpy(dtype=float), bins=edges)[0]
        for label, values in (
            ("Treino", results.loc[results["subset"].eq("train"), "dice"]),
            ("Teste", results.loc[results["subset"].eq("test"), "dice"]),
            ("Geral", results["dice"]),
        )
    }
    return pd.DataFrame({"Início": edges[:-1], "Fim": edges[1:], **counts})


def select_orcascore_dice_case(
    results: pd.DataFrame,
    target: float | None = None,
    *,
    preferred_exam_id: str | None = None,
) -> pd.Series | None:
    """Seleciona o maior Dice ou o caso mais próximo do valor solicitado.

    Um exame preferido tem prioridade quando está a até 0,01 do alvo.
    Empates são resolvidos pelo ID. Retorna ``None`` se não houver Dice válido.
    """
    valid = results.dropna(subset=["dice"]).sort_values("exam_id")
    if valid.empty:
        return None
    if target is None:
        return valid.nlargest(1, "dice").iloc[0]

    distance = valid["dice"].sub(target).abs()
    if preferred_exam_id is not None:
        preferred = valid.loc[
            valid["exam_id"].eq(preferred_exam_id) & distance.le(0.01)
        ]
        if not preferred.empty:
            return preferred.iloc[0]
    return valid.iloc[distance.argmin()]
