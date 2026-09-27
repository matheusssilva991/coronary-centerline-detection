"""Exporta e avalia a aorta do MM-WHS test com o avaliador oficial."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from nibabel.loadsave import load as load_nifti
from nibabel.loadsave import save as save_nifti
from nibabel.nifti1 import Nifti1Image
from numpy.typing import NDArray
from scipy import ndimage as ndi

from utils.project.ccta_datasets import (
    MMWHS_AORTA_LABEL_VALUE,
    align_ccta_volume_to_imagecas_view,
    load_ccta_volume,
)
from utils.project.dataframe import require_series_column
from utils.segmentation.pipeline_detection import (
    locate_and_filter_aorta_circles,
    segment_aorta_with_diagnostics,
)
from utils.segmentation.pipeline_preprocessing import preprocess_ccta_volume

EVALUATOR_FOLDER = "MMWHS_evaluation_testdata_label_encrypt_1mm_forpublic"
OFFICIAL_AORTA_METHOD = "mmwhs_official_1mm_wine"


@dataclass(frozen=True)
class AortaPrediction:
    """Guarda a máscara processada e os dados necessários para restaurá-la."""

    mask: NDArray[np.uint8]
    native_shape: tuple[int, int, int]
    flipped_axes: tuple[int, ...]
    circle_count: int


def predict_aorta(
    record: pd.Series, config: dict[str, Any], *, align_volume: bool = True
) -> AortaPrediction:
    """Reproduz as etapas do batch até a máscara final da aorta."""
    image = load_ccta_volume(record).astype(np.float32, copy=False)
    if image.ndim != 3:
        raise ValueError("A imagem de teste deve ser tridimensional.")
    native_shape = (int(image.shape[0]), int(image.shape[1]), int(image.shape[2]))
    if align_volume:
        aligned, flipped_axes = align_ccta_volume_to_imagecas_view(image, "MM-WHS")
    else:
        aligned, flipped_axes = image, ()
    spacing_values: list[float] = []
    for column in ("spacing_x_mm", "spacing_y_mm", "spacing_z_mm"):
        value = record.get(column)
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"Espaçamento inválido no inventário: {column}.")
        numeric = float(value)
        if not np.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"Espaçamento inválido no inventário: {column}.")
        spacing_values.append(numeric)
    processed = preprocess_ccta_volume(aligned, tuple(spacing_values), config)
    lcc_image = np.asarray(processed["lcc_image"])
    downscale_factors = processed["downscale_factors"]
    scaled_spacing = tuple(float(value) for value in processed["scaled_spacing"])
    del image, aligned, processed

    tracking = locate_and_filter_aorta_circles(
        lcc_image, downscale_factors, scaled_spacing, config["CIRCLE_DETECTION"]
    )
    if not tracking.original_circles:
        raise RuntimeError("Nenhum círculo da aorta foi detectado.")
    if not tracking.filtered_circles:
        raise RuntimeError("O filtro removeu todos os círculos da aorta.")
    segmentation = segment_aorta_with_diagnostics(
        lcc_image, tracking.filtered_circles, config["LEVEL_SET"], use_gpu=False
    )
    mask = np.asarray(segmentation.mask, dtype=np.uint8)
    if mask.shape != lcc_image.shape or not np.any(mask):
        raise RuntimeError(
            "A segmentação da aorta produziu uma máscara vazia ou inválida."
        )
    return AortaPrediction(
        mask=mask,
        native_shape=native_shape,
        flipped_axes=flipped_axes,
        circle_count=len(tracking.filtered_circles),
    )


def restore_native_mask(prediction: AortaPrediction) -> NDArray[np.uint8]:
    """Reamostra por vizinho mais próximo e desfaz inversões de visualização."""
    mask = prediction.mask
    if mask.ndim != 3 or any(size <= 0 for size in prediction.native_shape):
        raise ValueError("Shape da máscara ou da imagem nativa inválido.")
    if mask.shape == prediction.native_shape:
        restored = mask
    else:
        factors = tuple(
            target / source
            for target, source in zip(prediction.native_shape, mask.shape, strict=True)
        )
        restored = np.asarray(ndi.zoom(mask, zoom=factors, order=0, prefilter=False))
    if restored.shape != prediction.native_shape:
        raise ValueError(
            f"A máscara restaurada tem shape {restored.shape}; esperado {prediction.native_shape}."
        )
    for axis in prediction.flipped_axes:
        restored = np.flip(restored, axis=axis)
    restored = np.ascontiguousarray(restored > 0, dtype=np.uint8)
    if not np.any(restored):
        raise ValueError("A máscara restaurada ficou vazia.")
    return restored


def verify_run_result(run_dir: Path, exam_id: str, prediction: AortaPrediction) -> None:
    """Confere a máscara processada com o volume persistido no run original."""
    path = run_dir / "numeric/results_all.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Resultados do run não encontrados: {path}")
    results = pd.read_csv(
        path,
        usecols=["subset", "exam_id", "aorta_mask_voxels", "aorta_circle_count"],
        dtype={"subset": "string", "exam_id": "string"},
    )
    rows = results.loc[
        require_series_column(results, "subset").eq("test")
        & require_series_column(results, "exam_id").eq(exam_id)
    ]
    if len(rows) != 1:
        raise ValueError(f"O run não contém um único resultado para {exam_id}.")
    row = rows.iloc[0]
    expected_voxels = row["aorta_mask_voxels"]
    expected_circles = row["aorta_circle_count"]
    if pd.isna(expected_voxels) or pd.isna(expected_circles):
        raise ValueError(f"O run não contém métricas válidas da aorta para {exam_id}.")
    actual = (int(np.count_nonzero(prediction.mask)), prediction.circle_count)
    expected = (int(expected_voxels), int(expected_circles))
    if actual != expected:
        raise ValueError(
            f"A máscara não reproduz o run para {exam_id}: "
            f"voxels/círculos atuais={actual}, persistidos={expected}."
        )


def save_prediction(
    reference_path: Path, native_mask: NDArray[np.uint8], output_path: Path
) -> None:
    """Salva a máscara como label 820 em NIfTI int16 na geometria original."""
    if output_path.exists():
        raise FileExistsError(f"Predição já existe: {output_path}")
    reference = load_nifti(str(reference_path))
    if not isinstance(reference, Nifti1Image):
        raise TypeError("A imagem de referência deve ser NIfTI-1.")
    if reference.affine is None:
        raise ValueError("A imagem de referência não possui affine.")
    if native_mask.shape != reference.shape or not np.any(native_mask):
        raise ValueError("A máscara nativa não coincide com a imagem ou está vazia.")
    labels = np.where(native_mask > 0, MMWHS_AORTA_LABEL_VALUE, 0).astype(np.int16)
    header = reference.header.copy()
    header.set_data_dtype(np.int16)
    prediction = Nifti1Image(labels, reference.affine, header=header)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    prediction.set_qform(qform, code=qcode)
    prediction.set_sform(sform, code=scode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_nifti(prediction, str(output_path))


def validate_saved_prediction(reference_path: Path, prediction_path: Path) -> None:
    """Confere shape, orientação e tipo antes de chamar o avaliador."""
    reference = load_nifti(str(reference_path))
    prediction = load_nifti(str(prediction_path))
    if not isinstance(reference, Nifti1Image) or not isinstance(
        prediction, Nifti1Image
    ):
        raise TypeError("Imagem e predição devem ser NIfTI-1.")
    if reference.affine is None or prediction.affine is None:
        raise ValueError("Imagem e predição devem conter affine.")
    if prediction.shape != reference.shape or not np.allclose(
        prediction.affine, reference.affine, rtol=0, atol=1e-5
    ):
        raise ValueError("A predição não preservou a geometria CT original.")
    if prediction.get_data_dtype() != np.dtype(np.int16):
        raise ValueError("O avaliador exige um label int16.")


def preflight_evaluator(evaluator_dir: Path, exam_ids: list[str]) -> None:
    """Verifica Wine, executáveis e labels antes de uma avaliação em lote."""
    if shutil.which("wine") is None or shutil.which("winepath") is None:
        raise FileNotFoundError("wine e winepath precisam estar disponíveis no PATH.")
    required = [
        evaluator_dir / "zxhtransform.exe",
        evaluator_dir / "zxhCardWhsEvaluate.exe",
        *(
            evaluator_dir / "nii" / f"{exam_id}_label_encrypt_1mm.nii.gz"
            for exam_id in exam_ids
        ),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Arquivos do avaliador ausentes: " + ", ".join(missing))


def _wine_path(path: Path) -> str:
    """Converte um caminho Linux existente ou futuro para a sintaxe do Wine."""
    converted = subprocess.run(
        ["winepath", "-w", str(path.resolve())],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not converted:
        raise RuntimeError(f"winepath não converteu o caminho: {path}")
    return converted


def parse_dice_lo(path: Path, exam_id: str) -> float:
    """Lê o DiceLO da sexta coluna do texto tabulado produzido pelo avaliador."""
    lines = [
        line.strip()
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        raise ValueError(f"Resultado oficial deve ter uma linha: {path}")
    fields = [field.strip() for field in lines[0].split("\t") if field.strip()]
    expected_id = f"ct{exam_id.rsplit('_', maxsplit=1)[-1]}"
    if len(fields) != 10 or fields[8] != expected_id or fields[9] != "--decodeseg2":
        raise ValueError(f"Identidade ou formato inválido no resultado oficial: {path}")
    try:
        scores = [float(value) for value in fields[:8]]
    except ValueError as error:
        raise ValueError(
            f"Métrica não numérica no resultado oficial: {path}"
        ) from error
    if any(not np.isfinite(value) or value < 0 or value > 1 for value in scores):
        raise ValueError(f"Métrica fora de [0, 1] no resultado oficial: {path}")
    return scores[5]


def evaluate_with_wine(
    prediction_path: Path, exam_id: str, evaluator_dir: Path, output_dir: Path
) -> Path:
    """Reamostra para 1 mm e executa o Dice oficial com o label criptografado."""
    preflight_evaluator(evaluator_dir, [exam_id])
    transform = evaluator_dir / "zxhtransform.exe"
    evaluator = evaluator_dir / "zxhCardWhsEvaluate.exe"
    encrypted = evaluator_dir / "nii" / f"{exam_id}_label_encrypt_1mm.nii.gz"
    resampled = output_dir / f"{exam_id}_label_1mm.nii.gz"
    dice_path = output_dir / f"{exam_id}_dice.xls"
    if dice_path.exists():
        parse_dice_lo(dice_path, exam_id)
        return dice_path
    if not resampled.exists():
        subprocess.run(
            [
                "wine",
                str(transform),
                _wine_path(prediction_path),
                "-o",
                _wine_path(resampled),
                "-resave",
                "-spacing",
                "1",
                "1",
                "1",
                "-nearest",
                "-v",
                "0",
            ],
            check=True,
            cwd=evaluator_dir,
        )
    if not resampled.is_file():
        raise RuntimeError("O zxhtransform não produziu o NIfTI de 1 mm.")
    subprocess.run(
        [
            "wine",
            str(evaluator),
            _wine_path(resampled),
            _wine_path(encrypted),
            "0",
            "ALL",
            _wine_path(dice_path),
            f"ct{exam_id.rsplit('_', maxsplit=1)[-1]}",
            "--decodeseg2",
        ],
        check=True,
        cwd=evaluator_dir,
    )
    if not dice_path.is_file():
        raise RuntimeError("O avaliador terminou sem gerar o arquivo de Dice.")
    parse_dice_lo(dice_path, exam_id)
    return dice_path


def has_aorta_mask(row: Mapping[str, Any]) -> bool:
    """Identifica exames com máscara final da aorta persistida no CSV."""
    value = row.get("aorta_mask_voxels")
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and np.isfinite(float(value))
        and float(value) > 0
    )
