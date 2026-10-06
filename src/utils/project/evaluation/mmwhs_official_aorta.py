"""Exporta e avalia a aorta do MM-WHS test com o avaliador oficial."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
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

from utils.project.datasets.ccta import (
    MMWHS_AORTA_LABEL_VALUE,
    align_ccta_volume_to_imagecas_view,
    load_ccta_volume,
)
from utils.project.dataframe import require_series_column
from utils.segmentation.pipeline.detection import (
    locate_and_filter_aorta_circles,
    segment_aorta_with_diagnostics,
)
from utils.segmentation.pipeline.preprocessing import preprocess_ccta_volume

EVALUATOR_FOLDER = "MMWHS_evaluation_testdata_label_encrypt_1mm_forpublic"
LEGACY_WHS_METHOD = "mmwhs_official_1mm_wine"
OFFICIAL_AORTA_METHOD = "mmwhs_aorta_label820_1mm_wine"


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
    """Lê exclusivamente o DiceLO legado do protocolo de coração completo."""
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


def aorta_dice_result_path(output_dir: Path, exam_id: str) -> Path:
    """Resolve o artefato do Dice isolado, separado do XLS legado."""
    return output_dir / f"{exam_id}_aorta_dice.json"


def parse_aorta_dice(path: Path, exam_id: str) -> float:
    """Valida identidade, protocolo e Dice do resultado da aorta isolada."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or (
        payload.get("exam_id") != exam_id
        or payload.get("method") != OFFICIAL_AORTA_METHOD
        or payload.get("label") != MMWHS_AORTA_LABEL_VALUE
    ):
        raise ValueError(f"Identidade ou protocolo inválido no Dice da aorta: {path}")
    value = payload.get("dice")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Dice da aorta não numérico: {path}")
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Dice da aorta fora de [0, 1]: {path}")
    return float(value)


def parse_label_dice_output(stdout: str) -> float:
    """Extrai exatamente um escore da saída textual do modo por label."""
    tokens = [
        line.strip()
        for line in stdout.splitlines()
        if re.fullmatch(
            r"[-+]?(?:\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|\.\d+|nan|inf(?:inity)?|1\.#\w+)",
            line.strip(),
            re.IGNORECASE,
        )
    ]
    if len(tokens) != 1:
        raise ValueError("O avaliador deve retornar exatamente um Dice numérico.")
    try:
        value = float(tokens[0])
    except ValueError as error:
        raise ValueError("O avaliador retornou um Dice inválido.") from error
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("O avaliador retornou Dice fora de [0, 1].")
    return value


def _save_aorta_dice(path: Path, exam_id: str, dice: float) -> None:
    """Persiste o resultado validado com substituição atômica."""
    payload = {
        "exam_id": exam_id,
        "method": OFFICIAL_AORTA_METHOD,
        "label": MMWHS_AORTA_LABEL_VALUE,
        "dice": dice,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def evaluate_with_wine(
    prediction_path: Path, exam_id: str, evaluator_dir: Path, output_dir: Path
) -> Path:
    """Reamostra para 1 mm e compara somente o label 820 com a referência."""
    preflight_evaluator(evaluator_dir, [exam_id])
    transform = evaluator_dir / "zxhtransform.exe"
    evaluator = evaluator_dir / "zxhCardWhsEvaluate.exe"
    encrypted = evaluator_dir / "nii" / f"{exam_id}_label_encrypt_1mm.nii.gz"
    resampled = output_dir / f"{exam_id}_label_1mm.nii.gz"
    dice_path = aorta_dice_result_path(output_dir, exam_id)
    if dice_path.exists():
        parse_aorta_dice(dice_path, exam_id)
        return dice_path
    output_dir.mkdir(parents=True, exist_ok=True)
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
    evaluation = subprocess.run(
        [
            "wine",
            str(evaluator),
            "-label",
            "820",
            "820",
            _wine_path(resampled),
            _wine_path(encrypted),
            "0",
            "1",
            "--decodeseg2",
        ],
        check=True,
        cwd=evaluator_dir,
        capture_output=True,
        text=True,
    )
    dice = parse_label_dice_output(evaluation.stdout)
    _save_aorta_dice(dice_path, exam_id, dice)
    return dice_path


def has_aorta_mask(row: Mapping[str, Any]) -> bool:
    """Identifica exames com máscara final da aorta persistida no CSV."""
    value = row.get("aorta_mask_voxels")
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and np.isfinite(float(value))
        and float(value) > 0
    )
