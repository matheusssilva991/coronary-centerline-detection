"""Exporta e avalia a aorta do MM-WHS test com o avaliador oficial."""

from __future__ import annotations

import json
import logging
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

from utils.project.dataframe import require_series_column
from utils.project.datasets.ccta import (
    MMWHS_AORTA_LABEL_VALUE,
    align_ccta_volume_to_imagecas_view,
    load_ccta_volume,
)
from utils.segmentation.pipeline.detection import (
    locate_and_filter_aorta_circles,
    segment_aorta_with_diagnostics,
)
from utils.segmentation.pipeline.preprocessing import preprocess_ccta_volume

EVALUATOR_FOLDER = "MMWHS_evaluation_testdata_label_encrypt_1mm_forpublic"
LEGACY_WHS_METHOD = "mmwhs_official_1mm_wine"
OFFICIAL_AORTA_METHOD = "mmwhs_aorta_label820_1mm_wine"

LOGGER = logging.getLogger(__name__)


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
    """Reproduz o batch até a máscara final da aorta em um exame MM-WHS.

    Args:
        record (pd.Series): Linha do inventário CCTA com caminho e espaçamentos
            em ``mm``.
        config (dict[str, Any]): Configuração efetiva normalizada e já escalada
            para a resolução do run.
        align_volume (bool): Se ``True``, aplica o alinhamento visual
            MM-WHS → ImageCAS antes de segmentar.

    Returns:
        AortaPrediction: Estrutura com máscara binária, shape nativo e eixos
            invertidos para restauração.
    """
    # Etapa 1: carrega o volume nativo, valida espaçamento e alinha se ativo.
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
    # Etapa 2: pré-processa na resolução do run e libera buffers de entrada.
    processed = preprocess_ccta_volume(aligned, tuple(spacing_values), config)
    lcc_image = np.asarray(processed["lcc_image"])
    downscale_factors = processed["downscale_factors"]
    scaled_spacing = tuple(float(value) for value in processed["scaled_spacing"])
    del image, aligned, processed

    # Etapa 3: rastreia círculos, segmenta a aorta e guarda orientação nativa.
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
    """Reamostra a máscara por vizinho mais próximo e desfaz inversões visuais.

    Args:
        prediction (AortaPrediction): Predição produzida por ``predict_aorta`` com
            shape nativo e eixos invertidos.

    Returns:
        NDArray[np.uint8]: Máscara binária ``uint8`` não vazia restaurada na
            grade nativa do volume MM-WHS.
    """
    # Reamostra a máscara binária por vizinho mais próximo para a grade nativa.
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
    # Desfaz as inversões de alinhamento visual e confirma máscara binária não vazia.
    for axis in prediction.flipped_axes:
        restored = np.flip(restored, axis=axis)
    restored = np.ascontiguousarray(restored > 0, dtype=np.uint8)
    if not np.any(restored):
        raise ValueError("A máscara restaurada ficou vazia.")
    return restored


def verify_run_result(run_dir: Path, exam_id: str, prediction: AortaPrediction) -> None:
    """Confere se a máscara gerada reproduz voxels e círculos salvos no run original.

    Args:
        run_dir (Path): Diretório raiz do run MM-WHS com ``numeric/results_all.csv``.
        exam_id (str): Identificador do exame no subconjunto ``test``.
        prediction (AortaPrediction): Predição recém-gerada por ``predict_aorta``
            para validação de consistência.
    """
    # Lê o CSV consolidado do run e localiza a linha única do exame no conjunto test.
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
    # Compara voxels e círculos da predição atual com os valores persistidos.
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
    """Salva a máscara como label 820 em NIfTI ``int16`` mantendo geometria.

    Args:
        reference_path (Path): Caminho do NIfTI CT original usado como referência.
        native_mask (NDArray[np.uint8]): Máscara binária 3D na grade nativa de
            ``reference_path``.
        output_path (Path): Caminho do arquivo NIfTI de destino (falha se já existir).
    """
    # Carrega o CT de referência e valida geometria e máscara antes de gerar o rótulo.
    if output_path.exists():
        raise FileExistsError(f"Predição já existe: {output_path}")
    reference = load_nifti(str(reference_path))
    if not isinstance(reference, Nifti1Image):
        raise TypeError("A imagem de referência deve ser NIfTI-1.")
    if reference.affine is None:
        raise ValueError("A imagem de referência não possui affine.")
    if native_mask.shape != reference.shape or not np.any(native_mask):
        raise ValueError("A máscara nativa não coincide com a imagem ou está vazia.")
    # Codifica a aorta como label 820 em int16 preservando affine, qform e sform.
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
    """Confere shape, affine e dtype ``int16`` da predição antes de acionar o avaliador.

    Args:
        reference_path (Path): Caminho do volume CT original NIfTI-1.
        prediction_path (Path): Caminho do NIfTI-1 de rótulo salvo para avaliação.
    """
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
    """Verifica Wine, executáveis e rótulos cifrados antes de avaliar o lote.

    Args:
        evaluator_dir (Path): Pasta do avaliador MM-WHS contendo ``.exe`` e ``nii/``.
        exam_ids (list[str]): Lista de IDs de exames ``test`` com rótulos cifrados.
    """
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
    """Converte um caminho Linux existente ou futuro para a sintaxe do Wine.

    Args:
        path (Path): Caminho Linux a converter para o formato Windows do Wine.

    Returns:
        str: Caminho convertido pelo comando ``winepath -w``.
    """
    converted = subprocess.run(
        ["winepath", "-w", str(path.resolve())],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not converted:
        raise RuntimeError(f"winepath não converteu o caminho: {path}")
    return converted


def aorta_dice_result_path(output_dir: Path, exam_id: str) -> Path:
    """Resolve o caminho do JSON do Dice isolado, separado do XLS legado.

    Args:
        output_dir (Path): Diretório de saída por exame da avaliação da aorta.
        exam_id (str): Identificador do exame MM-WHS.

    Returns:
        Path: Caminho ``<output_dir>/<exam_id>_aorta_dice.json``.
    """
    return output_dir / f"{exam_id}_aorta_dice.json"


def parse_aorta_dice(path: Path, exam_id: str) -> float:
    """Valida identidade, protocolo e valor de Dice no artefato JSON da aorta isolada.

    Args:
        path (Path): Caminho do arquivo ``*_aorta_dice.json`` persistido.
        exam_id (str): Identificador esperado do exame (validação de correspondência).

    Returns:
        float: Valor de Dice validado no intervalo ``[0, 1]``.
    """
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
    """Extrai exatamente um escore Dice da saída textual do avaliador por label.

    Args:
        stdout (str): Saída padrão capturada da execução de ``zxhCardWhsEvaluate.exe``.

    Returns:
        float: Valor de Dice validado no intervalo ``[0, 1]``.
    """
    # Filtra linhas numéricas ou tokens especiais na saída do avaliador.
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
    # Valida conversão numérica única e garante escore finito dentro de [0, 1].
    try:
        value = float(tokens[0])
    except ValueError as error:
        raise ValueError("O avaliador retornou um Dice inválido.") from error
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("O avaliador retornou Dice fora de [0, 1].")
    return value


def _save_aorta_dice(path: Path, exam_id: str, dice: float) -> None:
    """Persiste o resultado validado com substituição atômica.

    Args:
        path (Path): Caminho de destino do arquivo JSON de Dice da aorta.
        exam_id (str): Identificador do exame MM-WHS.
        dice (float): Valor de Dice validado no intervalo ``[0, 1]``.
    """
    # Monta o payload oficial do label 820 e garante que o diretório exista.
    payload = {
        "exam_id": exam_id,
        "method": OFFICIAL_AORTA_METHOD,
        "label": MMWHS_AORTA_LABEL_VALUE,
        "dice": dice,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    # Grava em arquivo temporário com fsync e substitui atomicamente no destino final.
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
    """Reamostra a predição para 1 mm via Wine e avalia somente o label 820.

    Args:
        prediction_path (Path): Caminho do NIfTI ``int16`` nativo predito.
        exam_id (str): Identificador do exame MM-WHS ``test`` usado para localizar
            o gabarito cifrado.
        evaluator_dir (Path): Pasta com ``zxhtransform.exe``,
            ``zxhCardWhsEvaluate.exe`` e ``nii/``.
        output_dir (Path): Diretório do exame onde o volume reamostrado e o
            ``*_aorta_dice.json`` são gravados.

    Returns:
        Path: Caminho do arquivo JSON contendo o Dice oficial do label 820.
    """
    # Confere pré-requisitos do Wine e reutiliza JSON oficial já validado, se existir.
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
    # Reamostra a predição para 1 mm isotrópico antes de invocar o executável cifrado.
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
    # Avalia só o label 820 (aorta ascendente) e grava JSON separado do XLS legado.
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
    """Identifica exames com máscara final não vazia da aorta registrada no CSV.

    Args:
        row (Mapping[str, Any]): Registro tabular com a coluna
            ``aorta_mask_voxels``.

    Returns:
        bool: ``True`` se ``aorta_mask_voxels`` for finito e estritamente positivo.
    """
    value = row.get("aorta_mask_voxels")
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and np.isfinite(float(value))
        and float(value) > 0
    )


def evaluate_test_aorta_result(
    row: Mapping[str, Any],
    record: pd.Series,
    config: dict[str, Any],
    run_dir: Path,
    evaluator_dir: Path,
    *,
    rebuild_mask: bool,
    align_volume: bool = True,
) -> dict[str, Any]:
    """Avalia a aorta de um exame ``test`` sem alterar o resultado científico.

    Args:
        row (Mapping[str, Any]): Registro numérico atual do exame no run.
        record (pd.Series): Linha do inventário CCTA com caminho e espaçamentos
            em ``mm``.
        config (dict[str, Any]): Configuração efetiva normalizada e escalada do run.
        run_dir (Path): Diretório raiz do run MM-WHS.
        evaluator_dir (Path): Pasta com os binários oficiais e rótulos cifrados.
        rebuild_mask (bool): Se ``True``, reexecuta ``predict_aorta`` quando o
            NIfTI nativo ainda não foi exportado.
        align_volume (bool): Se ``True``, aplica alinhamento visual
            MM-WHS → ImageCAS ao reconstruir a máscara.

    Returns:
        dict[str, Any]: Cópia atualizada de ``row`` com ``aorta_dice``,
            ``aorta_evaluation_status``, ``aorta_ground_truth_evaluated``,
            ``aorta_evaluation_method`` e ``aorta_evaluation_error``.
    """
    updated = dict(row)
    exam_id = str(record["exam_id"])
    updated["aorta_evaluation_method"] = OFFICIAL_AORTA_METHOD
    updated["aorta_ground_truth_available"] = True
    updated["aorta_evaluation_error"] = None
    # Interrompe cedo quando o exame não produziu máscara da aorta no run.
    if not has_aorta_mask(row):
        updated["aorta_evaluation_status"] = "unavailable"
        updated["aorta_ground_truth_evaluated"] = False
        updated["aorta_dice"] = None
        return updated

    output_dir = run_dir / "evaluation" / "aorta" / "test" / exam_id
    prediction_path = output_dir / f"{exam_id}_label.nii.gz"
    try:
        # Etapa 1: reaproveita NIfTI exportado ou reconstrói a máscara nativa.
        reference_path = Path(str(record["path"]))
        if prediction_path.exists():
            validate_saved_prediction(reference_path, prediction_path)
        elif rebuild_mask:
            prediction = predict_aorta(
                record, {**config, "USE_GPU": False}, align_volume=align_volume
            )
            verify_run_result(run_dir, exam_id, prediction)
            native_mask = restore_native_mask(prediction)
            save_prediction(reference_path, native_mask, prediction_path)
            del native_mask, prediction
        else:
            raise FileNotFoundError(f"Predição da aorta ausente: {prediction_path}")
        # Etapa 2: calcula e persiste o Dice oficial do label 820 isoladamente.
        dice_path = evaluate_with_wine(
            prediction_path, exam_id, evaluator_dir, output_dir
        )
        updated["aorta_dice"] = parse_aorta_dice(dice_path, exam_id)
        updated["aorta_ground_truth_evaluated"] = True
        updated["aorta_evaluation_status"] = "success"
    except Exception as error:
        # Registra falha isolada da avaliação oficial sem descartar resultados do exame.
        LOGGER.exception("Falha na avaliação oficial da aorta de %s", exam_id)
        updated["aorta_dice"] = None
        updated["aorta_ground_truth_evaluated"] = False
        updated["aorta_evaluation_status"] = "error"
        updated["aorta_evaluation_error"] = str(error)
    return updated


def official_evaluation_complete(row: Mapping[str, Any], run_dir: Path) -> bool:
    """Confere se o Dice persistido na linha coincide com o arquivo oficial do exame.

    Args:
        row (Mapping[str, Any]): Registro numérico do exame no CSV consolidado.
        run_dir (Path): Diretório raiz do run contendo
            ``evaluation/aorta/test/<exam_id>/``.

    Returns:
        bool: ``True`` quando a avaliação oficial já foi concluída e validada
            com consistência numérica.
    """
    from utils.project.results.external import has_valid_aorta_dice

    # Verifica primeiro se o registro no CSV já indica conclusão oficial válida.
    if row.get("aorta_evaluation_method") != OFFICIAL_AORTA_METHOD:
        return False
    if row.get("aorta_evaluation_status") != "success" or not has_valid_aorta_dice(row):
        return False
    # Confirma a existência do JSON oficial e a igualdade numérica com o CSV.
    exam_id = str(row["exam_id"])
    dice_path = aorta_dice_result_path(
        run_dir / "evaluation" / "aorta" / "test" / exam_id, exam_id
    )
    if not dice_path.is_file():
        return False
    saved_dice = parse_aorta_dice(dice_path, exam_id)
    if not np.isclose(saved_dice, float(row["aorta_dice"]), rtol=0, atol=1e-6):
        raise ValueError(f"Dice do CSV diverge do arquivo oficial: {exam_id}")
    return True
