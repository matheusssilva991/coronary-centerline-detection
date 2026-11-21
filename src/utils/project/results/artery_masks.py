"""Exporta a previsão coronária na grade em que o pipeline a calculou."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from utils.project.results.columns import ARTERY_MASK_EXPORT_COLUMNS
from utils.utils.json_io import save_json_atomic

logger = logging.getLogger(__name__)
MASK_FILENAME = "artery_processed.nii.gz"
MANIFEST_FILENAME = "metadata.json"


def add_artery_mask_export_argument(parser: argparse.ArgumentParser) -> None:
    """Registra a opção de exportação arterial, desligada em runs novos por padrão."""
    parser.add_argument(
        "--save-artery-masks",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Salva a máscara arterial final na grade processada. Desativado por "
            "padrão; retomadas herdam a opção do run."
        ),
    )


def pending_artery_mask_export() -> dict[str, str | None]:
    """Inicializa os campos de exportação antes da segmentação arterial."""
    return dict.fromkeys(ARTERY_MASK_EXPORT_COLUMNS) | {
        "artery_mask_status": "not_generated"
    }


def _reference_geometry(
    path: Path,
) -> tuple[tuple[int, ...], np.ndarray, nib.Nifti1Header]:
    if path.name.lower().endswith((".nii", ".nii.gz")):
        image = nib.load(path)
        if image.affine is None:
            raise ValueError("O NIfTI de entrada não possui affine.")
        return (
            tuple(image.shape),
            image.affine,
            nib.Nifti1Header.from_header(image.header),
        )

    import SimpleITK as sitk

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    if reader.GetDimension() != 3:
        raise ValueError("O volume de entrada deve ser tridimensional.")
    affine = np.eye(4)
    affine[:3, :3] = np.asarray(reader.GetDirection()).reshape(3, 3) @ np.diag(
        reader.GetSpacing()
    )
    affine[:3, 3] = reader.GetOrigin()
    # A geometria do MetaImage é lida em LPS e o affine NIfTI é expresso em RAS.
    affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine
    header = nib.Nifti1Header()
    header.set_xyzt_units("mm")
    return tuple(reader.GetSize()), affine, header


def _processed_to_native(
    original_shape: Sequence[int],
    processed_shape: Sequence[int],
    method: str,
    flipped_axes: Sequence[int],
) -> np.ndarray:
    if method not in ("opencv", "scipy"):
        raise ValueError(f"Método de downscale não suportado: {method}")
    transform = np.eye(4)
    for axis, (original, processed) in enumerate(
        zip(original_shape, processed_shape, strict=True)
    ):
        if method == "opencv":
            # O resize trabalha com a grade de destino alinhada nas bordas.
            scale = original / processed
            transform[axis, 3] = (scale - 1.0) / 2.0
        else:
            # O zoom do SciPy alinha os centros dos voxels nas extremidades.
            scale = (original - 1) / (processed - 1) if processed > 1 else 1.0
        transform[axis, axis] = scale
    for axis in flipped_axes:
        transform[axis, :] *= -1
        transform[axis, 3] += original_shape[axis] - 1
    return transform


@dataclass(frozen=True)
class ArteryMaskExport:
    """Identifica a coorte de destino para a exportação de máscaras arteriais."""

    run_dir: Path
    dataset: str
    split: str
    resolution: str

    def save(
        self,
        exam_id: str,
        mask: np.ndarray,
        reference_path: str | Path,
        config: Mapping[str, Any],
        *,
        flipped_axes: Sequence[int] = (),
    ) -> dict[str, str | None]:
        """Salva a previsão final em (x, y, z) e retorna os campos do CSV.

        O NIfTI binário preserva os voxels, inclusive para uma predição vazia.
        As inversões aplicadas à entrada são representadas no affine. Falhas
        de exportação são registradas separadamente do resultado científico.
        """
        result = pending_artery_mask_export()
        relative_dir = Path("masks/artery") / self.split / exam_id
        try:
            self._write(
                exam_id,
                mask,
                Path(reference_path),
                config,
                flipped_axes,
                self.run_dir / relative_dir,
            )
        except Exception as error:
            logger.exception(
                "Falha ao salvar a máscara arterial de %s/%s", self.split, exam_id
            )
            result.update(artery_mask_status="error", artery_mask_error=str(error))
        else:
            result.update(
                artery_mask_status="saved",
                artery_mask_path=str(relative_dir / MASK_FILENAME),
                artery_mask_metadata_path=str(relative_dir / MANIFEST_FILENAME),
            )
        return result

    def _write(
        self,
        exam_id: str,
        mask: np.ndarray,
        reference_path: Path,
        config: Mapping[str, Any],
        flipped_axes: Sequence[int],
        output_dir: Path,
    ) -> None:
        array = np.asarray(mask)
        if array.ndim != 3 or not array.size or not ((array == 0) | (array == 1)).all():
            raise ValueError("A máscara arterial deve ser um volume binário 3D.")
        original_shape, original_affine, header = _reference_geometry(reference_path)
        if len(original_shape) != 3:
            raise ValueError("O volume de entrada deve ser tridimensional.")
        method = config["DOWNSCALE_METHOD"]
        transform = _processed_to_native(
            original_shape, array.shape, method, flipped_axes
        )
        processed_affine = original_affine @ transform
        header.set_data_dtype(np.uint8)
        header.set_intent("label")
        prediction = nib.Nifti1Image(
            array.astype(np.uint8, copy=False), processed_affine, header
        )
        prediction.set_sform(
            processed_affine, code=int(prediction.header["sform_code"]) or 1
        )
        metadata = {
            "schema_version": 1,
            "dataset": self.dataset,
            "split": self.split,
            "exam_id": exam_id,
            "resolution": self.resolution,
            "stage": "after_morphology",
            "source_filename": reference_path.name,
            "axis_order": "xyz",
            "foreground_label": 1,
            "original_shape": list(original_shape),
            "processed_shape": list(array.shape),
            "foreground_voxels": int(np.count_nonzero(array)),
            "visual_flip_axes": list(flipped_axes),
            "downscale_method": method,
            "opencv_interpolation": config.get("OPENCV_INTERPOLATION")
            if method == "opencv"
            else None,
            "processed_to_original_index": transform.tolist(),
            "original_affine_ras": original_affine.tolist(),
            "processed_affine_ras": processed_affine.tolist(),
            "effective_config_run_relative": "config/effective_pipeline_config.json",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=output_dir, prefix=".artery-"
        ) as temporary:
            staging = Path(temporary)
            mask_path = staging / MASK_FILENAME
            nib.save(prediction, mask_path)
            with mask_path.open("rb") as stream:
                metadata["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            save_json_atomic(metadata, staging / MANIFEST_FILENAME)
            # Publica o manifesto depois do NIfTI para sinalizar a exportação concluída.
            (output_dir / MANIFEST_FILENAME).unlink(missing_ok=True)
            os.replace(mask_path, output_dir / MASK_FILENAME)
            os.replace(staging / MANIFEST_FILENAME, output_dir / MANIFEST_FILENAME)

    def has_saved_prediction(self, exam_id: str) -> bool:
        """Confere se o NIfTI e o manifesto foram publicados para o exame."""
        directory = self.run_dir / "masks/artery" / self.split / exam_id
        return all(
            (directory / name).is_file() for name in (MASK_FILENAME, MANIFEST_FILENAME)
        )
