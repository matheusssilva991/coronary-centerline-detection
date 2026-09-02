#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

SPLIT="${SPLIT:-train}"
USE_GPU="${USE_GPU:-1}"
SAVE_VISUALS="${SAVE_VISUALS:-1}"
VISUAL_OUTPUT_DIR="${VISUAL_OUTPUT_DIR:-/media/matheus/HD/ImageCAS_pipeline_results}"
BASE_CONFIG="${BASE_CONFIG:-config/aorta_filter_envelope_generalization.json}"
NUM_BATCHES="${NUM_BATCHES:-5}"

case "$SPLIT" in
  train) SPLIT_CONFIG="${SPLIT_CONFIG:-config/imagecas_splits.json}" ;;
  val) SPLIT_CONFIG="${SPLIT_CONFIG:-config/imagecas_splits_val60.json}" ;;
  *)
    echo "SPLIT deve ser train ou val; não ajuste o método no conjunto de teste." >&2
    exit 2
    ;;
esac

GPU_ARGS=(--gpu)
[[ "$USE_GPU" == "0" ]] && GPU_ARGS=(--no-gpu)

VISUAL_ARGS=()
if [[ "$SAVE_VISUALS" == "1" ]]; then
  VISUAL_ARGS+=(--save-segmentation-visuals)
  [[ -n "$VISUAL_OUTPUT_DIR" ]] && VISUAL_ARGS+=(--visual-output-dir "$VISUAL_OUTPUT_DIR")
fi

run_selected_configuration() {
  echo
  echo "Executando configuração selecionada: Hough 18-29 px, envelope=2.25r, margem=10"
  uv run python src/segmentation_pipeline.py \
    --split "$SPLIT" \
    --split-config "$SPLIT_CONFIG" \
    --resolution mid \
    --config-file "$BASE_CONFIG" \
    --aorta-hough-radii-start-px 18 \
    --aorta-hough-radii-end-px 30 \
    --aorta-circle-filter robust \
    --aorta-circle-filter-min-coverage 0.40 \
    --aorta-circle-filter-max-trim-fraction 0.40 \
    --aorta-circle-filter-synthetic-tail-slices 5 \
    --aorta-circle-filter-mask-guided \
    --aorta-trajectory-radius-factor 2.25 \
    --aorta-trajectory-axial-margin-slices 10 \
    --run-group \
      "aorta_segmentation_experiments/${SPLIT}/selected_hough18_29_filter_envelope_p99_9_m300" \
    --num-batches "$NUM_BATCHES" \
    "${VISUAL_ARGS[@]}" \
    "${GPU_ARGS[@]}"
}

run_selected_configuration
