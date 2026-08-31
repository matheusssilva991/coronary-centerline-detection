#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

SPLIT="${SPLIT:-train}"
FACTORS="${FACTORS:-1.75 2.0 2.25}"
USE_GPU="${USE_GPU:-1}"
SAVE_VISUALS="${SAVE_VISUALS:-1}"
VISUAL_OUTPUT_DIR="${VISUAL_OUTPUT_DIR:-}"
BASE_CONFIG="${BASE_CONFIG:-config/article_cbeb_sensitivity.json}"

case "$SPLIT" in
  train) SPLIT_CONFIG="${SPLIT_CONFIG:-config/imagecas_splits.json}" ;;
  val) SPLIT_CONFIG="${SPLIT_CONFIG:-config/imagecas_splits_val60.json}" ;;
  *)
    echo "SPLIT deve ser train ou val; o conjunto de teste não deve ser usado no ajuste." >&2
    exit 2
    ;;
esac

GPU_ARG="--gpu"
[[ "$USE_GPU" == "0" ]] && GPU_ARG="--no-gpu"

VISUAL_ARGS=()
[[ "$SAVE_VISUALS" == "1" ]] && VISUAL_ARGS+=(--save-segmentation-visuals)
[[ -n "$VISUAL_OUTPUT_DIR" ]] && VISUAL_ARGS+=(--visual-output-dir "$VISUAL_OUTPUT_DIR")

for factor in $FACTORS; do
  factor_tag="${factor/./_}"
  variant="trajectory_envelope_k${factor_tag}_p99_9"
  echo
  echo "Running split=${SPLIT}, trajectory_radius_factor=${factor}"

  uv run python src/segmentation_pipeline.py \
    --split "$SPLIT" \
    --split-config "$SPLIT_CONFIG" \
    --resolution mid \
    --config-file "$BASE_CONFIG" \
    --aorta-level-set-mode fixed \
    --aorta-trajectory-radius-factor "$factor" \
    --run-group "aorta_segmentation_experiments/${SPLIT}/${variant}" \
    --num-batches 5 \
    "${VISUAL_ARGS[@]}" \
    "$GPU_ARG"
done
