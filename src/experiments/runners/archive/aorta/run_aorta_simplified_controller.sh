#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

SPLIT="${SPLIT:-train}"
AREA_RATIOS="${AREA_RATIOS:-2.6 2.7 2.8}"
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

for area_ratio in $AREA_RATIOS; do
  ratio_tag="${area_ratio/./_}"
  variant="adaptive_simplified_rp90_${ratio_tag}_p99_9"
  echo
  echo "Running split=${SPLIT}, oversegmented R_P90=${area_ratio}"

  uv run python src/segmentation_pipeline.py \
    --split "$SPLIT" \
    --split-config "$SPLIT_CONFIG" \
    --resolution mid \
    --config-file "$BASE_CONFIG" \
    --aorta-level-set-mode adaptive \
    --aorta-oversegmented-area-ratio-p90 "$area_ratio" \
    --run-group "aorta_segmentation_experiments/${SPLIT}/${variant}" \
    --num-batches 5 \
    "${VISUAL_ARGS[@]}" \
    "$GPU_ARG"
done
