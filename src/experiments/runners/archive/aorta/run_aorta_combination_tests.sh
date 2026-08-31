#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

SPLIT="${SPLIT:-train}"
VARIANTS="${VARIANTS:-envelope_corrected aggressive_simplified filtered_envelope combined}"
USE_GPU="${USE_GPU:-1}"
SAVE_VISUALS="${SAVE_VISUALS:-1}"
VISUAL_OUTPUT_DIR="${VISUAL_OUTPUT_DIR:-/media/matheus/HD/ImageCAS_pipeline_results}"
BASE_CONFIG="${BASE_CONFIG:-config/article_cbeb_sensitivity.json}"
TRAJECTORY_RADIUS_FACTOR="${TRAJECTORY_RADIUS_FACTOR:-2.25}"
TRAJECTORY_AXIAL_MARGIN="${TRAJECTORY_AXIAL_MARGIN:-5}"
AREA_RATIO_P90="${AREA_RATIO_P90:-2.7}"
FILTER_COVERAGE="${FILTER_COVERAGE:-0.40}"

case "$SPLIT" in
  train) SPLIT_CONFIG="${SPLIT_CONFIG:-config/imagecas_splits.json}" ;;
  val) SPLIT_CONFIG="${SPLIT_CONFIG:-config/imagecas_splits_val60.json}" ;;
  *)
    echo "SPLIT deve ser train ou val; o conjunto de teste não deve ser usado no ajuste." >&2
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

run_pipeline_variant() {
  local variant="$1"
  shift

  echo
  echo "Running split=${SPLIT}, variant=${variant}"
  uv run python src/segmentation_pipeline.py \
    --split "$SPLIT" \
    --split-config "$SPLIT_CONFIG" \
    --resolution mid \
    --config-file "$BASE_CONFIG" \
    --run-group "aorta_segmentation_experiments/${SPLIT}/${variant}" \
    --num-batches 5 \
    "${VISUAL_ARGS[@]}" \
    "${GPU_ARGS[@]}" \
    "$@"
}

for variant in $VARIANTS; do
  case "$variant" in
    envelope_corrected)
      run_pipeline_variant \
        "trajectory_envelope_k2_25_zmargin5_p99_9" \
        --aorta-circle-filter none \
        --aorta-level-set-mode fixed \
        --aorta-trajectory-radius-factor "$TRAJECTORY_RADIUS_FACTOR" \
        --aorta-trajectory-axial-margin-slices "$TRAJECTORY_AXIAL_MARGIN"
      ;;
    aggressive_simplified)
      run_pipeline_variant \
        "circle_filter_aggressive_simplified_rp90_2_7_p99_9" \
        --aorta-circle-filter robust \
        --aorta-circle-filter-min-coverage "$FILTER_COVERAGE" \
        --no-aorta-circle-filter-interpolate \
        --no-aorta-circle-filter-reject-oversegmented \
        --aorta-level-set-mode adaptive \
        --aorta-oversegmented-area-ratio-p90 "$AREA_RATIO_P90"
      ;;
    filtered_envelope)
      run_pipeline_variant \
        "circle_filter_aggressive_envelope_k2_25_zmargin5_p99_9" \
        --aorta-circle-filter robust \
        --aorta-circle-filter-min-coverage "$FILTER_COVERAGE" \
        --no-aorta-circle-filter-interpolate \
        --no-aorta-circle-filter-reject-oversegmented \
        --aorta-level-set-mode fixed \
        --aorta-trajectory-radius-factor "$TRAJECTORY_RADIUS_FACTOR" \
        --aorta-trajectory-axial-margin-slices "$TRAJECTORY_AXIAL_MARGIN"
      ;;
    combined)
      run_pipeline_variant \
        "circle_filter_aggressive_adaptive_rp90_2_7_envelope_k2_25_zmargin5_p99_9" \
        --aorta-circle-filter robust \
        --aorta-circle-filter-min-coverage "$FILTER_COVERAGE" \
        --no-aorta-circle-filter-interpolate \
        --no-aorta-circle-filter-reject-oversegmented \
        --aorta-level-set-mode adaptive \
        --aorta-oversegmented-area-ratio-p90 "$AREA_RATIO_P90" \
        --aorta-trajectory-radius-factor "$TRAJECTORY_RADIUS_FACTOR" \
        --aorta-trajectory-axial-margin-slices "$TRAJECTORY_AXIAL_MARGIN"
      ;;
    *)
      echo "Variante desconhecida: ${variant}" >&2
      echo "Use: envelope_corrected aggressive_simplified filtered_envelope combined" >&2
      exit 2
      ;;
  esac
done
