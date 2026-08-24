#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

USE_GPU="${USE_GPU:-1}"
SAVE_VISUALS="${SAVE_VISUALS:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_VAL="${RUN_VAL:-1}"
BASE_CONFIG="${BASE_CONFIG:-config/article_cbeb_sensitivity.json}"
FILTER_PROFILE="${FILTER_PROFILE:-coverage}"
COVERAGE="${COVERAGE:-0.65}"
RUN_WITHOUT_FALLBACK="${RUN_WITHOUT_FALLBACK:-0}"
RUN_WITH_FALLBACK="${RUN_WITH_FALLBACK:-1}"
COVERAGE_TAG="${COVERAGE/./}"

GPU_ARG="--gpu"
[[ "$USE_GPU" == "0" ]] && GPU_ARG="--no-gpu"

VISUAL_ARGS=()
[[ "$SAVE_VISUALS" == "1" ]] && VISUAL_ARGS+=(--save-segmentation-visuals)

run_variant() {
  local split="$1"
  local split_config="$2"
  local variant="$3"
  local fallback_arg="$4"

  echo
  echo "Running split=${split}, variant=${variant}, coverage=${COVERAGE}"
  uv run python src/segmentation_pipeline.py \
    --split "$split" \
    --split-config "$split_config" \
    --resolution mid \
    --config-file "$BASE_CONFIG" \
    --aorta-ostia-method standard \
    --aorta-circle-filter robust \
    --aorta-circle-filter-min-coverage "$COVERAGE" \
    --no-aorta-circle-filter-interpolate \
    "$fallback_arg" \
    --aorta-level-set-mode adaptive \
    --aorta-leak-correction circle_area_jump_pruning \
    --run-group "aorta_segmentation_experiments/${split}/${variant}" \
    --num-batches 5 \
    "${VISUAL_ARGS[@]}" \
    "$GPU_ARG"
}

run_aggressive() {
  local split="$1"
  local split_config="$2"

  # Reproduz o filtro agressivo histórico sobre o baseline P99.9/-300.
  COVERAGE=0.40
  COVERAGE_TAG=040
  run_variant \
    "$split" \
    "$split_config" \
    "circle_filter_aggressive_p999" \
    --no-aorta-circle-filter-reject-oversegmented
}

if [[ "$FILTER_PROFILE" != "coverage" && "$FILTER_PROFILE" != "aggressive" ]]; then
  echo "FILTER_PROFILE inválido: $FILTER_PROFILE (use coverage ou aggressive)" >&2
  exit 2
fi

echo "Base config: ${BASE_CONFIG}"
echo "Filter profile: ${FILTER_PROFILE}"
echo "Save visuals: ${SAVE_VISUALS}"

if [[ "$RUN_TRAIN" == "1" ]]; then
  if [[ "$FILTER_PROFILE" == "aggressive" ]]; then
    run_aggressive train config/imagecas_splits.json
  elif [[ "$RUN_WITHOUT_FALLBACK" == "1" ]]; then
    run_variant \
      train \
      config/imagecas_splits.json \
      "circle_filter_coverage_${COVERAGE_TAG}" \
      --no-aorta-circle-filter-reject-oversegmented
  fi
  if [[ "$FILTER_PROFILE" == "coverage" && "$RUN_WITH_FALLBACK" == "1" ]]; then
    run_variant \
      train \
      config/imagecas_splits.json \
      "circle_filter_coverage_${COVERAGE_TAG}_fallback" \
      --aorta-circle-filter-reject-oversegmented
  fi
fi

if [[ "$RUN_VAL" == "1" ]]; then
  if [[ "$FILTER_PROFILE" == "aggressive" ]]; then
    run_aggressive val config/imagecas_splits_val60.json
  elif [[ "$RUN_WITHOUT_FALLBACK" == "1" ]]; then
    run_variant \
      val \
      config/imagecas_splits_val60.json \
      "circle_filter_coverage_${COVERAGE_TAG}" \
      --no-aorta-circle-filter-reject-oversegmented
  fi
  if [[ "$FILTER_PROFILE" == "coverage" && "$RUN_WITH_FALLBACK" == "1" ]]; then
    run_variant \
      val \
      config/imagecas_splits_val60.json \
      "circle_filter_coverage_${COVERAGE_TAG}_fallback" \
      --aorta-circle-filter-reject-oversegmented
  fi
fi
