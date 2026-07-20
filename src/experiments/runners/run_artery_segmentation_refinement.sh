#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

SPLIT="${SPLIT:-train}"
SAMPLE_SIZE="${SAMPLE_SIZE:-30}"
START_INDEX="${START_INDEX:-0}"
RESOLUTION="${RESOLUTION:-mid}"
THRESHOLD_METHOD="${THRESHOLD_METHOD:-normal}"
AORTA_OSTIA_METHOD="${AORTA_OSTIA_METHOD:-bilateral_thin}"
RUN_NAME="${RUN_NAME:-artery_segmentation_refinement_${SPLIT}${SAMPLE_SIZE}_$(date +%Y-%m-%d_%H-%M-%S)}"
USE_GPU="${USE_GPU:-1}"

EXTRA_ARGS=()
if [[ -n "${REFINEMENT_VARIANTS:-}" ]]; then
  EXTRA_ARGS+=(--refinement-variants "$REFINEMENT_VARIANTS")
fi
if [[ -n "${MORPHOLOGY_PROFILES:-}" ]]; then
  EXTRA_ARGS+=(--morphology-profiles "$MORPHOLOGY_PROFILES")
fi
if [[ -n "${IDS:-}" ]]; then
  EXTRA_ARGS+=(--ids "$IDS")
fi
if [[ -n "${VARIANT_LIMIT:-}" ]]; then
  EXTRA_ARGS+=(--variant-limit "$VARIANT_LIMIT")
fi
if [[ -n "${RESUME_DIR:-}" ]]; then
  EXTRA_ARGS+=(--resume-dir "$RESUME_DIR")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

GPU_ARG="--gpu"
if [[ "$USE_GPU" == "0" ]]; then
  GPU_ARG="--no-gpu"
fi

uv run python src/experiments/artery_vesselness_fc_sweep.py \
  --stage refinement \
  --split "$SPLIT" \
  --sample-size "$SAMPLE_SIZE" \
  --start-index "$START_INDEX" \
  --resolution "$RESOLUTION" \
  --threshold-method "$THRESHOLD_METHOD" \
  --aorta-ostia-method "$AORTA_OSTIA_METHOD" \
  --run-name "$RUN_NAME" \
  "$GPU_ARG" \
  "${EXTRA_ARGS[@]}"
