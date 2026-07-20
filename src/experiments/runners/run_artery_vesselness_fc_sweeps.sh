#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

STAGE="${STAGE:-vesselness}"
SPLIT="${SPLIT:-train}"
SAMPLE_SIZE="${SAMPLE_SIZE:-30}"
START_INDEX="${START_INDEX:-0}"
RESOLUTION="${RESOLUTION:-mid}"
THRESHOLD_METHOD="${THRESHOLD_METHOD:-normal}"
AORTA_OSTIA_METHOD="${AORTA_OSTIA_METHOD:-bilateral_thin}"
RUN_NAME="${RUN_NAME:-artery_${STAGE}_${SPLIT}${SAMPLE_SIZE}_$(date +%Y-%m-%d_%H-%M-%S)}"
VESSELNESS_PROFILE="${VESSELNESS_PROFILE:-current}"

EXTRA_ARGS=()
if [[ -n "${PROFILES:-}" ]]; then
  EXTRA_ARGS+=(--profiles "$PROFILES")
fi
if [[ -n "${FC_VARIANTS:-}" ]]; then
  EXTRA_ARGS+=(--fc-variants "$FC_VARIANTS")
fi
if [[ -n "${VARIANT_LIMIT:-}" ]]; then
  EXTRA_ARGS+=(--variant-limit "$VARIANT_LIMIT")
fi
if [[ -n "${RESUME_DIR:-}" ]]; then
  EXTRA_ARGS+=(--resume-dir "$RESUME_DIR")
fi

uv run python src/experiments/artery_vesselness_fc_sweep.py \
  --stage "$STAGE" \
  --split "$SPLIT" \
  --sample-size "$SAMPLE_SIZE" \
  --start-index "$START_INDEX" \
  --resolution "$RESOLUTION" \
  --threshold-method "$THRESHOLD_METHOD" \
  --aorta-ostia-method "$AORTA_OSTIA_METHOD" \
  --vesselness-profile "$VESSELNESS_PROFILE" \
  --run-name "$RUN_NAME" \
  --gpu \
  "${EXTRA_ARGS[@]}"
