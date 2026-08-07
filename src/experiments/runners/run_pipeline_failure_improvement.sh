#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

MODE="${MODE:-baseline}"
SPLIT="${SPLIT:-val}"
RESOLUTION="${RESOLUTION:-mid}"
AORTA_OSTIA_METHOD="${AORTA_OSTIA_METHOD:-bilateral_thin}"
USE_GPU="${USE_GPU:-1}"
COHORT_FILE="${COHORT_FILE:-output/segmentation/analysis/pipeline_failure_analysis/focused_cohort.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/segmentation/analysis/pipeline_failure_improvement}"
RUN_NAME="${RUN_NAME:-${MODE}_${SPLIT}_$(date +%Y-%m-%d_%H-%M-%S)}"

if [[ "$SPLIT" == "test" ]]; then
  echo "Refusing to tune parameters on the test split." >&2
  exit 2
fi

if [[ ! -f "$COHORT_FILE" ]]; then
  uv run python src/experiments/pipeline_failure_analysis.py --split "$SPLIT"
fi

case "$MODE" in
  baseline) VARIANT_SET="baseline" ;;
  corrections) VARIANT_SET="corrections" ;;
  *)
    echo "MODE must be 'baseline' or 'corrections'." >&2
    exit 2
    ;;
esac

GPU_ARG="--gpu"
if [[ "$USE_GPU" == "0" ]]; then
  GPU_ARG="--no-gpu"
fi

EXTRA_ARGS=()
[[ "${DRY_RUN:-0}" == "1" ]] && EXTRA_ARGS+=(--dry-run)

echo "Mode: $MODE"
echo "Split: $SPLIT"
echo "Cohort: $COHORT_FILE"
echo "Aorta/ostia: $AORTA_OSTIA_METHOD"
echo "Run: $OUTPUT_ROOT/$RUN_NAME"

uv run python src/experiments/fuzzy_pipeline_comparison.py \
  --split "$SPLIT" \
  --ids-file "$COHORT_FILE" \
  --resolution "$RESOLUTION" \
  --variant-set "$VARIANT_SET" \
  --aorta-ostia-method "$AORTA_OSTIA_METHOD" \
  --output-root "$OUTPUT_ROOT" \
  --run-name "$RUN_NAME" \
  "$GPU_ARG" \
  "${EXTRA_ARGS[@]}"
