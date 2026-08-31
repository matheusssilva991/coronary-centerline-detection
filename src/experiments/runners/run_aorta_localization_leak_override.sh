#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

USE_GPU="${USE_GPU:-1}"
NUM_BATCHES="${NUM_BATCHES:-1}"
BASE_CONFIG="${BASE_CONFIG:-config/aorta_filter_envelope_generalization.json}"
VISUAL_OUTPUT_DIR="${VISUAL_OUTPUT_DIR:-/media/matheus/HD/ImageCAS_pipeline_results}"
FOCUSED_IDS="${FOCUSED_IDS:-11,464,790,792,308,341,513,886,2}"
VARIANTS="${VARIANTS:-override_disabled_reference override_reference05 override_reference10 override_mild05 override_strong05}"

GPU_ARGS=(--gpu)
[[ "$USE_GPU" == "0" ]] && GPU_ARGS=(--no-gpu)

VISUAL_ARGS=(--save-segmentation-visuals)
[[ -n "$VISUAL_OUTPUT_DIR" ]] && VISUAL_ARGS+=(--visual-output-dir "$VISUAL_OUTPUT_DIR")

run_variant() {
  local variant="$1"
  local balloon="$2"
  local alpha="$3"
  local threshold_percentile="$4"
  local min_improvement="$5"
  local override_flag="$6"

  echo
  echo "Executando ${variant}: R_P90>2.0, fill>=0.80, volume>=0.015"
  uv run python src/segmentation_pipeline.py \
    --split val \
    --split-config config/imagecas_splits_val60.json \
    --image-ids "$FOCUSED_IDS" \
    --resolution mid \
    --config-file "$BASE_CONFIG" \
    --aorta-circle-filter robust \
    --aorta-circle-filter-min-coverage 0.40 \
    --aorta-circle-filter-max-trim-fraction 0.40 \
    --aorta-circle-filter-synthetic-tail-slices 5 \
    --aorta-circle-filter-mask-guided \
    --aorta-level-set-mode adaptive \
    --aorta-level-set-balloon 0.6 \
    --aorta-level-set-radius-reduction-factor 0.10 \
    --aorta-level-set-iterations 26 \
    --aorta-opening-radius 2 \
    --aorta-trajectory-radius-factor 2.25 \
    --aorta-trajectory-axial-margin-slices 10 \
    --aorta-oversegmented-area-ratio-p90 2.0 \
    --aorta-conservative-balloon "$balloon" \
    --aorta-conservative-alpha "$alpha" \
    --aorta-conservative-threshold-percentile "$threshold_percentile" \
    --aorta-conservative-min-ratio-improvement "$min_improvement" \
    --aorta-localization-leak-min-area-ratio-p90 2.0 \
    --aorta-localization-leak-min-circle-fill-q25 0.80 \
    --aorta-localization-leak-min-volume-fraction 0.015 \
    "$override_flag" \
    --run-group "aorta_segmentation_experiments/val/localization_leak_override/focused/${variant}" \
    --num-batches "$NUM_BATCHES" \
    "${VISUAL_ARGS[@]}" \
    "${GPU_ARGS[@]}"
}

for variant in $VARIANTS; do
  case "$variant" in
    override_disabled_reference)
      run_variant "$variant" 0.50 1500 55 0.05 --no-aorta-localization-leak-override
      ;;
    override_reference05)
      run_variant "$variant" 0.50 1500 55 0.05 --aorta-localization-leak-override
      ;;
    override_reference10)
      run_variant "$variant" 0.50 1500 55 0.10 --aorta-localization-leak-override
      ;;
    override_mild05)
      run_variant "$variant" 0.55 1250 50 0.05 --aorta-localization-leak-override
      ;;
    override_strong05)
      run_variant "$variant" 0.40 1750 60 0.05 --aorta-localization-leak-override
      ;;
    *)
      echo "Variante desconhecida: ${variant}" >&2
      exit 2
      ;;
  esac
done
