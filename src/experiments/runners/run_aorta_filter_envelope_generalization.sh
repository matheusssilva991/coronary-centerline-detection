#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

SPLIT="${SPLIT:-train}"
# A referencia foi a unica variante mantida apos a avaliacao quantitativa e visual.
# As demais continuam disponiveis somente quando solicitadas explicitamente.
VARIANTS="${VARIANTS:-reference}"
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

run_variant() {
  local variant="$1"
  local radius_factor="$2"
  local axial_margin="$3"
  local radius_tag="${radius_factor//./_}"
  local run_name="filter_envelope_${variant}_k${radius_tag}_margin${axial_margin}_p99_9_m300"

  echo
  echo "Executando ${variant}: envelope=${radius_factor}r, margem=${axial_margin}"
  uv run python src/segmentation_pipeline.py \
    --split "$SPLIT" \
    --split-config "$SPLIT_CONFIG" \
    --resolution mid \
    --config-file "$BASE_CONFIG" \
    --aorta-circle-filter robust \
    --aorta-circle-filter-min-coverage 0.40 \
    --aorta-circle-filter-max-trim-fraction 0.40 \
    --aorta-circle-filter-synthetic-tail-slices 5 \
    --aorta-circle-filter-mask-guided \
    --aorta-level-set-mode fixed \
    --aorta-trajectory-radius-factor "$radius_factor" \
    --aorta-trajectory-axial-margin-slices "$axial_margin" \
    --run-group \
      "aorta_segmentation_experiments/${SPLIT}/filter_envelope_generalization/${run_name}" \
    --num-batches "$NUM_BATCHES" \
    "${VISUAL_ARGS[@]}" \
    "${GPU_ARGS[@]}"
}

for variant in $VARIANTS; do
  case "$variant" in
    reference)
      run_variant reference 2.25 10
      ;;
    balanced)
      run_variant balanced 2.35 10
      ;;
    conservative)
      run_variant conservative 2.40 12
      ;;
    anti_leak)
      run_variant anti_leak 2.15 12
      ;;
    *)
      echo "Variante desconhecida: ${variant}" >&2
      echo "Use: reference balanced conservative anti_leak" >&2
      exit 2
      ;;
  esac
done
