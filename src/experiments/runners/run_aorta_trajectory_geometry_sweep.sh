#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../../.." && pwd)"

SPLIT="${SPLIT:-train}"
VARIANTS="${VARIANTS:-r3_5_c6_0,r4_2_c7_0,r4_8_c8_0,r5_5_c9_0}"
BASE_CONFIG="${BASE_CONFIG:-config/aorta_filter_envelope_generalization.json}"
case "$SPLIT" in
  train) SPLIT_CONFIG="${SPLIT_CONFIG:-config/imagecas_splits.json}" ;;
  val) SPLIT_CONFIG="${SPLIT_CONFIG:-config/imagecas_splits_val60.json}" ;;
  *) echo 'SPLIT deve ser train ou val.' >&2; exit 2 ;;
esac

GPU_ARGS=(--gpu)
[[ "${USE_GPU:-1}" == 0 ]] && GPU_ARGS=(--no-gpu)
VISUAL_ARGS=()
if [[ "${SAVE_VISUALS:-0}" == 1 ]]; then
  VISUAL_OUTPUT_DIR="${VISUAL_OUTPUT_DIR:-/media/matheus/HD/ImageCAS_pipeline_results}"
  [[ -d "$VISUAL_OUTPUT_DIR" && -w "$VISUAL_OUTPUT_DIR" ]] || {
    echo "Diretório visual ausente ou sem escrita: $VISUAL_OUTPUT_DIR" >&2; exit 2;
  }
  VISUAL_ARGS=(--save-segmentation-visuals --visual-output-dir "$VISUAL_OUTPUT_DIR")
fi

IFS=',' read -ra SELECTED <<< "$VARIANTS"
for variant in "${SELECTED[@]}"; do
  case "$variant" in
    r3_5_c6_0) radius=3.5; center=6.0 ;;
    r4_2_c7_0) radius=4.2; center=7.0 ;;
    r4_8_c8_0) radius=4.8; center=8.0 ;;
    r5_5_c9_0) radius=5.5; center=9.0 ;;
    *) echo "Variante desconhecida: $variant" >&2; exit 2 ;;
  esac

  config_file="$(mktemp --suffix=.json)"
  trap 'rm -f "$config_file"' EXIT
  jq --argjson radius "$radius" --argjson center "$center" '
    .CIRCLE_DETECTION.trajectory_filter.max_radius_step_mm = $radius
    | .CIRCLE_DETECTION.trajectory_filter.max_center_step_mm = $center
  ' "$BASE_CONFIG" > "$config_file"

  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "split=$SPLIT variant=$variant radius=$radius center=$center"
  else
    uv run python src/segmentation_pipeline.py \
      --split "$SPLIT" --split-config "$SPLIT_CONFIG" --resolution mid \
      --config-file "$config_file" --num-batches "${NUM_BATCHES:-5}" \
      --run-group "aorta_segmentation_experiments/${SPLIT}/trajectory_geometry_${variant}_p99_9_m300" \
      "${GPU_ARGS[@]}" "${VISUAL_ARGS[@]}"
  fi
  rm -f "$config_file"
  trap - EXIT
done
