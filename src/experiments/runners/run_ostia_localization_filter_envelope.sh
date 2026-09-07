#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

SPLITS="${SPLITS:-train,val}"
USE_GPU="${USE_GPU:-1}"
SAVE_VISUALS="${SAVE_VISUALS:-0}"
NUM_BATCHES="${NUM_BATCHES:-5}"
BASE_CONFIG="${BASE_CONFIG:-config/pipeline_config.json}"
ENVELOPE_CONFIG="${ENVELOPE_CONFIG:-config/aorta_filter_envelope_generalization.json}"
VARIANTS_FILE="${VARIANTS_FILE:-}"
if [[ -n "$VARIANTS_FILE" ]]; then
  VARIANTS="${VARIANTS:-$(jq -er '.variants | keys_unsorted | join(",")' "$VARIANTS_FILE")}"
else
VARIANTS="${VARIANTS:-baseline_pad0,lower100_pad2}"
fi
RUN_FAMILY="${RUN_FAMILY:-ostia_localization}"
DRY_RUN="${DRY_RUN:-0}"

GPU_ARGS=(--gpu)
[[ "$USE_GPU" == "0" ]] && GPU_ARGS=(--no-gpu)

VISUAL_ARGS=()
if [[ "$SAVE_VISUALS" == "1" ]]; then
  VISUAL_OUTPUT_DIR="${VISUAL_OUTPUT_DIR:-/media/matheus/HD/ImageCAS_pipeline_results}"
  [[ -d "$VISUAL_OUTPUT_DIR" && -w "$VISUAL_OUTPUT_DIR" ]] || {
    echo "Diretório visual ausente ou sem escrita: $VISUAL_OUTPUT_DIR" >&2
    exit 2
  }
  VISUAL_ARGS+=(
    --save-segmentation-visuals
    --visual-output-dir "$VISUAL_OUTPUT_DIR"
  )
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

build_variant_config() {
  local variant="$1"
  local output_path="$2"

  if [[ -n "$VARIANTS_FILE" ]]; then
    # Cada variante parte da mesma referencia, sem acumular overrides.
    jq -en \
      --slurpfile base "$BASE_CONFIG" \
      --slurpfile envelope "$ENVELOPE_CONFIG" \
      --slurpfile sweep "$VARIANTS_FILE" \
      --arg variant "$variant" \
      'if ($sweep[0].variants | has($variant)) then
         $base[0] * $envelope[0] * $sweep[0].reference * $sweep[0].variants[$variant]
       else error("Variante desconhecida: " + $variant) end' > "$output_path"
    return
  fi

  jq -n \
    --slurpfile base "$BASE_CONFIG" \
    --slurpfile envelope "$ENVELOPE_CONFIG" \
    --arg variant "$variant" \
    '($base[0] * $envelope[0])
    | if $variant == "baseline_pad0" then
        .OSTIA_DETECTION.surface_padding_radius = 0
        | .OSTIA_DETECTION.lower_fraction = 0.85
      elif $variant == "baseline_pad1" then
        .OSTIA_DETECTION.surface_padding_radius = 1
      elif $variant == "baseline_pad2" then
        .OSTIA_DETECTION.surface_padding_radius = 2
      elif $variant == "lower100_pad1_inclusive" then
        .OSTIA_DETECTION.lower_fraction = 1.00
        | .OSTIA_DETECTION.surface_padding_radius = 1
      elif $variant == "lower100_pad2" then
        .OSTIA_DETECTION.lower_fraction = 1.00
        | .OSTIA_DETECTION.surface_padding_radius = 2
      elif $variant == "lower095_pad2" then
        .OSTIA_DETECTION.lower_fraction = 0.95
        | .OSTIA_DETECTION.surface_padding_radius = 2
      elif $variant == "lower100_pad3" then
        .OSTIA_DETECTION.lower_fraction = 1.00
        | .OSTIA_DETECTION.surface_padding_radius = 3
      elif $variant == "z30_pad1" then
        .OSTIA_DETECTION.max_z_diff_mm = 30
        | .OSTIA_DETECTION.surface_padding_radius = 1
      elif $variant == "z50_pad1" then
        .OSTIA_DETECTION.max_z_diff_mm = 50
        | .OSTIA_DETECTION.surface_padding_radius = 1
      elif $variant == "lateral025_pad1" then
        .OSTIA_DETECTION.min_lateral_factor = 0.25
        | .OSTIA_DETECTION.surface_padding_radius = 1
      elif $variant == "lateral055_pad1" then
        .OSTIA_DETECTION.min_lateral_factor = 0.55
        | .OSTIA_DETECTION.surface_padding_radius = 1
      elif $variant == "center065_pad1" then
        .OSTIA_DETECTION.min_center_distance_factor = 0.65
        | .OSTIA_DETECTION.surface_padding_radius = 1
      elif $variant == "center105_pad1" then
        .OSTIA_DETECTION.min_center_distance_factor = 1.05
        | .OSTIA_DETECTION.surface_padding_radius = 1
      elif $variant == "lower070_pad1" then
        .OSTIA_DETECTION.lower_fraction = 0.70
        | .OSTIA_DETECTION.surface_padding_radius = 1
      elif $variant == "lower100_pad1" then
        .OSTIA_DETECTION.lower_fraction = 1.00
        | .OSTIA_DETECTION.surface_padding_radius = 1
      else
        error("Variante desconhecida: " + $variant)
      end' \
    > "$output_path"
}

run_variant() {
  local split="$1"
  local variant="$2"
  local split_config="config/imagecas_splits.json"
  [[ "$split" == "val" ]] && split_config="${VAL_SPLIT_CONFIG:-config/imagecas_splits_val60.json}"

  local config_path="$TMP_DIR/${split}_${variant}.json"
  build_variant_config "$variant" "$config_path"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "${split}/${variant}: ${split_config}"
    jq '{OSTIA_DETECTION, VESSELNESS_AORTA}' "$config_path"
    return
  fi

  echo
  echo "Executando ${variant} no split ${split}"
  uv run python src/segmentation_pipeline.py \
    --split "$split" \
    --split-config "$split_config" \
    --resolution mid \
    --config-file "$config_path" \
    --aorta-hough-radii-start-px 18 \
    --aorta-hough-radii-end-px 30 \
    --aorta-circle-filter robust \
    --aorta-circle-filter-min-coverage 0.40 \
    --aorta-circle-filter-max-trim-fraction 0.40 \
    --aorta-circle-filter-synthetic-tail-slices 5 \
    --aorta-trajectory-radius-factor 2.25 \
    --aorta-trajectory-axial-margin-slices 10 \
    --run-group "aorta_segmentation_experiments/${split}/${RUN_FAMILY}/${variant}" \
    --num-batches "$NUM_BATCHES" \
    "${VISUAL_ARGS[@]}" \
    "${GPU_ARGS[@]}"
}

IFS=',' read -r -a variant_names <<< "$VARIANTS"
IFS=',' read -r -a split_names <<< "$SPLITS"

for split in "${split_names[@]}"; do
  case "$split" in
    train|val|test) ;;
    *)
      echo "SPLITS deve conter apenas train, val e/ou test: $split" >&2
      exit 2
      ;;
  esac
  for variant in "${variant_names[@]}"; do
    run_variant "$split" "$variant"
  done
done
