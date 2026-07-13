#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

# MODE=quick: 9 falhas conhecidas + 15 controles (24 imagens, triagem).
# MODE=ablation: seis variantes para isolar envelope e superfície (24 imagens).
# MODE=advanced: 14 hipóteses em 30 imagens independentes (índices 60-89).
# MODE=bilateral: 6 variantes em outras 30 imagens (índices 90-119).
# MODE=full: SAMPLE_SIZE imagens do split a partir de START_INDEX.
# O cache comprimido fica restrito aos mapas de vesselness e é reutilizado.
MODE="${MODE:-quick}"
SPLIT="${SPLIT:-val}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"
START_INDEX="${START_INDEX:-}"
RUN_NAME="${RUN_NAME:-aorta_mask_ostia_${MODE}}"
VARIANTS="${VARIANTS:-}"

EXTRA_ARGS=()
if [[ "$MODE" == "quick" ]]; then
  EXTRA_ARGS+=(--quick)
elif [[ "$MODE" == "ablation" ]]; then
  EXTRA_ARGS+=(--quick)
  if [[ -z "$VARIANTS" ]]; then
    VARIANTS="baseline,thin_surface,trajectory_only_f150,trajectory_only_f175,trajectory_only_f200,trajectory_f175_thin_surface"
  fi
elif [[ "$MODE" == "advanced" ]]; then
  SAMPLE_SIZE="${SAMPLE_SIZE:-30}"
  START_INDEX="${START_INDEX:-60}"
  if [[ -z "$VARIANTS" ]]; then
    VARIANTS="baseline,thin_surface,physical_shell_15mm,physical_shell_20mm,candidate_nms_3mm,candidate_nms_4mm,joint_pair,nms4_joint_pair,robust_score_p90_w30,robust_nms4_joint,thin_nms4_joint,conditional_mask_a175,conditional_mask_a200,physical20_nms4_joint"
  fi
elif [[ "$MODE" == "bilateral" ]]; then
  SAMPLE_SIZE="${SAMPLE_SIZE:-30}"
  START_INDEX="${START_INDEX:-90}"
  if [[ -z "$VARIANTS" ]]; then
    VARIANTS="baseline,thin_surface,thin_conditional_a200,bilateral_pair,bilateral_thin,bilateral_thin_conditional"
  fi
elif [[ "$MODE" != "full" ]]; then
  echo "MODE deve ser 'quick', 'ablation', 'advanced', 'bilateral' ou 'full'." >&2
  exit 2
fi
SAMPLE_SIZE="${SAMPLE_SIZE:-60}"
START_INDEX="${START_INDEX:-0}"
if [[ -n "$VARIANTS" ]]; then
  EXTRA_ARGS+=(--variants "$VARIANTS")
fi

uv run python src/experiments/aorta_mask_ostia_comparison.py \
  --split "$SPLIT" \
  --sample-size "$SAMPLE_SIZE" \
  --start-index "$START_INDEX" \
  --resolution mid \
  --num-batches 5 \
  --run-name "$RUN_NAME" \
  --gpu \
  "${EXTRA_ARGS[@]}"
