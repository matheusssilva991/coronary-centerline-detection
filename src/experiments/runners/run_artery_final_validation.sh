#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SPLIT="${SPLIT:-val}"
SAMPLE_SIZE="${SAMPLE_SIZE:-60}"
AORTA_OSTIA_METHOD="${AORTA_OSTIA_METHOD:-bilateral_thin}"
RUN_NAME="${RUN_NAME:-artery_final_candidates_${SPLIT}${SAMPLE_SIZE}_$(date +%Y-%m-%d_%H-%M-%S)}"

OPTIMIZATION_VARIANTS="validation_baseline_rg,validation_rg_gamma65,validation_fc_floor020_sigma090"

export SPLIT SAMPLE_SIZE AORTA_OSTIA_METHOD RUN_NAME OPTIMIZATION_VARIANTS

echo "Validacao final da segmentacao arterial"
echo "Split: ${SPLIT} | imagens: ${SAMPLE_SIZE}"
echo "Aorta/ostios: ${AORTA_OSTIA_METHOD}"
echo "Variantes: ${OPTIMIZATION_VARIANTS}"

bash "${SCRIPT_DIR}/run_artery_segmentation_optimization.sh"
