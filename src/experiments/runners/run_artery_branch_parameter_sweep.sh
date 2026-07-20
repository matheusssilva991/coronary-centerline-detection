#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${MODE:-all}"
SPLIT="${SPLIT:-train}"
SAMPLE_SIZE="${SAMPLE_SIZE:-30}"
AORTA_OSTIA_METHOD="${AORTA_OSTIA_METHOD:-standard}"
RUN_NAME="${RUN_NAME:-artery_branch_parameters_${MODE}_${SPLIT}${SAMPLE_SIZE}_$(date +%Y-%m-%d_%H-%M-%S)}"

SHARED_VARIANTS="rg_gamma55_current,rg_gamma65_current,rg_gamma65_relax094,rg_gamma65_switch1000,rg_gamma65_switch4000,rg_gamma65_floor072,rg_gamma65_floor084,rg_gamma65_no_smooth,rg_gamma65_neighborhood18,fc_floor020,fc_floor020_alpha018,fc_floor020_sigma090,fc_floor020_sigma110,fc_floor020_weight088,fc_floor020_weight092,fc_floor020_neighborhood18"
BRANCH_VARIANTS="rg_gamma55_current,rg_gamma65_current,rg_gamma65_branch_local_p90,rg_gamma65_branch_local_p95,rg_gamma65_left_permissive,rg_gamma65_right_permissive,fc_floor020,fc_left_permissive,fc_right_permissive"

case "$MODE" in
  shared)
    OPTIMIZATION_VARIANTS="$SHARED_VARIANTS"
    ;;
  branch)
    OPTIMIZATION_VARIANTS="$BRANCH_VARIANTS"
    ;;
  all)
    OPTIMIZATION_VARIANTS="${SHARED_VARIANTS},rg_gamma65_branch_local_p90,rg_gamma65_branch_local_p95,rg_gamma65_left_permissive,rg_gamma65_right_permissive,fc_left_permissive,fc_right_permissive"
    ;;
  *)
    echo "MODE deve ser shared, branch ou all." >&2
    exit 2
    ;;
esac

export SPLIT SAMPLE_SIZE AORTA_OSTIA_METHOD RUN_NAME OPTIMIZATION_VARIANTS
export MORPHOLOGY_PROFILES="${MORPHOLOGY_PROFILES:-current_c3_d2,conditioned_p10_f025,conditioned_p10_f050}"

echo "Sweep arterial: mode=${MODE}, split=${SPLIT}, imagens=${SAMPLE_SIZE}"
echo "Aorta/ostios: ${AORTA_OSTIA_METHOD}"
echo "Variantes: ${OPTIMIZATION_VARIANTS}"
echo "Morfologias: ${MORPHOLOGY_PROFILES}"

bash "${SCRIPT_DIR}/run_artery_segmentation_optimization.sh"
