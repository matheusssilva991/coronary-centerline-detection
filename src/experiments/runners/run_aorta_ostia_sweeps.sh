#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

# Sweep enxuto para testar localização da aorta e detecção dos óstios.
#
# Para usar uma GPU específica:
#   CUDA_VISIBLE_DEVICES=1 bash src/experiments/runners/run_aorta_ostia_sweeps.sh
#
# O script usa o threshold normal mais estável encontrado anteriormente e varia:
#   - LCC por fatia vs volume;
#   - miss count;
#   - fora da tolerância como parada vs miss;
#   - seleção do círculo por closest vs score.

uv run python src/experiments/aorta_ostia_parameter_sweep.py \
  --split train \
  --resolution mid \
  --run-name aorta_ostia_train60_quick \
  --sample-size 60 \
  --sample-source-splits train,val \
  --threshold-preset best_normal \
  --timeout-minutes 180 \
  --num-batches 5 \
  --gpu \
  --no-save-cache \
  --downscale-method opencv \
  --opencv-interpolation linear
