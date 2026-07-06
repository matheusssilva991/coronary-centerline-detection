#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

# Sweeps focados para testar localização da aorta e detecção dos óstios.
#
# Para usar uma GPU específica:
#   CUDA_VISIBLE_DEVICES=1 bash src/experiments/runners/run_aorta_ostia_sweeps.sh
#
# O script usa o threshold normal mais estável encontrado anteriormente.
# Rodada 1: isola a estratégia score sem tolerar outlier como miss.
# Rodada 2: refina tolerância geométrica em volta da melhor configuração atual.

uv run python src/experiments/aorta_ostia_parameter_sweep.py \
  --split train \
  --resolution mid \
  --run-name aorta_ostia_train60_score_stop \
  --sample-size 60 \
  --sample-source-splits train,val \
  --full-grid \
  --lcc-modes per_slice,per_volume \
  --miss-counts 5 \
  --tolerance-modes stop \
  --candidate-strategies closest,score \
  --threshold-preset best_normal \
  --timeout-minutes 180 \
  --num-batches 5 \
  --gpu \
  --no-save-cache \
  --downscale-method opencv \
  --opencv-interpolation linear

uv run python src/experiments/aorta_ostia_parameter_sweep.py \
  --split train \
  --resolution mid \
  --run-name aorta_ostia_train60_geometry_best \
  --sample-size 60 \
  --sample-source-splits train,val \
  --full-grid \
  --lcc-modes per_volume \
  --miss-counts 5 \
  --tolerance-modes stop \
  --candidate-strategies closest \
  --tol-radius-mm 7,9 \
  --tol-distance-mm 18,20 \
  --neighbor-distance-thresholds 3,5 \
  --threshold-preset best_normal \
  --timeout-minutes 180 \
  --num-batches 5 \
  --gpu \
  --no-save-cache \
  --downscale-method opencv \
  --opencv-interpolation linear
