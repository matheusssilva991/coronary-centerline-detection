#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

# Compara recuperação padrão, descarte de trajetória curta e busca estendida.
# Variáveis opcionais: SAMPLE_SIZE, SPLIT, RUN_NAME e CUDA_VISIBLE_DEVICES.
SAMPLE_SIZE="${SAMPLE_SIZE:-60}"
SPLIT="${SPLIT:-val}"
RUN_NAME="${RUN_NAME:-aorta_recovery_${SPLIT}${SAMPLE_SIZE}}"

uv run python src/experiments/aorta_recovery_comparison.py \
  --split "$SPLIT" \
  --sample-size "$SAMPLE_SIZE" \
  --resolution mid \
  --num-batches 5 \
  --run-name "$RUN_NAME" \
  --gpu
