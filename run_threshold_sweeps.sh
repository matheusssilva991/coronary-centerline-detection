#!/usr/bin/env bash
set -euo pipefail

# Validação dos melhores parâmetros encontrados no treino.
#
# Para usar uma GPU específica:
#   CUDA_VISIBLE_DEVICES=1 bash run_threshold_sweeps.sh
#
# Este script roda apenas duas configurações:
#   1. threshold normal com melhor configuração do treino;
#   2. threshold fuzzy com melhor configuração do treino.

uv run python src/experiments/threshold_parameter_sweep.py \
  --split val \
  --resolution mid \
  --run-name val_normal_threshold_best \
  --methods percentile \
  --threshold-methods normal \
  --percentiles 10.75 \
  --max-threshold-percentiles 99.8 \
  --num-batches 5 \
  --gpu \
  --no-save-cache \
  --downscale-method opencv \
  --opencv-interpolation linear

uv run python src/experiments/threshold_parameter_sweep.py \
  --split val \
  --resolution mid \
  --run-name val_fuzzy_threshold_best \
  --methods percentile \
  --threshold-methods fuzzy \
  --percentiles 10.5 \
  --max-threshold-percentiles 99.7 \
  --fuzzy-object-percentiles 99.8 \
  --fuzzy-dense-percentiles 99.96 \
  --fuzzy-soft-margins 100 \
  --fuzzy-smooth-radii 0 \
  --fuzzy-mask-strategies object_argmax \
  --fuzzy-dense-membership-thresholds 0.5 \
  --num-batches 5 \
  --gpu \
  --no-save-cache \
  --downscale-method opencv \
  --opencv-interpolation linear
