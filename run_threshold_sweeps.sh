#!/usr/bin/env bash
set -euo pipefail

# Use uma GPU especifica antes de chamar este script:
#   CUDA_VISIBLE_DEVICES=1 bash run_threshold_sweeps.sh
#
# Sweep 1: refina os parametros do fuzzy atual (object_argmax).
# Sweep 2: testa melhorias fuzzy mais permissivas:
#   - dense_suppression: usa fuzzy apenas para remover voxels muito densos.
#   - normal_dense_suppression: threshold normal + remocao fuzzy de fundo denso.

python src/experiments/lower_threshold_sweep.py \
  --split train \
  --resolution mid \
  --run-name fuzzy_threshold_object_argmax_refined \
  --methods percentile \
  --threshold-methods fuzzy \
  --percentiles 10.25,10.5 \
  --max-threshold-percentiles 99.7 \
  --fuzzy-object-percentiles 99.8,99.85 \
  --fuzzy-dense-percentiles 99.94,99.96 \
  --fuzzy-soft-margins 100,120 \
  --fuzzy-smooth-radii 0 \
  --fuzzy-mask-strategies object_argmax \
  --fuzzy-dense-membership-thresholds 0.5 \
  --num-batches 5 \
  --gpu \
  --no-save-cache \
  --downscale-method opencv \
  --opencv-interpolation linear

python src/experiments/lower_threshold_sweep.py \
  --split train \
  --resolution mid \
  --run-name fuzzy_threshold_strategy_refined \
  --methods percentile \
  --threshold-methods fuzzy \
  --percentiles 10.5 \
  --max-threshold-percentiles 99.7 \
  --fuzzy-object-percentiles 99.8 \
  --fuzzy-dense-percentiles 99.96 \
  --fuzzy-soft-margins 120 \
  --fuzzy-smooth-radii 0 \
  --fuzzy-mask-strategies dense_suppression,normal_dense_suppression \
  --fuzzy-dense-membership-thresholds 0.55,0.6,0.65,0.7 \
  --num-batches 5 \
  --gpu \
  --no-save-cache \
  --downscale-method opencv \
  --opencv-interpolation linear
