d#!/usr/bin/env bash
set -euo pipefail

# Use a GPU especifica antes de chamar este script:
#   CUDA_VISIBLE_DEVICES=1 bash run_threshold_sweeps.sh
#
# Objetivo dos dois sweeps:
# 1. Normal: refinar ao redor do melhor normal atual:
#    lower p10.75 + max threshold p99.8.
# 2. Fuzzy: refinar ao redor do melhor fuzzy atual:
#    lower p10.5 + fuzzy object p99.8 + dense p99.97 + smooth radius 0.

python src/experiments/lower_threshold_sweep.py \
  --split train \
  --resolution mid \
  --run-name normal_threshold_refined_p10p75_upper \
  --methods percentile \
  --threshold-methods normal \
  --percentiles 10.6,10.75,10.9 \
  --max-threshold-percentiles 99.75,99.8,99.85 \
  --num-batches 5 \
  --gpu \
  --no-save-cache \
  --downscale-method opencv \
  --opencv-interpolation linear

python src/experiments/lower_threshold_sweep.py \
  --split train \
  --resolution mid \
  --run-name fuzzy_threshold_refined_permissive_p10 \
  --methods percentile \
  --threshold-methods fuzzy \
  --percentiles 10,10.25,10.5 \
  --max-threshold-percentiles 99.7 \
  --fuzzy-object-percentiles 99.8 \
  --fuzzy-dense-percentiles 99.96,99.97,99.98,99.99 \
  --fuzzy-soft-margins 120 \
  --fuzzy-smooth-radii 0 \
  --num-batches 5 \
  --gpu \
  --no-save-cache \
  --downscale-method opencv \
  --opencv-interpolation linear
