CUDA_VISIBLE_DEVICES=1 bash -lc '
set -e

python src/experiments/lower_threshold_sweep.py \
  --split train \
  --resolution mid \
  --run-name normal_threshold_fine_p10_upper \
  --methods percentile \
  --threshold-methods normal \
  --percentiles 10.25,10.5,10.75 \
  --max-threshold-percentiles 99.6,99.7,99.8 \
  --num-batches 5 \
  --gpu \
  --no-save-cache \
  --downscale-method opencv \
  --opencv-interpolation linear

python src/experiments/lower_threshold_sweep.py \
  --split train \
  --resolution mid \
  --run-name fuzzy_threshold_permissive_p10 \
  --methods percentile \
  --threshold-methods normal,fuzzy \
  --percentiles 10,10.5 \
  --max-threshold-percentiles 99.7 \
  --fuzzy-object-percentiles 99.8 \
  --fuzzy-dense-percentiles 99.97,99.99 \
  --fuzzy-soft-margins 120 \
  --fuzzy-smooth-radii 0,1 \
  --num-batches 5 \
  --gpu \
  --no-save-cache \
  --downscale-method opencv \
  --opencv-interpolation linear
'