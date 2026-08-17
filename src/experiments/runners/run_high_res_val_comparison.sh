#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

# Executa sequencialmente as três configurações de treino em alta resolução.
# O script para imediatamente se qualquer execução falhar.
#
# Para selecionar uma GPU física específica:
#   CUDA_VISIBLE_DEVICES=1 bash src/experiments/runners/run_high_res_val_comparison.sh

echo "[1/3] Baseline histórico do artigo: P99.7"
uv run python src/segmentation_pipeline.py \
  --split val \
  --resolution high \
  --config-file config/article_cbeb_sensitivity.json \
  --upper-threshold-percentile 99.7 \
  --num-batches 5 \
  --gpu

echo "[2/3] Baseline atual do projeto: P99.8"
uv run python src/segmentation_pipeline.py \
  --split val \
  --resolution high \
  --config-file config/pipeline_config.json \
  --num-batches 5 \
  --gpu

echo "[3/3] Configuração do artigo CBEB: P99.9"
uv run python src/segmentation_pipeline.py \
  --split val \
  --resolution high \
  --config-file config/article_cbeb_sensitivity.json \
  --upper-threshold-percentile 99.9 \
  --num-batches 5 \
  --gpu

echo "As três execuções foram concluídas."
