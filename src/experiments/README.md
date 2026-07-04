# Experiments

Scripts executáveis para estudos auxiliares do pipeline.

Esta pasta é diferente de `tests/`: aqui ficam experimentos manuais ou
diagnósticos longos, geralmente rodados no PC ou servidor. Testes automatizados
de unidade/CI devem ficar em `tests/` se forem criados no futuro.

## Scripts atuais

- `compare_cpu_gpu.py`: compara saídas de CPU/GPU e ajuda a identificar etapas
  com diferença numérica relevante.
- `fuzzy_pipeline_comparison.py`: versão executável do notebook de comparação
  fuzzy/FC, útil para rodar no servidor.
- `threshold_parameter_sweep.py`: executa vários runs do pipeline variando
  limiar inferior, limiar superior e parâmetros do threshold fuzzy, mantendo
  `region growing`.

## Notebooks relacionados

- `src/experiments/fuzzy_pipeline_comparison.ipynb`: compara threshold normal, fuzzy
  threshold, region growing e fuzzy connectedness.
- `src/eda/threshold_pipeline_comparison_analysis.ipynb`: analisa resultados
  consolidados das variantes de threshold, RG e FC.

Os helpers compartilhados ficam em `src/utils/experiments/`.

## Exemplos

```bash
uv run python src/experiments/fuzzy_pipeline_comparison.py --split train --sample-size 30 --no-gpu

uv run python src/experiments/fuzzy_pipeline_comparison.py \
  --split val \
  --sample-size 15 \
  --run-name fuzzy_val_fc_test \
  --variants normal_rg,normal_threshold_fc,normal_fc_semi_permissive

uv run python src/experiments/threshold_parameter_sweep.py \
  --split train \
  --percentiles 1,2,5,10 \
  --num-batches 5 \
  --gpu \
  --no-save-cache
```
