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

## Notebooks relacionados

- `src/fuzzy_pipeline_comparison.ipynb`: compara threshold normal, fuzzy
  threshold, region growing e fuzzy connectedness.

Os helpers compartilhados ficam em `src/utils/experiments/`.

## Exemplos

```bash
uv run python src/experiments/fuzzy_pipeline_comparison.py --split train --sample-size 30 --no-gpu

uv run python src/experiments/fuzzy_pipeline_comparison.py \
  --split val \
  --sample-size 15 \
  --run-name fuzzy_val_fc_test \
  --variants normal_rg,normal_threshold_fc,normal_fc_semi_permissive
```
