# Experiments

Scripts executáveis para estudos auxiliares do pipeline.

Esta pasta é diferente de `tests/`: aqui ficam experimentos manuais ou
diagnósticos longos, geralmente rodados no PC ou servidor. Testes automatizados
de unidade/CI devem ficar em `tests/` se forem criados no futuro.

## Scripts atuais

- `compare_cpu_gpu.py`: compara saídas de CPU/GPU e ajuda a identificar etapas
  com diferença numérica relevante.
- `aorta_ostia_parameter_sweep.py`: executa runs do pipeline variando parâmetros
  da localização da aorta e da detecção dos óstios. Também salva métricas sobre
  quantidade de fatias, círculos detectados e cobertura da aorta.
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

## Runners

Os scripts `.sh` usados para rodar sweeps longos ficam em
`src/experiments/runners/`. Eles fazem `cd` automático para a raiz do projeto,
então podem ser chamados a partir de qualquer pasta.

```bash
bash src/experiments/runners/run_threshold_sweeps.sh

bash src/experiments/runners/run_aorta_ostia_sweeps.sh
```

Para escolher uma GPU específica:

```bash
CUDA_VISIBLE_DEVICES=1 bash src/experiments/runners/run_aorta_ostia_sweeps.sh
```

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

uv run python src/experiments/aorta_ostia_parameter_sweep.py \
  --split train \
  --run-name aorta_ostia_train_quick \
  --threshold-preset best_normal \
  --timeout-minutes 180 \
  --gpu \
  --no-save-cache
```
