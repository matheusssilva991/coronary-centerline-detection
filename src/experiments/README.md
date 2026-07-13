# Experiments

Scripts executáveis para estudos auxiliares do pipeline.

Esta pasta é diferente de `tests/`: aqui ficam experimentos manuais ou
diagnósticos longos, geralmente rodados no PC ou servidor. Testes automatizados
de unidade/CI devem ficar em `tests/` se forem criados no futuro.

## Scripts atuais

- `compare_cpu_gpu.py`: compara saídas de CPU/GPU e ajuda a identificar etapas
  com diferença numérica relevante.
- `aorta_recovery_comparison.py`: compara somente os ajustes ainda úteis da
  recuperação inicial da trajetória da aorta.
- `aorta_mask_ostia_comparison.py`: compara refinamento da máscara pela
  trajetória dos círculos e estratégias de seleção dos candidatos de óstio.
- `fuzzy_pipeline_comparison.py`: versão executável do notebook de comparação
  das quatro combinações mantidas entre threshold normal/fuzzy e RG/FC.
- `threshold_parameter_sweep.py`: executa vários runs do pipeline variando
  limiar inferior, limiar superior e parâmetros do threshold fuzzy, mantendo
  `region growing`.

## Notebooks relacionados

- `src/eda/threshold_pipeline_comparison_analysis.ipynb`: analisa resultados
  consolidados das variantes de threshold, RG e FC.

Os helpers compartilhados ficam em `src/utils/experiments/`.

## Runners

Os scripts `.sh` usados para rodar sweeps longos ficam em
`src/experiments/runners/`. Eles fazem `cd` automático para a raiz do projeto,
então podem ser chamados a partir de qualquer pasta.

```bash
bash src/experiments/runners/run_threshold_sweeps.sh

bash src/experiments/runners/run_aorta_recovery_adjustments.sh

# Triagem: 9 falhas conhecidas + 15 controles, em 9 variantes.
bash src/experiments/runners/run_aorta_mask_ostia_comparison.sh

# Ablação da trajetória: baseline, superfície fina e fatores 1.50/1.75/2.00.
MODE=ablation RUN_NAME=aorta_trajectory_ablation \
  bash src/experiments/runners/run_aorta_mask_ostia_comparison.sh

# Sweep avançado: 14 variantes em 30 imagens novas do split de validação.
MODE=advanced RUN_NAME=aorta_ostia_advanced_val30 \
  bash src/experiments/runners/run_aorta_mask_ostia_comparison.sh

# Seleção bilateral: seis variantes em outras 30 imagens inéditas.
MODE=bilateral RUN_NAME=aorta_ostia_bilateral_val30 \
  bash src/experiments/runners/run_aorta_mask_ostia_comparison.sh

# Confirmação das variantes selecionadas em 60 imagens.
MODE=full SAMPLE_SIZE=60 RUN_NAME=aorta_mask_ostia_val60 \
  bash src/experiments/runners/run_aorta_mask_ostia_comparison.sh

# Confirma somente baseline e as variantes vencedoras da triagem.
MODE=full SAMPLE_SIZE=60 \
VARIANTS=baseline,combined_local,trajectory_local \
RUN_NAME=aorta_mask_ostia_selected_val60 \
  bash src/experiments/runners/run_aorta_mask_ostia_comparison.sh
```

Para escolher uma GPU específica:

```bash
CUDA_VISIBLE_DEVICES=1 bash src/experiments/runners/run_threshold_sweeps.sh
```

O comparador de aorta/óstios salva `summary.csv`, `image_results.csv`,
`pairwise_by_image.csv` e `pairwise_summary.csv` em
`output/segmentation/analysis/aorta_mask_ostia_comparison/<run>/results/`.
Por padrão, ele reaproveita e salva somente o cache comprimido de vesselness.
Use `--no-cache` ao chamar o script Python para desabilitar esse comportamento.

## Exemplos

```bash
uv run python src/experiments/fuzzy_pipeline_comparison.py --split train --sample-size 30 --no-gpu

uv run python src/experiments/fuzzy_pipeline_comparison.py \
  --split val \
  --sample-size 15 \
  --run-name fuzzy_val_fc_test \
  --variants normal_rg,fuzzy_threshold_rg,normal_threshold_fc,fuzzy_threshold_fc

uv run python src/experiments/threshold_parameter_sweep.py \
  --split train \
  --percentiles 1,2,5,10 \
  --num-batches 5 \
  --gpu \
  --no-save-cache

```
