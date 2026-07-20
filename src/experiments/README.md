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
- `artery_vesselness_fc_sweep.py`: otimiza o mapa de vesselness arterial, os
  parâmetros de fuzzy connectedness e o pós-processamento comum a RG/FC.
  Reutiliza a parte comum do pipeline, os mapas de vesselness e as máscaras
  brutas dentro de cada imagem, sem criar cache volumétrico.
  As grades ficam em `artery_vesselness_fc_variants.json` para serem ajustadas
  sem aumentar o script.
- `threshold_parameter_sweep.py`: executa vários runs do pipeline variando
  limiar inferior, limiar superior e parâmetros do threshold fuzzy, mantendo
  `region growing`.

## Notebooks relacionados

- [`threshold_pipeline_comparison_analysis.ipynb`](../eda/threshold_pipeline_comparison_analysis.ipynb): analisa resultados
  consolidados das variantes de threshold, RG e FC.
- [`src/eda/README.md`](../eda/README.md): catálogo completo das análises, entradas, saídas e custo
  aproximado de execução.

Os helpers compartilhados ficam em `src/utils/experiments/`.

## Runners

Os scripts `.sh` usados para rodar sweeps longos ficam em
`src/experiments/runners/`. Eles fazem `cd` automático para a raiz do projeto,
então podem ser chamados a partir de qualquer pasta.

```bash
bash src/experiments/runners/run_threshold_sweeps.sh

bash src/experiments/runners/run_aorta_recovery_adjustments.sh

# Etapa 1: seis mapas de vesselness, cada um avaliado com RG e FC.
bash src/experiments/runners/run_artery_vesselness_fc_sweeps.sh

# Etapa 2: refina FC após escolher o melhor perfil da etapa 1.
STAGE=fc VESSELNESS_PROFILE=current RUN_NAME=artery_fc_train30 \
  bash src/experiments/runners/run_artery_vesselness_fc_sweeps.sh

# Etapa 3: refina RG/FC e compara quatro morfologias nas mesmas máscaras.
bash src/experiments/runners/run_artery_segmentation_refinement.sh

# Etapa 4: critérios de RG/FC, dilatação condicionada e recuperação por ramo.
bash src/experiments/runners/run_artery_segmentation_optimization.sh

# Etapa 5: parâmetros compartilhados e específicos para cada artéria.
bash src/experiments/runners/run_artery_branch_parameter_sweep.sh

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

O sweep arterial salva `image_results.csv`, `ranking.csv`,
`variant_parameters.csv` e `pairwise_vs_reference.csv` em
`output/segmentation/analysis/artery_vesselness_fc_sweep/<run>/results/`.
No estágio `vesselness`, RG e FC compartilham o mesmo mapa em memória. No
estágio `fc`, todas as variantes compartilham o perfil selecionado. No estágio
`refinement`, são executadas 10 segmentações por imagem e cada máscara bruta é
reutilizada em quatro perfis morfológicos, gerando 40 avaliações. O ranking
inclui Dice, sensibilidade e precisão antes/depois da morfologia. Para retomar
uma execução interrompida:

```bash
RESUME_DIR=output/segmentation/analysis/artery_vesselness_fc_sweep/<run> \
  STAGE=vesselness \
  bash src/experiments/runners/run_artery_vesselness_fc_sweeps.sh
```

Para retomar o refinamento:

```bash
RESUME_DIR=output/segmentation/analysis/artery_vesselness_fc_sweep/<run> \
  bash src/experiments/runners/run_artery_segmentation_refinement.sh
```

Um teste curto de configuração pode ser feito sem processar imagens:

```bash
DRY_RUN=1 bash src/experiments/runners/run_artery_segmentation_refinement.sh
```

O estágio `optimization` executa 16 segmentações e avalia cada máscara com a
morfologia atual e três dilatações condicionadas por vesselness. As variantes
de recuperação repetem somente ramos com menos de 500 voxels, usando sementes
locais mais amplas e parâmetros relaxados uma única vez. Para executar ou
retomar:

```bash
bash src/experiments/runners/run_artery_segmentation_optimization.sh

RESUME_DIR=output/segmentation/analysis/artery_vesselness_fc_sweep/<run> \
  bash src/experiments/runners/run_artery_segmentation_optimization.sh
```

O sweep de parâmetros por ramo usa `standard` como perfil de aorta/óstios para
preservar o controle histórico. Para repetir o estudo com a estratégia
bilateral, execute:

```bash
AORTA_OSTIA_METHOD=bilateral_thin \
  bash src/experiments/runners/run_artery_branch_parameter_sweep.sh
```

`MODE=shared` avalia somente parâmetros comuns, `MODE=branch` avalia apenas
referências e overrides independentes e `MODE=all` executa os dois grupos. Por
padrão, cada máscara bruta é reutilizada na morfologia histórica e nas duas
dilatações condicionadas selecionadas nos estudos anteriores.

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
