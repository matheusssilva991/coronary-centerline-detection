# Experimentos mantidos

Esta pasta contém somente experimentos ainda úteis para comparação ou ajuste
do pipeline. Testes automatizados continuam em `tests/`; resultados derivados
ficam em `output/segmentation/analysis/`.

## Scripts

- `fuzzy_pipeline_comparison.py`: compara threshold normal/fuzzy combinado com
  region growing ou fuzzy connectedness.
- `compare_cpu_gpu.py`: compara resultados intermediários dos backends CPU e
  GPU para localizar diferenças numéricas.
- `threshold_parameter_sweep.py`: varia limites inferior/superior e parâmetros
  do threshold fuzzy, mantendo o restante do pipeline controlado.
- `pipeline_failure_analysis.py`: seleciona, a partir dos quatro runs de
  validação, uma coorte focada de falhas e controles para testar correções.
- `pipeline_parameter_validation.py`: executa a análise OFAT de sensibilidade
  do artigo ou o diagnóstico dos grupos escalados para high resolution. Compara
  configurações controladas no split de validação e mede sucesso dos óstios e
  Dice. A referência congelada em
  `config/article_cbeb_sensitivity.json` usa P99.9 como referência do artigo do
  CBEB. Seus resultados alimentam a análise OFAT em
  `src/eda/pipeline_sensitivity_analysis.ipynb` e a investigação específica
  dos percentis em `src/eda/upper_threshold_analysis.ipynb`.
- `hybrid_resolution_pipeline.py`: localiza a aorta e os óstios em mid
  resolution, reescala as coordenadas `(y, x, z)` e executa somente o
  pré-processamento e a segmentação arterial em high resolution. Salva o
  progresso após cada exame em
  `output/segmentation/analysis/hybrid_resolution_pipeline/`.

Helpers reutilizáveis ficam em `src/utils/experiments/`.

## Runners

O runner de threshold executa sua seleção de parâmetros:

```bash
bash src/experiments/runners/run_threshold_sweeps.sh
```

O runner de falhas compara correções isoladas de RG, FC e fuzzy threshold:

```bash
MODE=corrections bash src/experiments/runners/run_pipeline_failure_improvement.sh
```

Para selecionar uma GPU:

```bash
CUDA_VISIBLE_DEVICES=1 bash src/experiments/runners/run_threshold_sweeps.sh
```

## Exemplos

```bash
uv run python src/experiments/fuzzy_pipeline_comparison.py \
  --split val \
  --sample-size 60 \
  --variants normal_rg,fuzzy_threshold_rg,normal_threshold_fc,fuzzy_threshold_fc

uv run python src/experiments/threshold_parameter_sweep.py \
  --split train \
  --percentiles 1,2,5,10 \
  --num-batches 5 \
  --gpu

uv run python src/experiments/compare_cpu_gpu.py --help

uv run python src/experiments/pipeline_parameter_validation.py \
  --split val \
  --sample-size 30 \
  --resolution mid \
  --gpu
```

Para diagnosticar somente a localização dos óstios em high resolution, sem
calcular vesselness arterial, RG/FC ou pós-processamento, execute primeiro uma
triagem curta:

```bash
uv run python src/experiments/pipeline_parameter_validation.py \
  --study resolution_scaling \
  --ostia-only \
  --split val \
  --sample-size 12 \
  --resolution high \
  --config-path config/article_cbeb_sensitivity.json \
  --run-name high_res_scaling_ostia_val12 \
  --gpu
```

O estudo mantém os raios da Hough escalados na referência e desativa um grupo
por variante: geometria dos círculos, rastreamento auxiliar, iterações e
morfologia do level set, erosão da superfície e quantidade de candidatos. A
variante combinada `morphology_radii_unscaled` mantém nos valores mid-res os
dois raios morfológicos 3D mais sensíveis à anisotropia entre XY e Z. Depois da
triagem, repita somente `all_scaled` e as duas melhores variantes em 60 imagens
usando `--variants` e outro `--run-name`.

Depois da triagem inicial, a rodada abaixo concentra a análise em falhas de
óstios da validação high-res e em controles próximos do limite de 7 mm. Ela
isola os três parâmetros do rastreamento que antes eram alterados em conjunto e
testa um número intermediário de iterações do level set. Os IDs explícitos são
validados contra o split informado para evitar vazamento do conjunto de teste.

```bash
uv run python src/experiments/pipeline_parameter_validation.py \
  --study resolution_scaling \
  --ostia-only \
  --split val \
  --ids 180,269,293,961,188,960,753,209,340,556,470,450,150,755,557,539,9,464,296,122,190,460,978,325 \
  --resolution high \
  --config-path config/article_cbeb_sensitivity.json \
  --variants all_scaled,canny_sigma_mid,neighbor_distance_mid,local_roi_padding_mid,level_set_iterations_50 \
  --run-name high_res_scaling_ostia_failures_val24 \
  --gpu
```

Os primeiros 20 exames representam falhas unilaterais ou bilaterais em high
resolution que tiveram sucesso na referência mid-res. Os quatro últimos são
controles high-res próximos do limite de tolerância. Variantes sem mudança na
triagem de 12 imagens ficam fora desta rodada, mas continuam disponíveis para
reprodutibilidade.

Após concluir essa rodada, acrescente os refinamentos de Canny e level set ao
mesmo diretório. `--append` ignora as variantes já completas e preserva os
resultados anteriores:

```bash
uv run python src/experiments/pipeline_parameter_validation.py \
  --study resolution_scaling \
  --ostia-only \
  --split val \
  --ids 180,269,293,961,188,960,753,209,340,556,470,450,150,755,557,539,9,464,296,122,190,460,978,325 \
  --resolution high \
  --config-path config/article_cbeb_sensitivity.json \
  --variants canny_sigma_4,canny_sigma_5,level_set_iterations_60,canny_sigma_3_level_set_50,canny_sigma_4_level_set_50,canny_sigma_5_level_set_50,roi_padding_40_level_set_50 \
  --run-name high_res_scaling_ostia_failures_val24 \
  --append \
  --gpu
```

Depois de selecionar a configuração vencedora na coorte difícil, execute apenas
ela nas 270 imagens de validação:

```bash
uv run python src/experiments/pipeline_parameter_validation.py \
  --study resolution_scaling \
  --split val \
  --sample-size 270 \
  --resolution high \
  --config-path config/article_cbeb_sensitivity.json \
  --variants canny_sigma_3_level_set_50 \
  --run-name high_res_canny3_ls50_full_val270 \
  --gpu
```

O baseline não precisa ser recalculado: nas 24 imagens pareadas, `all_scaled`
reproduziu exatamente Dice, distâncias dos óstios e volume da aorta do run
P99.9 já concluído em
`output/segmentation/runs/high_res/legacy_low_ostia_accuracy/p99_9/val/2026-08-11_13-31-45`.
Quando a execução for interrompida, repita o mesmo comando com `--append`; os
pares já salvos em `results/image_results.csv` serão ignorados.

Para fazer a triagem pareada das cinco variantes híbridas em cinco imagens:

```bash
uv run python src/experiments/hybrid_resolution_pipeline.py \
  --split train \
  --sample-size 5 \
  --config-path config/article_cbeb_sensitivity.json \
  --upper-threshold-percentile 99.9 \
  --variants recommended \
  --run-name hybrid_tuning_train_5 \
  --gpu
```

As variantes recomendadas isolam uma mudança por vez:

- `baseline_high_scaled`: configuração high atual, usada como referência;
- `morphology_mid_radii`: mantém RG/Frangi high e usa fechamento 3, dilatação 2;
- `rg_mid_thresholds`: usa divisor 7 e fração mínima de vesselness 0,078;
- `rg_mid_thresholds_morphology_mid`: combina os dois ajustes anteriores;
- `artery_sigmas_physical_x2`: testa sigmas `[3, 4, 5, 6]` em high.

As quatro variantes com os sigmas originais compartilham o mesmo mapa de
vesselness. O mapa com sigmas duplicados só é calculado depois e o mapa anterior
é liberado, reduzindo tempo e pico de memória. Depois da triagem, confirme as
variantes promissoras nas 30 imagens, por exemplo:

```bash
uv run python src/experiments/hybrid_resolution_pipeline.py \
  --split train \
  --sample-size 30 \
  --config-path config/article_cbeb_sensitivity.json \
  --upper-threshold-percentile 99.9 \
  --variants baseline_high_scaled,morphology_mid_radii,rg_mid_thresholds_morphology_mid \
  --run-name hybrid_confirmation_train_30 \
  --gpu
```

Use `--sample-size 0` para todo o split ou `--ids 1,2,3` para exames
específicos. Se a execução for interrompida, retome no mesmo diretório:

```bash
uv run python src/experiments/hybrid_resolution_pipeline.py \
  --split train \
  --sample-size 5 \
  --config-path config/article_cbeb_sensitivity.json \
  --upper-threshold-percentile 99.9 \
  --variants recommended \
  --resume-dir output/segmentation/analysis/hybrid_resolution_pipeline/SEU_RUN \
  --gpu
```

O CSV registra variante, parâmetros efetivos, óstios antes/depois da reescala,
Dice high antes/depois da morfologia, razão entre volumes predito/GT e tempos.
O resumo inclui diferença pareada e vitórias contra `baseline_high_scaled`. O
experimento não interpola máscaras mid para high: apenas as coordenadas dos
óstios são transferidas.

Para confirmar no conjunto completo de validação os parâmetros que precisam ser
recalculados sobre o novo baseline P99.9:

```bash
uv run python src/experiments/pipeline_parameter_validation.py \
  --split val \
  --sample-size 270 \
  --resolution mid \
  --variants baseline,ostia_z30,ostia_z50,rg_vessel_05,rg_vessel_09 \
  --run-name sensitivity_cbeb_p999_val_270 \
  --gpu
```

O experimento processa uma imagem por vez. Variantes que alteram somente a
seleção dos óstios ou o Region Growing compartilham downsampling, threshold,
LCC, vesselness, círculos e máscara da aorta em memória. Nenhum cache volumétrico
é salvo em disco. Variações do percentil superior compartilham apenas o
carregamento/downsampling, pois mudam todas as etapas posteriores.

A validação pode ser dividida sem separar os resultados. Execute a primeira
parte com um `--run-name` fixo:

```bash
uv run python src/experiments/pipeline_parameter_validation.py \
  --split val --sample-size 270 --resolution mid --gpu \
  --variants baseline,upper_p995,upper_p997 \
  --run-name sensitivity_cbeb_p999_val_270
```

Depois anexe as variantes restantes ao mesmo run:

```bash
uv run python src/experiments/pipeline_parameter_validation.py \
  --split val --sample-size 270 --resolution mid --gpu \
  --variants rg_vessel_05,rg_vessel_09 \
  --run-name sensitivity_cbeb_p999_val_270 \
  --append
```

`--append` confere IDs, split, resolução, configuração, método de aorta/óstios
e backend antes de combinar os CSVs. Variantes já concluídas são ignoradas.

As variantes `ostia_z30` e `ostia_z50` estão disponíveis para a sensibilidade
do limite axial, mas não alteraram os resultados na triagem de 30 imagens. Para
provocar uma análise mais informativa da localização, podem ser anexadas duas
variantes OFAT da região de busca: 70% e 100% da extensão inferior da aorta. O
baseline utiliza 85%:

```bash
uv run python src/experiments/pipeline_parameter_validation.py \
  --split val --sample-size 270 --resolution mid --gpu \
  --variants ostia_lower_70,ostia_lower_100 \
  --run-name sensitivity_cbeb_p999_val_270 \
  --append
```

O sweep de threshold mantém apenas configurações, resumos e o CSV consolidado
por imagem. Para preservar excepcionalmente os runs internos completos, use
`--keep-pipeline-runs`.

Experimentos encerrados e as razões para descarte estão documentados em
[`output/segmentation/analysis/EXPERIMENTS_ARCHIVE.md`](../../output/segmentation/analysis/EXPERIMENTS_ARCHIVE.md).

## Cobertura da trajetória circular da aorta

O runner abaixo testa, por padrão, cobertura mínima de 65% com rejeição da
trajetória quando a máscara filtrada continua `oversegmented`. Ele executa
treino-30 e validação-60, salva os HTMLs para inspeção visual e usa como base
`config/article_cbeb_sensitivity.json` (`-300 HU`/P99.9):

```bash
bash src/experiments/runners/run_aorta_circle_coverage_tests.sh
```

Use `RUN_TRAIN=0` ou `RUN_VAL=0` para executar somente um subconjunto, e
`SAVE_VISUALS=0` quando forem necessários apenas os resultados numéricos.

Para repetir o filtro agressivo nas 30 imagens de treino e gerar os HTMLs, use:

```bash
FILTER_PROFILE=aggressive RUN_VAL=0 SAVE_VISUALS=1 \
  bash src/experiments/runners/run_aorta_circle_coverage_tests.sh
```

Esse perfil usa cobertura mínima de 40%, não interpola outliers isolados e não
aplica fallback. O resultado é salvo em
`runs/mid_res/aorta_segmentation_experiments/train/circle_filter_aggressive_p99_9/`.

Os parâmetros `COVERAGE`, `RUN_WITHOUT_FALLBACK` e `RUN_WITH_FALLBACK` permitem
reutilizar o runner. Por exemplo, a comparação histórica de 60% com e sem
fallback pode ser reproduzida com:

```bash
COVERAGE=0.60 RUN_WITHOUT_FALLBACK=1 \
  bash src/experiments/runners/run_aorta_circle_coverage_tests.sh
```
