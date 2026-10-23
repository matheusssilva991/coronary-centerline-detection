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
  `src/eda/sensitivity/pipeline_sensitivity_analysis.ipynb` e a investigação específica
  dos percentis em `src/eda/sensitivity/upper_threshold_analysis.ipynb`.
- O estudo `artery_region_growing` de `pipeline_parameter_validation.py`
  mantém disponível somente a grade dos parâmetros ativos do primeiro RG.
  Estratégias descartadas estão registradas em
  `output/segmentation/docs/experiments/artery_region_growing.md`.

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
[`output/segmentation/docs/experiments/README.md`](../../output/segmentation/docs/experiments/README.md).

## Filtro e envelope da trajetória da aorta

O runner `run_aorta_filter_envelope_generalization.sh` mantém fixos o threshold
`-300 HU`/P99.9, o filtro robusto com cobertura mínima de 40%, o limite de corte
de 40%, cinco círculos sintéticos e o level set fixo. A configuração de
referência usa `balloon=0.6`, semente de 10% do raio e 26 iterações, combinação
que obteve `30/30` aortas adequadas no treino e `56/60` na validação visual.

As quatro variantes alteram somente o raio e a margem axial do envelope:

| Variante | Raio | Margem axial | Objetivo |
| --- | ---: | ---: | --- |
| `reference` | `2.25r` | 10 | Reproduzir a referência atual |
| `balanced` | `2.35r` | 10 | Pequena proteção contra subsegmentação |
| `conservative` | `2.40r` | 12 | Preservar mais as extremidades da aorta |
| `anti_leak` | `2.15r` | 12 | Restringir lateralmente sem cortar cedo no eixo z |

Execute primeiro no treino:

```bash
bash src/experiments/runners/run_aorta_filter_envelope_generalization.sh
```

Depois, confirme as mesmas variantes nas 60 imagens de validação:

```bash
SPLIT=val bash src/experiments/runners/run_aorta_filter_envelope_generalization.sh
```

Use `SAVE_VISUALS=0` para omitir os HTMLs. O runner reproduz apenas a
configuração selecionada: Hough `18-29 px`, filtro robusto, cinco círculos
sintéticos e envelope `2.25r` com margem axial 10.

As tentativas encerradas de level set adaptativo isolado, correção condicional
e recuperação de trajetórias curtas estão resumidas no arquivo de experimentos.
Seus runners foram removidos para evitar novas execuções acidentais.

O sweep de raios e o override de vazamento foram encerrados e removidos. A
comparação completa de 90 imagens selecionou `18-29 px`; as demais faixas e o
override não apresentaram ganho robusto e permanecem apenas no histórico.

## Sensibilidade da localização dos óstios com filtro e envelope

O runner `run_ostia_localization_filter_envelope.sh` mantém fixos o filtro
robusto da trajetória e o envelope `2.25r` com
margem axial 10. Ele varia somente os parâmetros de localização dos óstios e o
padding externo da superfície morfológica. Por padrão, executa as 30 imagens de
treino e as 60 de validação:

```bash
bash src/experiments/runners/run_ostia_localization_filter_envelope.sh
```

O padrao executa duas variantes: `baseline_pad0` (controle) e
`lower100_pad2` (combinacao selecionada). O padding usa raio em
voxels, deslocando a casca para fora sem alterar a mascara da aorta.

Para rodar somente a combinacao principal:

```bash
VARIANTS=lower100_pad2 \
  bash src/experiments/runners/run_ostia_localization_filter_envelope.sh
```

Para executar somente a validação ou usar GPU:

```bash
SPLITS=val USE_GPU=1 \
  bash src/experiments/runners/run_ostia_localization_filter_envelope.sh
```

Os resultados ficam em `output/segmentation/runs/mid_res/aorta_segmentation_experiments/{split}/ostia_localization/`, separados por variante.

Uma variante opcional, `lower100_pad3`, amplia o padding para 3 voxels.
Use `DRY_RUN=1` para conferir as configuracoes sem processar imagens e
`SAVE_VISUALS=1` para gerar os HTMLs. Os runs historicos com lower_fraction=1
excluiam a ultima fatia; os novos incluem toda a superficie.

### Validação dos óstios em 270 exames

Para confirmar na validacao completa (270 exames), use
`VAL_SPLIT_CONFIG=config/imagecas_splits.json`. Sem essa variavel, o runner
continua selecionando as 60 imagens de validacao.

```bash
SPLITS=val VAL_SPLIT_CONFIG=config/imagecas_splits.json \
VARIANTS_FILE= VARIANTS=baseline_pad0,lower100_pad2,lower100_pad3 \
RUN_FAMILY=ostia_validation270 USE_GPU=1 SAVE_VISUALS=0 \
bash src/experiments/runners/run_ostia_localization_filter_envelope.sh
```

O comando executa tres runs sequenciais com a mesma aorta (filtro + envelope),
variando a superficie de busca dos ostios. Selecione apenas um nome em
`VARIANTS` para executar cada run separadamente.

### Confirmacao final nos 700 exames de teste

O runner de ostios aceita `SPLITS=test`, usando os 700 IDs de
`config/imagecas_splits.json`. Para referencia com filtro + envelope:

```bash
SPLITS=test VARIANTS_FILE= VARIANTS=baseline_pad0 \
RUN_FAMILY=ostia_comparison USE_GPU=1 SAVE_VISUALS=0 \
bash src/experiments/runners/run_ostia_localization_filter_envelope.sh
```

Para o candidato, substituir por `VARIANTS=lower100_pad2`. O pad0 nesta
comparacao tambem usa filtro + envelope; nao e o P99.9 puro.
Os resultados ficam em `aorta_segmentation_experiments/test/ostia_comparison/`.
`SAVE_VISUALS=1` espelha a mesma estrutura no disco externo.

### Sensibilidade da trajetória (historico)

`runners/run_aorta_trajectory_geometry_sweep.sh` compara quatro pares de
limites físicos de mudança de raio/centro e isola o efeito do filtro robusto.

```bash
SPLIT=train bash src/experiments/runners/run_aorta_trajectory_geometry_sweep.sh
```

`VARIANTS` aceita uma lista separada por vírgulas para executar apenas parte
do sweep. Os visuais ficam desativados por padrão; use `SAVE_VISUALS=1` para
salvá-los no disco externo configurado por `VISUAL_OUTPUT_DIR`.

## Ablação do pipeline refinado: Level Set e mapa único

`pipeline_component_ablation.py` executa por padrão três variantes na resolução reduzida:

- `full_pipeline`: controle com o pipeline refinado completo.
- `no_level_set`: usa os círculos detectados com raio completo (fator 1,0),
  mantendo abertura morfológica, envelope e maior componente; não calcula
  gradientes nem executa a evolução Level Set.
- `single_vesselness`: calcula Frangi apenas uma vez com os parâmetros de
  `VESSELNESS_ARTERY` e usa o mesmo mapa nos óstios e no Region Growing.

A variante adicional `no_downsampling` deve ser selecionada explicitamente. Ela
altera somente `DOWNSCALE_FACTORS` de `[2, 2, 1]` para `[1, 1, 1]`, sem chamar
`scale_config_to_resolution`: raios em pixels, sigmas, 26 iterações de Level Set,
limites do Region Growing e demais parâmetros continuam numericamente idênticos.
As escalas físicas correspondentes aos valores em voxels mudam com a resolução;
esta é uma ablação sem reajuste, não o pipeline high-resolution otimizado.
O snapshot `config/no_downsampling.json` salva a configuração efetiva original.

```bash
uv run python src/experiments/pipeline_component_ablation.py --gpu --variants no_downsampling --split test
```

O comando usa `IMAGECAS_BASE_PATH`; alternativamente, acrescente `--base-path`.
Para executar o controle e essa ablação juntos, use
`--variants full_pipeline,no_downsampling`. A seleção padrão das três variantes
anteriores permanece inalterada.

A configuração congelada `config/article_cbeb_ablation.json` corresponde a
`effective_base_config` do run refinado `fuzzy_comparison/current_pipeline/test700`.
A variante de referência usa threshold normal e Region Growing. Os modos
experimentais são salvos separadamente da configuração científica. Não há
compartilhamento de intermediários entre variantes nem ajuste de parâmetros pelo
conjunto de teste.

A variante sem Level Set foi corrigida para não usar as sementes contraídas
(fator 0,10), cuja expansão dependia da evolução removida. Essa comparação agora
usa a aproximação geométrica dos círculos no lugar do refinamento Level Set,
sem alterar o pós-processamento. O manifesto registra `circle_mask` e
`aorta_circle_radius_factor=1.0`. Runs antigos com `initial_mask` são preservados;
a validação de retomada rejeita misturá-los com a variante corrigida. Execute a
nova definição em uma pasta nova. O fator 1,0 corresponde diretamente ao raio
medido pela Hough; não foi escolhido por ganho no conjunto de teste.

Os comandos abaixo devem ser executados na raiz do repositório. Defina
`IMAGECAS_BASE_PATH` com a pasta contendo os arquivos `ID.img.nii.gz` e
`ID.label.nii.gz`, ou forneça `--base-path`. O padrão usa os 700 IDs de teste de
`config/imagecas_splits.json`. `--split train` e `--split val` usam, respectivamente,
30 e 270 IDs. `--ids` aceita somente IDs do split escolhido. `--sample-size` e
`--start-index` permitem pilotos, sem alterar a configuração.

Inspeção sem carregar imagens nem criar arquivos:

```bash
uv run python src/experiments/pipeline_component_ablation.py --dry-run
```

Controle e ambas as ablações no teste completo:

```bash
uv run python src/experiments/pipeline_component_ablation.py --gpu
```

Teste sem Level Set, com controle no mesmo run:

```bash
uv run python src/experiments/pipeline_component_ablation.py --gpu --variants full_pipeline,no_level_set
```

Teste com mapa único, com controle no mesmo run:

```bash
uv run python src/experiments/pipeline_component_ablation.py --gpu --variants full_pipeline,single_vesselness
```

Para executar apenas a ablação, use `--variants no_level_set` ou
`--variants single_vesselness`. Para um piloto sintético/operacional com imagens
reais de treino, use `--split train --sample-size 3`. `--no-gpu` solicita CPU;
se GPU for solicitada mas indisponível, os helpers existentes usam CPU e essa
condição fica registrada em `runtime_environment.json`.

Os runs são criados em uma pasta nova dentro de
`output/segmentation/analysis/pipeline_component_ablation/runs/`, com nome único.
`--output-root` e `--run-name` permitem escolher o destino; pastas existentes nunca
são sobrescritas como uma nova execução. Os resultados são salvos após cada exame:

- `results/image_results.csv`: Dice observado, status dos óstios, erros e tempo por
  par variante/exame; falhas sem saída preservam Dice ausente.
- `summary/ablation_summary.csv` e `summary/ablation_table.md`: número concluído e
  planejado, sucesso dos óstios, Dice global, Dice condicionado e tempo médio.
- `run_config.json`, `config/split_ids.json` e `config/{variante}.json`: coorte,
  configuração efetiva, hash e modos experimentais.
- `runtime_environment.json`: ambiente e disponibilidade de GPU. Picos de RAM e
  VRAM não são medidos e permanecem vazios, não zero.

O resumo end-to-end inclui todos os exames concluídos; atribui zero somente aos
Dice ausentes acompanhados de erro explícito. Inicialização incorreta com saída
mantém seu Dice observado. O Dice condicionado usa a mesma regra operacional no
subconjunto de inicializações aceitas. Uma tabela parcial indica
`images/planned_images`; não deve ser apresentada como resultado do teste completo.
O cronômetro inclui carregamento e etapas completas, sincronizando CUDA quando
ativa; não inclui a gravação dos CSVs após cada exame.

Para retomar, repita as opções de seleção/configuração da execução original e
informe a pasta retornada pelo script:

```bash
uv run python src/experiments/pipeline_component_ablation.py --gpu --resume-dir output/segmentation/analysis/pipeline_component_ablation/runs/NOME_DO_RUN
```

Se o run contém somente duas variantes, repita também seu `--variants`. Retomadas
validam configuração, modos, dataset, split e IDs; pares já registrados, inclusive
falhas explícitas, não são reexecutados. O CSV por exame é a referência para
retomar; tabelas agregadas são reconstruídas. Não execute dois processos na mesma
pasta de retomada.

## RAM e VRAM em um único exame: Mid versus High

`pipeline_resource_benchmark.py` executa o pipeline completo em processos
independentes e sequenciais. Por padrão, usa o exame 642: 275 fatias, óstios aceitos e Dice histórico 0,7497 em Mid e 0,7261 em High. O caso
foi escolhido entre os exames de teste com Dice >= 0,65 nas duas resoluções,
priorizando o número de fatias e, nos empates, o menor Dice entre as resoluções.
É o máximo de fatias nesse grupo de boa segmentação; o banco completo também
contém um caso de 277 fatias, cujo Dice High é 0,4272. Não representa o pior
consumo de memória de todo o banco.

```bash
uv run python src/experiments/pipeline_resource_benchmark.py \
  --image-id 642 \
  --base-path /run/media/matheus/HD/DatasetsCCTA/ImageCAS/1-1000 \
  --resolutions mid,high \
  --gpu-index 0 \
  --interval 0.1
```

No servidor, substitua `--base-path` pelo caminho do ImageCAS nesse equipamento.
`--dry-run` verifica as entradas sem carregar os volumes. A GPU é obrigatória;
falhas de CUDA não são tratadas como uma medição equivalente em CPU. A configuração
base congelada é a mesma do artigo; em High aplicam-se os ajustes normais por
resolução, inclusive 70 iterações Level Set. Não se trata de `no_downsampling`.

- RAM: `ru_maxrss` do Linux, convertido de KiB para bytes. É o pico de memória
  residente do worker, incluindo imports, contexto CUDA e buffers de imagem.
- VRAM: máximo observado por PID via `nvidia-smi`, em intervalos de 0,1 s por
  padrão. A memória total da placa ou de outros processos não entra nessa medida.
  Picos menores que o intervalo de observação podem não ser capturados; o intervalo
  efetivo depende também da duração da consulta ao driver.
- Tempo: somente `process_image`, incluindo carregamento, pré-processamento,
  aorta, óstios, artérias e métricas; sincroniza CUDA no começo e no fim. Imports
  e inicialização do contexto CUDA ficam fora do cronômetro.

Cada execução cria uma pasta nova em
`output/segmentation/analysis/pipeline_resource_benchmark/runs/`, contendo
`resource_summary.csv`, `results.json`, `mid.json`, `high.json`, logs e amostras de
VRAM. As medidas originais ficam em bytes; a tabela também apresenta GiB (2³⁰
bytes), explicitamente distintos de GB decimais. Configuração efetiva, GPU,
UUID, driver, hostname, Dice e número de fatias acompanham a medição.

Recursos de um exame em uma máquina não devem ser apresentados como média do
banco nem combinados sem identificação com tempos de execução de outro servidor.
