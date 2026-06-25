# Organizacao dos outputs de segmentacao

Esta pasta guarda os resultados do pipeline de segmentacao, comparacoes de
backend e analises derivadas. A ideia principal e separar:

- execucoes completas do pipeline;
- resultados oficiais usados como referencia;
- experimentos e sweeps de parametros;
- comparacoes CPU/GPU;
- outputs exploratorios de notebooks.

## Estrutura atual

```text
output/segmentation/
  runs/
  canonical/
  analysis/
  backend_comparison/
  README.md
```

Pastas antigas, como `8.final_results/` ou `val_diff/`, quando existirem, devem
ser tratadas como historico. Resultados novos devem ir para `runs/`, `analysis/`
ou `backend_comparison/`.

## `runs/`: execucoes completas do pipeline

Cada execucao nova do `src/segmentation_pipeline.py` cria uma pasta em:

```text
output/segmentation/runs/<resolucao>_res/<timestamp>/
```

Exemplos:

```text
output/segmentation/runs/mid_res/2026-06-05_20-00-43/
output/segmentation/runs/high_res/2026-06-15_09-24-23/
```

Alguns estudos podem ficar agrupados por tema antes do timestamp, por exemplo:

```text
output/segmentation/runs/mid_res/train_diff/<timestamp>/
output/segmentation/runs/mid_res/downscale_method/<timestamp>/
output/segmentation/runs/mid_res/circle_detection_gpu_diff/<timestamp>/
```

Use esses agrupamentos quando quiser comparar muitas execucoes do mesmo tipo sem
misturar tudo na raiz de `mid_res/`.

### Estrutura de uma run

```text
<run>/
  config/
  logs/
  numeric/
  visual/        # criada apenas quando houver artefatos visuais
```

### `config/`

Guarda os arquivos que explicam como a run foi produzida.

```text
config/
  effective_pipeline_config.json
  split_ids.json
```

- `effective_pipeline_config.json`: configuracao efetiva usada no run, ja com
  overrides, resolucao e parametros escalados quando aplicavel.
- `split_ids.json`: IDs processados por split e, quando usado, o arquivo de
  split alternativo.

Use essa pasta para responder:

- quais parametros foram usados;
- quais IDs foram processados;
- se era `mid_res` ou `high_res`;
- se o split foi o padrao ou um split alternativo.

### `logs/`

```text
logs/
  pipeline.log
```

Use para investigar falhas, retomadas, mensagens detalhadas e progresso da
execucao.

### `numeric/`

Guarda os CSVs e metadados numéricos do pipeline.

```text
numeric/
  ostios_<split>_summary.csv
  ostios_<split>_metadata.json
  ostios_<split>_lote_<n>_summary.csv
  ostios_<split>_batch_timings.csv
```

Arquivos principais:

- `ostios_<split>_summary.csv`: resultado consolidado final do split.
- `ostios_<split>_metadata.json`: metadados da execucao e resumo agregado.
- `ostios_<split>_lote_<n>_summary.csv`: resultado parcial de cada lote.
- `ostios_<split>_batch_timings.csv`: tempo individual de cada lote.

Em caso de queda do servidor, os lotes ja salvos continuam em `numeric/`. Use
`--resume-dir` e `--resume-batch` para continuar a partir do lote desejado.

O arquivo de tempos por lote registra:

- `batch_number`;
- `num_images`;
- `started_at`;
- `finished_at`;
- `duration_seconds`;
- `duration_minutes`;
- `duration_hours`;
- `result_file`.

O metadata final tambem registra tempos em segundos, minutos e horas, incluindo
tempo total conhecido por lote e tempo apenas da execucao atual.

### `visual/`

Reservado para HTMLs, PNGs ou outros exemplos visuais ligados diretamente a uma
run. O pipeline nao cria essa pasta quando nao ha nada visual para salvar.

Sugestao:

```text
visual/
  ostia_examples/
  artery_examples/
  hough_examples/
```

Use `visual/` quando a figura depende daquela run especifica. Exemplos soltos ou
mais exploratorios devem ir para `analysis/visual_examples/`.

## `canonical/`: resultados oficiais de referencia

`canonical/` guarda os resultados que devem ser considerados referencia atual
para notebooks, comparacoes e relatorios.

Estrutura:

```text
canonical/
  mid_res/
    train/<timestamp>/
    val/<timestamp>/
    test/<timestamp>/
  high_res/
    train/<timestamp>/
    val/<timestamp>/
    test/<timestamp>/
```

Exemplos atuais em `mid_res`:

```text
canonical/mid_res/train/2026-06-03_09-31-08/
canonical/mid_res/val/2026-06-05_14-46-06/
canonical/mid_res/test/2026-06-05_20-00-43/
```

Cada pasta canonica deve manter a mesma estrutura de uma run:

```text
config/
logs/
numeric/
visual/        # se existir
```

Regra pratica: `runs/` pode ter varias tentativas; `canonical/` deve apontar ou
conter apenas o resultado escolhido como referencia.

## `analysis/`: experimentos e analises derivadas

Use `analysis/` para outputs que nao sao uma execucao direta do pipeline
principal.

Estrutura atual:

```text
analysis/
  fuzzy_sweep/
  fuzzy_pipeline_comparison/
  visual_examples/
```

### `analysis/fuzzy_pipeline_comparison/`

Resultados do notebook `src/fuzzy_pipeline_comparison.ipynb`. Use para comparar:

- threshold HU normal;
- threshold fuzzy alpha-cut;
- region growing;
- fuzzy connectedness;
- combinacao de threshold fuzzy + fuzzy connectedness.

Estrutura esperada:

```text
fuzzy_pipeline_comparison/
  <run_name>/
    summary/ranking.csv
    results/image_results.csv
    parameters/variant_parameters.csv
```

### `analysis/fuzzy_sweep/`

Historico de varreduras exploratorias antigas. Ela pode ter subpastas como:

```text
fuzzy_sweep/
  runs/<run_name>/
    summary/
    results/
    parameters/
    partial/
    debug/       # opcional
```

### Pastas antigas de sweep

Qualquer pasta antiga de varredura que ainda existir em `analysis/` deve ser
tratada como historico. Os scripts de sweep foram removidos para manter os
experimentos atuais mais simples.


### `analysis/visual_examples/`

Figuras e exemplos visuais que nao pertencem claramente a uma run especifica.

Exemplo:

```text
analysis/visual_examples/ostia_3d/
```

## `backend_comparison/`: comparacao CPU/GPU

Resultados do comparador de backend ficam em:

```text
backend_comparison/<resolucao>_res/<timestamp>/
```

Arquivos principais:

```text
stage_comparison.csv
ostia_comparison.csv
timing_comparison.csv
run_config.json
```

Use essa pasta para investigar diferencas entre CPU e GPU por etapa do pipeline,
especialmente vesselness, localizacao de circulos, segmentacao de aorta,
deteccao de ostios e segmentacao arterial.

## Comandos uteis

### Rodar treino em mid resolution

```bash
uv run python src/segmentation_pipeline.py \
  --resolution mid \
  --split train \
  --num-batches 5
```

Saida:

```text
output/segmentation/runs/mid_res/<timestamp>/
```

### Rodar high resolution

```bash
uv run python src/segmentation_pipeline.py \
  --resolution high \
  --split train \
  --num-batches 5
```

Saida:

```text
output/segmentation/runs/high_res/<timestamp>/
```

### Rodar com split alternativo

```bash
uv run python src/segmentation_pipeline.py \
  --resolution mid \
  --split train \
  --num-batches 5 \
  --split-config config/imagecas_splits_train90.json
```

Para voltar ao split padrao, remova `--split-config`.

### Retomar uma execucao que caiu

Se caiu antes de terminar o lote 3:

```bash
uv run python src/segmentation_pipeline.py \
  --resolution mid \
  --split train \
  --num-batches 5 \
  --resume-batch 3 \
  --resume-dir output/segmentation/runs/mid_res/<timestamp>
```

O pipeline volta a salvar os proximos lotes como `lote_3`, `lote_4`, etc. Se
voce retomou do lote 11, os arquivos novos continuam como `lote_11`,
`lote_12`, e assim por diante.

### Retomar varios splits

```bash
uv run python src/segmentation_pipeline.py \
  --split all \
  --num-batches 5 \
  --resume-batches train=0,val=3,test=0 \
  --resume-dir output/segmentation/runs/mid_res/<timestamp>
```

Use `0` para splits que nao devem ser reprocessados naquela retomada.

### Apenas consolidar lotes existentes

```bash
uv run python src/segmentation_pipeline.py \
  --merge-only \
  --split train \
  --resume-dir output/segmentation/runs/mid_res/<timestamp>
```

Esse modo nao reprocessa imagens. Ele junta os CSVs de lote existentes e
atualiza o metadata.

### Forcar CPU ou GPU

```bash
uv run python src/segmentation_pipeline.py --split train --gpu
uv run python src/segmentation_pipeline.py --split train --no-gpu
```

## Como encontrar rapidamente

- Resultado bruto do pipeline:
  `runs/<resolucao>_res/<timestamp>/numeric/`
- Resultado bruto agrupado por estudo:
  `runs/<resolucao>_res/<tema>/<timestamp>/numeric/`
- Configuracao e IDs:
  `runs/<resolucao>_res/<timestamp>/config/`
- Log:
  `runs/<resolucao>_res/<timestamp>/logs/pipeline.log`
- Resultado oficial:
  `canonical/<resolucao>_res/<split>/<timestamp>/numeric/`
- Comparacao fuzzy/FC atual:
  `analysis/fuzzy_pipeline_comparison/<run_name>/`
- Sweeps exploratorios antigos:
  `analysis/fuzzy_sweep/runs/<run_name>/`
- Comparacao CPU/GPU:
  `backend_comparison/<resolucao>_res/<timestamp>/`
