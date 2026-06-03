# Organização dos outputs de segmentação

Esta pasta guarda os resultados do pipeline de segmentação e das análises
derivadas dele. A ideia é separar claramente:

- execuções completas do pipeline;
- resultados numéricos;
- exemplos visuais;
- análises feitas em notebooks;
- resultados oficiais/canônicos;
- resultados antigos que ficam apenas como histórico.

## Estrutura geral

```text
output/segmentation/
  runs/
  canonical/
  analysis/
  8.final_results/
  val_diff/
  README.md
```

## `runs/`: execuções do pipeline

Cada vez que o `segmentation_pipeline.py` é executado sem `--resume-dir`, uma
nova pasta é criada em:

```text
output/segmentation/runs/<resolucao>_res/<timestamp>/
```

Exemplo:

```text
output/segmentation/runs/mid_res/2026-06-03_09-31-08/
```

Dentro de cada run, a estrutura principal é:

```text
config/
logs/
numeric/
```

A pasta `visual/` é criada sob demanda, apenas quando algum notebook ou script
salva exemplos visuais daquele run.

### `config/`

Guarda informações que explicam como aquele run foi produzido.

```text
config/
  effective_pipeline_config.json
  split_ids.json
```

- `effective_pipeline_config.json`: configuração efetiva usada no run.
- `split_ids.json`: IDs processados em cada conjunto (`train`, `val`, `test`) e,
  quando aplicável, qual arquivo de split foi usado.

Use essa pasta quando quiser responder perguntas como:

- Esse resultado usou `mid_res` ou `high_res`?
- Quais `sigmas`, raios de Hough e thresholds estavam configurados?
- Quais IDs foram processados?
- Foi usado o split padrão ou um split alternativo, como `train90`?

### `logs/`

Guarda o log da execução.

```text
logs/
  pipeline.log
```

Use essa pasta para investigar falhas, retomadas de lote, tempos de execução e
mensagens detalhadas do pipeline.

### `numeric/`

Guarda os resultados tabulares e metadados gerados diretamente pelo pipeline.

```text
numeric/
  ostios_<split>_summary.csv
  ostios_<split>_metadata.json
  ostios_<split>_lote_<n>_summary.csv
  ostios_<split>_batch_timings.csv
```

Exemplos:

```text
numeric/ostios_train_summary.csv
numeric/ostios_train_metadata.json
numeric/ostios_train_lote_1_summary.csv
numeric/ostios_train_batch_timings.csv
```

Arquivos de lote:

- `ostios_train_lote_1_summary.csv`
- `ostios_train_lote_2_summary.csv`
- etc.

Arquivo consolidado:

- `ostios_train_summary.csv`

O consolidado é gerado pela junção dos lotes. Se o servidor cair, os lotes já
salvos continuam nessa pasta e podem ser usados com `--resume-dir` ou
`--merge-only`.

O arquivo `ostios_<split>_batch_timings.csv` guarda o tempo de cada lote. Ele é
atualizado incrementalmente após cada lote salvo.

Colunas principais:

- `batch_number`: número do lote.
- `num_images`: quantidade de imagens no lote.
- `started_at`: horário de início do lote.
- `finished_at`: horário de término do lote.
- `duration_seconds`: duração em segundos.
- `duration_minutes`: duração em minutos.
- `duration_hours`: duração em horas.
- `result_file`: CSV de resultado daquele lote.

Em uma retomada com `--resume-batch`, o pipeline usa esse arquivo para somar os
tempos conhecidos dos lotes anteriores com os novos lotes processados. Se um run
antigo não tiver esse arquivo, o pipeline consegue retomar pelos CSVs, mas não
consegue reconstruir o tempo dos lotes antigos.

O metadata final registra:

- `execution_time_seconds`, `execution_time_minutes`, `execution_time_hours`:
  tempo total conhecido pelos lotes.
- `current_run_execution_time_seconds`, `current_run_execution_time_minutes`,
  `current_run_execution_time_hours`: tempo apenas da execução atual.
- `batch_timing_summary`: resumo dos tempos por lote e lista de lotes sem tempo
  salvo.
- `batch_timings`: registros individuais de tempo por lote.

### `visual/`

Reservado para exemplos visuais associados a um run específico.

Essa pasta não é criada automaticamente pelo pipeline quando não há nada para
salvar nela. Ela deve aparecer apenas quando algum notebook/script gerar HTML,
PNG ou outro artefato visual ligado diretamente àquele run.

Sugestão de organização:

```text
visual/
  ostia_examples/
    correct/
    tolerable/
    wrong/
    not_found/
  artery_examples/
  hough_examples/
```

Use essa pasta quando o exemplo visual depende diretamente daquele run. Por
exemplo: HTML 3D de um caso em que os óstios foram encontrados incorretamente
durante aquela execução.

## `canonical/`: resultados oficiais atuais

O `canonical/` serve para apontar quais runs devem ser considerados os
resultados oficiais atuais para notebooks e comparações.

Exemplo:

```text
output/segmentation/canonical/
  mid_res/
    train/
    val/
    test/
  high_res/
    train/
    val/
    test/
```

A motivação é simples: você pode ter muitos runs em `runs/`, mas normalmente
quer que os notebooks usem um resultado de referência.

Exemplo de situação:

```text
runs/mid_res/2026-06-01_10-00-00/
runs/mid_res/2026-06-02_10-00-00/
runs/mid_res/2026-06-03_10-00-00/
```

Se o run de `2026-06-03_10-00-00` virou o resultado oficial de teste, o
`canonical/mid_res/test/` deve apontar para ele ou conter uma cópia dos arquivos
numéricos dele.

Assim os notebooks podem sempre procurar:

```text
output/segmentation/canonical/mid_res/test/
```

em vez de depender de um timestamp fixo.

Se `canonical/` estiver vazio, os helpers dos notebooks ainda usam os resultados
legados em `8.final_results/` como fallback.

## `analysis/`: análises derivadas

Use `analysis/` para outputs que não são uma execução direta do pipeline, mas
sim produtos de notebooks, comparações ou estudos de caso.

Estrutura atual:

```text
analysis/
  bad_cases/
  cases_analysis/
  ia_vs_math/
  resolution_comparison/
  subset_reports/
  visual_examples/
```

### `analysis/bad_cases/`

Casos ruins exportados por notebooks ou scripts de análise.

Exemplos:

- casos com óstios não encontrados;
- casos com apenas um óstio correto;
- casos em que `mid_res` falha e `high_res` acerta;
- casos em que ambos falham.

### `analysis/cases_analysis/`

Outputs do notebook de análise de casos.

Sugestão:

```text
cases_analysis/
  cache/
  visual/
```

- `cache/`: arquivos intermediários reutilizáveis.
- `visual/`: HTMLs, imagens ou visualizações 3D.

### `analysis/ia_vs_math/`

Resultados de comparação entre IA e método matemático.

Use aqui para tabelas e gráficos de comparação, especialmente quando ambos os
métodos foram filtrados para os mesmos IDs.

### `analysis/resolution_comparison/`

Comparações entre resoluções, como `mid_res` vs `high_res`.

Exemplos:

- Dice por resolução;
- status dos óstios por resolução;
- casos que melhoram/pioram ao mudar resolução;
- interseção de erros entre resoluções.

### `analysis/subset_reports/`

Relatórios agregados por subset (`train`, `val`, `test`) ou por tamanho de
treino, como experimentos com `train30`, `train90`, `train150`.

### `analysis/visual_examples/`

Exemplos visuais soltos que não pertencem claramente a um único run ou que foram
gerados por notebooks exploratórios.

Exemplo atual:

```text
analysis/visual_examples/ostia_3d/
```

## `8.final_results/` e `val_diff/`: legado

Essas pastas são resultados antigos.

```text
8.final_results/
val_diff/
```

Elas devem ser mantidas como referência histórica, mas os novos resultados devem
ir para `runs/`.

Não é necessário mover tudo antigo imediatamente. Quando algum resultado antigo
for escolhido como referência atual, ele pode ser copiado ou referenciado em
`canonical/`.

## Como rodar e encontrar os resultados

### Rodar treino em mid resolution

```bash
uv run python src/segmentation_pipeline.py \
  --resolution mid \
  --split train \
  --num-batches 5
```

Saída:

```text
output/segmentation/runs/mid_res/<timestamp>/
```

CSV final:

```text
output/segmentation/runs/mid_res/<timestamp>/numeric/ostios_train_summary.csv
```

Metadata:

```text
output/segmentation/runs/mid_res/<timestamp>/numeric/ostios_train_metadata.json
```

IDs usados:

```text
output/segmentation/runs/mid_res/<timestamp>/config/split_ids.json
```

### Rodar treino com split alternativo de 90 imagens

```bash
uv run python src/segmentation_pipeline.py \
  --resolution mid \
  --split train \
  --num-batches 5 \
  --split-config config/imagecas_splits_train90.json
```

Para voltar ao split normal, remova `--split-config`.

### Retomar uma execução que caiu

Se caiu no lote 3:

```bash
uv run python src/segmentation_pipeline.py \
  --resolution mid \
  --split train \
  --num-batches 5 \
  --resume-batch 3 \
  --resume-dir output/segmentation/runs/mid_res/<timestamp>
```

O pipeline procura os lotes em:

```text
output/segmentation/runs/mid_res/<timestamp>/numeric/
```

### Apenas consolidar lotes já processados

```bash
uv run python src/segmentation_pipeline.py \
  --merge-only \
  --split train \
  --resume-dir output/segmentation/runs/mid_res/<timestamp>
```

## Regra prática

- Resultado bruto do pipeline: `runs/<resolucao>_res/<timestamp>/numeric/`
- Configuração e IDs do run: `runs/<resolucao>_res/<timestamp>/config/`
- Log do run: `runs/<resolucao>_res/<timestamp>/logs/`
- Visual ligado a um run: `runs/<resolucao>_res/<timestamp>/visual/`
- Visual/análise exploratória: `analysis/`
- Resultado oficial para notebooks: `canonical/`
- Resultado antigo: `8.final_results/` ou `val_diff/`
