# Referências oficiais do pipeline

`canonical/` é a fonte única das análises do baseline atual, por resolução e
conjunto. Somente resultados completos e verificados são promovidos. Em
15/11/2026, mid foi atualizado para as reexecuções `complete_metrics_v1`, com
diagnósticos completos da aorta, estágios morfológicos e HD/AHD. High mantém as
referências anteriores.

```text
canonical/
  references_manifest.json
  mid_res/{train,val,test}/current -> runs/canonical_metrics/complete_metrics_v1/...
  mid_res/{train,val,test}/before_complete_metrics_v1/
    numeric/results_<split>.csv
    config/effective_pipeline_config.json
    config/experiment_settings.json
    source_manifest.json
  high_res/{train,val,test}/current -> run original
```

## Referências atuais

| Resolução | Conjunto | N | Origem | Variante | Dice geral |
|---|---|---:|---|---|---:|
| Mid | Train | 30 | `runs/canonical_metrics/complete_metrics_v1/mid_res/train` | Baseline reexecutado | 0,6288 |
| Mid | Val | 270 | `runs/canonical_metrics/complete_metrics_v1/mid_res/val` | Baseline reexecutado | 0,6009 |
| Mid | Test | 700 | `runs/canonical_metrics/complete_metrics_v1/mid_res/test` | Baseline reexecutado | 0,5996 |
| High | Train | 30 | `runs/high_res/aorta_segmentation_experiments/train/filter_envelope_lower100_pad2/2026-09-15_11-08-42` | Run completo | 0,5720 |
| High | Val | 270 | `runs/high_res/aorta_segmentation_experiments/val/filter_envelope_lower100_pad2/2026-09-12_07-51-48` | Run completo | 0,5155 |
| High | Test | 700 | `runs/high_res/aorta_segmentation_experiments/test/filter_envelope_lower100_pad2/2026-09-13_14-06-44` | Run completo | 0,5146 |

Caminhos relativos a `output/segmentation/`. O Dice geral inclui as falhas;
os snapshots mid preservam o Dice registrado/avaliável separadamente quando
disponível. As coortes contêm os mesmos IDs das referências anteriores.

Mid usa threshold normal de -300 HU a P99.9, filtro robusto, envelope e
`lower100/pad2`, com level set b0.6/r0.10/i26. A fonte de verdade dos parâmetros
é o snapshot registrado, não o nome ou a descrição de uma variante. Algumas
descrições antigas ainda mencionavam P10.75/P99.8 sem corresponder à configuração.
Em high, consulte a configuração efetiva no próprio run, incluindo a escala e
os ajustes específicos da resolução.

## Proveniência e tempo

As tabelas mid atuais foram reexecutadas com os mesmos IDs e as configurações
efetivas das referências anteriores, acrescentando a avaliação HD/AHD. Os
manifests e planos registram os hashes, IDs, protocolo e configurações utilizados.
As três referências anteriores permanecem em `before_complete_metrics_v1`.
Os Dice gerais, volumes da aorta e status dos óstios foram reproduzidos sem
alteração. As cinco falhas de superfície na validação e sete no teste já existiam;
as métricas de etapas não executadas são ausentes, sem valores fabricados.

HD/AHD referem-se à máscara arterial após morfologia, em mm, entre centros de
voxels das superfícies de seis vizinhos. HD usa o máximo bidirecional; AHD usa
a média aritmética das duas médias direcionais. Há 30, 265 e 689 pares de
distâncias válidas no treino, validação e teste, respectivamente. No teste, quatro
predições arteriais vazias também não têm distâncias definidas. As médias de
distância são condicionadas aos casos válidos; não representam toda a coorte.

Não são criados tempos de batch ou metadados de execução fictícios para mid.
Sem manifest equivalente de batches, o tempo aparece como indisponível. Tempos
por exame eventualmente registrados na fonte não equivalem a tempo de batch ou
a uma comparação em hardware controlado. High continua usando seus artefatos
originais, sem duplicação.

## Referências anteriores preservadas

Os links antigos foram substituídos, mas os runs reais continuam nos destinos:

| Resolução | Train | Val | Test |
|---|---|---|---|
| Mid | `runs/mid_res/current_baseline_p99_9/train/2026-08-06_18-43-37` | `runs/mid_res/current_baseline_p99_9/val/2026-08-06_22-43-14` | `runs/mid_res/current_baseline_p99_9/test/2026-08-06_10-04-22` |
| High | `runs/high_res/reference/train/2026-04-21_08-42-13` | `runs/high_res/reference/val/2026-04-21_08-42-13` | `runs/high_res/reference/test/2026-04-28_14-28-44` |

`references_manifest.json` registra as referências anteriores e as promovidas.
Comparações históricas devem continuar apontando explicitamente para esses runs,
em vez de seguir automaticamente a referência móvel de `canonical`.

## Reexecução com todas as métricas atuais

Os três CSVs mid atuais já contêm os novos diagnósticos e métricas. High ainda
não tem HD/AHD e os estágios intermediários recentes. Para repetir referências,
consulte a seção de reexecução em `src/experiments/README.md` e use
`rebuild_canonical_metrics.py`.

O script usa os mesmos IDs e o snapshot **já escalonado** de cada referência,
sem carregar o `pipeline_config.json` corrente nem escalar parâmetros outra vez.
Somente a avaliação HD/AHD é habilitada adicionalmente. Mid mantém o fluxo dos
experimentos compartilhados; high mantém o pipeline principal. Os resultados
são novas medições, não preenchimento retroativo dos CSVs antigos.

Uma promoção validada move `current` para `before_<run-name>` no mesmo diretório,
cria um link para o novo run e atualiza `references_manifest.json`, preservando
a referência anterior no histórico. Os notebooks canônicos passam a ler o novo
run; comparações históricas com fontes explícitas continuam usando os runs antigos.
Falhas e etapas não executadas preservam métricas ausentes e status explicativos.
Não são fabricados tempos de batch, valores de distância ou métricas de falhas.
