# Referências oficiais do pipeline

`canonical/` é a fonte das análises do baseline atual por resolução e conjunto.
Em 21/11/2026, os três splits de mid passaram a apontar para os runs SciPy
cúbico do Curupira da série de 20/11/2026.
A reexecução `complete_metrics_v1` foi promovida para `high/test` em 21/11/2026.

```text
canonical/
  references_manifest.json
  mid_res/{train,val,test}/current -> runs/mid_res/server_mid_comparison/<split>/scipy_cubic/<run>
  mid_res/{train,val,test}/before_server_mid_scipy_cubic -> runs/mid_res/historical_baselines/opencv_linear_complete_metrics_v1/<split>
  mid_res/{train,val,test}/before_complete_metrics_v1/
  high_res/{train,val}/current -> run original
  high_res/test/current -> runs/high_res/complete_metrics_v1/test
  high_res/test/before_complete_metrics_v1 -> run original
```

## Referências atuais

| Resolução | Conjunto | N | Origem | Variante | Dice geral |
|---|---|---:|---|---|---:|
| Mid | Train | 30 | `runs/mid_res/server_mid_comparison/train/scipy_cubic/2026-11-20_09-42-08` | SciPy cúbico | 0,6156 |
| Mid | Val | 270 | `runs/mid_res/server_mid_comparison/val/scipy_cubic/2026-11-20_09-43-42` | SciPy cúbico | 0,6061 |
| Mid | Test | 700 | `runs/mid_res/server_mid_comparison/test/scipy_cubic/2026-11-20_09-46-25` | SciPy cúbico | 0,6099 |
| High | Train | 30 | `runs/high_res/aorta_segmentation_experiments/train/filter_envelope_lower100_pad2/2026-09-15_11-08-42` | Run completo | 0,5720 |
| High | Val | 270 | `runs/high_res/aorta_segmentation_experiments/val/filter_envelope_lower100_pad2/2026-09-12_07-51-48` | Run completo | 0,5155 |
| High | Test | 700 | `runs/high_res/complete_metrics_v1/test` | Reexecução com HD/AHD | 0,5151 |

Os caminhos são relativos a `output/segmentation/`. O Dice geral inclui as
falhas. Mid usa SciPy cúbico na imagem, zoom de ordem zero no rótulo, threshold
normal de -300 HU a P99.9, filtro robusto, envelope e `lower100/pad2`, com level
set b0.6/r0.10/i26. Consulte o snapshot efetivo de cada run para os parâmetros.

## Proveniência e tempo

Os consolidados mid foram conferidos contra os cinco lotes e os IDs oficiais
de cada split. O manifesto registra os hashes dos CSVs, das configurações e dos
metadados promovidos. A configuração das duas variantes da série varia somente
no método de redução da imagem. O SciPy atingiu Dice geral de 60,91% nos 1.000
exames e ambos os óstios corretos ou toleráveis em 872 exames.

As distâncias mid são de superfície após a morfologia. HD usa o máximo
bidirecional e AHD usa a média das duas médias direcionais. Há 30, 266 e 690
pares válidos no treino, validação e teste. Na validação, dois exames ficaram
com `not_evaluated` e dois com `prediction_empty` na avaliação de distâncias.
No teste, essas contagens são sete e três. As médias de distância consideram
somente os casos válidos.

Os tempos vêm de `numeric/batch_timings_<split>.csv` e do metadata de cada run.
Os pares OpenCV e SciPy se sobrepuseram na mesma GPU. Os tempos refletem essa
concorrência.

No novo `high/test`, os 700 IDs coincidem com os do run anterior. O Dice
geral passou de 51,46% para 51,51%, com diferença por exame em 8 casos e
mudança no status dos óstios em 6. Foram registrados 695 pares válidos de HD/AHD
de superfície, 3 casos `not_evaluated` e 2 `prediction_empty`. A reexecução
foi feita por exame e o tempo consolidado de lotes está indisponível.

## Referências anteriores preservadas

O baseline OpenCV de 15/11/2026 (`complete_metrics_v1`) está acessível por
`mid_res/{train,val,test}/before_server_mid_scipy_cubic`, com os artefatos em
`runs/mid_res/historical_baselines/opencv_linear_complete_metrics_v1/`. O run
anterior de `high/test` está acessível por `high_res/test/before_complete_metrics_v1`.
Os snapshots mid `before_complete_metrics_v1` e as referências históricas abaixo
também estão disponíveis.

| Resolução | Train | Val | Test |
|---|---|---|---|
| Mid | `runs/mid_res/current_baseline_p99_9/train/2026-08-06_18-43-37` | `runs/mid_res/current_baseline_p99_9/val/2026-08-06_22-43-14` | `runs/mid_res/current_baseline_p99_9/test/2026-08-06_10-04-22` |
| High | `runs/high_res/reference/train/2026-04-21_08-42-13` | `runs/high_res/reference/val/2026-04-21_08-42-13` | `runs/high_res/reference/test/2026-04-28_14-28-44` |

`references_manifest.json` registra as referências anteriores e as promovidas.
Comparações históricas devem continuar apontando explicitamente para esses runs,
em vez de seguir automaticamente a referência móvel de `canonical`.

## Reexecução com todas as métricas atuais

Os três CSVs mid e o de `high/test` contêm os diagnósticos e as distâncias de
superfície. Em high, os CSVs de treino e validação estão sem as colunas de
HD/AHD.

O [guia da série no servidor](../docs/structure/server_mid_comparison.md)
usa a configuração OpenCV arquivada para repetir a comparação de interpolação.
Para reutilizar a configuração efetiva de um split SciPy, passe o snapshot do
split em `current/config/effective_pipeline_config.json` com `--config-file` e
`--config-replace`.

A opção `--distance-metrics` atual produz o protocolo 3, com HD/AvgHD da
máscara inteira e HD95 de superfície. O HD/AHD desses CSVs canônicos usa a
definição histórica de superfície descrita acima. Cada comparação deve
identificar o protocolo e os denominadores válidos.

Uma promoção validada move `current` para `before_<run-name>` no mesmo diretório,
cria um link para o novo run e atualiza `references_manifest.json`, preservando
a referência anterior no histórico. Os notebooks canônicos passam a ler o novo
run; comparações históricas com fontes explícitas continuam usando os runs antigos.
Falhas e etapas não executadas preservam métricas ausentes e status explicativos.
Tempos publicados devem vir dos manifests efetivamente medidos na execução.
