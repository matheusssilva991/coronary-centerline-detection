# Baseline de desenvolvimento histórico P10.75/P99.8

Esta pasta contém a execução completa adotada como baseline de desenvolvimento
em 7 de agosto de 2026. A configuração efetiva daquele momento foi preservada
dentro de cada run; o `config/pipeline_config.json` atual pode ter evoluído
desde então.

Este grupo serve para reproduzir e comparar o estado de desenvolvimento daquele
momento. Ele não deve ser interpretado automaticamente como o melhor resultado:
o canonical P99.9 foi mantido justamente porque apresentou desempenho superior
na avaliação final.

## Estrutura

```text
historical_baselines/baseline_development_p10_75_p99_8/
  train/2026-08-07_01-43-41/
  val/2026-08-07_01-43-41/
  test/2026-08-07_01-43-41/
```

Os três splits foram processados na mesma execução original. Para manter o
layout padronizado dos resultados, os CSVs foram separados por split e cada
pasta recebeu o mesmo snapshot de configuração e log.

## Configuração principal

- threshold normal com limite inferior adaptativo P10.75 e superior P99.8;
- perfil de aorta e óstios `standard`;
- recuperação inicial da trajetória de círculos ativada;
- level set fixo com 31 iterações;
- segmentação arterial por region growing;
- downsampling OpenCV linear com fatores `[2, 2, 1]`.

O limite inferior adaptativo calcula o percentil 10,75 entre as intensidades
válidas do exame e restringe o valor resultante à faixa configurada. Já o
limite superior P99.8 remove a cauda mais densa do histograma. A recuperação
inicial dos círculos permite reiniciar a busca nas primeiras fatias quando a
trajetória ainda não possui círculos suficientes.

Em relação ao grupo `current_baseline_p99_9`, este baseline altera
simultaneamente o limite inferior, o percentil superior e a recuperação inicial
da localização. Por isso, a diferença entre os grupos não deve ser atribuída
somente ao P99.8.

## Resultados

| Split | Imagens | Sucesso dos óstios | Dice geral | Dice com óstios válidos |
|---|---:|---:|---:|---:|
| Train | 30 | 93,33% (28/30) | 0,5997 | 0,6098 |
| Val | 270 | 82,22% (222/270) | 0,5802 | 0,6402 |
| Test | 700 | 81,86% (573/700) | 0,5815 | 0,6407 |

Óstios válidos correspondem aos status `both correct` e `both tolerable`.
O resultado P99.9 permanece no `canonical`, pois apresentou Dice superior no
conjunto de teste.

Os três diretórios compartilham o mesmo timestamp porque train, val e test foram
processados na mesma execução original e depois organizados por split. Para
reprodução, use sempre o snapshot existente em cada diretório, não a versão
atual de `config/pipeline_config.json`.
