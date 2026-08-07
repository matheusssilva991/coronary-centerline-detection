# Baseline atual do pipeline

Esta pasta contém a execução completa da configuração atualmente definida em
`config/pipeline_config.json`, realizada em resolução média em 7 de agosto de
2026.

## Estrutura

```text
current_baseline/
  train/2026-08-07_01-43-41/
  val/2026-08-07_01-43-41/
  test/2026-08-07_01-43-41/
```

Os três splits foram processados na mesma execução original. Para manter o
layout padronizado dos resultados, os CSVs foram separados por split e cada
pasta recebeu o mesmo snapshot de configuração e log.

## Resultados

| Split | Imagens | Sucesso dos óstios | Dice geral | Dice com óstios válidos |
|---|---:|---:|---:|---:|
| Train | 30 | 93,33% (28/30) | 0,5997 | 0,6098 |
| Val | 270 | 82,22% (222/270) | 0,5802 | 0,6402 |
| Test | 700 | 81,86% (573/700) | 0,5815 | 0,6407 |

Óstios válidos correspondem aos status `both correct` e `both tolerable`.
O resultado P99.9 permanece no `canonical`, pois apresentou Dice superior no
conjunto de teste.
