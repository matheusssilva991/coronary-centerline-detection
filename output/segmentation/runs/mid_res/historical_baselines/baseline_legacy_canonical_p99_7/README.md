# Baseline legacy canonical P99.7

Resultados que formavam a referência oficial antes da promoção da configuração
P99.9. O nome destaca o percentil superior P99.7 presente nos snapshots; os
três runs são históricos e não representam a configuração padrão atual.

Os três runs foram produzidos em momentos diferentes e preservam o schema e a
configuração disponíveis em cada data. Eles foram reunidos nesta pasta quando
deixaram de ser a referência principal, sem reprocessamento e sem alteração dos
CSVs originais.

O objetivo deste grupo é permitir auditoria de tabelas, notebooks e resultados
anteriores. Ele também mostra o ganho obtido após a reorganização dos
thresholds e das configurações usadas no artigo, principalmente nos conjuntos
de validação e teste.

| Split | Run | Imagens | Sucesso dos ostios | Dice geral | Dice com ostios validos |
|---|---|---:|---:|---:|---:|
| Train | `2026-06-17_10-40-05` | 30 | 93,33% | 0,5996 | 0,6100 |
| Val | `2026-06-05_14-46-06` | 270 | 80,74% | 0,5424 | 0,6053 |
| Test | `2026-06-05_20-00-43` | 700 | 80,71% | 0,5452 | 0,6109 |

Estes runs nao devem ser usados como baseline atual sem que a escolha seja
explicitamente justificada. Para reproduzir qualquer um deles, consulte o
`effective_pipeline_config.json` do próprio split, pois não é seguro assumir
que os três snapshots sejam idênticos ao pipeline atual.
