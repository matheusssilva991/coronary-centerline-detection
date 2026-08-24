# Baselines históricos

Esta pasta reúne configurações que já foram usadas como referência principal,
mas não representam mais o baseline atual. Cada nome descreve a característica
mais importante da configuração preservada.

| Baseline | Configuração | Uso histórico |
|---|---|---|
| [`baseline_development_p10_75_p99_8/`](baseline_development_p10_75_p99_8/README.md) | Limite inferior adaptativo P10.75 e superior P99.8 | Baseline de desenvolvimento de agosto de 2026 |
| [`baseline_legacy_canonical_p99_7/`](baseline_legacy_canonical_p99_7/README.md) | Snapshots legacy com limite superior P99.7 | Canonical anterior à promoção do P99.9 |

O baseline ativo permanece em
[`../current_baseline_p99_9/`](../current_baseline_p99_9/README.md), com limite
inferior fixo em `-300 HU` e limite superior P99.9. Para reproduzir resultados
históricos, deve-se usar o `effective_pipeline_config.json` de cada run.
