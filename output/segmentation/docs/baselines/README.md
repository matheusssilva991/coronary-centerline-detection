# Catálogo de baselines

Esta pasta reúne a referência atual e configurações que já foram usadas como
baseline. Os dados permanecem em `runs/`; estes documentos registram contexto,
métricas e decisões.

| Documento | Configuração | Situação |
|---|---|---|
| [`current_p99_9.md`](current_p99_9.md) | Limite inferior fixo em -300 HU e superior P99.9 | Baseline e canonical atuais |
| [`development_p10_75_p99_8.md`](development_p10_75_p99_8.md) | Limite inferior adaptativo P10.75 e superior P99.8 | Baseline de desenvolvimento histórico |
| [`legacy_canonical_p99_7.md`](legacy_canonical_p99_7.md) | Snapshots legacy com limite superior P99.7 | Canonical anterior |
| [`high_resolution_legacy.md`](high_resolution_legacy.md) | P99.7, P99.8 e P99.9 em high resolution | Série com baixa acurácia dos óstios |

Para reproduzir um resultado, use sempre o `effective_pipeline_config.json` do
run indicado no documento, não a configuração atual do projeto.
