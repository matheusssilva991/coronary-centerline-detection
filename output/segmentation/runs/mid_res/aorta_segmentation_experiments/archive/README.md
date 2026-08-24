# Arquivo dos experimentos de aorta

Esta pasta contém famílias encerradas ou executadas sobre configurações que não
são mais o baseline. Os arquivos são preservados para auditoria e comparação,
mas não representam candidatos ativos para promoção ao pipeline.

## Grupos

| Pasta | Configuração de origem | Situação |
|---|---|---|
| [`p10_75_p99_8/`](p10_75_p99_8/README.md) | Threshold inferior P10.75 e superior P99.8 | Histórico anterior à promoção do baseline P99.9/-300 |

Cada run mantém `config/`, `numeric/` e `logs/`. Os HTMLs foram removidos após a
inspeção visual porque ocupavam aproximadamente 1,2 GB e pertenciam somente a
variantes descartadas.
