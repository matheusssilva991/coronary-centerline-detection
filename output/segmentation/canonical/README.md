# Resultados canonical

`canonical/` identifica os resultados adotados como referencia oficial por
resolucao e split. As entradas sao links simbolicos para os arquivos reais em
`runs/`, evitando duplicacao.

```text
canonical/
  mid_res/
    train/<timestamp>/
    val/<timestamp>/
    test/<timestamp>/
  high_res/
```

## Referencia atual

O canonical `mid_res` aponta para a configuracao
[`current_baseline_p99_9`](../runs/mid_res/current_baseline_p99_9/README.md):

| Split | Destino |
|---|---|
| Train | `runs/mid_res/current_baseline_p99_9/train/2026-08-06_18-43-37` |
| Val | `runs/mid_res/current_baseline_p99_9/val/2026-08-06_22-43-14` |
| Test | `runs/mid_res/current_baseline_p99_9/test/2026-08-06_10-04-22` |

`canonical/high_res/` ainda nao possui uma referencia promovida. Os runs
high-res existentes sao historicos e estao documentados em
[`runs/high_res/README.md`](../runs/high_res/README.md).

Somente resultados completos e validados devem ser promovidos para esta pasta.
