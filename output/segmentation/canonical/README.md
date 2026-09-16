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
[`current_baseline_p99_9`](../docs/baselines/current_p99_9.md):

| Split | Destino |
|---|---|
| Train | `runs/mid_res/current_baseline_p99_9/train/2026-08-06_18-43-37` |
| Val | `runs/mid_res/current_baseline_p99_9/val/2026-08-06_22-43-14` |
| Test | `runs/mid_res/current_baseline_p99_9/test/2026-08-06_10-04-22` |

`canonical/high_res/` aponta para `runs/high_res/reference`: train e val de
`2026-04-21_08-42-13`, test de `2026-04-28_14-28-44`. Os splits seguem os IDs
atuais (train 30, val 270, test 700). A referência histórica tem sucesso dos
óstios de 71,43% e Dice médio 0,4684 no teste.

Somente resultados completos e validados devem ser promovidos para esta pasta.
