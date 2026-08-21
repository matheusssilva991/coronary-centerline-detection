# Runs históricos com baixa acurácia dos óstios

Esta pasta organiza os runs completos que anteriormente estavam soltos na raiz
de `runs/high_res`. Eles comparam P99.7, P99.8 e P99.9 nos splits disponíveis,
mantendo a configuração histórica de alta resolução: threshold normal, Canny
sigma 6, 70 iterações do level set e region growing. As taxas de sucesso dos
óstios ficaram baixas, principalmente em validação e teste, e por isso esses
runs são tratados como legado, não como análise definitiva do artigo.

## Estrutura

```text
legacy_low_ostia_accuracy/
  p99_7/train/2026-08-08_01-28-23/
  p99_7/val/2026-08-08_07-40-07/
  p99_7/test/2026-08-12_09-52-53/
  p99_8/train/2026-08-08_02-54-08/
  p99_8/val/2026-08-08_20-54-51/
  p99_8/test/2026-08-17_10-28-49/
  p99_9/train/2026-08-08_04-19-59/
  p99_9/val/2026-08-11_13-31-45/
```

Não existe um run P99.9 completo para o conjunto de teste nessa série.

## Leitura dos resultados

| Split | P99.7 | P99.8 | P99.9 | Interpretação |
|---|---:|---:|---:|---|
| Train, sucesso dos óstios | 80,00% | 76,67% | 80,00% | P99.8 foi inferior. |
| Train, Dice geral | 0,5316 | 0,5080 | 0,5375 | P99.9 obteve o maior Dice. |
| Val, sucesso dos óstios | 63,70% | 55,19% | 67,04% | P99.9 foi o melhor para os óstios. |
| Val, Dice geral | 0,4821 | 0,4419 | 0,4782 | P99.7 teve vantagem pequena sobre P99.9. |
| Test, sucesso dos óstios | 50,57% | 55,14% | não executado | P99.8 superou P99.7. |
| Test, Dice geral | 0,3892 | 0,4394 | não executado | Ambos ficaram abaixo dos resultados mid-res. |

P99.8 em validação foi claramente inferior. P99.7 e P99.9 apresentaram uma
troca entre Dice e localização dos óstios: P99.7 teve Dice geral 0,0039 maior,
enquanto P99.9 recuperou nove sucessos adicionais. Por isso, o run P99.9 de
validação foi usado como baseline da investigação posterior de escala dos
parâmetros em high resolution.

Esses resultados não devem ser promovidos a canonical. Uma nova série high-res
deverá ser executada após a escolha dos parâmetros corrigidos de Canny e level
set. Os estudos de seleção desses parâmetros ficam em
`analysis/pipeline_parameter_validation/runs/high_res_scaling_*`.
