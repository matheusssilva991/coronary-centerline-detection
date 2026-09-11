# Runs em resolucao media

Esta pasta organiza os resultados `mid_res` por objetivo experimental.

| Grupo | Finalidade | Situacao |
|---|---|---|
| `current_baseline_p99_9/` | Configuração fixa -300 HU/P99.9 usada como referência do artigo | [Baseline atual](../../docs/baselines/current_p99_9.md) |
| `historical_baselines/` | Baselines anteriores identificados pela configuração que representam | [Baselines históricos](../../docs/baselines/README.md) |
| `bilateral_thin/` | Perfil bilateral e correcao fina da aorta | [Histórico removido](../../docs/aorta/bilateral_thin.md) |
| `fuzzy_comparison/` | Threshold normal/fuzzy com RG/FC | [Comparação metodológica](../../docs/methods/README.md) |
| `aorta_segmentation_experiments/` | Controle do level set e filtros de círculos | [Experimentos de aorta](../../docs/aorta/README.md) |

Cada grupo separa os resultados por `train`, `val` e `test` quando esses splits
foram executados. A estrutura interna de cada run esta descrita em
[`../README.md`](../README.md).
