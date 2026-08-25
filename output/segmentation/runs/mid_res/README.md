# Runs em resolucao media

Esta pasta organiza os resultados `mid_res` por objetivo experimental.

| Grupo | Finalidade | Situacao |
|---|---|---|
| [`current_baseline_p99_9/`](current_baseline_p99_9/README.md) | Configuração fixa -300 HU/P99.9 usada como referência do artigo | Baseline e canonical atuais |
| [`historical_baselines/`](historical_baselines/README.md) | Baselines anteriores identificados pela configuração que representam | Histórico agrupado |
| [`bilateral_thin/`](bilateral_thin/README.md) | Perfil bilateral e correcao fina da aorta | Histórico; removido do runtime |
| [`fuzzy_comparison/`](fuzzy_comparison/README.md) | Threshold normal/fuzzy com RG/FC | Comparacao metodologica |
| [`aorta_segmentation_experiments/`](aorta_segmentation_experiments/README.md) | Controle do level set e filtros de círculos; resultados antigos ficam em `archive/` | Experimentos P99.9 ativos e histórico identificado |

Cada grupo separa os resultados por `train`, `val` e `test` quando esses splits
foram executados. A estrutura interna de cada run esta descrita em
[`../README.md`](../README.md).
