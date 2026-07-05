# Analises de segmentacao

Esta pasta guarda resultados exploratorios, comparacoes e figuras usadas para
avaliar o pipeline. Ela nao segue a mesma estrutura das execucoes finais em
`output/segmentation/runs`.

## Subpastas

- `aorta_ostia_sweep/`: resultados dos sweeps focados em localização da aorta,
  quantidade/cobertura de círculos e detecção dos óstios.
- `fuzzy_comparison_eda/`: tabelas e figuras geradas pela análise comparativa
  das variantes fuzzy/normal.
- `fuzzy_membership_functions/`: figuras das funções de pertinência fuzzy.
- `fuzzy_pipeline_comparison/`: resultados do notebook
  `src/experiments/fuzzy_pipeline_comparison.ipynb`, comparando normal, fuzzy alpha-cut,
  fuzzy threshold e fuzzy connectedness.
- `threshold_sweep/`: tabelas consolidadas das varreduras de limiar HU. Os runs
  individuais antigos foram removidos para manter apenas os CSVs resumidos.
- `visual_examples/`: figuras e exemplos visuais para inspecao/documentos.

## Regra pratica

Para escolher parametros, use primeiro os arquivos de resumo em
`fuzzy_pipeline_comparison/<run_name>/summary` e os CSVs consolidados em
`threshold_sweep/`. Abra arquivos detalhados apenas quando precisar entender um
caso especifico.

Arquivos principais em `threshold_sweep/`:

- `threshold_sweep_summary_all.csv`: tabela consolidada de todos os runs de
  threshold mantidos.
- `threshold_sweep_ranking.csv`: ranking consolidado por desempenho.
- `latest_best_pairwise.csv`: comparação par-a-par das melhores configurações
  do treino.
- `val_threshold_pairwise_comparison.csv`: comparação par-a-par dos melhores
  thresholds na validação.
