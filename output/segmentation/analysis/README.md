# Analises de segmentacao

Esta pasta guarda resultados exploratorios, comparacoes e figuras usadas para
avaliar o pipeline. Ela nao segue a mesma estrutura das execucoes finais em
`output/segmentation/runs`.

## Subpastas

- `fuzzy_sweep/`: historico de varreduras exploratorias antigas.
- `fuzzy_pipeline_comparison/`: resultados do notebook
  `src/fuzzy_pipeline_comparison.ipynb`, comparando normal, fuzzy alpha-cut,
  contextual fuzzy e fuzzy connectedness.
- `cases_analysis/`: estudos de casos especificos, geralmente com cache e
  exemplos visuais.
- `bad_cases/`: relatorios ou amostras de casos problemáticos.
- `ia_vs_math/`: comparacoes entre resultados da IA e metodo matematico.
- `resolution_comparison/`: comparacoes entre resolucoes.
- `subset_reports/`: relatorios de subconjuntos de treino, validacao ou teste.
- `visual_examples/`: figuras e exemplos visuais para inspecao/documentos.

## Regra pratica

Para escolher parametros, use primeiro os arquivos de resumo em
`fuzzy_pipeline_comparison/<run_name>/summary`. Abra os CSVs detalhados apenas
quando precisar entender um caso especifico.
