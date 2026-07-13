# Comparacao threshold/RG/FC

Saidas geradas por `src/eda/threshold_pipeline_comparison_analysis.ipynb`.

## Pastas

- `tables/`: tabelas finais em CSV.
- `figures/`: figuras em PNG com distribuicao de Dice, status dos ostios e
  maiores variacoes entre variantes.
- `qualitative_3d/`: exemplos HTML com aorta, ostios e arterias segmentadas.

## CSVs principais

- `tables/dice_stats_by_variant.csv`: estatisticas do Dice por variante.
- `tables/pair_outcome_counts.csv`: todas as comparacoes par-a-par entre as
  variantes carregadas. Para `n` variantes, o esperado e `n * (n - 1) / 2`
  linhas.

O notebook gera `pair_outcome_counts.csv` automaticamente a partir das variantes
presentes em `results_df`; nao e mais necessario manter uma lista fixa de pares.
