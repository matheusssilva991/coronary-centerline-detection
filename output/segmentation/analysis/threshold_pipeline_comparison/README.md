# Comparação histórica threshold/RG/FC

As duas tabelas em `tables/` preservam o resumo histórico da comparação.
O notebook atual carrega os resultados por exame dos runs selecionados e
calcula suas próprias tabelas e gráficos.

Em 20/11/2026, as figuras e os HTMLs antigos sem consumidores foram removidos.
Os caminhos, tamanhos e hashes constam em `../../results_catalog.json`.

## CSVs principais

- `tables/dice_stats_by_variant.csv`: estatisticas do Dice por variante.
- `tables/pair_outcome_counts.csv`: todas as comparacoes par-a-par entre as
  variantes carregadas. Para `n` variantes, o esperado e `n * (n - 1) / 2`
  linhas.

Essas tabelas históricas identificam as variantes avaliadas naquela execução.
As análises atuais devem usar os CSVs por exame de seus runs de origem.
