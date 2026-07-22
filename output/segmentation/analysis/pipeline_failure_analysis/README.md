# Coorte focada de falhas

Tabelas geradas a partir dos quatro runs de validação com threshold
normal/fuzzy e RG/FC. O processamento de imagens não é repetido nesta etapa.

- `focused_cohort.csv`: IDs de falhas severas e controles estáveis usados pelo
  runner `run_pipeline_failure_improvement.sh`.
- `case_catalog.csv`: métricas e categorias por exame.
- `category_summary.csv`: contagem de cada padrão de falha.
- `selected_cases.csv`: exemplos representativos.
- `variant_summary.csv`: resumo das quatro variantes originais.
- `analysis_metadata.json`: split e parâmetros da seleção.

Para regenerar:

```bash
uv run python src/experiments/pipeline_failure_analysis.py --split val
```
