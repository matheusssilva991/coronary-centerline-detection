# Coorte compacta de falhas

Dados gerados a partir dos quatro runs de validação com threshold
normal/fuzzy e RG/FC. O processamento de imagens não é repetido nesta etapa.
Não há um notebook associado: o script abaixo lê os resultados oficiais e
seleciona os exames usados nos experimentos de correção.

- `focused_cohort.csv`: somente `IMG_ID`, tipo e papéis de falha necessários ao
  runner `run_pipeline_failure_improvement.sh`.
- `category_summary.csv`: contagem de cada padrão de falha.
- `analysis_metadata.json`: origem, split e parâmetros usados na seleção.

O catálogo detalhado e o ranking não são persistidos porque duplicavam as
tabelas dos runs e não eram consumidos por nenhuma etapa posterior.

Para regenerar:

```bash
uv run python src/experiments/pipeline_failure_analysis.py --split val
```
