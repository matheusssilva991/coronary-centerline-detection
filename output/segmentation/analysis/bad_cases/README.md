# Catálogo de casos ruins

Catálogos compactos produzidos por
`src/eda/results/segmentation_results_eda.ipynb`, separados por treino,
validação e teste, nas resoluções mid e high.

- `bad_cases_{train,val,test}_{mid_res,high_res}.csv`: IDs e motivos das falhas;
- arquivos `.json` com o mesmo nome: a mesma seleção em formato estruturado.

Um exame é selecionado por falha dos óstios ou Dice arterial estritamente menor
que 0,30. Ambos corretos e ambos toleráveis contam como sucesso. Falhas dos
óstios têm precedência sobre Dice baixo, evitando duplicações. Dice ausente
não é classificado como baixo automaticamente.

O notebook carrega as referências oficiais de `canonical` para as duas
resoluções. Mid consolida normal+RG de `train30_val60`/`test700` e baseline de
`sensitivity_refined_pipeline_val270`. High aponta para os runs completos
refinados de setembro de 2026. A origem é mostrada na análise e na exportação.

**A exportação é opcional:** `EXPORT_BAD_CASES = False` não altera esta pasta.
Ao habilitá-la, apenas os grupos carregados são atualizados, inclusive os sem
falhas. Os novos arquivos acrescentam `source_run`, `variant`, `dice_artery` e
`status`, quando disponíveis. Catálogos históricos sem essas colunas continuam
legíveis; não devem ser considerados atualizados apenas pela execução da análise.

Esses arquivos servem à seleção de casos para inspeção. Eles não contêm
novas máscaras e não substituem os summaries originais em `runs/`.
