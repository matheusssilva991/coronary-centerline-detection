# Validação de parâmetros

Destino reservado aos resultados compactos de
`src/experiments/pipeline_parameter_validation.py`.

O experimento avalia sensibilidade de parâmetros em train/val e pode alimentar
os notebooks:

- `src/eda/pipeline_sensitivity_analysis.ipynb`;
- `src/eda/upper_threshold_analysis.ipynb`.

Somente configurações, summaries por variante e tabelas necessárias para
reprodução devem ser persistidos. Figuras e tabelas já exibidas nos notebooks
não precisam ser duplicadas. A pasta pode permanecer vazia quando os artefatos
forem mantidos apenas nos runs de origem.
