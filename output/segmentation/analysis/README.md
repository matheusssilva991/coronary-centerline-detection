# Analises de segmentacao

Esta pasta guarda resultados derivados de experimentos, notebooks e figuras de
apoio. Ela e diferente de `output/segmentation/runs/`, que guarda execucoes
oficiais do pipeline.

O catálogo dos notebooks que produzem estas análises está em
[`src/eda/README.md`](../../../src/eda/README.md).

## Organizacao atual

- `EXPERIMENTS_ARCHIVE.md`: decisões e métricas resumidas dos sweeps encerrados.
- `aorta_circle_slices/`: métricas entre número de fatias e círculos da aorta.
- `aorta_bad_cases/`: atalhos leves para comparar rapidamente máscaras 3D da
  aorta classificadas visualmente como ruins entre variantes. Há catálogos separados
  para [treino](aorta_bad_cases/train_p99_9_m300/README.md) e
  [validação](aorta_bad_cases/val_p99_9_m300/README.md).
- `bad_cases/`: catálogo compacto de casos ruins usado pelas EDAs qualitativas.
- `image_slices/`: cortes axiais exportados para inspeção ou publicação.
- `visual_examples/`: exemplos 3D independentes de uma run específica.
- `pipeline_failure_analysis/`: catálogo e coorte focada de validação usada
  pelo runner de correções.
- `pipeline_parameter_validation/`: resultados compactos usados tanto pela
  análise OFAT de sensibilidade quanto pela investigação dos thresholds
  adaptativos. Em `runs/`, somente execuções cuja referência foi conferida
  contra um run histórico equivalente devem ser mantidas.
- `hybrid_resolution_pipeline/`: resultados compactos do experimento que
  localiza os óstios em mid resolution e segmenta as artérias em high
  resolution. O sweep recomendado compara morfologia, critérios do RG e sigmas
  do Frangi reutilizando as etapas comuns. Cada run mantém apenas configurações,
  CSV por imagem e resumo pareado por variante.
- `threshold_pipeline_comparison/`: tabelas e figuras da comparação entre
  threshold normal/fuzzy e segmentação por RG/FC.

Cada subpasta possui um README com a origem dos arquivos e a regra de
persistência. Pastas vazias funcionam apenas como destino reservado para uma
execução futura; os resultados exibidos diretamente em notebooks não são
duplicados aqui.

## Regra pratica

Consulte `EXPERIMENTS_ARCHIVE.md` para entender por que parâmetros e abordagens
antigas foram retirados. Grades sem ganho, execuções inválidas e diagnósticos
temporários são resumidos no histórico e removidos para não serem confundidos
com confirmações positivas.

## Política de armazenamento

`analysis/` deve guardar apenas artefatos usados para interpretação ou
reprodução compacta:

- tabelas consolidadas que servem de entrada para outras análises;
- configurações e comandos necessários para reproduzir o experimento;
- catálogos de casos ruins e métricas de círculos da aorta;
- pequenos catálogos de diagnóstico ainda ativos.

Figuras, tabelas derivadas e visualizações 3D que já aparecem em notebooks não
são persistidas nesta pasta. Elas permanecem como outputs das células e podem
ser regeneradas a partir dos runs oficiais.

Runs completos do pipeline, caches, lotes intermediários, dry runs e grades já
descartadas não devem permanecer aqui. Runs oficiais pertencem a
`output/segmentation/runs/`. O sweep de threshold remove por padrão seus runs
internos depois de copiar o summary por imagem; use `--keep-pipeline-runs`
somente quando precisar investigar uma falha.

O runner de correções pode criar temporariamente `pipeline_failure_improvement/`.
Depois da interpretação, seus resultados devem ser resumidos em
`EXPERIMENTS_ARCHIVE.md` e removidos; apenas a coorte de entrada em
`pipeline_failure_analysis/` é persistente.
