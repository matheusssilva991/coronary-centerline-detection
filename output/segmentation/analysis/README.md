# Analises de segmentacao

Esta pasta guarda resultados derivados de experimentos, notebooks e figuras de
apoio. Ela e diferente de `output/segmentation/runs/`, que guarda execucoes
oficiais do pipeline.

O catálogo dos notebooks que produzem estas análises está em
[`src/eda/README.md`](../../../src/eda/README.md).

## Organizacao atual

- `threshold_pipeline_comparison/`: analise do notebook
  [`threshold_pipeline_comparison_analysis.ipynb`](../../../src/eda/threshold_pipeline_comparison_analysis.ipynb).
  - `tables/`: CSVs finais para leitura rapida.
  - `figures/`: graficos em PNG.
  - `qualitative_3d/`: exemplos HTML 3D e casos qualitativos selecionados.
- `EXPERIMENTS_ARCHIVE.md`: decisões e métricas resumidas dos sweeps encerrados.
- `aorta_mask_ostia_comparison/aorta_ostia_bilateral_final_val90/`: confirmação
  final da nova opção bilateral em 90 imagens independentes de validação.
- `artery_vesselness_fc_sweep/`: runs de seleção do mapa de vesselness arterial,
  refinamento de RG/FC e ablação do pós-processamento morfológico. Cada run
  mantém seus parâmetros e tabelas na subpasta `results/`. O estágio mais novo
  também compara dilatação condicionada por vesselness e recuperação local de
  ramos subsegmentados.
- `fuzzy_membership_functions/`: figuras das funcoes de pertinencia fuzzy.
- `image_slices/`: fatias de CCTA exportadas para publicação.
- `segmentation_results/`: figuras e tabelas da análise canônica de Dice.
- `visual_examples/`: exemplos visuais avulsos.

## Arquivos principais

### `threshold_pipeline_comparison/tables/`

- `dice_stats_by_variant.csv`: media, mediana, minimo, maximo e desvio do Dice
  por variante.
- `pair_outcome_counts.csv`: contagem par-a-par de exames que melhoraram,
  pioraram ou ficaram iguais em relacao a cada comparacao.

O arquivo `pair_outcome_counts.csv` e gerado automaticamente com todas as
combinacoes possiveis entre as variantes carregadas no notebook. Para 4
variantes, o esperado e ter 6 comparacoes.

## Regra pratica

Use os CSVs em `tables/` para a analise final. Consulte
`EXPERIMENTS_ARCHIVE.md` para entender por que parâmetros e abordagens antigas
foram retirados. Os resultados preliminares de aorta/ostios foram resumidos no
arquivo historico e removidos para evitar que triagens sejam confundidas com a
confirmacao final.
