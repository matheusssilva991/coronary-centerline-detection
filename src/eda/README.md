# Notebooks de EDA

Esta pasta reúne análises exploratórias, comparações de resultados e figuras
metodológicas do pipeline coronário. Os notebooks não substituem os scripts de
experimentos: eles leem resultados já produzidos ou executam poucos casos para
inspeção qualitativa.

## Como executar

Na raiz do projeto:

```bash
uv run jupyter lab
```

Os notebooks resolvem a raiz do repositório automaticamente. Para usar um
dataset fora dos caminhos conhecidos, configure:

```bash
export IMAGECAS_BASE_PATH=/caminho/para/ImageCAS/1-1000
```

## Catálogo

### EDA de imagens

| Notebook | Objetivo | Entrada principal | Saída | Custo |
|---|---|---|---|---|
| [image_intensity_eda.ipynb](image_intensity_eda.ipynb) | Distribuição HU e percentis da ROI | Volumes ImageCAS | Figuras e tabelas exibidas | Médio para KDE |
| [preprocessing_visualization.ipynb](preprocessing_visualization.ipynb) | Fatias axiais completas, MIP, downscale, threshold, LCC, Hough e vesselness | ImageCAS e `pipeline_config.json` | Figuras exibidas no notebook | Baixo para fatias; médio, ou alto com vesselness |

### Resultados quantitativos

| Notebook | Objetivo | Entrada principal | Saída | Custo |
|---|---|---|---|---|
| [segmentation_results_eda.ipynb](segmentation_results_eda.ipynb) | Status dos óstios, distâncias e Dice por split | Resultados canônicos | `analysis/segmentation_results/` | Baixo |
| [split_resolution_analysis.ipynb](split_resolution_analysis.ipynb) | Comparar train/val/test entre mid e high | Resultados canônicos | Tabelas e gráficos exibidos | Baixo |
| [ia_vs_pipeline_analysis.ipynb](ia_vs_pipeline_analysis.ipynb) | Comparar IA e pipeline somente nos IDs comuns | `output/ia_results` e resultados canônicos | Tabelas e gráficos exibidos | Baixo |
| [bad_cases_results_analysis.ipynb](bad_cases_results_analysis.ipynb) | Quantificar casos ruins em mid e high | Summaries canônicos | `analysis/bad_cases/` | Baixo |
| [threshold_pipeline_comparison_analysis.ipynb](threshold_pipeline_comparison_analysis.ipynb) | Comparar threshold normal/fuzzy e RG/FC | Runs de comparação | Tabelas, gráficos e 3D no notebook | Baixo; alto na seção 3D |
| [aorta_circle_slice_analysis.ipynb](aorta_circle_slice_analysis.ipynb) | Relacionar fatias e círculos detectados | Summary com métricas da aorta | `analysis/aorta_circle_slices/` | Baixo |
| [pipeline_parameter_validation_eda.ipynb](pipeline_parameter_validation_eda.ipynb) | Análise OFAT de sensibilidade: sucesso dos óstios, Dice e erros qualitativos no split `val` | Run de `pipeline_parameter_validation.py` | Resultados no notebook; somente o run compacto é persistido | Baixo na análise; alto nos casos 3D |

### Análises qualitativas

| Notebook | Objetivo | Entrada principal | Saída | Custo |
|---|---|---|---|---|
| [bad_cases_qualitative_analysis.ipynb](bad_cases_qualitative_analysis.ipynb) | Rerodar casos ruins e gerar comparação 3D | ImageCAS, bad cases e config | `analysis/cases_analysis/` | Alto |

### Figuras metodológicas

| Notebook | Objetivo | Entrada principal | Saída | Custo |
|---|---|---|---|---|
| [fuzzy_membership_functions.ipynb](fuzzy_membership_functions.ipynb) | Gerar funções de pertinência fuzzy | 60 imagens de teste por padrão | Figura e centros exibidos no notebook | Médio |
| [morphological_operations_example.ipynb](morphological_operations_example.ipynb) | Ilustrar operações morfológicas | Imagem sintética | Figuras exibidas | Baixo |

Os caminhos de saída da tabela são relativos a
`output/segmentation/analysis/`.

## Casos ruins

Os dois notebooks de casos ruins têm responsabilidades diferentes:

- `bad_cases_results_analysis.ipynb` é quantitativo e compara frequências,
  Dice e casos compartilhados entre resoluções.
- `bad_cases_qualitative_analysis.ipynb` seleciona exemplos e executa o
  pipeline para gerar visualizações 3D da aorta, dos óstios e das artérias.

## Convenções

- Imports e configuração ficam no início de cada notebook.
- Caminhos locais devem ser resolvidos por `notebook_env`, nunca escritos de
  forma absoluta nas células.
- Figuras, CSVs e HTMLs derivados permanecem apenas nos notebooks. Em
  `output/segmentation/analysis/`, persistem somente entradas compactas usadas
  por outras análises, como bad cases, métricas de círculos e runs de
  sensibilidade.
- Seções com vesselness, segmentação completa ou visualização 3D são as mais
  demoradas e podem ser executadas isoladamente após o carregamento.
