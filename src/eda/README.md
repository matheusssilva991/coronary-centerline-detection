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
| [ccta_dataset_analysis.ipynb](ccta_dataset_analysis.ipynb) | Caracterizar e visualizar volumes CCTA do ImageCAS, OrCaScore e MM-WHS | CCTA dos três bancos e um par CTI/CTAI do OrCaScore | Inventário, orientação, geometria, amostra HU e vistas axiais | Médio ao carregar dez volumes |
| [image_intensity_eda.ipynb](image_intensity_eda.ipynb) | Distribuição HU e percentis da ROI | Volumes ImageCAS | Figuras e tabelas exibidas | Médio para KDE |
| [preprocessing_visualization.ipynb](preprocessing_visualization.ipynb) | Fatias axiais completas, MIP, downscale, threshold, LCC, Hough e vesselness | ImageCAS e `pipeline_config.json` | Figuras exibidas no notebook | Baixo para fatias; médio, ou alto com vesselness |

### Resultados quantitativos

| Notebook | Objetivo | Entrada principal | Saída | Custo |
|---|---|---|---|---|
| [segmentation_results_eda.ipynb](segmentation_results_eda.ipynb) | Status dos óstios, distâncias e Dice por split | Resultados canônicos | `analysis/segmentation_results/` | Baixo |
| [split_resolution_analysis.ipynb](split_resolution_analysis.ipynb) | Comparar train/val/test entre mid e high | Resultados canônicos | Tabelas e gráficos exibidos | Baixo |
| [resolution_filter_envelope_dice_comparison.ipynb](resolution_filter_envelope_dice_comparison.ipynb) | Comparar o Dice em mid/high antes e depois do padrão filtro + envelope/lower100/pad2 | Baselines canônicos e runs selecionados de train/val/test | Médias, deltas pareados, IC95%, Wilcoxon/Holm e gráficos exibidos | Baixo |
| [ia_vs_pipeline_analysis.ipynb](ia_vs_pipeline_analysis.ipynb) | Comparar IA e pipeline somente nos IDs comuns | `output/ia_results` e resultados canônicos | Tabelas e gráficos exibidos | Baixo |
| [bad_cases_results_analysis.ipynb](bad_cases_results_analysis.ipynb) | Quantificar casos ruins em mid e high | Summaries canônicos | `analysis/bad_cases/` | Baixo |
| [segmentation_method_comparison.ipynb](segmentation_method_comparison.ipynb) | Resumir threshold normal/fuzzy e RG/FC; comparar a melhor variante com o baseline | Runs de comparação | Tabelas e gráficos pareados no notebook | Baixo |
| [aorta_circle_slice_analysis.ipynb](aorta_circle_slice_analysis.ipynb) | Comparar a cobertura dos círculos com as fatias ocupadas pela máscara final e medir expansão ou recuo axial | Summary com métricas da aorta e rótulos visuais | Tabelas agregadas e valores por exame | Baixo |
| [aorta_volume_quality_analysis.ipynb](aorta_volume_quality_analysis.ipynb) | Relacionar volume da máscara, sucesso dos óstios e avaliação visual da aorta; confirmar no `val` o corte exploratório obtido no `train` | Runs `mid_res` de treino e validação com métricas volumétricas e rótulos visuais | Tabelas separadas por coorte e comparação do corte no notebook | Baixo |
| [aorta_ostia_pipeline_comparison.ipynb](aorta_ostia_pipeline_comparison.ipynb) | Comparar P99.9 puro, filtro + envelope e novo padrão lower100/pad2 | Treino 30, validação 270, teste 700; revisão visual histórica 30/60 | Dice geral/condicionado, sucesso dos óstios, Wilcoxon pareado com Holm e efeito rank-biserial; qualidade visual separada | Baixo |
| [pipeline_sensitivity_analysis.ipynb](pipeline_sensitivity_analysis.ipynb) | Análise OFAT: Dice, sucesso dos óstios, efeitos relativos e amplitude por parâmetro no split `val` | Runs de `pipeline_parameter_validation.py` | Tabelas e gráficos exibidos no notebook | Baixo |
| [upper_threshold_analysis.ipynb](upper_threshold_analysis.ipynb) | Comparar P99.9, P99.7 e P99.5; investigar thresholds HU e histogramas de quatro casos representativos | Mesmos runs de validação e, opcionalmente, volumes ImageCAS | Tabelas e gráficos exibidos no notebook | Baixo; moderado ao carregar intensidades |

### Figuras metodológicas

| Notebook | Objetivo | Entrada principal | Saída | Custo |
|---|---|---|---|---|

Os caminhos de saída da tabela são relativos a
`output/segmentation/analysis/`.

## Casos ruins

`bad_cases_results_analysis.ipynb` compara quantitativamente as frequências,
o Dice e os casos ruins compartilhados entre resoluções.

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
