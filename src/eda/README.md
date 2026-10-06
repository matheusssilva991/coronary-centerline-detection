# Notebooks de EDA

Esta pasta reúne análises exploratórias, comparações de resultados e figuras
metodológicas do pipeline coronário. Os notebooks não substituem os scripts de
experimentos: eles leem resultados já produzidos ou executam poucos casos para
inspeção qualitativa.

As funções de apresentação exclusivas de cada análise ficam em células marcadas
com `presentation-helpers` no próprio notebook. Carregamento, validação, cálculos
e visualizações genéricas continuam nos módulos `utils` compartilhados.

Os resumos por conjunto, a leitura de variantes, os pareamentos e as estatísticas
ficam em `utils.comparison_utils`, sem dependências de apresentação. Os módulos
de `utils.visualization.results` organizam rótulos e gráficos; novos imports de
cálculos devem usar diretamente `comparison_utils`.

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

### Imagens e datasets — `images/`

| Notebook | Objetivo | Entrada principal | Saída | Custo |
|---|---|---|---|---|
| [ccta_dataset_analysis.ipynb](images/ccta_dataset_analysis.ipynb) | Caracterizar e visualizar volumes CCTA do ImageCAS, OrCaScore e MM-WHS | CCTA dos três bancos e pares CTI/CTAI do OrCaScore | Inventário, orientação, geometria, comparação da amostragem do OrCaScore, amostra HU e vistas axiais | Médio ao carregar dez volumes |
| [mmwhs_heart_label_visualization.ipynb](images/mmwhs_heart_label_visualization.ipynb) | Inspecionar os sete rótulos cardíacos do MM-WHS em 2D e 3D | Imagem e rótulo NIfTI de um exame MM-WHS | Corte axial central e cena K3D multirrótulo | Baixo para 2D; médio para a malha 3D |
| [preprocessing_visualization.ipynb](images/preprocessing_visualization.ipynb) | Fatias axiais completas, MIP, downscale, threshold, LCC, Hough e vesselness | ImageCAS e `pipeline_config.json` | Figuras exibidas no notebook | Baixo para fatias; médio, ou alto com vesselness |

### Intensidades HU — `intensity/`

| Notebook | Objetivo | Entrada principal | Saída | Custo |
|---|---|---|---|---|
| [mmwhs_heart_intensity_analysis.ipynb](intensity/mmwhs_heart_intensity_analysis.ipynb) | Comparar HU dos sete labels cardíacos, sua união e dois fundos, com peso igual por exame | Pares CT/label de treino MM-WHS; 10 exames sorteados por padrão | Estatísticas exatas por exame e distribuições médias, sem downscale ou threshold | Médio; um volume carregado por vez |
| [aorta_hu_threshold_comparison.ipynb](intensity/aorta_hu_threshold_comparison.ipynb) | Comparar HU da aorta prevista ImageCAS com a referência MM-WHS e aplicar intervalos cardíacos aos dois bancos | 10 CTs/labels MM-WHS e 5 aortas ImageCAS visualmente boas, com snapshot do run revisado | Distribuições, limites HU, MIPs, fatias e preservação das regiões, sem modificar runs | Médio/alto; reconstrói somente a aorta ImageCAS na CPU, um exame por vez |
| [image_intensity_eda.ipynb](intensity/image_intensity_eda.ipynb) | Distribuição HU e percentis da ROI | Volumes ImageCAS | Figuras e tabelas exibidas | Médio para KDE |

### Resultados gerais — `results/`

| Notebook | Objetivo | Entrada principal | Saída | Custo |
|---|---|---|---|---|
| [external_ccta_visual_assessment.ipynb](results/external_ccta_visual_assessment.ipynb) | Comparar avaliações visuais mid/high por treino e teste no MM-WHS e OrCaScore e Dice da aorta no treino e teste MM-WHS | `avaliacao_visual.xlsx` e `results_all.csv` dos quatro runs | Visão geral por split, distribuições por etapa e Dice separado por protocolo, geral e por exame | Baixo |
| [segmentation_results_eda.ipynb](results/segmentation_results_eda.ipynb) | Status dos óstios, distâncias e Dice por split | Resultados canônicos | `analysis/segmentation_results/` | Baixo |
| [split_resolution_analysis.ipynb](results/split_resolution_analysis.ipynb) | Comparar train/val/test entre mid e high | Resultados canônicos | Tabelas e gráficos exibidos | Baixo |
| [bad_cases_results_analysis.ipynb](results/bad_cases_results_analysis.ipynb) | Quantificar casos ruins em mid e high | Summaries canônicos | `analysis/bad_cases/` | Baixo |

### Comparações de métodos e variantes — `comparisons/`

| Notebook | Objetivo | Entrada principal | Saída | Custo |
|---|---|---|---|---|
| [resolution_filter_envelope_dice_comparison.ipynb](comparisons/resolution_filter_envelope_dice_comparison.ipynb) | Comparar o Dice em mid/high antes e depois do padrão filtro + envelope/lower100/pad2 | Baselines canônicos e runs selecionados de train/val/test | Médias, deltas pareados, IC95%, Wilcoxon/Holm e gráficos exibidos | Baixo |
| [ia_vs_pipeline_analysis.ipynb](comparisons/ia_vs_pipeline_analysis.ipynb) | Comparar IA e pipeline somente nos IDs comuns | `output/ia_results` e resultados canônicos | Tabelas e gráficos exibidos | Baixo |
| [segmentation_method_comparison.ipynb](comparisons/segmentation_method_comparison.ipynb) | Resumir threshold normal/fuzzy e RG/FC; comparar a melhor variante com o baseline | Runs de comparação | Tabelas e gráficos pareados no notebook | Baixo |
| [aorta_ostia_pipeline_comparison.ipynb](comparisons/aorta_ostia_pipeline_comparison.ipynb) | Comparar P99.9 puro, filtro + envelope e novo padrão lower100/pad2 | Treino 30, validação 270, teste 700; revisão visual histórica 30/60 | Dice geral/condicionado, sucesso dos óstios, Wilcoxon pareado com Holm e efeito rank-biserial; qualidade visual separada | Baixo |

### Diagnósticos da aorta — `aorta/`

| Notebook | Objetivo | Entrada principal | Saída | Custo |
|---|---|---|---|---|
| [aorta_circle_slice_analysis.ipynb](aorta/aorta_circle_slice_analysis.ipynb) | Comparar a cobertura dos círculos com as fatias ocupadas pela máscara final e medir expansão ou recuo axial | Summary com métricas da aorta e rótulos visuais | Tabelas agregadas e valores por exame | Baixo |
| [aorta_volume_quality_analysis.ipynb](aorta/aorta_volume_quality_analysis.ipynb) | Relacionar volume da máscara, sucesso dos óstios e avaliação visual da aorta; confirmar no `val` o corte exploratório obtido no `train` | Runs `mid_res` de treino e validação com métricas volumétricas e rótulos visuais | Tabelas separadas por coorte e comparação do corte no notebook | Baixo |

### Sensibilidade e thresholds — `sensitivity/`

| Notebook | Objetivo | Entrada principal | Saída | Custo |
|---|---|---|---|---|
| [pipeline_sensitivity_analysis.ipynb](sensitivity/pipeline_sensitivity_analysis.ipynb) | Análise OFAT: Dice, sucesso dos óstios, efeitos relativos e amplitude por parâmetro no split `val` | Runs de `pipeline_parameter_validation.py` | Tabelas e gráficos exibidos no notebook | Baixo |
| [upper_threshold_analysis.ipynb](sensitivity/upper_threshold_analysis.ipynb) | Comparar P99.9, P99.7 e P99.5; investigar thresholds HU e histogramas de quatro casos representativos | Mesmos runs de validação e, opcionalmente, volumes ImageCAS | Tabelas e gráficos exibidos no notebook | Baixo; moderado ao carregar intensidades |

Os caminhos de saída da tabela são relativos a
`output/segmentation/analysis/`.

## Casos ruins

[`results/bad_cases_results_analysis.ipynb`](results/bad_cases_results_analysis.ipynb) compara quantitativamente as frequências,
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
