# Notebooks de EDA

Esta pasta reúne análises exploratórias, comparações de resultados e figuras
metodológicas do pipeline coronário. Os notebooks leem resultados já produzidos ou executam poucos casos para
inspeção qualitativa.

As funções de apresentação exclusivas de notebooks mais extensos ficam em submódulos
`view` dentro da respectiva subpasta de categoria (`src/eda/<grupo>/view/<analise>_views.py`),
importados em células marcadas com `presentation-helpers`, enquanto notebooks leves
com 1 ou 2 funções curtas podem mantê-las diretamente nessa célula. Carregamento,
validação, cálculos e visualizações genéricas continuam nos módulos `utils`
compartilhados.

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

### Análises que usam a revisão visual histórica da aorta

Quatro notebooks leem as classificações manuais por exame do catálogo
`config/aorta_visual_reviews.json`. Ele contém os IDs revisados, as observações
e os caminhos das execuções correspondentes. As seguintes análises precisam
dele para reproduzir a seleção dos exames ou as tabelas de revisão visual.

- [Cobertura dos círculos por fatia](aorta/aorta_circle_slice_analysis.ipynb)
- [Volume e qualidade da aorta](aorta/aorta_volume_quality_analysis.ipynb)
- [Comparação da aorta e dos óstios](comparisons/aorta_ostia_pipeline_comparison.ipynb)
- [Comparação de intensidades HU](intensity/aorta_hu_threshold_comparison.ipynb)

O catálogo foi restaurado da revisão `4ce263f` em 20/11/2026. As fontes usadas
pelas seleções atuais desses notebooks estão disponíveis. Algumas variantes
antigas do catálogo já tiveram seus runs removidos; a ausência dessas fontes
está registrada em `output/segmentation/results_catalog.json`. A limpeza
preserva os resultados usados pelas revisões disponíveis.

## Catálogo

### Imagens e datasets — `images/`

| Notebook | Objetivo | Entrada principal | Saída | Custo |
|---|---|---|---|---|
| [artery_volume_analysis.ipynb](images/artery_volume_analysis.ipynb) | Quantificar volumes em mL e relacioná-los ao Dice, óstios e volume predito em Mid/High | Máscaras ImageCAS, classe 1; canonical e configurações efetivas por split | CSV incremental em `imagecas_artery_volume/artery_volumes.csv`; figuras Plotly de distribuição, Dice/delta, faixas e predição × referência; análise de outliers | Médio na primeira execução; baixo com CSV |
| [ccta_dataset_analysis.ipynb](images/ccta_dataset_analysis.ipynb) | Caracterizar e visualizar volumes CCTA do ImageCAS, OrCaScore e MM-WHS | CCTA dos três bancos e pares CTI/CTAI do OrCaScore | Inventário, orientação, geometria, comparação da amostragem do OrCaScore, amostra HU e vistas axiais | Médio ao carregar dez volumes |
| [orcascore_calcification_visualization.ipynb](images/orcascore_calcification_visualization.ipynb) | Visualizar CT com e sem contraste e calcificações de referência do OrCaScore, sem reorientar os dados | Par CTAI/CTI e referência de treino; padrão TRV1P3; `ORCASCORE_BASE_PATH` ou descoberta nos discos montados | MIPs e fatias centrais; CTI, rótulos e sobreposição recortados; 3D opcional; PNG/TIFF a 600 dpi e PDF em `output/figures/orcascore_calcification/` ou `ORCASCORE_FIGURES_ROOT` | Baixo/médio; dois volumes e referência |
| [mmwhs_heart_label_visualization.ipynb](images/mmwhs_heart_label_visualization.ipynb) | Inspecionar os sete rótulos cardíacos do MM-WHS em 2D e 3D | Imagem e rótulo NIfTI de um exame MM-WHS | Corte axial central e cena K3D multirrótulo | Baixo para 2D; médio para a malha 3D |
| [totalsegmentator_label_visualization.ipynb](images/totalsegmentator_label_visualization.ipynb) | Inspecionar CT e máscaras selecionáveis de aorta, coração e outras estruturas do TotalSegmentator | Um exame com `ct.nii.gz` e `segmentations/`; caminho por `TOTALSEGMENTATOR_BASE_PATH` ou descoberta nos discos montados | Fatias completas axiais, coronais e sagitais; resumo volumétrico e 3D opcional | Baixo/médio para 2D; médio para 3D |
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
| [imagecas_artery_distance_analysis.ipynb](results/imagecas_artery_distance_analysis.ipynb) | Analisar Dice, AHD e HD da máscara arterial inteira em um exame ImageCAS | Exame e rótulo ImageCAS, configuração do pipeline e CSV canônico selecionável | Tabela de métricas, histogramas por sentido e malhas K3D das máscaras e das faixas de distância | Alto para prever a artéria, baixo para consultar o CSV |
| [external_aorta_assessment.ipynb](results/external_aorta_assessment.ipynb) | Comparar avaliações visuais mid/high por treino e teste no MM-WHS e OrCaScore e Dice da aorta no treino e teste MM-WHS | `avaliacao_visual.xlsx` e `results_all.csv` dos quatro runs | Avaliação visual, Dice separado por protocolo e alertas contextualizados por exame; comparação automática no notebook dedicado | Baixo |
| [segmentation_results_eda.ipynb](results/segmentation_results_eda.ipynb) | Status e distâncias dos óstios, Dice e HD/AHD arteriais | `canonical` por resolução e split | Tabelas e gráficos, com comparação mid/high por ID | Baixo |
| [split_resolution_analysis.ipynb](results/split_resolution_analysis.ipynb) | Comparar train/val/test entre mid e high | Referências oficiais de `canonical` | Tabelas e gráficos; tempos ausentes permanecem indisponíveis | Baixo |

### Comparações de métodos e variantes — `comparisons/`

| Notebook | Objetivo | Entrada principal | Saída | Custo |
|---|---|---|---|---|
| [resolution_filter_envelope_dice_comparison.ipynb](comparisons/resolution_filter_envelope_dice_comparison.ipynb) | Comparar o Dice em mid/high antes e depois do padrão filtro + envelope/lower100/pad2 | Baselines canônicos e runs selecionados de train/val/test | Médias, deltas pareados, IC95%, Wilcoxon/Holm e gráficos exibidos | Baixo |
| [ia_vs_pipeline_analysis.ipynb](comparisons/ia_vs_pipeline_analysis.ipynb) | Comparar IA e referência oficial nos IDs comuns, com Wilcoxon e Holm | Resultados IA e `canonical` mid/high | Tabelas e gráficos exibidos | Baixo |
| [orcascore_artery_ia_vs_pipeline.ipynb](comparisons/orcascore_artery_ia_vs_pipeline.ipynb) | Medir concordância arterial entre o pipeline OrCaScore e o modelo de IA mid de 4 ou 12 canais | Run externo com máscaras e predições NIfTI sob `CCTA_RESULTS_ROOT` | Dice por exame e conjunto, histograma e sobreposição K3D com óstios | Médio/alto, leitura e alinhamento por exame |
| [segmentation_method_comparison.ipynb](comparisons/segmentation_method_comparison.ipynb) | Comparar normal/fuzzy e RG/FC no pipeline refinado, por conjunto, com Wilcoxon e Holm | Runs atuais train30_val60 e test700 | Tabelas, gráficos pareados e estatísticas no notebook | Baixo |
| [aorta_ostia_pipeline_comparison.ipynb](comparisons/aorta_ostia_pipeline_comparison.ipynb) | Comparar P99.9 puro, filtro + envelope e novo padrão lower100/pad2 | Treino 30, validação 270, teste 700; revisão visual histórica 30/60 | Dice geral/condicionado, sucesso dos óstios, Wilcoxon pareado com Holm e efeito rank-biserial; qualidade visual separada | Baixo |

### Diagnósticos da aorta — `aorta/`

| Notebook | Objetivo | Entrada principal | Saída | Custo |
|---|---|---|---|---|
| [aorta_automatic_quality_assessment.ipynb](aorta/aorta_automatic_quality_assessment.ipynb) | Comparar alertas automáticos ImageCAS, MM-WHS e OrCaScore por subconjunto e resolução | Somente CSVs canônicos ImageCAS e runs externos explícitos | Gráfico percentual e tabela n (%) dos bancos e Geral; cobertura e transições mid/high; ImageCAS aorta × óstios; MM-WHS diagnóstico × Dice (corte exploratório ≥ 0,70, teste somente com protocolo corrigido); sem segmentações | Baixo |
| [aorta_high_sigma_vesselness_experiment.ipynb](aorta/aorta_high_sigma_vesselness_experiment.ipynb) | Comparar a localização por Hough na tomografia e no Frangi de sigma alto | Um exame de treino do MM-WHS ou ImageCAS | MIP axial do Frangi, máscaras K3D, cobertura, volume, tempos e Dice quando houver referência de aorta | Médio/alto |
| [aorta_circle_slice_analysis.ipynb](aorta/aorta_circle_slice_analysis.ipynb) | Comparar a cobertura dos círculos com as fatias ocupadas pela máscara final e medir expansão ou recuo axial | Summary com métricas da aorta e rótulos visuais | Tabelas agregadas e valores por exame | Baixo |
| [aorta_volume_quality_analysis.ipynb](aorta/aorta_volume_quality_analysis.ipynb) | Relacionar volume da máscara, sucesso dos óstios e avaliação visual da aorta; confirmar no `val` o corte exploratório obtido no `train` | Runs `mid_res` de treino e validação com métricas volumétricas e rótulos visuais | Tabelas separadas por coorte e comparação do corte no notebook | Baixo |

### Sensibilidade e thresholds — `sensitivity/`

| Notebook | Objetivo | Entrada principal | Saída | Custo |
|---|---|---|---|---|
| [pipeline_sensitivity_analysis.ipynb](sensitivity/pipeline_sensitivity_analysis.ipynb) | Análise OFAT do pipeline refinado: Dice, óstios e amplitude por parâmetro nas 270 imagens de validação | Run sensitivity_refined_pipeline_val270 | Tabelas e gráficos exibidos no notebook | Baixo |
| [upper_threshold_analysis.ipynb](sensitivity/upper_threshold_analysis.ipynb) | Comparar P99.9, P99.7 e P99.5; investigar thresholds HU e histogramas de quatro casos representativos | Mesmos runs de validação e, opcionalmente, volumes ImageCAS | Tabelas e gráficos exibidos no notebook | Baixo; moderado ao carregar intensidades |

Os caminhos de saída da tabela são relativos a
`output/segmentation/analysis/`.

## Fontes das análises

As análises do baseline atual usam `canonical`. A seção de HD/AHD de superfície
usa os CSVs canônicos de mid e high e exibe a disponibilidade por conjunto.
Comparações históricas, fuzzy e de sensibilidade mantêm fontes explícitas.

## Convenções

- Imports e configuração ficam no início de cada notebook.
- Caminhos locais devem ser resolvidos por `notebook_env`, nunca escritos de
  forma absoluta nas células.
- Figuras, CSVs e HTMLs derivados permanecem apenas nos notebooks. Em
  `output/segmentation/analysis/`, persistem somente entradas compactas usadas
  por outras análises, como bad cases, métricas de círculos e runs de
  sensibilidade.
- A análise de volume arterial também persiste o CSV compacto
  `imagecas_artery_volume/artery_volumes.csv` após cada exame. Ela retoma arquivos
  pendentes/com erro, sem reler máscaras concluídas. `FORCE_RECOMPUTE = True`
  recalcula tudo; use-o quando as máscaras concluídas forem modificadas.
  Sem o dataset montado, é possível consultar o CSV com aviso de verificação
  incompleta. O volume usa somente `label == 1`, na geometria nativa.
  `OUTLIER_METHOD` seleciona IQR (padrão: 1,5×) ou média ± DP (padrão: 2×).
  A exclusão de outliers é apenas descritiva, usa os volumes de referência
  disponíveis e não altera os CSVs nem as referências canônicas.
- Seções com vesselness, segmentação completa ou visualização 3D são as mais
  demoradas e podem ser executadas isoladamente após o carregamento.
