# Correções do artigo CBEB 2026 — pipeline refinado

## 1. Quantificar falhas de localização dos óstios

> Quantify failures in coronary ostia localization.

**Tabela 1 — Sucesso e falhas da inicialização (teste, n = 700 por resolução).**

| Resultado | Reduzida | Original |
| --- | --- | --- |
| Inicialização aceita | 597 (85,29%) | 538 (76,86%) |
| Inicialização não aceita | 103 (14,71%) | 162 (23,14%) |
| ↳ Um ponto intersecta a máscara | 28 (4,00%) | 81 (11,57%) |
| ↳ Nenhum ponto intersecta a máscara | 68 (9,71%) | 78 (11,14%) |
| ↳ Óstios não encontrados | 0 (0,00%) | 3 (0,43%) |
| ↳ Erro do pipeline | 7 (1,00%) | 0 (0,00%) |

**Legenda:** aceita = ambos os pontos intersectam a máscara coronária ou estão a até 7 mm dela. Falhas exclusivas; percentuais sobre 700 exames. Os sete erros ocorreram na superfície da aorta. IC95% de sucesso (Wilson): reduzida, 82,47–87,72%; original, 73,59–79,83%. A máscara coronária substitui a anotação dos óstios.

## 2. Desempenho de ponta a ponta, intervalos de confiança e distribuição do Dice

> Separate performance conditional on successful initialization from true end-to-end performance.
>
> Report confidence intervals and the distribution of Dice scores.

**Tabela 2 — Dice global e condicionado à inicialização aceita.**

| Resolução/cenário | N | Média ± DP | IC95% média | Mediana [Q1–Q3] |
| --- | --- | --- | --- | --- |
| Reduzida · global | 700 | 0,5996 ± 0,1844 | 0,5857–0,6127 | 0,6677 [0,5684–0,7113] |
| Reduzida · condicionado | 597 | 0,6560 ± 0,0967 | 0,6481–0,6636 | 0,6791 [0,6179–0,7188] |
| Original · global | 700 | 0,5151 ± 0,1823 | 0,5017–0,5286 | 0,5784 [0,4543–0,6377] |
| Original · condicionado | 538 | 0,5799 ± 0,1084 | 0,5706–0,5890 | 0,6080 [0,5411–0,6495] |

**Legenda:** Dice em 0–1; DP amostral; Q1–Q3 = quartis. IC95% por bootstrap percentil (10.000 reamostragens, seed 42). Global inclui todos os exames e atribui zero às sete falhas sem saída; condicionado inclui apenas inicializações aceitas. High usa os 700 IDs de teste do run com 1.000 exames. Condicionamento não demonstra efeito causal.

## 3. Comparação com deep learning

> Formally compare computational cost with deep learning approaches.

**Tabela 3 — Precisão e recursos computacionais: pipeline e FCNs diretas em Mid e High.**

| Método / resolução | DSC | Tempo/volume (base: 1.000) | Pico RAM | VRAM |
| --- | --- | --- | --- | --- |
| Pipeline refinado / Mid | 0,6024 | 58,80 s | 2,88 GB | 2,12 GB |
| Direct FCN (4 canais) / Mid | 0,7132 | 244,80 s | — | 15,45 GB |
| Direct FCN (12 canais) / Mid | **0,7195** | 244,80 s | — | 18,61 GB |
| Direct FCN + AG (4 canais) / Mid | 0,7172 | 244,80 s | — | 19,3 GB |
| Pipeline refinado / High | 0,5098 | 300,37 s | 5,46 GB | 7,95 GB |
| Direct FCN (4 canais) / High | 0,7655 | 244,80 s | — | 22,93 GB |
| Direct FCN + AG (4 canais) / High | **0,7785** | 244,80 s | — | 23,7 GB |

**Legenda:** Mid = resolução reduzida; High = original; AG = Attention Gate; — = indisponível. DSC nos mesmos 200 exames; negrito = melhor IA por resolução. Tempo da IA = 68 h / 1.000 volumes; pipeline High = 300.370,08 s / 1.000; Mid = 41.157,83 s / 700, projetando 16h19min57s para 1.000. A IA foi executada com testes simultâneos; os protocolos não permitem concluir vantagem de latência isolada.

**Memória:** máximos observados por PID nas mesmas 30 imagens de treino, amostrados a cada 0,5 s, sem visuais. Medição em Ryzen 5 5500X3D/RTX 5060 Ti, distinta dos tempos do servidor; inclui memória retida entre exames e pode perder picos breves. Pipeline em GB decimais; VRAM da IA conforme informada, sem confirmação de pico por processo ou base de conversão.

**Fontes:** tempo [Mid](../../output/segmentation/runs/mid_res/fuzzy_comparison/current_pipeline/test700/summary/ranking.csv) e [High](../../output/segmentation/runs/high_res/benchmark_server/full1000/2026-10-12_09-08-06/numeric/metadata_full.json); memória [Mid](../../output/segmentation/runs/mid_res/2026-10-22_15-18-08/numeric/results_train.csv) e [High](../../output/segmentation/runs/high_res/2026-10-22_18-52-53/numeric/results_train.csv); IA: dados fornecidos pelo autor.

**Tabela 4 — Dice (média ± DP) nos 200 exames do conjunto de IA (quatro folds).**

| Método | Reduzida | Original |
| --- | --- | --- |
| Direct FCN (12 canais) | 0,7195 ± 0,0501 | — |
| Direct FCN (4 canais) | 0,7132 ± 0,0528 | 0,7655 ± 0,0597 |
| Direct FCN (4 canais, AG) | 0,7172 ± 0,0495 | 0,7785 ± 0,0555 |
| Tree GRU | 0,5916 ± 0,1429 | — |
| Graph GCN | 0,5865 ± 0,1306 | — |
| Pipeline refinado | 0,6024 ± 0,1747 | 0,5098 ± 0,1868 |

**Legenda:** média ± DP amostral entre 200 exames pareados dos quatro folds, incluindo falhas. A coorte contém sete exames de treino, 39 de validação e 154 de teste do pipeline; não é uma avaliação exclusiva no teste independente. High usa o run de 1.000 exames. AG = Attention Gate; — = indisponível.

## 4. Ablações e sensibilidade

> Perform ablation studies for dual-vesselness, Level Set refinement, and downsampling.

**Tabela 5 — Sensibilidade do pipeline refinado (validação, n = 270, resolução reduzida).**

| Parâmetro | Valor | Sucesso óstios (%) | DSC (média ± DP) | Mediana DSC | Δ DSC |
| --- | --- | --- | --- | --- | --- |
| Percentil superior | **99,9** | **85,19** | **0,6009 ± 0,1776** | **0,6593** | — |
| Percentil superior | 99,7 | 82,59 | 0,5841 ± 0,1919 | 0,6532 | -0,0168 |
| Percentil superior | 99,5 | 80,00 | 0,5596 ± 0,1964 | 0,6233 | -0,0413 |
| Distância máxima em z | 30 mm | 84,81 | 0,6008 ± 0,1781 | 0,6593 | -0,0001 |
| Distância máxima em z | **40 mm** | **85,19** | **0,6009 ± 0,1776** | **0,6593** | — |
| Distância máxima em z | 50 mm | 85,19 | 0,6009 ± 0,1776 | 0,6593 | 0,0000 |
| Vesselness mínimo do RG | 5,0% | 85,19 | 0,5720 ± 0,1713 | 0,6250 | -0,0289 |
| Vesselness mínimo do RG | **7,8%** | **85,19** | **0,6009 ± 0,1776** | **0,6593** | — |
| Vesselness mínimo do RG | 9,0% | 85,19 | 0,5941 ± 0,1827 | 0,6593 | -0,0068 |

**Legenda:** um parâmetro alterado por vez nos mesmos 270 exames. Negrito = referência; Δ DSC = diferença para ela; DP amostral. Falhas sem saída recebem zero. Diferenças menores que 0,00005 aparecem como 0,0000; — = referência.

**Tabela 6 — Ablação dos componentes do pipeline refinado (teste, n = 700 por variante).**

| Configuração | Sucesso óstios | DSC end-to-end |
| --- | --- | --- |
| Pipeline completo | 597/700 (85,29%) | 0,5996 |
| Sem dual-vesselness (mapa único) | 536/700 (76,57%) | 0,5616 |
| Sem Level Set (círculos completos) | 56/700 (8,00%) | 0,2689 |
| Sem downsampling | 24/700 (3,43%) | 0,0228 |

**Legenda:** mesmos 700 IDs; referência em Mid. Mapa único compartilha a parametrização arterial com os óstios; sem Level Set usa círculos completos e mantém o pós-processamento; sem downsampling altera apenas os fatores para [1, 1, 1], sem reescalar parâmetros. Falhas sem saída recebem zero. O controle foi executado em outro ambiente; a comparação não isola integralmente o efeito dos componentes.

## 5. Robustez e falhas de inicialização

> Moderate claims of robustness while initialization remains a critical source of failure.

**Síntese das Tabelas 1 e 2:** falhas de inicialização em 14,71% (Mid) e 23,14% (High). O Dice condicionado é maior em 5,64 e 6,47 pontos percentuais, sem demonstrar ganho causal pela correção dos óstios.
