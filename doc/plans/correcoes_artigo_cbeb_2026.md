# Correções do artigo CBEB 2026 — pipeline refinado

## 1. Quantificar falhas de localização dos óstios

> Quantify failures in coronary ostia localization.

**Tabela 1 — Sucesso e falhas da inicialização (teste, n = 700 por resolução).**

| Resultado | Reduzida | Original |
| --- | --- | --- |
| Inicialização aceita | 597 (85,29%) | 538 (76,86%) |
| Inicialização não aceita | 103 (14,71%) | 162 (23,14%) |
| ↳ Um ponto intersecta a máscara | 28 (4,00%) | 80 (11,43%) |
| ↳ Nenhum ponto intersecta a máscara | 68 (9,71%) | 79 (11,29%) |
| ↳ Óstios não encontrados | 0 (0,00%) | 3 (0,43%) |
| ↳ Erro do pipeline | 7 (1,00%) | 0 (0,00%) |

**Legenda:** inicialização aceita = ambos os pontos intersectam a máscara coronária ou ficam a até 7 mm dela. As quatro categorias de falha são exclusivas e somam a linha “não aceita”; percentuais relativos a 700 exames. “Um” e “nenhum” indicam interseção estrita nos pares reprovados. Os sete erros ocorreram na obtenção da superfície da aorta. IC95% de sucesso (Wilson): reduzida, 82,47–87,72%; original, 73,59–79,83%. A máscara coronária é uma referência substituta, sem anotação independente dos óstios.

## 2. Desempenho de ponta a ponta, intervalos de confiança e distribuição do Dice

> Separate performance conditional on successful initialization from true end-to-end performance.
>
> Report confidence intervals and the distribution of Dice scores.

**Tabela 2 — Dice global e condicionado à inicialização aceita.**

| Resolução/cenário | N | Média ± DP | IC95% média | Mediana [Q1–Q3] |
| --- | --- | --- | --- | --- |
| Reduzida · global | 700 | 0,5996 ± 0,1844 | 0,5857–0,6127 | 0,6677 [0,5684–0,7113] |
| Reduzida · condicionado | 597 | 0,6560 ± 0,0967 | 0,6481–0,6636 | 0,6791 [0,6179–0,7188] |
| Original · global | 700 | 0,5146 ± 0,1837 | 0,5011–0,5281 | 0,5784 [0,4550–0,6377] |
| Original · condicionado | 538 | 0,5805 ± 0,1072 | 0,5712–0,5894 | 0,6072 [0,5424–0,6495] |

**Legenda:** Dice em escala 0–1; DP amostral; Q1–Q3 = primeiro e terceiro quartis. IC95% percentil bootstrap da média, com 10.000 reamostragens por exame e seed 42. Global = todos os exames; condicionado = subconjunto com inicialização aceita. Os sete erros sem Dice na resolução reduzida recebem zero na avaliação global; os demais valores observados são preservados. O cenário condicionado não equivale a sementes ideais nem isola o efeito causal da inicialização.

## 3. Comparação com deep learning

> Formally compare computational cost with deep learning approaches.

**Tabela 3 — Precisão e recursos computacionais: pipeline e FCNs diretas em Mid e High.**

| Método / resolução | DSC | Tempo/volume | Pico RAM | VRAM informada |
| --- | --- | --- | --- | --- |
| Pipeline refinado / Mid | 0,6024 | 58,80 s | — | — |
| Direct FCN (4 canais) / Mid | 0,7132 | Não isolado | — | 15,45 GB |
| Direct FCN (12 canais) / Mid | **0,7195** | Não isolado | — | 18,61 GB |
| Direct FCN + AG (4 canais) / Mid | 0,7172 | Não isolado | — | 19,3 GB |
| Pipeline refinado / High | 0,5069 | — | — | — |
| Direct FCN (4 canais) / High | 0,7655 | Não isolado | — | 22,93 GB |
| Direct FCN + AG (4 canais) / High | **0,7785** | Não isolado | — | 23,7 GB |

**Legenda:** Mid = resolução reduzida; High = resolução original; AG = Attention Gate. Negrito indica a IA de maior DSC em cada resolução: Direct FCN de 12 canais em Mid e Direct FCN+AG de quatro canais em High. DSC calculado nos mesmos 200 exames dos quatro folds de IA. O tempo de 58,80 s/volume corresponde à média amortizada dos 700 exames da execução reduzida do pipeline no servidor, não aos 200 pares. A VRAM foi informada pelo autor; os registros de FCN+AG em Mid e High correspondem à CEPEDI. O protocolo de medição não confirma pico de VRAM por processo. Não há dados completos de DSC e VRAM para FCN+AG de 12 canais em Mid nem para FCNs de 12 canais em High. — = medida indisponível, não uso nulo de memória; o pipeline também solicita GPU.

**Legenda sobre comparabilidade do custo:** as 68 horas reportadas para a segmentação direta incluem execução concorrente de vários testes pela automação ImageCAS; não representam a latência isolada desta rede e não permitem calcular tempo por volume nem speedup frente ao pipeline. Os tempos de patches dependem da segmentação direta e não são custos isolados; os resultados de grafos e árvores em Low não substituem medidas da FCN+AG original. Uma comparação controlada exige os mesmos exames e equipamento, um processo por vez, lote de um volume, modelos previamente treinados e a mesma fronteira de medição (carregamento, pré-processamento, inferência/segmentação e pós-processamento). Treinamento deve ser contabilizado separadamente; tempo por volume, pico de RAM do processo e pico de VRAM por processo devem ser medidos para ambos. A comparação na resolução original e a comparação entre FCN+AG original e pipeline reduzido devem ser identificadas separadamente. Os registros atuais permitem comparar precisão e informar o consumo reportado, mas não demonstram vantagem de velocidade ou memória do método determinístico.

**Tabela 4 — Dice (média ± DP) nos 200 exames do conjunto de IA (quatro folds).**

| Método | Reduzida | Original |
| --- | --- | --- |
| Direct FCN (12 canais) | 0,7195 ± 0,0501 | — |
| Direct FCN (4 canais) | 0,7132 ± 0,0528 | 0,7655 ± 0,0597 |
| Direct FCN (4 canais, AG) | 0,7172 ± 0,0495 | 0,7785 ± 0,0555 |
| Tree GRU | 0,5916 ± 0,1429 | — |
| Graph GCN | 0,5865 ± 0,1306 | — |
| Pipeline refinado | 0,6024 ± 0,1747 | 0,5069 ± 0,1899 |

**Legenda:** valores em média ± desvio-padrão amostral (DP, n−1), calculados entre os 200 exames individuais; não representam o desvio entre as quatro médias dos folds. AG = Attention Gate; — = resultado indisponível. Cada média inclui os 200 IDs únicos disponíveis nos quatro folds de IA, sem restringir a comparação aos 154 IDs do teste determinístico. O pipeline refinado foi pareado com esses mesmos 200 exames, preservando as falhas no Dice global. Em relação à divisão do determinístico, a coorte contém sete exames de treino, 39 de validação e 154 de teste; portanto, esta comparação não constitui avaliação exclusiva em seu teste independente. As médias diferem das da Tabela 2, que usa 700 exames de teste. A independência dos pacientes em relação ao treinamento das redes depende dos splits dos respectivos folds. Proximidade entre médias não demonstra equivalência; esta tabela compara precisão, não custo computacional.

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

**Legenda:** resultados do pipeline refinado, com um parâmetro alterado por vez; as linhas em negrito repetem a mesma referência em cada grupo (percentil 99,9, distância z de 40 mm e vesselness mínimo de 7,8%). DSC em escala 0–1; DP amostral (n−1), calculado sobre os mesmos 270 exames usados na média e na mediana. Falhas sem saída entram como zero no DSC global; o valor observado é preservado quando há segmentação. Δ DSC = diferença em relação à referência; diferenças menores que 0,00005 são exibidas como 0,0000. — = configuração de referência. Sensibilidade de parâmetros não constitui ablação de componentes.

**Tabela 6 — Ablação dos componentes do pipeline refinado (teste; 700 exames planejados por variante).**

| Configuração | Sucesso óstios | DSC end-to-end |
| --- | --- | --- |
| Pipeline completo | 597/700 (85,29%) | 0,5996 |
| Sem dual-vesselness (mapa único) | 536/700 (76,57%) | 0,5616 |
| Sem Level Set (círculos completos) | 56/700 (8,00%) | 0,2689 |
| Sem downsampling (parcial) | 6/165 (3,64%) | 0,0202 |

**Legenda:** as variantes utilizam a coorte planejada de 700 IDs de teste e a configuração científica da referência reduzida. As três primeiras linhas estão completas (700 exames cada); sem downsampling apresenta somente os primeiros 165 exames concluídos, não o resultado final dos 700. Os 535 exames pendentes não entram no denominador nem recebem Dice zero. A comparação direta das médias parciais com as médias completas envolve conjuntos de tamanhos diferentes. “Mapa único” compartilha o mapa com parametrização arterial (`VESSELNESS_ARTERY`) entre localização dos óstios e extração arterial. “Sem Level Set” usa a máscara geométrica dos círculos com raio completo (fator 1,0), sem evolução e mantendo o pós-processamento; os resultados do teste antigo com sementes reduzidas e do piloto de três exames não entram nesta tabela. Dice global inclui falhas sem saída como zero: sete no mapa único, nove sem Level Set e 13 no resultado parcial sem downsampling. As reduções observadas em relação à referência são 0,0380 e 0,3307, respectivamente. O controle provém de execução anterior em outro ambiente, com diferenças nos intermediários da aorta; a comparação não isola integralmente o efeito de cada componente. A ablação de downsampling está em andamento; a variante `no_downsampling` altera apenas os fatores para [1, 1, 1], mantendo numericamente todos os demais parâmetros da referência reduzida, sem reescala automática. As escalas físicas dos parâmetros em voxels mudam com a resolução. A execução original otimizada existente não substitui essa ablação.

## 5. Robustez e falhas de inicialização

> Moderate claims of robustness while initialization remains a critical source of failure.

**Legenda das Tabelas 1 e 2:** a inicialização foi reprovada em 14,71% dos exames reduzidos e 23,14% dos exames originais. O Dice condicionado supera o global em 5,64 e 6,59 pontos percentuais, respectivamente. Essas diferenças descrevem subconjuntos distintos e não representam ganhos causais garantidos ao corrigir a inicialização.
