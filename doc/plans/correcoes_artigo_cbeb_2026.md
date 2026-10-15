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

**Tabela 3 — Precisão e custo computacional na resolução reduzida.**

| Método | DSC | Tempo/volume | Pico RAM | Pico VRAM |
| --- | --- | --- | --- | --- |
| Pipeline refinado | 0,6026 | 58,80 s | — | — |
| Direct FCN (12 canais) | 0,7227 | — | — | — |
| Tree GRU | 0,6035 | — | — | — |
| Graph GCN | 0,6007 | — | — | — |

**Legenda:** DSC médio nos mesmos 154 exames de teste comuns aos métodos, preservando a comparação de precisão. O DSC global do pipeline nos 700 exames é 0,5996 (Tabela 2). O tempo disponível do pipeline é a média amortizada nos 700 volumes da última execução no servidor, variante normal+Region Growing: 41.157,83 s / 700 = 58,80 s/volume; não é uma medição restrita aos 154 pares. — = medida ainda indisponível, não ausência de uso do recurso. A execução solicita GPU; pico de VRAM não foi medido. Tempos e picos de memória das redes ainda precisam de medição sob protocolo e equipamento comparáveis; os dados atuais não demonstram vantagem computacional sobre DL.

**Tabela 4 — Dice no subconjunto comum de teste (n = 154).**

| Método | Reduzida | Original |
| --- | --- | --- |
| Direct FCN (12 canais) | 0,7227 | — |
| Direct FCN (4 canais) | 0,7171 | 0,7694 |
| Direct FCN (4 canais, AG) | 0,7209 | 0,7832 |
| Tree GRU | 0,6035 | — |
| Graph GCN | 0,6007 | — |
| Pipeline refinado | 0,6026 | 0,5057 |

**Legenda:** AG = Attention Gate; — = resultado indisponível. Foram pareados os 154 IDs comuns ao teste do pipeline e aos arquivos de IA dos quatro folds; 46 dos 200 IDs de IA estão fora desse teste. O Dice do pipeline é global nesse subconjunto, por isso difere da Tabela 2. Proximidade entre médias não demonstra equivalência; esta tabela compara precisão, não custo computacional.

## 4. Ablações e sensibilidade

> Perform ablation studies for dual-vesselness, Level Set refinement, and downsampling.

**Tabela 5 — Sensibilidade do pipeline refinado (validação, n = 270, resolução reduzida).**

| Configuração | Sucesso óstios | Dice global | Δ Dice |
| --- | --- | --- | --- |
| Referência: P99,9 / z40 / 7,8% | 85,19% | 0,6009 | 0,0000 |
| Percentil 99,7 | 82,59% | 0,5841 | -0,0168 |
| Percentil 99,5 | 80,00% | 0,5596 | -0,0413 |
| Distância z: 30 mm | 84,81% | 0,6008 | -0,0001 |
| Distância z: 50 mm | 85,19% | 0,6009 | 0,0000 |
| Vesselness mínimo: 5,0% | 85,19% | 0,5720 | -0,0289 |
| Vesselness mínimo: 9,0% | 85,19% | 0,5941 | -0,0068 |

**Legenda:** um parâmetro é alterado por vez, mantendo os demais na referência: percentil 99,9, distância z de 40 mm e vesselness mínimo de 7,8%. Δ Dice = diferença em relação à referência. Erros explícitos entram como zero no Dice global salvo. Sensibilidade de parâmetros não constitui ablação de componentes.

**Tabela 6 — Ablação dos componentes do pipeline refinado (teste, n = 700).**

| Configuração | Sucesso óstios | DSC end-to-end |
| --- | --- | --- |
| Pipeline completo | 85,29% | 0,5996 |
| Sem dual-vesselness (mapa único) | A medir | A medir |
| Sem Level Set | A medir | A medir |
| Sem downsampling | A medir | A medir |

**Legenda:** referência = pipeline refinado na resolução reduzida. “Mapa único” utiliza o mesmo mapa de vesselness na localização dos óstios e na extração arterial; a parametrização compartilhada deve ser explicitada. “Sem Level Set” substitui o refinamento pela máscara geométrica dos círculos com raio completo (fator 1,0), sem evolução do contorno e com o mesmo pós-processamento. Essa definição corrige o teste anterior com sementes reduzidas a 10% do raio, que dependiam de expansão. As variantes preservam os demais refinamentos e os mesmos IDs. Para remover o downsampling, manter a equivalência física das escalas e restrições espaciais, sem reajustar parâmetros pelo teste. A medir = ablação controlada ainda não realizada. Os 76,86% de sucesso e DSC 0,5146 da execução original existente não preenchem essa linha: ela também usa ajustes específicos, incluindo 70 versus 26 iterações de Level Set. DSC end-to-end inclui todos os exames e atribui zero às falhas sem saída.

## 5. Robustez e falhas de inicialização

> Moderate claims of robustness while initialization remains a critical source of failure.

**Legenda das Tabelas 1 e 2:** a inicialização foi reprovada em 14,71% dos exames reduzidos e 23,14% dos exames originais. O Dice condicionado supera o global em 5,64 e 6,59 pontos percentuais, respectivamente. Essas diferenças descrevem subconjuntos distintos e não representam ganhos causais garantidos ao corrigir a inicialização.
