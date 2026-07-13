# Histórico de experimentos encerrados

Este arquivo registra as decisões tomadas antes da limpeza dos sweeps. Os
diretórios brutos foram removidos porque continham muitas configurações
descartadas e execuções completas duplicadas. Runs oficiais continuam em
`output/segmentation/runs/`.

## Configurações mantidas

| Etapa | Configuração mantida | Evidência principal |
|---|---|---|
| Threshold normal | piso percentil 10,75 e teto percentil 99,8 | Dice médio 0,6407 e sucesso dos óstios 93,3% em 30 imagens de treino |
| Threshold fuzzy | piso percentil 10,5; centros P99,8/P99,96; margem 100 HU; sem suavização; `object_argmax` | Dice médio 0,6429 em 30 imagens de treino |
| Vesselness da aorta | sigmas `[2.0, 2.25, 2.5]`, gamma 30 | sucesso 88,3% e Dice 0,6183 em 60 imagens de validação |
| Rastreamento da aorta | candidato geométrico válido mais próximo e recuperação precoce em 8 fatias | sucesso 90,0% e Dice 0,6301 em 60 imagens de validação |
| Region growing | implementação refinada atual, com busca local e múltiplas sementes | melhor método de segmentação arterial nos comparativos amplos |
| Fuzzy connectedness | alpha 0,16; sigma HU 100; vizinhança 26; piso vesselness 0,018; peso vesselness 0,9 | melhor configuração FC mantida para comparação científica |
| Aorta/óstios bilateral | erosão 2; seleção bilateral; 50 candidatos por lado; correção condicional com razão de área 2,0 e raio 1,75 | 77/90 sucessos contra 73/90 do baseline no bloco final de validação |

## Abordagens mantidas para comparação

1. Threshold normal + region growing.
2. Threshold fuzzy + region growing.
3. Threshold normal + fuzzy connectedness.
4. Threshold fuzzy + fuzzy connectedness.

## Abordagens descartadas

| Abordagem | Motivo da retirada |
|---|---|
| Frangi modificado | chegou a produzir Dice zero e não superou o Frangi normal |
| `dense_suppression` e `normal_dense_suppression` | não superaram a máscara fuzzy `object_argmax` |
| RG simples do artigo | não superou o region growing refinado usado no pipeline |
| LCC por volume | não melhorou a combinação padrão por fatia |
| Candidato circular por score | piorou a localização em relação ao candidato geométrico mais próximo |
| Círculo fora da tolerância como miss | aumentou muito o tempo e piorou óstios/Dice |
| Seleção inicial por score | não melhorou a primeira detecção circular |
| Refinamento ponderado de círculos | não superou a média local |
| Referência média/mediana do rastreamento | não superou o último círculo válido |
| Grades alternativas de Canny/Hough | não trouxeram ganho sobre sigma 3 e 15 picos |
| Grades alternativas do level set | não superaram 31 iterações, balloon 0,8 e leak radius 2 |
| Restrição de óstios pela trajetória | sucesso 85,0% e Dice 0,6075, abaixo do baseline |
| Seleção global do par de óstios | combinação completa teve sucesso 83,3% e Dice 0,6080 |
| FC competitivo/percentil e afinidades alternativas | não superaram alpha fixo com produto ponderado |

## Experimentos de máscara da aorta e seleção dos óstios

Os diretórios brutos de triagem e seleção foram removidos após a confirmação
final. Foi mantido somente o run final de 90 imagens, além deste resumo.

| Etapa | Resultado | Decisão |
|---|---|---|
| Triagem inicial, 24 imagens | `thin_surface` foi o ganho isolado mais seguro; estratégias locais, externas, distância física e região z curta foram instáveis | hipóteses usadas apenas para orientar a ablação |
| Ablação de trajetória, 24 imagens | `trajectory_f175_thin_surface` chegou a 17/24 contra 15/24 do baseline | exigiu confirmação independente |
| Confirmação de trajetória, 60 imagens | baseline 51/60; combinação de trajetória 50/60 | trajetória global descartada |
| Estratégias avançadas, 30 imagens | NMS de 3/4 mm e combinações joint caíram para 1--2 sucessos; correção condicional preservou 27/30 | NMS e seleção joint descartados |
| Seleção bilateral, 30 imagens | baseline 22/30; combinação bilateral fina e condicional 24/30 | candidata selecionada |
| Confirmação, 60 imagens | sucesso 49/60 para 52/60; Dice médio 0,5578 para 0,5949 | candidata mantida |
| Confirmação final, 90 imagens | sucesso 73/90 para 77/90; Dice médio 0,5802 para 0,5919; quatro recuperados e nenhuma perda | perfil promovido como opção do pipeline |

Nos dois blocos independentes de 60 e 90 imagens, a abordagem passou de
122/150 (81,3%) para 129/150 (86,0%) sucessos. Houve oito recuperações e uma
regressão, com Dice médio de 0,5712 para 0,5931. O método foi incorporado como
`bilateral_thin`, mas `standard` continua sendo o padrão para preservar
reprodutibilidade com os runs históricos.

## Observação

Os valores acima servem como histórico de seleção, não como resultado final do
artigo. Comparações finais devem usar os mesmos IDs, resolução e ambiente para
as quatro abordagens mantidas.
