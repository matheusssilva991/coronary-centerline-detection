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
| Vesselness da aorta | baseline final `[2.5, 3.0]`, gamma 30 | pertence ao melhor run completo P99.9; `[2.0, 2.25, 2.5]` teve ganho isolado em 60 imagens, mas não compõe o baseline promovido |
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

## Otimização arterial encerrada

Os resultados brutos de `artery_vesselness_fc_sweep` foram removidos após a
validação, pois nenhuma variante superou o pipeline de referência. Na seleção
de treino, algumas combinações chegaram a aproximadamente 0,607 de Dice médio,
mas o ganho não se confirmou em validação.

No bloco final com 60 imagens de validação:

| Variante | Dice médio | Decisão |
|---|---:|---|
| Baseline RG, morfologia atual | 0,6137 | mantido como referência |
| RG gamma 65 + dilatação condicionada | 0,6005 | descartado |
| FC piso 0,020, sigma HU 90 + dilatação condicionada | 0,5952 | descartado |

Também foram descartados ajustes por ramo, recuperação de ramos pequenos,
dilatação condicionada e grades alternativas de vesselness/FC. As opções RG e
FC continuam no pipeline; somente as grades experimentais sem ganho foram
retiradas.

## Diagnóstico focado de falhas

Os resultados antigos de `pipeline_failure_improvement` foram removidos, mas o
runner e a coorte de validação foram restaurados para continuar as variantes
`corrections`. A primeira execução (`baseline_val_2026-07-20_14-41-38`) era
inválida: os 46 exames de cada variante falharam porque a seleção bilateral não
recebia os círculos detectados. O erro foi corrigido no comparador principal.

A repetição válida em 46 casos difíceis confirmou apenas o comportamento já
medido pelas quatro abordagens, sem uma correção nova para promover:

| Variante | Sucesso dos óstios | Dice médio |
|---|---:|---:|
| Threshold normal + RG | 80,4% | 0,5272 |
| Threshold fuzzy + RG | 82,6% | 0,5171 |
| Threshold normal + FC | 80,4% | 0,4836 |
| Threshold fuzzy + FC | 82,6% | 0,4676 |

Como a coorte é intencionalmente concentrada em falhas, esses números não devem
ser usados como estimativa global. Ela serve apenas para selecionar correções
antes de uma validação mais ampla.

## Código experimental removido

Foram removidos os scripts e runners encerrados de recuperação/rastreamento da
aorta, seleção de óstios e otimização arterial. Além dos três comparadores
principais e do runner de threshold, permanece o diagnóstico focado de falhas
com seu runner de correções.

## Sensibilidade dos parâmetros do artigo

A primeira execução de sensibilidade,
`sensitivity_canonical_val_30`, foi removida porque sua configuração de
referência não reproduzia o baseline histórico: obteve sucesso dos óstios de
70,0% e Dice médio de 0,5237 nas primeiras 30 imagens de validação.

Ela foi substituída por `sensitivity_normal_rg_val_30`, cuja referência foi
comparada imagem a imagem com
`fuzzy_comparison/val/normal_rg/2026-06-23_14-47-01`. As duas execuções obtiveram
22/30 sucessos dos óstios (73,3%), e a diferença entre os Dice médios foi de
apenas 0,00012. Esse run corrigido foi mantido como triagem da análise OFAT.

Na triagem, o percentil superior foi o parâmetro mais sensível: P99,5 reduziu
o Dice médio para 0,4947, enquanto P99,9 o elevou para 0,5579. Os limites em z
de 30 e 50 mm reproduziram a referência nesse bloco; os divisores 5 e 9 do RG
tiveram variação inferior a 0,0002 no Dice médio; e os pisos de vesselness de
5% e 9% reduziram o Dice para 0,5318 e 0,5256, respectivamente.

O CSV de AUC da comparação por imagem também foi removido. AUC não representa
uma métrica adequada para a curva de Dice ordenada por exame e deixou de fazer
parte das figuras e conclusões do notebook.

## Limpeza dos artefatos de análise

O comparativo qualitativo acumulava HTMLs de seleções antigas. Foram mantidos
somente os três casos registrados em `selected_qualitative_cases.csv`, cada um
com o pipeline original e a melhor variante: óstios iguais (imagem 997),
óstios diferentes (imagem 327) e Dice próximo das médias (imagem 178).

As cópias completas dos runs internos da confirmação bilateral da aorta foram
removidas. Permaneceram os CSVs consolidados, as configurações das duas
variantes e os arquivos de identificação da amostra, suficientes para consultar
o resultado e reproduzir o experimento. Exemplos visuais gerais, casos ruins,
figuras metodológicas e resultados compactos das análises ativas foram
preservados.

Posteriormente, o diretório consolidado `aorta_mask_ostia_comparison` também foi
removido; suas métricas finais permanecem documentadas neste histórico. Os PNGs
e CSVs de `fuzzy_membership_functions` foram excluídos porque a figura e a
tabela já ficam incorporadas ao notebook correspondente.

O último `pipeline_failure_improvement` também foi removido: ele era apenas a
saída do runner de correções e suas métricas já estavam registradas neste
histórico. O catálogo `pipeline_failure_analysis` foi preservado porque seu
`focused_cohort.csv` é uma entrada direta do runner.
