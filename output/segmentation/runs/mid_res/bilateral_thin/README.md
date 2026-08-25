# Perfil bilateral thin (histórico)

Este grupo preserva os resultados históricos do antigo perfil
`bilateral_thin` em resolução média. O
objetivo foi reduzir dois problemas observados no perfil histórico: máscaras da
aorta excessivamente largas próximas aos óstios e seleção dos dois candidatos
no mesmo lado da aorta.

O perfil não é um método independente de segmentação. Ele reaproveita o mesmo
pipeline e aplica, de uma vez, ajustes específicos na máscara da aorta e na
seleção dos óstios.

## Como o perfil funciona

1. O level set produz e pós-processa a máscara da aorta normalmente.
2. Em cada fatia acompanhada pela Hough, a área segmentada é comparada com a
   área do círculo detectado, aproximada por `pi * raio²`.
3. Quando a área da máscara ultrapassa duas vezes a área esperada, somente essa
   fatia é limitada a uma região circular com raio `1,75` vezes o raio
   interpolado da trajetória.
4. A superfície usada para procurar os óstios é extraída com erosão de raio 2,
   formando uma casca mais fina que a do perfil `standard`.
5. Os candidatos são divididos pelos lados do centro interpolado da aorta. O
   algoritmo avalia pares com um candidato de cada lado e escolhe o par válido
   com maior soma de vesselness.

Essa seleção bilateral reduz a chance de escolher dois máximos próximos do
mesmo ramo, mas depende de uma trajetória de círculos bem localizada.

## Diferenças para o perfil standard

| Etapa | `standard` | `bilateral_thin` |
|---|---|---|
| Correção pela área dos círculos | Desativada | Ativada acima da razão 2,0 |
| Raio da erosão da superfície | 4 | 2 |
| Seleção do par | Greedy, um óstio após o outro | Bilateral, um candidato de cada lado |
| Segmentação arterial | Region growing | Region growing |

## Configuracao principal

- threshold normal com limite inferior adaptativo P10.75 e superior P99.9;
- correcao da aorta guiada pela trajetoria, com razao de area limite `2.0`;
- erosao dos ostios com raio `2`;
- selecao bilateral dos candidatos;
- segmentacao arterial por region growing.

## Run disponivel

| Split | Run | Imagens | Sucesso dos ostios | Dice geral | Dice com ostios validos |
|---|---|---:|---:|---:|---:|
| Train | `2026-08-11_13-49-22` | 30 | 93,33% | 0,6018 | 0,6281 |

O perfil, seu alias e sua flag de CLI foram removidos do runtime. Apesar de
ganhos históricos em algumas coortes, ele não pertence ao baseline atual e
adicionava uma segunda família de parâmetros sem benefício confirmado nos
runs recentes. O único run completo disponível usa 30 imagens de treino. A
configuração exata continua preservada em
`config/effective_pipeline_config.json` dentro do run.
