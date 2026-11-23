# Baseline histórico P99.9/-300 HU

Este documento cataloga a série OpenCV linear que foi adotada como referência
quantitativa do artigo em resolução média. A série SciPy adotada em novembro
está listada nas [referências atuais](../../canonical/README.md).

Em 24 de agosto de 2026, as chaves desta configuração também foram promovidas
para `config/pipeline_config.json`. As opções experimentais de aorta, threshold
fuzzy e fuzzy connectedness continuam disponíveis no arquivo principal, mas
permanecem desativadas no caminho padrão.

O teste principal desta configuração foi manter o limite inferior fixo em
`-300 HU` e usar o percentil 99,9 do próprio volume como limite superior. Com
isso, o intervalo aceito acompanha a faixa de intensidades altas de cada exame,
sem introduzir o limite inferior adaptativo usado em experimentos posteriores.

## Configuracao principal

- threshold normal entre `-300 HU` e o percentil superior `99.9`;
- perfil de aorta e ostios `standard`;
- level set fixo com 31 iteracoes;
- segmentacao arterial por region growing;
- downsampling OpenCV linear com fatores `[2, 2, 1]`.

As etapas de localização da aorta, seleção dos óstios e segmentação arterial
seguem o comportamento histórico do perfil `standard`. O percentil superior é
a principal diferença estudada nesta série.

## Runs e resultados

| Split | Run | Imagens | Sucesso dos ostios | Dice geral | Dice com ostios validos |
|---|---|---:|---:|---:|---:|
| Train | `2026-08-06_18-43-37` | 30 | 90,00% | 0,6148 | 0,6290 |
| Val | `2026-08-06_22-43-14` | 270 | 83,70% | 0,5879 | 0,6469 |
| Test | `2026-08-06_10-04-22` | 700 | 82,57% | 0,5930 | 0,6558 |

Sucesso considera `both ostia correct` e `both ostia tolerable`. O manifesto
das referências registra estes três runs como fontes históricas. O split de
teste fornece os valores usados na comparação do artigo. Os snapshots de
train e val registram a mesma configuração nos conjuntos de desenvolvimento.
