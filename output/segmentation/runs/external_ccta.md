# CCTA externas: MM-WHS e OrCaScore

Os diretórios `mid_res/external_ccta/` e `high_res/external_ccta/` guardam
snapshots compactos dos runs externos. Cada run mantém `config/`, `numeric/` e
`metadata.json`; no MM-WHS, `evaluation/aorta/test/` contém somente os arquivos
pequenos `*_dice.xls` produzidos pelo avaliador oficial. Máscaras NIfTI, HTMLs,
logs e cópias de segurança permanecem no disco externo configurado por
`CCTA_RESULTS_ROOT` ou `--output-root`.

| Banco | Resolução | Run | Exames persistidos | Situação |
|---|---|---|---:|---|
| MM-WHS | mid | `2026-09-22_18-51-28` | 60/60 | 54 sucessos; 6 erros de aorta vazia no teste; Dice oficial em 34/40 testes |
| MM-WHS | high | `2026-09-22_17-47-45` | 60/60 | 60 sucessos; Dice oficial em 40/40 testes |
| OrCaScore | mid | `2026-09-19_10-23-50` | 72/72 | 63 sucessos; 9 erros |
| OrCaScore | high | `2026-09-22_19-12-41` | **71/72** | Parcial: falta `test/TEV2P6`; 67 sucessos e 4 erros nos exames persistidos |

Os quatro runs foram iniciados com `--subset all`: `results_train.csv` e
`results_test.csv` separam os exames, enquanto `results_all.csv` os reúne. O
estado `complete_with_errors` indica que todos os exames foram tentados, mas
alguns falharam; `incomplete` significa que ainda falta ao menos um resultado.
Esses snapshots não substituem o run original e não devem ser usados como
diretório de retomada. Após concluir o OrCaScore high no disco externo, seu
snapshot precisa ser atualizado antes de comparações que exigem 72/72 exames.

No MM-WHS, o Dice do `train` é calculado diretamente contra o label aberto na
resolução de trabalho. O Dice do `test` só é calculado quando a execução usa
`--test-aorta-dice`, com os labels criptografados e o avaliador
oficial via Wine em 1 mm. Não combine as médias dos dois protocolos.
