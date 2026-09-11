# Arquivo dos experimentos de aorta

Esta pasta documenta famílias encerradas ou executadas sobre configurações que
não são mais o baseline. Os resultados brutos foram removidos depois da
consolidação; permanecem somente tabelas e conclusões necessárias para evitar a
repetição de variantes descartadas.

## Grupos

| Documento | Configuração de origem | Situação |
|---|---|---|
| [`archive_p10_75_p99_8.md`](archive_p10_75_p99_8.md) | Threshold inferior P10.75 e superior P99.8 | Histórico anterior à promoção do baseline P99.9/-300 |
| [`archive_p99_9_m300.md`](archive_p99_9_m300.md) | Threshold inferior -300 HU e superior P99.9 | Variantes dominadas ou substituídas por testes mais claros |
| [`trajectory_geometry_3_5_6.md`](trajectory_geometry_3_5_6.md) | Limites geométricos 3.5/6 mm | Configuração não promovida |

Os CSVs, snapshots, logs e HTMLs dessas famílias foram removidos. Os HTMLs
sozinhos ocupavam aproximadamente 1,2 GB e pertenciam somente a variantes
descartadas.

## Famílias removidas após consolidação

Os diretórios abaixo foram excluídos em 2026-08-31 porque eram sweeps encerrados
e seus achados já estavam consolidados nos READMEs e nas referências ativas:

- `aorta_opening_sweep`: abertura com raio 0/1 não trouxe ganho útil;
- `aorta_parameter_focused`: grade ampla substituída pelo candidato ativo
  `b0.6/r0.10/i26`;
- `rp90_leak_sweep`: reduções de `R_P90` não corrigiram visualmente os casos de
  vazamento sem introduzir risco de perda de preenchimento;
- `localization_leak_override`: perfis entre `balloon=0.50` e `0.20` só mudaram
  o exame 11; os perfis mais fortes foram rejeitados pela perda de preenchimento
  e não resolveram 464, 790 ou 792.
- `tail_filter_limits`: variar o início da busca entre 25% e 35% não alterou
  as máscaras; permitir cortes de 45%/50% recuperou parcialmente o exame 315,
  mas subsegmentou o 428 e reduziu seu Dice de 0,2743 para zero. Os seis runs e
  o mecanismo que forçava o corte até o limite foram removidos.

As referências preservadas fora do arquivo continuam sendo o baseline fixo,
`b0.6/r0.10/i26`, filtro robusto, envelope e suas combinações validadas.
