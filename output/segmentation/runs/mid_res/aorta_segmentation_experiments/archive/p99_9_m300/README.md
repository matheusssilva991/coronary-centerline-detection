# Variantes arquivadas sobre P99.9/-300 HU

Estas variantes foram executadas sobre o baseline atual, mas não permaneceram
como candidatas ativas. Os dados brutos foram removidos; os nomes, métricas e
motivos abaixo preservam a rastreabilidade das decisões.

## Treino

| Variante histórica | Dice | Óstios válidos | Motivo do arquivamento |
|---|---:|---:|---|
| `adaptive_current_p99_9_m300` | 0,6030 | 26/30 | Inferior ao level set fixo |
| `adaptive_simplified_rp90_2_6_p99_9` | 0,6030 | 26/30 | Limiar R_P90 inferior piorou o resultado |
| `adaptive_simplified_rp90_2_7_p99_9` | 0,6148 | 27/30 | Reproduziu o baseline, sem ganho |
| `adaptive_simplified_rp90_2_8_p99_9` | 0,6148 | 27/30 | Reproduziu o baseline, sem ganho |
| `circle_filter_coverage_080` | 0,6030 | 26/30 | Inferior ao baseline e sem melhora visual consistente |
| `circle_filter_coverage_065_fallback` | 0,6030 | 26/30 | Inferior ao baseline e sem melhora visual consistente |
| `trajectory_envelope_k1_75_p99_9` | 0,6297 | 28/30 | Envelope sem margem axial causou cortes em extremidades |
| `trajectory_envelope_k2_0_p99_9` | 0,6297 | 28/30 | Superado pelo envelope corrigido com margem axial |
| `trajectory_envelope_k2_25_p99_9` | 0,6297 | 28/30 | Superado pelo envelope corrigido com margem axial |
| `circle_filter_aggressive_simplified_rp90_2_7_p99_9` | 0,5848 | 25/30 | Igual ao filtro agressivo histórico, sem ganho adicional |
| `circle_filter_aggressive_envelope_k2_25_zmargin5_p99_9` | 0,5982 | 26/30 | Combinação não preservou o melhor de cada método |
| `circle_filter_aggressive_adaptive_rp90_2_7_envelope_k2_25_zmargin5_p99_9` | 0,5982 | 26/30 | Controlador adicional não mudou o resultado da combinação |
| `robust_tail_filter_cov040_maxtrim040_adaptive_levelset_p99_9_m300` | 0,6057 | 26/30 | Igual ao filtro protegido fixo e inferior aos candidatos ativos |
| `robust_tail_filter_cov040_maxtrim040_fixed_levelset_p99_9_m300` | 0,6057 | 26/30 | Etapa intermediária substituída pela continuação sintética com envelope |
| `robust_tail_filter_cov040_maxtrim040_envelope_k2_25_margin5_fixed_levelset_p99_9_m300` | 0,6191 | 27/30 | Perdeu os óstios de 315 e 676; substituído pela versão com círculos sintéticos |
| `robust_tail_filter_cov040_maxtrim040_synthetic_tail5_fixed_levelset_p99_9_m300` | 0,6148 | 27/30 | Reproduziu o baseline arterial, mas não avaliava o envelope conjuntamente |
| `robust_filter_synthetic10_envelope_k2_25_margin10_p99_9_m300` | 0,6144 | 27/30 | Mesmo resultado arterial de cinco círculos, porém com vazamentos adicionais em 44 e 330 |
| `robust_filter_cov033_synthetic5_envelope_k2_25_margin10_p99_9_m300` | 0,6144 | 27/30 | Cobertura global de 33% reproduziu exatamente a variante ativa anterior |
| `robust_filter_cov040_fallback030_rp90_2_5_synthetic5_envelope_k2_25_margin10_p99_9_m300` | 0,6144 | 27/30 | O fallback foi tentado apenas no 603, mas nenhuma cauda geométrica incompatível foi encontrada |
| `incomplete_robust_filter_synthetic5_envelope_k2_25_margin10_p99_9_m300` | — | — | Inicialização incompleta anterior ao run final de cinco círculos |

## Validação

| Variante histórica | Dice | Óstios válidos | Motivo do arquivamento |
|---|---:|---:|---|
| `adaptive_current_p99_9_m300` | 0,5609 | 47/60 | Inferior ao baseline e ao filtro robusto |
| `circle_filter_coverage_080` | 0,5674 | 47/60 | Cobertura conservadora não trouxe ganho consistente |
| `circle_filter_coverage_065_fallback` | 0,5727 | 48/60 | Inferior ao filtro robusto de cobertura 40% |

## Conclusão

Os testes intermediários mostraram que o filtro puro reduz vazamentos, mas pode
encerrar cedo demais a trajetória usada na localização dos óstios. A combinação
ativa com cinco círculos sintéticos, envelope `2.25r` e margem axial de dez
fatias recuperou `315` e `676` e preservou a melhora visual dos demais casos.
O exame `603` continua como falha residual. Reduzir a cobertura mínima de 40%
para 33% ou repetir o filtro com 30% não alterou seus círculos: o limitante não
é mais a cobertura, mas a ausência de uma cauda classificada como
geometricamente incompatível. Por isso, o próximo experimento usa o perfil
`R_z` da máscara para propor o ponto de corte.
