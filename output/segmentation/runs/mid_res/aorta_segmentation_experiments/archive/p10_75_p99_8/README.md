# Experimentos arquivados P10.75/P99.8

Este grupo reúne os testes de segmentação da aorta executados com o antigo
baseline de desenvolvimento:

- limite inferior adaptativo no percentil 10,75;
- limite superior no percentil 99,8;
- vesselness da aorta com sigmas `[2.0, 2.25, 2.5]`;
- recuperação inicial da trajetória de círculos ativada.

Os resultados foram arquivados porque o baseline fixo `-300 HU`/P99.9 obteve
melhor desempenho nos conjuntos completos. A mudança de pasta não altera os
CSVs ou snapshots de configuração. Os HTMLs das variantes descartadas foram
removidos após a análise visual. As duas referências visuais preservadas para
auditoria também ficam aqui, com nomes que registram configuração e coorte.

## Conteúdo

- `train/`: controladores adaptativos, podas de vazamento, filtros de círculos
  e recuperação dos óstios avaliados nas 30 imagens de treino, além de
  `visual_reference_standard_p10_75_p99_8_train30`;
- `val/`: filtros por cobertura avaliados em 60 ou 270 imagens de validação. A
  referência visual de 60 imagens está em
  `visual_reference_standard_p10_75_p99_8_val60`.

## Decisão

Nenhuma dessas combinações demonstrou melhora visual consistente da máscara da
aorta. O filtro agressivo alterou mais casos, mas também provocou truncamento e
afinamento. Os filtros de 60%/65% com fallback produziram ganhos numéricos
localizados, porém ficaram abaixo do canonical P99.9 na validação completa.

Os próximos runs devem ser criados fora de `archive/`, diretamente em
`aorta_segmentation_experiments/train/` ou `val/`, sempre sobre o baseline
P99.9/-300.
