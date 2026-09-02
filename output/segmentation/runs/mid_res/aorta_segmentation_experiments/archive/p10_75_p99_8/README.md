# Experimentos arquivados P10.75/P99.8

Este grupo reúne os testes de segmentação da aorta executados com o antigo
baseline de desenvolvimento:

- limite inferior adaptativo no percentil 10,75;
- limite superior no percentil 99,8;
- vesselness da aorta com sigmas `[2.0, 2.25, 2.5]`;
- recuperação inicial da trajetória de círculos ativada.

Os resultados foram descartados porque o baseline fixo `-300 HU`/P99.9 obteve
melhor desempenho nos conjuntos completos. Depois da consolidação, CSVs,
snapshots, logs e referências visuais foram removidos; este README preserva a
decisão experimental.

## Decisão

Nenhuma dessas combinações demonstrou melhora visual consistente da máscara da
aorta. O filtro agressivo alterou mais casos, mas também provocou truncamento e
afinamento. Os filtros de 60%/65% com fallback produziram ganhos numéricos
localizados, porém ficaram abaixo do canonical P99.9 na validação completa.

Novos runs devem ser criados fora de `archive/`, diretamente em
`aorta_segmentation_experiments/train/` ou `val/`, sempre sobre o baseline
P99.9/-300.
