# Casos ruins da aorta

Esta pasta reúne atalhos para visualizações 3D de exames classificados
manualmente como problemáticos. Os arquivos HTML são links simbólicos para os
runs originais; portanto, não duplicam as visualizações nem ocupam espaço
relevante em disco.

## Catálogos

- `train_p99_9_m300/`: comparação das mesmas 30 imagens de treino usando o
  pipeline normal e o filtro agressivo, ambos com faixa de threshold de
  `-300 HU` a `P99.9`.
- `val_p99_9_m300/`: comparação equivalente nas 60 imagens revisadas do
  conjunto de validação, com atalhos por exame e por tipo de falha.

As classificações e métricas derivadas são analisadas em
[`aorta_volume_quality_analysis.ipynb`](../../../../src/eda/aorta_volume_quality_analysis.ipynb).
