# Métricas de círculos por fatia

Contém a tabela usada por
[`aorta_circle_slice_analysis.ipynb`](../../../../src/eda/aorta_circle_slice_analysis.ipynb)
para relacionar o tamanho axial do exame, a trajetória de círculos da aorta e o
resultado dos óstios.

- `aorta_circle_slice_metrics.csv`: uma linha por exame, com número de fatias,
  círculos detectados/interpolados, cobertura axial e métricas dos raios.

O CSV é persistido porque pode ser reutilizado sem executar novamente a
localização da aorta.
