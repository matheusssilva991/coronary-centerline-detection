# Métricas de círculos por fatia

Contém tabelas históricas que podem ser consultadas por
[`aorta_circle_slice_analysis.ipynb`](../../../../src/eda/aorta_circle_slice_analysis.ipynb)
para relacionar o tamanho axial do exame e a trajetória de círculos da aorta.

O notebook atual lê diretamente o summary selecionado e apresenta somente
fatias da imagem, cobertura dos círculos, raio, posição axial e a relação com a
qualidade visual da aorta. Ele não gera novos arquivos nesta pasta.

- `aorta_circle_slice_metrics.csv`: uma linha por exame, com número de fatias,
  círculos detectados/interpolados, cobertura axial e métricas dos raios.

O CSV existente é mantido como referência de execuções anteriores e pode ser
reutilizado sem executar novamente a localização da aorta.
