# Experimentos Descartados do Region Growing Arterial

Esta pasta preserva apenas o histórico das estratégias removidas. Os CSVs e o
código experimental foram excluídos após a avaliação pareada em 30 imagens de
treino e 60 de validação.

| Estratégia | Dice treino | Dice validação | Conclusão |
|---|---:|---:|---|
| Baseline | 0,6288 | 0,6067 | Referência |
| Segunda passagem (`second_current`) | 0,6325 | 0,6114 | Ganho combinado de 0,44%, abaixo do mínimo operacional |
| Retry por poucos voxels | 0,6284 | 0,6179 | Média elevada por poucos casos extremos; 4 melhoras e 11 perdas |
| Retry por extensão | 0,6288 | 0,6113 | Efeito quase nulo na maioria dos exames |
| Relaxamento esquerdo | 0,6280 | 0,6132 | Piorou 57 de 90 exames |
| Relaxamento direito | 0,6194 | 0,6008 | Piora nos dois conjuntos |
| Restrição HU no óstio, 100 HU | 0,4077 | 0,3987 | Restrição excessiva |
| Restrição HU no óstio, 150 HU | 0,5102 | 0,4720 | Restrição excessiva |
| Referência HU da aorta, 150 HU | 0,4755 | 0,4597 | Restrição excessiva |

O maior ganho médio, do retry por poucos voxels, foi dominado pelos exames 835
e 341 da validação. A mediana do delta foi zero e o teste de Wilcoxon pareado
resultou em `p=0,7333`. A segunda passagem foi mais estável (`p=0,0146`), mas
seu ganho médio combinado foi somente `0,0044` em Dice. Como nenhuma abordagem
atingiu ganho mínimo entre 1% e 5%, nenhuma foi incorporada ao pipeline.

O pipeline mantém somente o RG original, com pós-processamento morfológico
único. Novos estudos devem partir de uma hipótese diferente, sem reativar essas
estratégias automaticamente.
