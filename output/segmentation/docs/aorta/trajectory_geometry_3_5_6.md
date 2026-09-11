# Geometria 3.5/6: nao promovida

Limites de salto: raio 3.5 mm, centro 6 mm. Mantem lower_fraction=0.85 e
padding=0; nao foi combinada com pad2/pad3.

Treino: Dice 0.590650, 26/30 sucessos. Inspecao visual boa com pequenas
subsegmentacoes proximas aos ostios em 315 e 428.
Validacao: Dice 0.572742, 52/60. Revisao visual: 227, 772 e 802 bons;
790 perdeu o inicio da aorta e sobresegmentou; 11 com ostios incorretos.

Numericos organizados em train/ e val/. Os HTMLs continuam em seus locais
originais no disco externo, sob `aorta_segmentation_experiments/{split}/`
`trajectory_geometry_r3_5_c6_0_p99_9_m300/{timestamp}/visual/`.
