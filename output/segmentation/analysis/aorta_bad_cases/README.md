# Comparacao visual da aorta e do novo padrao

O padrao operacional agora e **filtro + envelope + lower100/pad2**:
Hough 18-29, geometria 4.8/8 mm, cinco circulos sinteticos, envelope 2.25r
com margem axial 10 e level set b0.6/r0.10/i26. Threshold -300 HU/P99.9.
O pad2 altera a selecao dos ostios, nao a mascara da aorta.

## Catalogos visuais

- [Treino 30](train_p99_9_m300/): baseline historico, correcao revisada e novo pad2.
- [Validacao 60](val_p99_9_m300/): os mesmos tres visuais por exame selecionado.

Os arquivos `new_default_pad2.html` sao links para os runs de 06/09/2026,
no disco `/media/matheus/HD/Results_dataset_ccta/imagecas`. Nao ha copia dos HTMLs.
Os links antigos preservam a revisao original, inclusive diferencas de versao.
Nao atribuir automaticamente seus rotulos aos novos runs: contagens de voxels
iguais nao garantem igualdade espacial das mascaras.

## Comparacao quantitativa completa

Both correct e both tolerable contam como sucesso. Medias incluem falhas.

| Coorte | P99.9 puro: Dice / ostios | Filtro + envelope pad0 | Novo pad2 |
|---|---:|---:|---:|
| Treino 30 | 0.6148 / 27 | 0.5907 / 26 | 0.6288 / 29 |
| Validacao 270 | 0.5879 / 226 | 0.5804 / 221 | 0.5992 / 229 |
| Teste 700 | 0.5930 / 578 | 0.5874 / 572 | 0.6018 / 599 |

O novo padrao melhora os agregados, mas o Wilcoxon de Dice contra o puro
nao foi significativo: p bruto 0.353 na validacao e 0.172 no teste.
No teste, recuperou sucesso dos ostios em 57 exames e perdeu em 36.
Nao foram produzidos HTMLs dos 270/700; nao confundir desempenho numerico
com confirmacao visual das aortas dessas coortes.

A revisao historica b0.6/r0.10/i26 encontrou 30/30 e 56/60 aortas globalmente
adequadas; as quatro ruins da validacao eram 11, 464, 790 e 792.
Esses numeros pertencem aos runs revisados, nao ao novo teste de 700 imagens.

Analise, caminhos exatos e Wilcoxon com Holm no
[notebook de comparacao](../../../../src/eda/aorta_ostia_pipeline_comparison.ipynb).
