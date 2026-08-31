# Comparacao visual da aorta

Esta pasta compara as duas referencias relevantes para a segmentacao da aorta
em `mid resolution`:

- **Baseline:** level set fixo, threshold entre `-300 HU` e `P99.9`.
- **Correcao de referencia `b0.6/r0.10/i26`:** filtro robusto da trajetoria,
  cinco circulos sinteticos, envelope espacial de `2.25r` com margem axial de
  10 fatias e level set fixo com `balloon=0.6`, semente de 10% do raio e 26
  iteracoes.

Os HTMLs sao links simbolicos para o disco externo e nao duplicam os arquivos.
Para abri-los, o disco deve estar montado em
`/media/matheus/HD/ImageCAS_pipeline_results`.

## Catalogos

- [`train_p99_9_m300/`](train_p99_9_m300/): sete falhas do baseline e dois
  casos sensiveis na regiao dos ostios entre as 30 imagens de treino.
- [`val_p99_9_m300/`](val_p99_9_m300/): oito falhas do baseline e um caso
  sensivel na regiao dos ostios entre as 60 imagens revisadas da validacao.

Na revisao visual, a correcao de referencia produziu `30/30` aortas globalmente
adequadas no treino e `56/60` na validacao. Os casos ainda ruins na validacao
sao `11`, `464`, `790` e `792`.

| Split | Variante | Aortas boas (visual) | Sucesso dos ostios | Dice arterial medio |
|---|---|---:|---:|---:|
| Treino | Baseline | 23/30 | 27/30 | 0.6148 |
| Treino | Correcao | 30/30 | 26/30 | 0.5826 |
| Validacao | Baseline | 52/60 | 48/60 | 0.5650 |
| Validacao | Correcao | 56/60 | 51/60 | 0.5851 |

Assim, `b0.6/r0.10/i26` e a referencia para qualidade visual da correcao da
aorta, mas ainda nao substitui automaticamente o baseline de todo o pipeline:
o ganho na validacao veio acompanhado de perda de Dice no treino.

As classificacoes quantitativas e visuais continuam sendo analisadas em
[`aorta_volume_quality_analysis.ipynb`](../../../../src/eda/aorta_volume_quality_analysis.ipynb).
