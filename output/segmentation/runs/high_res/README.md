# Runs em alta resolução

As execuções históricas de high resolution estão agrupadas por objetivo para
evitar timestamps soltos sem contexto.

```text
high_res/
  reference/<split>/<timestamp>/
  aorta_segmentation_experiments/<split>/
    filter_envelope_lower100_pad2/<timestamp>/
    no_filter_no_envelope_lower100_pad2/<timestamp>/
    filter_envelope_lower100_pad2_opening_radius8/<timestamp>/
  legacy_low_ostia_accuracy/
    p99_7/<split>/<timestamp>/
    p99_8/<split>/<timestamp>/
    p99_9/<split>/<timestamp>/
```

A referência de 71,43% de sucesso dos óstios e Dice 0,4684 está em
`runs/high_res/reference`, com links em `canonical/high_res`.

Os experimentos seguem a organização do mid-res por estudo, split, variante e
timestamp. A variante `opening_radius8` identifica o treino com raio diferente
do padrão efetivo 4; não deve ser misturada à comparação principal.
Os snapshots e logs históricos conservam seus bytes e caminhos originais.
Para retomar runs movidos, use o novo caminho em `--resume-dir`.

`legacy_low_ostia_accuracy/` contém a comparação histórica dos percentis
superiores 99.7, 99.8 e 99.9 usando Canny sigma 6 e 70 iterações do level set.
Esses runs apresentaram baixa acurácia dos óstios e não representam a futura
configuração high-res corrigida. Consulte o
[catálogo dessa série](../../docs/baselines/high_resolution_legacy.md) e o
`run_index.csv` antes de reutilizar qualquer resultado.

Experimentos compactos que investigam a escala dos parâmetros permanecem em
`output/segmentation/analysis/pipeline_parameter_validation/runs/`, pois não são
runs oficiais do pipeline principal.
