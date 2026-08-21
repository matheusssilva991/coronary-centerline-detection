# Runs em alta resolução

As execuções históricas de high resolution estão agrupadas por objetivo para
evitar timestamps soltos sem contexto.

```text
high_res/
  legacy_low_ostia_accuracy/
    p99_7/<split>/<timestamp>/
    p99_8/<split>/<timestamp>/
    p99_9/<split>/<timestamp>/
```

`legacy_low_ostia_accuracy/` contém a comparação histórica dos percentis
superiores 99.7, 99.8 e 99.9 usando Canny sigma 6 e 70 iterações do level set.
Esses runs apresentaram baixa acurácia dos óstios e não representam a futura
configuração high-res corrigida. Consulte o README e o `run_index.csv` antes de
reutilizar qualquer resultado.

Experimentos compactos que investigam a escala dos parâmetros permanecem em
`output/segmentation/analysis/pipeline_parameter_validation/runs/`, pois não são
runs oficiais do pipeline principal.
