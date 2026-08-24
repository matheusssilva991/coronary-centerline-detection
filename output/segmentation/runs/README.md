# Execucoes do pipeline

`runs/` guarda resultados completos produzidos pelo pipeline de segmentacao.
As execucoes sao separadas por resolucao e, quando fazem parte de um estudo,
por um grupo com nome descritivo.

```text
runs/
  mid_res/
    <grupo>/<split>/<timestamp>/
  high_res/
    <grupo>/<split>/<timestamp>/
```

- [`mid_res/README.md`](mid_res/README.md): baselines, comparacoes fuzzy e
  experimentos em resolucao media.
- [`high_res/README.md`](high_res/README.md): resultados e historico em alta
  resolucao.

## Estrutura de uma execucao

```text
<timestamp>/
  config/
    effective_pipeline_config.json
    split_ids.json
  logs/
    pipeline.log
  numeric/
    ostios_<split>_summary.csv
    ostios_<split>_metadata.json
    ostios_<split>_lote_<n>_summary.csv
    ostios_<split>_batch_timings.csv
  visual/                       # existe somente quando solicitado
```

| Pasta | Finalidade |
|---|---|
| `config/` | Configuracao efetiva e IDs realmente processados |
| `logs/` | Progresso, avisos, falhas e retomadas |
| `numeric/` | Resultados consolidados, lotes e tempos de execucao |
| `visual/` | HTMLs ou imagens vinculados especificamente ao run |

O arquivo `effective_pipeline_config.json` e a fonte principal para conferir
os parametros. O nome da pasta descreve o objetivo do estudo, mas nao substitui
o snapshot da configuracao.

## Organizacao

- Um run de referencia deve continuar em `runs/` e ser apontado por
  `canonical/`, evitando duplicacao.
- Variantes do mesmo estudo devem compartilhar um grupo e possuir um README.
- Runs incompletos devem ser identificados como tal no README do grupo.
- Resultados descartados podem manter CSVs e configuracoes, mas visuais grandes
  devem ser removidos quando nao forem mais necessarios.
