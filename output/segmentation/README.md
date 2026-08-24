# Resultados de segmentacao

Esta pasta concentra as execucoes e as analises produzidas pelo pipeline. Este
README funciona apenas como mapa da estrutura; os detalhes de configuracao e os
testes realizados ficam nos READMEs de cada subpasta.

```text
output/segmentation/
  runs/                # execucoes completas do pipeline
  canonical/           # referencias oficiais escolhidas
  analysis/            # tabelas e artefatos derivados de analises
  backend_comparison/  # diagnosticos comparando CPU e GPU
```

## Pastas

| Pasta | Conteudo | Documentacao |
|---|---|---|
| `runs/` | Resultados completos, separados por resolucao, estudo, split e data | [`runs/README.md`](runs/README.md) |
| `canonical/` | Links para os resultados adotados como referencia atual | [`canonical/README.md`](canonical/README.md) |
| `analysis/` | Dados compactos usados por notebooks e experimentos | [`analysis/README.md`](analysis/README.md) |
| `backend_comparison/` | Comparacoes numericas e de tempo entre CPU e GPU | [`backend_comparison/README.md`](backend_comparison/README.md) |

## Convencoes

- `mid_res/` e `high_res/` identificam a resolucao efetiva do pipeline.
- `train/`, `val/` e `test/` identificam o conjunto processado.
- Pastas no formato `AAAA-MM-DD_HH-MM-SS` identificam uma execucao.
- Grupos nomeados, como `fuzzy_comparison/`, reúnem runs do mesmo experimento.
- Resultados brutos pertencem a `runs/`; artefatos derivados pertencem a
  `analysis/`.

Consulte primeiro o README da pasta de interesse antes de reutilizar um run.
