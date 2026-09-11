# Resultados de segmentacao

Esta pasta concentra as execucoes e as analises produzidas pelo pipeline. Este
README funciona apenas como mapa da estrutura; a interpretação dos resultados
e o histórico dos experimentos ficam no catálogo em `docs/`.

```text
output/segmentation/
  runs/                # execucoes completas do pipeline
  canonical/           # referencias oficiais escolhidas
  analysis/            # tabelas e artefatos derivados de analises
  docs/                # catalogo de resultados e decisoes experimentais
```

## Pastas

| Pasta | Conteudo | Documentacao |
|---|---|---|
| `runs/` | Resultados completos, separados por resolucao, estudo, split e data | [`runs/README.md`](runs/README.md) |
| `canonical/` | Links para os resultados adotados como referencia atual | [`canonical/README.md`](canonical/README.md) |
| `analysis/` | Dados compactos usados por notebooks e experimentos | [`analysis/README.md`](analysis/README.md) |
| `docs/` | Catálogo central de baselines, métodos e experimentos | [`docs/README.md`](docs/README.md) |

## Convencoes

- `mid_res/` e `high_res/` identificam a resolucao efetiva do pipeline.
- `train/`, `val/`, `test/` e `full/` identificam a coorte processada. Cada
  execução do pipeline processa somente uma delas; `full` usa todos os exames
  como uma coorte única, sem criar a divisão train/val/test.
- Pastas no formato `AAAA-MM-DD_HH-MM-SS` identificam uma execucao.
- Grupos nomeados, como `fuzzy_comparison/`, reúnem runs do mesmo experimento.
- Resultados brutos pertencem a `runs/`; artefatos derivados pertencem a
  `analysis/`.
- Em cada run, `results_<split>.csv` contém uma linha por imagem. Agregados são
  calculados sob demanda nas EDAs; o metadata guarda apenas identidade e
  rótulos principais, e a configuração completa fica no snapshot do run.
- Os HTMLs 3D podem ser mantidos fora do repositório com
  `--visual-output-dir /caminho/externo`; CSVs, configurações e logs continuam
  dentro do run. Os arquivos são salvos diretamente em `visual/`, sem repetir
  o nome do split dentro dessa pasta, e a hierarquia do run é espelhada na raiz
  externa.

Consulte primeiro o [catálogo](docs/README.md) antes de reutilizar um run.
