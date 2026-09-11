# Catálogo dos resultados

Esta pasta centraliza a documentação interpretativa dos resultados. Os dados
continuam em `runs/`, os resultados oficiais em `canonical/` e os artefatos
derivados em `analysis/`. Assim, um relatório histórico não precisa ficar
misturado aos CSVs, logs ou visualizações que descreve.

## Categorias

| Categoria | Conteúdo |
|---|---|
| [`structure/`](structure/README.md) | Organização de `runs`, `canonical` e `analysis`, convenções e política de armazenamento |
| [`baselines/`](baselines/README.md) | Baseline atual, referências anteriores e resultados legacy de alta resolução |
| [`methods/`](methods/README.md) | Comparações entre threshold normal/fuzzy e segmentação por RG/FC |
| [`aorta/`](aorta/README.md) | Experimentos ativos e arquivados de localização e segmentação da aorta |
| [`experiments/`](experiments/README.md) | Decisões consolidadas de sweeps encerrados e abordagens descartadas |

## Regra de manutenção

- Documentação de uma família de resultados pertence a esta pasta.
- READMEs locais são mantidos apenas quando explicam como consumir artefatos da
  própria pasta, como catálogos de casos ruins ou métricas por fatia.
- Resultados brutos nunca devem ser movidos para `docs/`.
- Ao encerrar um sweep, registre configuração, coorte, métricas e decisão em
  `experiments/`, depois remova artefatos intermediários sem valor de auditoria.

