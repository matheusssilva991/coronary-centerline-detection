# Estrutura dos resultados

```text
output/segmentation/
  runs/       # execuções completas do pipeline
  canonical/  # referências oficiais escolhidas
  analysis/   # tabelas e artefatos derivados
  docs/       # catálogo e decisões sobre os resultados
```

## Convenções

- `mid_res/` e `high_res/` indicam a resolução efetiva.
- `train/`, `val/`, `test/` e `full/` identificam a coorte processada.
- Diretórios `AAAA-MM-DD_HH-MM-SS` representam uma execução.
- Cada run deve preservar a configuração efetiva e os identificadores usados.
- Runs completos pertencem a `runs/`; tabelas derivadas pertencem a
  `analysis/`; interpretação e histórico pertencem a `docs/`.
- HTMLs 3D grandes devem ser gravados fora do repositório com
  `--visual-output-dir`.

Consulte também os índices operacionais de [`runs/`](../../runs/README.md),
[`canonical/`](../../canonical/README.md) e
[`analysis/`](../../analysis/README.md).
