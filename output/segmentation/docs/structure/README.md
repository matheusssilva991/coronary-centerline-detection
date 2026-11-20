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

## Inventário e retenção

O [inventário](../../results_catalog.json) classifica as referências canônicas,
fontes ativas dos notebooks, marcos históricos documentados e candidatos em
avaliação. Cada entrada registra consumidores, arquivos, tamanhos e hashes.
As entradas retiradas conservam essa identificação para auditoria.

Em 20/11/2026, foram removidos dez runs sem consumidores nem vínculo com
marcos documentados e 33 visualizações derivadas sem uso atual. Foram
retirados 120 arquivos, somando 89.169.814 bytes (85,04 MiB).
Os resultados de interpolação e refatoração recentes estão catalogados como
candidatos; as referências anteriores e os estudos documentados permanecem
consultáveis nos caminhos registrados.

A comparação antes/depois executou dez notebooks de resultados salvos.
Foram conferidos 151 DataFrames/Séries e 94 eventos de exibição de tabelas
ou gráficos. Os fingerprints dessas análises foram idênticos.
A conferência utilizou os resultados salvos das execuções científicas.

O catálogo visual da aorta foi recuperado da revisão `4ce263f`. Seis entradas
de variantes antigas já não tinham os respectivos diretórios; o inventário
identifica essas ausências e a proveniência das revisões disponíveis.

## Série do servidor

O [guia ImageCAS mid](server_mid_comparison.md) contém os seis comandos para
OpenCV/SciPy em treino, validação e teste, as condições de medição e os
critérios de conferência antes de uma eventual promoção.
