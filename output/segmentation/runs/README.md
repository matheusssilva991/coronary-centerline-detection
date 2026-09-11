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
    results_<split>.csv
    results_<split>_lote_<n>.csv
    metadata_<split>.json
    batch_timings_<split>.csv
    integrity_<split>.json       # somente quando a coorte estiver incompleta
  visual/                       # existe somente quando solicitado
```

| Pasta | Finalidade |
|---|---|
| `config/` | Configuracao efetiva e IDs realmente processados |
| `logs/` | Progresso, avisos, falhas e retomadas |
| `numeric/` | Resultados por imagem, metadata compacto, lotes e tempos de execucao |
| `visual/` | HTMLs ou imagens vinculados especificamente ao run |

O arquivo `effective_pipeline_config.json` e a fonte principal para conferir
os parametros. O nome da pasta descreve o objetivo do estudo, mas nao substitui
o snapshot da configuracao.

`results_<split>.csv` preserva uma linha por imagem para análises pareadas,
contendo somente métricas e diagnósticos científicos do exame. Entre eles estão
Dice, volumes, fatias, círculos e máscara da aorta, óstios, distâncias,
coordenadas, estados e erros. O metadata contém somente o split, a resolução,
o hash e os rótulos principais usados para identificar a variante. A
configuração completa fica exclusivamente em `config/effective_pipeline_config.json`.

Não há um `summary_<split>.csv`: Dice, quartis, contagens e demais agregados são
calculados sob demanda a partir de `results_<split>.csv` com
`summarize_split_results`. IDs completos permanecem em `config/split_ids.json`,
e durações permanecem em `batch_timings_<split>.csv`; nenhum desses dados é
repetido no metadata.

Antes de publicar o resumo, o pipeline exige igualdade exata entre os IDs do
consolidado e os IDs esperados. Se houver IDs ausentes, inesperados ou
duplicados, `results_<split>.csv` é preservado para diagnóstico, mas o metadata
de execução completa não é gerado; a divergência fica em
`integrity_<split>.json`.

## Migração do layout legado

Inspecione primeiro e aplique explicitamente:

```bash
uv run python src/experiments/migrate_run_result_layout.py
uv run python src/experiments/migrate_run_result_layout.py --apply
```

O modo padrão não altera arquivos. Runs íntegros são projetados para a whitelist
científica, recebem os nomes novos e têm metadata compacto regenerado. Antes
de retirar colunas antigas de configuração, a migração preserva somente os
rótulos principais no metadata; o restante permanece no snapshot efetivo.
Leitores e retomadas continuam
aceitando os nomes anteriores; encontrar o mesmo número de lote nos dois
formatos é tratado como erro. Runs parciais já marcados permanecem intocados no
layout legado até serem concluídos.

## Organizacao

- Um run de referencia deve continuar em `runs/` e ser apontado por
  `canonical/`, evitando duplicacao.
- Variantes do mesmo estudo devem compartilhar um grupo e possuir um README.
- Runs incompletos devem ser identificados como tal no README do grupo.
- Resultados descartados podem manter CSVs e configuracoes, mas visuais grandes
  devem ser removidos quando nao forem mais necessarios.
