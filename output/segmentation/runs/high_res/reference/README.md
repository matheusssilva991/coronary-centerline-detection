# Referência histórica high resolution

Recuperada do commit `7afdaef764314d1910e3b4d7fda3140f33992cd3`.
Origem: https://github.com/matheusssilva991/coronary-centerline-detection/commit/7afdaef764314d1910e3b4d7fda3140f33992cd3

Os splits foram renomeados conforme os IDs atuais: val histórico virou train
(30 exames), train histórico virou val (270 exames), test permaneceu test (700).

| Split atual | Run | Exames | Sucesso dos óstios | Dice médio |
| --- | --- | ---: | ---: | ---: |
| train | 2026-04-21_08-42-13 | 30 | 90,00% | 0,5590316715 |
| val | 2026-04-21_08-42-13 | 270 | 72,22% | 0,4807136924 |
| test | 2026-04-28_14-28-44 | 700 | 71,43% | 0,4684355232 |

Cada run contém `numeric/results_<split>.csv`, `numeric/metadata_<split>.json`
(schema v3), `config/split_ids.json` e `config/legacy_pipeline_config.json`.
Colunas científicas seguem o schema atual; medições indisponíveis ficam vazias.
Os agregados são calculados dos resultados persistidos, sem recalcular segmentações.
Sucesso significa ambos corretos ou ambos toleráveis.

Não existe snapshot completo efetivo nem arquivos de lotes/timings nesta origem.
A configuração permanece identificada como `legacy_metadata`, sem SHA-256 de
snapshot efetivo. Tempo agregado atual fica null por ausência de timings de lotes;
o tempo histórico informado permanece no metadata original em `provenance`.
Dice final é usado no agregado pós-morfologia; Dice pré-morfologia indisponível
permanece null. Não foram inventados lotes, timings ou diagnósticos.

`provenance/original_results.csv` e `provenance/original_metadata.json` preservam
os bytes originais e os nomes de split históricos dentro do conteúdo.
`provenance/origin.json` documenta o mapeamento para os splits atuais.
As seções de configuração registradas dos três runs são iguais.

Esta pasta é uma referência histórica, não a configuração candidata moderna
com filtro e envelope em `../aorta_segmentation_experiments/<split>/filter_envelope_lower100_pad2`.
