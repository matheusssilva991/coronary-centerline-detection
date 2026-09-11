# Candidatos de localizacao dos ostios

Runs em `train/ostia_localization/` e `val/ostia_localization/`.
Confirmacao completa em `val/ostia_validation270/`. Testes finais novos em
`test/ostia_comparison/`. Dentro de cada familia: variante / timestamp.

Todos usam filtro robusto 4.8/8, envelope 2.25r, margem axial 10,
cinco circulos sinteticos e level set b0.6/r0.10/i26.

| Pasta | Uso |
|---|---|
| baseline_pad0 | Controle com lower_fraction=0.85 e padding zero; apenas runs antigos de ostia_localization incluem fallback |
| lower100_pad2 | Novo padrao operacional: superficie completa e padding de 2 voxels |
| lower100_pad3 | Alternativa historica: superficie completa e padding de 3 voxels |

| Variante | Timestamp treino (30) | Timestamp validacao (60) | Dice treino / val | Sucesso treino / val |
|---|---|---|---|---|
| pad2 | 2026-09-06_08-08-25 | 2026-09-06_08-47-17 | 0.628770 / 0.606670 | 29/30 / 52/60 |
| pad3 | 2026-09-06_08-27-47 | 2026-09-06_09-24-47 | 0.640534 / 0.603242 | 30/30 / 53/60 |

O pad3 recupera 428 no treino, 187/513 na validacao e perde 907 frente ao pad2.
307 permanece incorreto. A revisao visual considerou 187 bem localizado com
pad2 apesar da classificacao automatica; preservar essa distincao na analise.
As contagens de voxels da aorta sao iguais entre os pads.

Cada timestamp contem `config/`, `numeric/` e `logs/`. O split esta no nome
dos CSVs. Os HTMLs seguem o mesmo caminho relativo sob
`/media/matheus/HD/ImageCAS_pipeline_results/segmentation/runs/mid_res/`.
Os runs antigos com fallback foram removidos depois que seus resultados foram
registrados em `output/segmentation/docs/experiments/README.md`.

## Validacao completa: 270 imagens

| Variante | Timestamp | Dice | Sucesso |
|---|---|---:|---:|
| baseline_pad0 | 2026-09-07_07-15-02 | 0.580372 | 221/270 |
| lower100_pad2 | 2026-09-07_07-15-15 | 0.599190 | 229/270 |
| lower100_pad3 | 2026-09-07_07-15-29 | 0.593756 | 229/270 |

Estes runs nao usam fallback. Pad2 foi selecionado por maior Dice com o
mesmo sucesso total de pad3. Todos os tres permanecem para comparacao.
Sucesso soma ambos corretos e ambos toleraveis.

## Teste completo: 700 imagens e promocao

| Variante | Timestamp | Dice | Sucesso |
|---|---|---:|---:|
| P99.9 puro historico | 2026-08-06_10-04-22 | 0.593020 | 578/700 |
| baseline_pad0 | 2026-09-07_10-37-23 | 0.587438 | 572/700 |
| lower100_pad2 | 2026-09-07_10-37-38 | 0.601776 | 599/700 |

Pad2 foi promovido em `config/pipeline_config.json` em 07/09/2026, mantendo
RG, -300 HU e P99.9. O perfil do artigo desativa as novas correcoes
explicitamente e os snapshots dos runs permanecem inalterados.

A melhora agregada nao implica superioridade estatistica de Dice:
Wilcoxon bilateral contra o puro p=0.172041 (teste) e p=0.352612 (validacao).
Nenhuma comparacao foi significativa apos Holm na familia de nove testes
(tres pares em cada split). O teste teve 57 sucessos de ostios recuperados
e 36 perdidos frente ao puro; as novas aortas ainda requerem inspecao visual.
