# Comparação das máscaras da aorta no treino

Atalhos para as visualizações 3D das 30 imagens de treino processadas com
limite inferior `-300 HU` e superior `P99.9`. O catálogo contém somente exames
com aorta visualmente ruim em pelo menos uma variante; falhas exclusivas dos
óstios não são armazenadas aqui.

## Runs comparados

- **Normal:** `normal_p99_9_m300/2026-08-24_11-09-38`.
- **Agressivo:** `circle_filter_aggressive_p99_9/2026-08-24_11-32-23`.
- **Cobertura 80%:** `circle_filter_coverage_080/2026-08-24_14-10-36`.
- **Cobertura 65% + fallback:** `circle_filter_coverage_065_fallback/2026-08-24_14-54-43`.
- **Level set adaptativo:** `adaptive_current_p99_9_m300/2026-08-24_16-41-06`.

## Inspeção rápida

| IMG_ID | Avaliação visual | Normal | Agressivo | 80% | 65% + fallback | Adaptativo |
|---:|---|---|---|---|---|---|
| 44 | Ruim no normal, coberturas e adaptativo | [abrir](by_image/img_44/normal.html) | [abrir](by_image/img_44/aggressive.html) | [abrir](by_image/img_44/coverage_080.html) | [abrir](by_image/img_44/coverage_065_fallback.html) | [abrir](by_image/img_44/adaptive_current.html) |
| 175 | Pequeno vazamento no normal, coberturas e adaptativo | [abrir](by_image/img_175/normal.html) | [abrir](by_image/img_175/aggressive.html) | [abrir](by_image/img_175/coverage_080.html) | [abrir](by_image/img_175/coverage_065_fallback.html) | [abrir](by_image/img_175/adaptive_current.html) |
| 330 | Ruim no normal, coberturas e adaptativo | [abrir](by_image/img_330/normal.html) | [abrir](by_image/img_330/aggressive.html) | [abrir](by_image/img_330/coverage_080.html) | [abrir](by_image/img_330/coverage_065_fallback.html) | [abrir](by_image/img_330/adaptive_current.html) |
| 428 | Adaptativo ficou quase bom, mas removeu a quina usada por um óstio | [abrir](by_image/img_428/normal.html) | [abrir](by_image/img_428/aggressive.html) | [abrir](by_image/img_428/coverage_080.html) | [abrir](by_image/img_428/coverage_065_fallback.html) | [abrir](by_image/img_428/adaptive_current.html) |
| 603 | Vazamento em todas as variantes | [abrir](by_image/img_603/normal.html) | [abrir](by_image/img_603/aggressive.html) | [abrir](by_image/img_603/coverage_080.html) | [abrir](by_image/img_603/coverage_065_fallback.html) | [abrir](by_image/img_603/adaptive_current.html) |
| 608 | Ruim no normal, coberturas e adaptativo | [abrir](by_image/img_608/normal.html) | [abrir](by_image/img_608/aggressive.html) | [abrir](by_image/img_608/coverage_080.html) | [abrir](by_image/img_608/coverage_065_fallback.html) | [abrir](by_image/img_608/adaptive_current.html) |
| 752 | Ruim no normal, coberturas e adaptativo | [abrir](by_image/img_752/normal.html) | [abrir](by_image/img_752/aggressive.html) | [abrir](by_image/img_752/coverage_080.html) | [abrir](by_image/img_752/coverage_065_fallback.html) | [abrir](by_image/img_752/adaptive_current.html) |
| 760 | Adaptativo manteve visualmente o vazamento do normal | [abrir](by_image/img_760/normal.html) | [abrir](by_image/img_760/aggressive.html) | [abrir](by_image/img_760/coverage_080.html) | [abrir](by_image/img_760/coverage_065_fallback.html) | [abrir](by_image/img_760/adaptive_current.html) |

## Atalhos por variante

- [`normal_bad_aorta/`](normal_bad_aorta/): 7 exames.
- [`aggressive_bad_aorta/`](aggressive_bad_aorta/): 2 exames.
- [`coverage_080_bad_aorta/`](coverage_080_bad_aorta/): 8 exames.
- [`coverage_065_fallback_bad_aorta/`](coverage_065_fallback_bad_aorta/): 8 exames.
- [`adaptive_current_bad_aorta/`](adaptive_current_bad_aorta/): 8 exames.

Nos exames sem alteração da máscara, as coberturas e o adaptativo mantiveram a
classificação visual do normal. Por isso, `44`, `175`, `330`, `603`, `608` e
`752` continuam ruins mesmo sem diferença na quantidade de voxels.
