# Comparação das máscaras da aorta na validação

Atalhos para as visualizações 3D das 60 imagens de validação processadas com
limite inferior `-300 HU` e superior `P99.9`. O catálogo contém somente exames
com aorta visualmente ruim em pelo menos uma variante.

## Runs comparados

- **Normal:** `normal_p99_9_m300/2026-08-24_11-14-04`.
- **Agressivo:** `circle_filter_aggressive_p99_9/2026-08-24_11-56-44`.
- **Cobertura 80%:** `circle_filter_coverage_080/2026-08-24_14-10-36`.
- **Cobertura 65% + fallback:** `circle_filter_coverage_065_fallback/2026-08-24_14-54-43`.
- **Level set adaptativo:** `adaptive_current_p99_9_m300/2026-08-24_16-41-23`.

## Inspeção rápida

| IMG_ID | Avaliação visual | Normal | Agressivo | 80% | 65% + fallback | Adaptativo |
|---:|---|---|---|---|---|---|
| 11 | Vazamento no normal, coberturas e adaptativo | [abrir](by_image/img_11/normal.html) | [abrir](by_image/img_11/aggressive.html) | [abrir](by_image/img_11/coverage_080.html) | [abrir](by_image/img_11/coverage_065_fallback.html) | [abrir](by_image/img_11/adaptive_current.html) |
| 134 | Ruim em todas as variantes | [abrir](by_image/img_134/normal.html) | [abrir](by_image/img_134/aggressive.html) | [abrir](by_image/img_134/coverage_080.html) | [abrir](by_image/img_134/coverage_065_fallback.html) | [abrir](by_image/img_134/adaptive_current.html) |
| 184 | Ruim no normal, coberturas e adaptativo | [abrir](by_image/img_184/normal.html) | [abrir](by_image/img_184/aggressive.html) | [abrir](by_image/img_184/coverage_080.html) | [abrir](by_image/img_184/coverage_065_fallback.html) | [abrir](by_image/img_184/adaptive_current.html) |
| 296 | Ruim no normal, coberturas e adaptativo | [abrir](by_image/img_296/normal.html) | [abrir](by_image/img_296/aggressive.html) | [abrir](by_image/img_296/coverage_080.html) | [abrir](by_image/img_296/coverage_065_fallback.html) | [abrir](by_image/img_296/adaptive_current.html) |
| 307 | Ruim somente no agressivo | [abrir](by_image/img_307/normal.html) | [abrir](by_image/img_307/aggressive.html) | [abrir](by_image/img_307/coverage_080.html) | [abrir](by_image/img_307/coverage_065_fallback.html) | [abrir](by_image/img_307/adaptive_current.html) |
| 384 | Adaptativo adicionou tortuosidade e tornou a aorta incorreta | [abrir](by_image/img_384/normal.html) | [abrir](by_image/img_384/aggressive.html) | [abrir](by_image/img_384/coverage_080.html) | [abrir](by_image/img_384/coverage_065_fallback.html) | [abrir](by_image/img_384/adaptive_current.html) |
| 444 | Adaptativo reduziu o vazamento, mas ele permaneceu | [abrir](by_image/img_444/normal.html) | [abrir](by_image/img_444/aggressive.html) | [abrir](by_image/img_444/coverage_080.html) | [abrir](by_image/img_444/coverage_065_fallback.html) | [abrir](by_image/img_444/adaptive_current.html) |
| 464 | Ruim em todas as variantes | [abrir](by_image/img_464/normal.html) | [abrir](by_image/img_464/aggressive.html) | [abrir](by_image/img_464/coverage_080.html) | [abrir](by_image/img_464/coverage_065_fallback.html) | [abrir](by_image/img_464/adaptive_current.html) |
| 597 | Ruim em todas as variantes | [abrir](by_image/img_597/normal.html) | [abrir](by_image/img_597/aggressive.html) | [abrir](by_image/img_597/coverage_080.html) | [abrir](by_image/img_597/coverage_065_fallback.html) | [abrir](by_image/img_597/adaptive_current.html) |
| 602 | Adaptativo reduziu o volume, mas o vazamento permaneceu | [abrir](by_image/img_602/normal.html) | [abrir](by_image/img_602/aggressive.html) | [abrir](by_image/img_602/coverage_080.html) | [abrir](by_image/img_602/coverage_065_fallback.html) | [abrir](by_image/img_602/adaptive_current.html) |
| 705 | Ruim no normal, coberturas e adaptativo | [abrir](by_image/img_705/normal.html) | [abrir](by_image/img_705/aggressive.html) | [abrir](by_image/img_705/coverage_080.html) | [abrir](by_image/img_705/coverage_065_fallback.html) | [abrir](by_image/img_705/adaptive_current.html) |
| 720 | Vazamento no normal, coberturas e adaptativo | [abrir](by_image/img_720/normal.html) | [abrir](by_image/img_720/aggressive.html) | [abrir](by_image/img_720/coverage_080.html) | [abrir](by_image/img_720/coverage_065_fallback.html) | [abrir](by_image/img_720/adaptive_current.html) |
| 790 | Vazamento permaneceu ruim | [abrir](by_image/img_790/normal.html) | [abrir](by_image/img_790/aggressive.html) | [abrir](by_image/img_790/coverage_080.html) | [abrir](by_image/img_790/coverage_065_fallback.html) | [abrir](by_image/img_790/adaptive_current.html) |
| 792 | Subsegmentada em todas as variantes | [abrir](by_image/img_792/normal.html) | [abrir](by_image/img_792/aggressive.html) | [abrir](by_image/img_792/coverage_080.html) | [abrir](by_image/img_792/coverage_065_fallback.html) | [abrir](by_image/img_792/adaptive_current.html) |
| 838 | Adaptativo reduziu o volume, mas o vazamento permaneceu | [abrir](by_image/img_838/normal.html) | [abrir](by_image/img_838/aggressive.html) | [abrir](by_image/img_838/coverage_080.html) | [abrir](by_image/img_838/coverage_065_fallback.html) | [abrir](by_image/img_838/adaptive_current.html) |

## Casos alterados que permaneceram bons

- `384`: máscara um pouco mais modelada, embora tenha ocorrido perda de um dos
  óstios; a qualidade visual da aorta permaneceu boa.
- `437`: pequena adição na máscara, sem influência visual relevante.
- No adaptativo, `437` também permaneceu visualmente correta.

## Atalhos por variante

- [`normal_bad_aorta/`](normal_bad_aorta/): 13 exames.
- [`aggressive_bad_aorta/`](aggressive_bad_aorta/): 8 exames.
- [`coverage_080_bad_aorta/`](coverage_080_bad_aorta/): 13 exames.
- [`coverage_065_fallback_bad_aorta/`](coverage_065_fallback_bad_aorta/): 13 exames.
- [`adaptive_current_bad_aorta/`](adaptive_current_bad_aorta/): 14 exames.

Os casos ruins do normal sem mudança de voxels foram mantidos como ruins nas
coberturas e no adaptativo. A única regressão visual adicional do adaptativo
foi o exame `384`; o exame `437` permaneceu correto.
