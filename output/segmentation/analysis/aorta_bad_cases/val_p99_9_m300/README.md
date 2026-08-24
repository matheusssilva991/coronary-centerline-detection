# Comparação normal versus filtro agressivo na validação

Atalhos das visualizações 3D das mesmas 60 imagens de validação processadas
com limite inferior `-300 HU` e limite superior `P99.9`. Somente os 18 exames
com alguma falha visual de aorta ou óstios em pelo menos uma solução aparecem
neste catálogo.

## Runs comparados

- **Normal:** `normal_p99_9_m300/2026-08-24_11-14-04`.
- **Agressivo:** `circle_filter_aggressive_p99_9/2026-08-24_11-56-44`.

## Inspeção rápida por exame

| IMG_ID | Problema observado | Normal | Agressivo |
|---:|---|---|---|
| 11 | Aorta e óstios ruins no normal | [abrir](by_image/img_11/normal.html) | [abrir](by_image/img_11/aggressive.html) |
| 116 | Óstios ruins nas duas soluções | [abrir](by_image/img_116/normal.html) | [abrir](by_image/img_116/aggressive.html) |
| 134 | Aorta e óstios ruins nas duas soluções | [abrir](by_image/img_134/normal.html) | [abrir](by_image/img_134/aggressive.html) |
| 184 | Aorta ruim no normal | [abrir](by_image/img_184/normal.html) | [abrir](by_image/img_184/aggressive.html) |
| 187 | Óstios ruins nas duas soluções | [abrir](by_image/img_187/normal.html) | [abrir](by_image/img_187/aggressive.html) |
| 296 | Aorta ruim no normal | [abrir](by_image/img_296/normal.html) | [abrir](by_image/img_296/aggressive.html) |
| 307 | Aorta ruim no agressivo e óstios ruins nas duas | [abrir](by_image/img_307/normal.html) | [abrir](by_image/img_307/aggressive.html) |
| 384 | Óstios ruins no agressivo | [abrir](by_image/img_384/normal.html) | [abrir](by_image/img_384/aggressive.html) |
| 444 | Aorta ruim nas duas e óstios ruins no agressivo | [abrir](by_image/img_444/normal.html) | [abrir](by_image/img_444/aggressive.html) |
| 464 | Aorta ruim nas duas e óstios ruins no agressivo | [abrir](by_image/img_464/normal.html) | [abrir](by_image/img_464/aggressive.html) |
| 597 | Aorta e óstios ruins nas duas soluções | [abrir](by_image/img_597/normal.html) | [abrir](by_image/img_597/aggressive.html) |
| 602 | Aorta e óstios ruins nas duas soluções | [abrir](by_image/img_602/normal.html) | [abrir](by_image/img_602/aggressive.html) |
| 705 | Aorta ruim no normal e óstios ruins nas duas | [abrir](by_image/img_705/normal.html) | [abrir](by_image/img_705/aggressive.html) |
| 720 | Aorta ruim no normal | [abrir](by_image/img_720/normal.html) | [abrir](by_image/img_720/aggressive.html) |
| 790 | Aorta e óstios ruins nas duas soluções | [abrir](by_image/img_790/normal.html) | [abrir](by_image/img_790/aggressive.html) |
| 792 | Aorta subsegmentada e óstios ruins nas duas | [abrir](by_image/img_792/normal.html) | [abrir](by_image/img_792/aggressive.html) |
| 835 | Óstios ruins nas duas soluções | [abrir](by_image/img_835/normal.html) | [abrir](by_image/img_835/aggressive.html) |
| 838 | Aorta e óstios ruins no normal | [abrir](by_image/img_838/normal.html) | [abrir](by_image/img_838/aggressive.html) |

## Atalhos por categoria

- [`normal_bad_aorta/`](normal_bad_aorta/): 13 exames.
- [`normal_bad_ostia/`](normal_bad_ostia/): 12 exames.
- [`aggressive_bad_aorta/`](aggressive_bad_aorta/): 8 exames.
- [`aggressive_bad_ostia/`](aggressive_bad_ostia/): 13 exames.

Os arquivos são links simbólicos para os HTMLs originais e não duplicam o
espaço ocupado pelas visualizações.
