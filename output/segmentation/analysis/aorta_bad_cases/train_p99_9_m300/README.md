# Comparação normal versus filtro agressivo

Atalhos das visualizações 3D das mesmas 30 imagens de treino processadas com
limite inferior `-300 HU` e limite superior `P99.9`.

## Runs comparados

- **Normal:** `normal_p99_9_m300/2026-08-24_11-09-38`.
- **Agressivo:** `circle_filter_aggressive_p99_9/2026-08-24_11-32-23`.

## Inspeção rápida por exame

Cada pasta abaixo contém `normal.html` e `aggressive.html`, permitindo abrir as
duas versões lado a lado sem copiar os arquivos originais.

| IMG_ID | Problema observado | Normal | Agressivo |
|---:|---|---|---|
| 44 | Aorta ruim no normal | [abrir](by_image/img_44/normal.html) | [abrir](by_image/img_44/aggressive.html) |
| 175 | Pequeno vazamento no normal | [abrir](by_image/img_175/normal.html) | [abrir](by_image/img_175/aggressive.html) |
| 315 | Óstios ruins no agressivo | [abrir](by_image/img_315/normal.html) | [abrir](by_image/img_315/aggressive.html) |
| 330 | Aorta ruim no normal | [abrir](by_image/img_330/normal.html) | [abrir](by_image/img_330/aggressive.html) |
| 428 | Aorta subsegmentada e óstios ruins no agressivo | [abrir](by_image/img_428/normal.html) | [abrir](by_image/img_428/aggressive.html) |
| 447 | Óstios ruins no agressivo | [abrir](by_image/img_447/normal.html) | [abrir](by_image/img_447/aggressive.html) |
| 603 | Vazamento nas duas variantes e óstios ruins no agressivo | [abrir](by_image/img_603/normal.html) | [abrir](by_image/img_603/aggressive.html) |
| 608 | Aorta ruim no normal | [abrir](by_image/img_608/normal.html) | [abrir](by_image/img_608/aggressive.html) |
| 676 | Óstios ruins no agressivo | [abrir](by_image/img_676/normal.html) | [abrir](by_image/img_676/aggressive.html) |
| 752 | Aorta ruim no normal | [abrir](by_image/img_752/normal.html) | [abrir](by_image/img_752/aggressive.html) |
| 760 | Aorta ruim no normal | [abrir](by_image/img_760/normal.html) | [abrir](by_image/img_760/aggressive.html) |

## Atalhos por categoria

- [`normal_bad_aorta/`](normal_bad_aorta/): 44, 175, 330, 603, 608, 752 e 760.
- [`aggressive_bad_aorta/`](aggressive_bad_aorta/): 428 e 603.
- [`aggressive_bad_ostia/`](aggressive_bad_ostia/): 315, 428, 447, 603 e 676.

O filtro agressivo corrigiu visualmente as aortas 44, 175, 330, 608, 752 e
760, mas introduziu subsegmentação no exame 428. O exame 603 permaneceu com
vazamento nas duas abordagens.
