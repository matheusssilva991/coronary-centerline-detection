# Aorta no treino: baseline versus correcao de referencia

Comparacao visual das sete falhas do baseline e de dois exames sensiveis na
regiao dos ostios. Cada pasta em `by_image/` contem somente a mascara do
baseline e a mascara obtida com `b0.6/r0.10/i26`.

## Runs

- **Baseline:** `baseline_fixed_levelset_p99_9_m300/2026-08-24_11-09-38`.
- **Correcao de referencia:**
  `levelset_b0_6_r0_10_i26_p99_9_m300/2026-08-29_18-03-58`.

## Casos

| IMG_ID | Baseline | Correcao | Avaliacao visual |
|---:|---|---|---|
| 44 | [abrir](by_image/img_44/baseline.html) | [abrir](by_image/img_44/corrected_levelset.html) | Vazamento corrigido. |
| 175 | [abrir](by_image/img_175/baseline.html) | [abrir](by_image/img_175/corrected_levelset.html) | Pequeno vazamento corrigido. |
| 330 | [abrir](by_image/img_330/baseline.html) | [abrir](by_image/img_330/corrected_levelset.html) | Vazamento corrigido. |
| 603 | [abrir](by_image/img_603/baseline.html) | [abrir](by_image/img_603/corrected_levelset.html) | Vazamento corrigido pela nova evolucao. |
| 608 | [abrir](by_image/img_608/baseline.html) | [abrir](by_image/img_608/corrected_levelset.html) | Resultado corrigido. |
| 752 | [abrir](by_image/img_752/baseline.html) | [abrir](by_image/img_752/corrected_levelset.html) | Resultado corrigido. |
| 760 | [abrir](by_image/img_760/baseline.html) | [abrir](by_image/img_760/corrected_levelset.html) | Vazamento corrigido. |
| 315 | [abrir](by_image/img_315/baseline.html) | [abrir](by_image/img_315/corrected_levelset.html) | Globalmente correta, mas um pouco fina na regiao dos ostios. |
| 428 | [abrir](by_image/img_428/baseline.html) | [abrir](by_image/img_428/corrected_levelset.html) | Globalmente correta, mas um pouco fina na regiao dos ostios. |

Resultado visual resumido: a correcao elevou os casos bons de `23/30` para
`30/30`. Os exames `315` e `428` motivam o teste de abertura, pois uma operacao
agressiva pode prejudicar a superficie onde os ostios sao procurados.

## Novo padrao lower100/pad2

Run `2026-09-06_08-08-25`, geometria 4.8/8 e mesmo level set b0.6/r0.10/i26.
Os visuais abaixo sao do novo run; os rotulos da tabela anterior continuam
pertencendo a revisao historica. Nao representam uma nova revisao automatica.

- Exame 44: [abrir pad2](by_image/img_44/new_default_pad2.html).
- Exame 175: [abrir pad2](by_image/img_175/new_default_pad2.html).
- Exame 315: [abrir pad2](by_image/img_315/new_default_pad2.html).
- Exame 330: [abrir pad2](by_image/img_330/new_default_pad2.html).
- Exame 428: [abrir pad2](by_image/img_428/new_default_pad2.html).
- Exame 603: [abrir pad2](by_image/img_603/new_default_pad2.html).
- Exame 608: [abrir pad2](by_image/img_608/new_default_pad2.html).
- Exame 752: [abrir pad2](by_image/img_752/new_default_pad2.html).
- Exame 760: [abrir pad2](by_image/img_760/new_default_pad2.html).
