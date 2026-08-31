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
