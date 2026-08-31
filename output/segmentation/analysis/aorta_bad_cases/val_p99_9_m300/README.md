# Aorta na validacao: baseline versus correcao de referencia

Comparacao visual das oito falhas do baseline e de um exame sensivel na regiao
dos ostios, entre as 60 imagens revisadas da validacao.

## Runs

- **Baseline:** `baseline_fixed_levelset_p99_9_m300/2026-08-24_11-14-04`.
- **Correcao de referencia:**
  `levelset_b0_6_r0_10_i26_p99_9_m300/2026-08-29_18-04-15`.

## Casos

| IMG_ID | Baseline | Correcao | Avaliacao visual |
|---:|---|---|---|
| 11 | [abrir](by_image/img_11/baseline.html) | [abrir](by_image/img_11/corrected_levelset.html) | Vazamento permanece; `R_P90=3.81`. |
| 134 | [abrir](by_image/img_134/baseline.html) | [abrir](by_image/img_134/corrected_levelset.html) | Corrigida. |
| 444 | [abrir](by_image/img_444/baseline.html) | [abrir](by_image/img_444/corrected_levelset.html) | Corrigida. |
| 464 | [abrir](by_image/img_464/baseline.html) | [abrir](by_image/img_464/corrected_levelset.html) | Sobresegmentacao lateral perto dos ostios. |
| 597 | [abrir](by_image/img_597/baseline.html) | [abrir](by_image/img_597/corrected_levelset.html) | Aorta curta, mas visualmente plausivel. |
| 602 | [abrir](by_image/img_602/baseline.html) | [abrir](by_image/img_602/corrected_levelset.html) | Corrigida. |
| 790 | [abrir](by_image/img_790/baseline.html) | [abrir](by_image/img_790/corrected_levelset.html) | Vazamento em cauda; `R_P90=2.44`. |
| 792 | [abrir](by_image/img_792/baseline.html) | [abrir](by_image/img_792/corrected_levelset.html) | Permanece subsegmentada. |
| 513 | [abrir](by_image/img_513/baseline.html) | [abrir](by_image/img_513/corrected_levelset.html) | Globalmente correta, mas um pouco fina na regiao dos ostios. |

Resultado visual resumido: a correcao elevou os casos bons de `52/60` para
`56/60`. Os vazamentos `11` e `790` compartilham `R_P90 > 2.0`, mais de 200
circulos e baixa confianca media da Hough; nenhum dos 56 casos bons ultrapassou
esse limite de `R_P90`. O exame `464` e um vazamento lateral localizado e nao e
separado com a mesma clareza pelas metricas globais. O `792` apresenta o
problema oposto, de subsegmentacao.

As metricas usadas nessa comparacao estao em
[`leak_indicator_summary.csv`](leak_indicator_summary.csv). A linha
`good_cohort_p90` representa o percentil 90 calculado somente sobre as 56 aortas
visualmente boas, e nao um novo limiar adotado pelo pipeline.
