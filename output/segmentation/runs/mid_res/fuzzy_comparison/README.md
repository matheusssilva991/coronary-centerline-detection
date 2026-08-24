# Comparação dos métodos fuzzy

Esta pasta reúne as execuções usadas para comparar os métodos de threshold e
segmentação arterial em resolução média.

O estudo separa duas decisões do pipeline. A primeira é como formar a máscara
de intensidades candidata: threshold normal ou classificação fuzzy. A segunda é
como propagar a segmentação a partir dos óstios: region growing ou fuzzy
connectedness. A combinação fatorial permite observar o efeito de cada decisão
sem confundir RG com FC.

## Estrutura

```text
fuzzy_comparison/
  train/
    <variant>/<timestamp>/
  val/
    <variant>/<timestamp>/
  test/
    <variant>/<timestamp>/
```

Cada execução mantém a estrutura padrão com `config/`, `numeric/` e, quando
disponível, `logs/`.

## Variantes

- `normal_rg`: threshold normal com region growing.
- `normal_fc`: threshold normal com fuzzy connectedness.
- `th_fuzzy_rg`: threshold fuzzy com region growing.
- `th_fuzzy_fc`: threshold fuzzy com fuzzy connectedness.

O split aparece antes da variante para deixar explícito quais resultados podem
ser usados para ajuste (`train` e `val`) e quais devem permanecer reservados
para avaliação final (`test`).

## Configuração do estudo

As variantes alteram somente o método de threshold e o método de segmentação
arterial. As demais etapas seguem a configuração registrada no snapshot de cada
run.

| Variante | Threshold | Segmentação arterial |
|---|---|---|
| `normal_rg` | Normal | Region growing |
| `normal_fc` | Normal | Fuzzy connectedness |
| `th_fuzzy_rg` | Fuzzy de três classes | Region growing |
| `th_fuzzy_fc` | Fuzzy de três classes | Fuzzy connectedness |

No threshold fuzzy, cada voxel recebe pertinências para fundo mole, objeto e
fundo denso. Após a agregação espacial das pertinências, apenas a classe objeto
é preservada como entrada das etapas seguintes. No fuzzy connectedness, os
óstios dão origem às sementes e a conectividade se propaga por caminhos cuja
força é limitada pelo elo de menor afinidade, combinando vesselness e
similaridade de intensidade.

O FC substitui somente a etapa de crescimento arterial; ele não substitui a
localização da aorta nem a detecção dos óstios. Da mesma forma, o threshold
fuzzy atua antes da vesselness e pode ser combinado tanto com RG quanto com FC.

## Runs

| Split | Variante | Run | Imagens | Sucesso dos óstios | Dice geral |
|---|---|---|---:|---:|---:|
| Train | `normal_rg` | `2026-06-29_09-42-27` | 30 | 90,00% | 0,5990 |
| Val | `normal_rg` | `2026-06-23_14-47-01` | 270 | 81,48% | 0,5755 |
| Val | `normal_fc` | `2026-06-23_08-17-16` | 270 | 81,48% | 0,5535 |
| Val | `th_fuzzy_rg` | `2026-06-25_16-16-10` | 270 | 82,59% | 0,5829 |
| Val | `th_fuzzy_fc` | `2026-06-26_13-22-38` | 270 | 82,59% | 0,5569 |
| Test | `normal_rg` | `2026-06-22_10-18-31` | 700 | 80,86% | 0,5784 |
| Test | `normal_fc` | `2026-06-19_09-20-42` | 700 | 80,86% | 0,5647 |
| Test | `th_fuzzy_rg` | `2026-06-20_08-33-25` | 700 | 80,86% | 0,5809 |
| Test | `th_fuzzy_fc` | `2026-06-20_23-26-14` | 700 | 80,86% | 0,5628 |

Sucesso considera os casos com ambos os óstios corretos ou toleráveis. Esta
série é metodológica e não substitui o canonical atual. Na validação, o
threshold fuzzy com RG apresentou o maior Dice geral (`0,5829`). No teste, o
mesmo método manteve uma vantagem pequena sobre o threshold normal com RG
(`0,5809` contra `0,5784`). As duas variantes com FC ficaram abaixo das versões
com RG nessa configuração, embora permaneçam relevantes como comparação de
método para o artigo.
