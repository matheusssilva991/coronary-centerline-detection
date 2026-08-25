# Experimentos de segmentação da aorta

Esta pasta reúne os runs mid resolution usados para estudar localização por
círculos, controle adaptativo do level set, remoção de vazamentos e recuperação
dos óstios. Cada variante preserva `config/`, `numeric/` e `logs/`. A pasta
`visual/` é mantida em referências, variantes promissoras ou execuções criadas
explicitamente para inspeção anatômica.

O objetivo deste grupo é investigar falhas que acontecem antes da segmentação
arterial. Uma trajetória incorreta de círculos pode deslocar o level set,
produzir vazamentos na máscara da aorta e, por consequência, oferecer uma
superfície errada para a seleção dos óstios. Por isso, Dice arterial, sucesso
dos óstios e inspeção dos HTMLs precisam ser analisados em conjunto.

## Famílias de testes

### Referências visuais e level set adaptativo

- As referências visuais antigas e `fixed_reference` preservam a trajetória
  original e o level set fixo. Servem como controle imagem a imagem para as
  correções testadas sobre P10.75/P99.8.
- `adaptive_*` executa o level set em checkpoints, calcula volume, preenchimento
  dos círculos, razão entre área segmentada e área circular e continuidade
  axial. A máscara nominal pode ser classificada como `adequate`,
  `undersegmented`, `oversegmented` ou `localization_suspected`.
- Quando o controlador encontra sobre ou subsegmentação, ele pode tentar uma
  nova evolução conservadora ou permissiva a partir de um checkpoint anterior.
  Uma alternativa rejeitada devolve integralmente a máscara nominal. Esses
  runs mostraram que métricas geométricas ajudam no diagnóstico, mas ainda não
  garantem melhora visual da aorta.

### Correções aplicadas depois do level set (removidas)

- `neck_pruning_*` erode a máscara para tentar romper uma conexão estreita
  entre a aorta e um vazamento. As componentes que ainda tocam o núcleo definido
  pelos círculos são reconstruídas por dilatação. Os sufixos `r2`, `r3` e `r6`
  indicam o raio da erosão; raios maiores podem romper também a aorta correta.
- `area_jump_pruning` analisa a razão axial entre área da máscara e área dos
  círculos. Ao detectar crescimento abrupto nas fatias inferiores, procura uma
  seção estreita anterior e tenta remover a região que surgiu depois dela. A
  correção é rejeitada se perder preenchimento, fatias ou volume em excesso.
- Essas famílias foram removidas do runtime após não apresentarem ganho
  agregado; permanecem documentadas somente para interpretar runs antigos.
  Elas modificavam a máscara já segmentada e não corrigiam uma trajetória de
  círculos localizada sobre a estrutura anatômica errada.

### Filtros da trajetória circular

- `circle_filter_aggressive` atua **antes** do level set. Com cobertura mínima
  de 40%, permite remover cedo uma cauda persistente com saltos incompatíveis
  de centro, raio ou confiança Hough. Foi a variante que mais alterou as
  máscaras, mas também produziu cortes prematuros e afinamento próximo aos
  óstios; por isso permanece somente para inspeção experimental.
- `circle_filter_conservative` exige que a trajetória cubra pelo menos 80% das
  fatias. Assim, somente divergências muito tardias podem ser removidas.
- `circle_filter_coverage_060` e `circle_filter_coverage_065` usam o mesmo
  filtro robusto com coberturas mínimas de 60% e 65%. Cobertura menor aumenta a
  chance de corrigir vazamentos, mas também aumenta o risco de truncar uma
  trajetória válida.
- Variantes com `_fallback` repetem somente o level set com os círculos
  originais quando a máscara filtrada continua classificada como
  `oversegmented`. O fallback evita aceitar algumas intervenções ruins, mas não
  prova que a máscara aceita ficou anatomicamente correta.

### Recuperação dos óstios

- `ostia_symmetric_band` não altera a localização dos círculos nem a evolução
  do level set. Ele amplia a região da superfície em que os candidatos de
  óstio são buscados, tentando recuperar óstios perdidos após uma correção da
  aorta. Recuperou alguns exames e piorou outros, portanto foi descartado.

As variantes de correção são condicionais: quando o critério não é acionado, a
imagem deve conservar o resultado nominal. Os campos de diagnóstico no summary
indicam se o filtro foi aplicado, aceito, rejeitado ou substituído pelo
fallback.

## Baseline dos próximos experimentos

Novos testes desta pasta devem usar `config/article_cbeb_sensitivity.json` como
base. Essa configuração mantém threshold normal com limite inferior fixo em
`-300 HU` e limite superior P99.9, que foi o melhor run completo em mid
resolution. O filtro de círculos e o level set adaptativo continuam desativados
no baseline e são ligados apenas pela CLI do experimento.

Os runs anteriores a 24 de agosto foram produzidos sobre o antigo baseline de
desenvolvimento P10.75/P99.8. Eles permanecem úteis como histórico, mas não
devem ser comparados diretamente aos novos runs P99.9 sem considerar essa
mudança de configuração.

Eles foram movidos para [`archive/p10_75_p99_8/`](archive/p10_75_p99_8/README.md).
As pastas `train/` e `val/` na raiz ficam reservadas aos novos experimentos
executados sobre P99.9. As referências antigas usadas pela EDA estão no
arquivo P10.75/P99.8.

Para gerar novamente o perfil agressivo sobre o novo baseline nas 30 imagens
de treino, incluindo um HTML por exame, execute:

```bash
FILTER_PROFILE=aggressive RUN_VAL=0 SAVE_VISUALS=1 \
  bash src/experiments/runners/run_aorta_circle_coverage_tests.sh
```

O novo run deve ser organizado em
`train/circle_filter_aggressive_p99_9/<timestamp>/`. O sufixo `p99_9` evita
confusão com `circle_filter_aggressive/2026-08-23_10-07-44`,
que foi executado com P10.75/P99.8.

Nos runs ativos, `train/` e `val/` identificam a coorte do experimento. Dentro
de cada run, `visual/{split}/` é mantido porque faz parte da estrutura padrão
do pipeline e permite que uma execução futura contenha mais de um split.

## Referências visuais arquivadas

| Split | Variante | Run | Uso |
|---|---|---|---|
| Train | `visual_reference_standard_p10_75_p99_8_train30` | `2026-08-22_08-11-56` | Inspeção das 30 imagens de treino no notebook de qualidade da aorta |
| Val | `visual_reference_standard_p10_75_p99_8_val60` | `2026-08-22_09-27-56` | Inspeção das 60 imagens de validação no mesmo notebook |

Essas referências usam o baseline antigo e agora estão em
`archive/p10_75_p99_8/{train,val}/`. O notebook aponta explicitamente para os
caminhos arquivados. Elas devem ser substituídas por referências P99.9 quando
os novos HTMLs forem revisados.

## Histórico arquivado

### Treino P10.75/P99.8

| Variante | Run | Descrição |
|---|---|---|
| `adaptive_initial` | `2026-08-22_13-22-12` | Primeira versão adaptativa preservada, executada com HTMLs |
| `fixed_reference` | `2026-08-22_17-43-27` | Level set fixo de referência |
| `adaptive_refined` | `2026-08-22_19-24-59` | Controlador adaptativo conservador |
| `neck_pruning_r2` | `2026-08-23_07-35-37` | Poda por colo, erosão 2 |
| `neck_pruning_r3` | `2026-08-23_08-10-14` | Poda por colo, erosão 3 |
| `neck_pruning_r6` | `2026-08-23_08-38-50` | Poda por colo, erosão 6 |
| `area_jump_pruning` | `2026-08-23_09-25-00` | Poda por salto axial de área |
| `circle_filter_aggressive` | `2026-08-23_10-07-44` | Filtro de trajetória agressivo |
| `circle_filter_conservative` | `2026-08-23_10-48-27` | Filtro restrito a cobertura extrema |
| `circle_filter_coverage_060` | `2026-08-23_14-15-00` | Cobertura mínima de 60% das fatias |
| `circle_filter_coverage_060_fallback` | `2026-08-23_14-39-42` | Cobertura de 60% com rejeição de resultado sobresegmentado |
| `circle_filter_coverage_065_fallback` | `2026-08-23_18-02-01` | Cobertura de 65% com fallback; Dice 0,6068 e 27/30 sucessos |
| `ostia_symmetric_band` | `2026-08-23_11-32-16` | Recuperação de candidatos em banda interna/externa |

## Decisões

- `circle_filter_conservative` foi o melhor resultado agregado desta sequência:
  Dice médio `0,6068` e sucesso dos óstios em `27/30` exames.
- `ostia_symmetric_band` recuperou os casos 603 e 676 e melhorou o caso 428,
  mas reduziu fortemente o Dice de 194, 631 e 965. O resultado final foi Dice
  médio `0,5878`, com sucesso em `27/30`; por isso a variante foi descartada.
- `circle_filter_coverage_060` aplicou o filtro em três exames, mas reduziu o
  Dice médio para `0,5976`. Com o fallback, um resultado ainda sobresegmentado
  foi rejeitado e o Dice voltou a `0,6068`, com sucesso em `27/30`.
- `circle_filter_coverage_065_fallback` aplicou o filtro apenas ao exame 608 e
  reproduziu exatamente o melhor resultado do filtro conservador de 80%:
  Dice `0,6068`, sem alterar a taxa de sucesso dos óstios. Em relação ao
  controle com a mesma configuração adaptativa, o Dice do exame 608 subiu de
  `0,3697` para `0,6760`. Esse é um ganho arterial indireto e não deve ser
  interpretado sozinho como melhora anatômica da máscara da aorta.
- Os HTMLs das variantes descartadas foram removidos para economizar espaço.
  CSVs, configurações e logs foram preservados para manter o histórico
  quantitativo reproduzível.
- Os visuais de `circle_filter_coverage_060` sem fallback foram removidos em
  treino e validação. Os visuais de `circle_filter_conservative` em validação
  também foram removidos porque o resultado foi superado pela cobertura de 60%
  com fallback. Os dados numéricos desses runs continuam disponíveis.

### Validação P10.75/P99.8

| Variante | Run | Descrição |
|---|---|---|
| `adaptive_no_circle_filter_full_270` | `2026-08-24_07-21-55` | Controle adaptativo sem filtro de círculos nas 270 imagens; Dice 0,5797 e 222/270 sucessos dos óstios |
| `circle_filter_conservative` | `2026-08-23_12-15-12` | Cobertura mínima de 80%; Dice 0,5859 e 48/60 sucessos |
| `circle_filter_coverage_065` | `2026-08-23_13-15-52` | Cobertura mínima de 65%; Dice 0,5892 e 49/60 sucessos |
| `circle_filter_coverage_060` | `2026-08-23_15-04-34` | Cobertura mínima de 60%; Dice 0,5892 e 49/60 sucessos |
| `circle_filter_coverage_060_fallback` | `2026-08-23_15-50-01` | Cobertura de 60% com fallback; Dice 0,5917 e 49/60 sucessos |
| `circle_filter_coverage_065_fallback` | `2026-08-23_18-26-53` | Cobertura de 65% com fallback; Dice 0,5917 e 49/60 sucessos |
| `circle_filter_coverage_065_fallback_full_270` | `2026-08-23_22-42-20` | Validação completa; Dice 0,5831 e 223/270 sucessos |

O filtro de 60% produziu o mesmo resultado agregado do filtro de 65% na coorte
de validação antes do fallback. Com o fallback, o resultado filtrado do exame
790 foi rejeitado por continuar `oversegmented`, restaurando seu resultado de
referência. Os exames 838 e 907 tiveram alterações favoráveis nos indicadores
indiretos, mas a inspeção visual posterior não confirmou uma melhora clara e
reprodutível da anatomia da aorta.

Nesta coorte de 60 imagens, a combinação atingiu o maior Dice agregado da série
(`0,5917`), mas o filtro efetivamente alterou somente três exames. Portanto, o
resultado ainda deve ser tratado como evidência localizada, não como validação
definitiva para todos os casos.

A cobertura de 65% com fallback reproduziu exatamente os 60 resultados da
cobertura de 60% com fallback. Na validação, o filtro preservou os ganhos
numéricos dos exames 838 e 907 e rejeitou a trajetória filtrada do exame 790.
No treino, a faixa de 65% evitou acionar o filtro em trajetórias limítrofes,
como os exames 428 (62,5%) e 330 (64,7%), mantendo somente o ganho numérico do
exame 608.

Entre as coberturas avaliadas, 65% com fallback é a opção mais conservadora:
produziu o melhor resultado agregado da validação sem as intervenções extras do
limite de 60%. Ainda assim, a evidência permanece concentrada em poucos exames
e deve ser confirmada em uma coorte maior antes de alterar o padrão do pipeline.

Na validação completa de 270 imagens, o filtro foi acionado em seis exames. As
trajetórias dos exames 534, 838, 850, 907 e 934 foram aceitas; todas melhoraram
ou preservaram o Dice em relação ao baseline de desenvolvimento. O exame 790
continuou `oversegmented` após o filtro e foi restaurado pelo fallback. O ganho
agregado sobre esse baseline foi de `0,0029` no Dice, com um sucesso adicional
dos óstios e sem perdas de sucesso.

As regressões dos exames 248 e 861 ocorreram sem aplicação do filtro e são
atribuídas ao controle adaptativo do level set. Além disso, o resultado completo
ainda ficou abaixo do canonical P99.9 (`0,5831` contra `0,5879` no Dice e
`223/270` contra `226/270` sucessos). Assim, o filtro de círculos com fallback
permanece uma hipótese experimental com efeitos localizados; ele não foi
validado como melhoria visual da aorta e a configuração adaptativa completa
não deve substituir o canonical.

Novos runs podem ser criados diretamente nesta hierarquia com
`--run-group aorta_segmentation_experiments/<split>/<variante>`.
