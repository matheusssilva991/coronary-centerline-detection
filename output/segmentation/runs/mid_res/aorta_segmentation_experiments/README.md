# Experimentos de segmentação da aorta

Esta pasta contém os runs ativos em mid resolution usados para estudar a
trajetória de círculos, a máscara da aorta e seu efeito sobre os óstios e a
segmentação arterial. Todos os candidatos atuais usam threshold inferior de
`-300 HU` e superior `P99.9`.

## Runs ativos

| Pasta | Método | Situação |
|---|---|---|
| `baseline_fixed_levelset_p99_9_m300` | Círculos originais + level set fixo | Referência quantitativa |
| `levelset_b0_6_r0_10_i26_p99_9_m300` | Level set fixo com pressão, semente e iterações refinadas | Melhor configuração isolada do level set |
| `robust_filter_synthetic5_envelope_k2_25_margin10_p99_9_m300` | Filtro robusto, cinco círculos sintéticos e envelope | Referência visual do filtro combinado |
| `filter_envelope_generalization/` | Confirmação histórica do envelope `2.25r` com margem axial 10 | Evidência da seleção |
| `selected_hough18_29_filter_envelope_p99_9_m300/` | Hough `18-29 px`, filtro robusto, cinco círculos sintéticos e envelope `2.25r` | Configuração ativa selecionada em 90 imagens |

`train/` e `val/` identificam a coorte. Dentro de cada timestamp, `config/`
registra a configuração efetiva, `numeric/` contém CSVs e metadados e `logs/`
contém o log da execução. Os HTMLs podem ser gravados fora do repositório com
`--visual-output-dir`.

## Nomes

Os nomes descrevem os mecanismos usados:

- `robust_tail_filter`: remove uma cauda persistentemente incompatível da
  trajetória de círculos antes do level set;
- `cov040`: exige cobertura original mínima de 40% das fatias;
- `maxtrim040`: rejeita cortes que removeriam mais de 40% da trajetória;
- `fixed_levelset`: identifica a evolução com número fixo de iterações;
- `trajectory_envelope_k2_25_margin5`: limita a máscara ao tubo de raio
  `2.25r` e prolonga o envelope por cinco fatias nas extremidades;
- `synthetic5`: prolonga a última região estável com cinco círculos estimados
  pela tendência mediana de centro e raio;
- `mask_guided_tail`: usa a máscara nominal para detectar excesso de área
  persistente quando a geometria dos círculos, sozinha, não autoriza o corte;
- `margin10`: mantém o envelope por mais dez fatias além dos círculos extremos,
  reduzindo o risco de cortar a região dos óstios;
- `p99_9_m300`: threshold superior P99.9 e inferior `-300 HU`.

## Comparação ativa

O runner [`run_aorta_filter_envelope_generalization.sh`](../../../../../src/experiments/runners/run_aorta_filter_envelope_generalization.sh)
reproduz o filtro, a continuação sintética e o envelope atuais. Nenhum run
altera automaticamente o baseline do projeto.

```bash
SPLIT=train bash src/experiments/runners/run_aorta_filter_envelope_generalization.sh
```

O runner ativo reproduz somente a configuração selecionada. Grades anteriores
de envelope, raios e controle adaptativo foram encerradas para evitar novas
execuções acidentais.

Na combinação atual, cinco círculos sintéticos recuperaram os exames `315` e
`676`. Dez círculos não alteraram Dice ou sucesso dos óstios e reintroduziram
pequenos vazamentos visuais em `44` e `330`, por isso essa variação foi
arquivada.

## Arquivo histórico

Runs executados com configuração antiga ou variantes dominadas estão em
[`archive/`](archive/README.md). Eles preservam configurações, CSVs e logs para
auditoria, mas não são candidatos ativos para o pipeline.
