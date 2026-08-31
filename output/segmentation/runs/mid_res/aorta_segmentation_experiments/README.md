# Experimentos de segmentação da aorta

Esta pasta contém os runs ativos em mid resolution usados para estudar a
trajetória de círculos, a máscara da aorta e seu efeito sobre os óstios e a
segmentação arterial. Todos os candidatos atuais usam threshold inferior de
`-300 HU` e superior `P99.9`.

## Runs ativos

| Pasta | Método | Situação |
|---|---|---|
| `baseline_fixed_levelset_p99_9_m300` | Círculos originais + level set fixo | Referência quantitativa |
| `mask_guided_tail_fillloss015_rp90_2_5_rz2_5_synthetic5_envelope_k2_25_margin10_p99_9_m300` | Filtro robusto, cinco círculos sintéticos, envelope `2.25r`, margem axial de 10 fatias e fallback guiado por `R_z` | Melhor candidato atual; corrige o exame 603 sem perder sucessos de óstios |
| `filter_envelope_generalization/` | Variações próximas do envelope atual | Confirmação de estabilidade antes da validação completa |

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
- `adaptive_levelset` ou `fixed_levelset`: identifica o controlador de
  iterações;
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

Os runs de `filter_envelope_generalization/` comparam quatro combinações
próximas da referência visual. Todos usam filtro robusto, cinco círculos
sintéticos e, por padrão, nenhuma recuperação posterior; os nomes registram o
fator radial `k` e a margem axial.

Na combinação atual, cinco círculos sintéticos recuperaram os exames `315` e
`676`. Dez círculos não alteraram Dice ou sucesso dos óstios e reintroduziram
pequenos vazamentos visuais em `44` e `330`, por isso essa variação foi
arquivada.

## Arquivo histórico

Runs executados com configuração antiga ou variantes dominadas estão em
[`archive/`](archive/README.md). Eles preservam configurações, CSVs e logs para
auditoria, mas não são candidatos ativos para o pipeline.
