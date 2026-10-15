# Referências oficiais do pipeline

`canonical/` é a fonte única das análises do baseline atual, por resolução e
conjunto. Somente resultados completos e verificados são promovidos. A promoção
atual consolida resultados existentes: **não representa novas execuções**.

```text
canonical/
  references_manifest.json
  mid_res/{train,val,test}/current/
    numeric/results_<split>.csv
    config/effective_pipeline_config.json
    config/experiment_settings.json
    source_manifest.json
  high_res/{train,val,test}/current -> run original
```

## Referências atuais

| Resolução | Conjunto | N | Origem | Variante | Dice geral |
|---|---|---:|---|---|---:|
| Mid | Train | 30 | `runs/mid_res/fuzzy_comparison/current_pipeline/train30_val60` | `normal_rg` | 0,6288 |
| Mid | Val | 270 | `analysis/pipeline_parameter_validation/runs/sensitivity_refined_pipeline_val270` | `baseline` | 0,6009 |
| Mid | Test | 700 | `runs/mid_res/fuzzy_comparison/current_pipeline/test700` | `normal_rg` | 0,5996 |
| High | Train | 30 | `runs/high_res/aorta_segmentation_experiments/train/filter_envelope_lower100_pad2/2026-09-15_11-08-42` | Run completo | 0,5720 |
| High | Val | 270 | `runs/high_res/aorta_segmentation_experiments/val/filter_envelope_lower100_pad2/2026-09-12_07-51-48` | Run completo | 0,5155 |
| High | Test | 700 | `runs/high_res/aorta_segmentation_experiments/test/filter_envelope_lower100_pad2/2026-09-13_14-06-44` | Run completo | 0,5146 |

Caminhos relativos a `output/segmentation/`. O Dice geral inclui as falhas;
os snapshots mid preservam o Dice registrado/avaliável separadamente quando
disponível. As coortes contêm os mesmos IDs das referências anteriores.

Mid usa threshold normal de -300 HU a P99.9, filtro robusto, envelope e
`lower100/pad2`, com level set b0.6/r0.10/i26. A fonte de verdade dos parâmetros
é o snapshot registrado, não o nome ou a descrição de uma variante. Algumas
descrições antigas ainda mencionavam P10.75/P99.8 sem corresponder à configuração.
Em high, consulte a configuração efetiva no próprio run, incluindo a escala e
os ajustes específicos da resolução.

## Proveniência e tempo

As tabelas mid são extrações compactas de uma única variante/split dos sweeps,
no schema aceito pelos carregadores usuais. Cada `source_manifest.json` registra
origem, variante, IDs, contagem, hashes do CSV/config original e sobrescritas.
A configuração foi reconstruída somente a partir do snapshot e das sobrescritas
registradas; os arquivos de origem permanecem intactos. O código reutilizável de
consolidação fica em `utils.comparison_utils.canonical`.

Não são criados tempos de batch ou metadados de execução fictícios para mid.
Sem manifest equivalente de batches, o tempo aparece como indisponível. Tempos
por exame eventualmente registrados na fonte não equivalem a tempo de batch ou
a uma comparação em hardware controlado. High continua usando seus artefatos
originais, sem duplicação.

## Referências anteriores preservadas

Os links antigos foram substituídos, mas os runs reais continuam nos destinos:

| Resolução | Train | Val | Test |
|---|---|---|---|
| Mid | `runs/mid_res/current_baseline_p99_9/train/2026-08-06_18-43-37` | `runs/mid_res/current_baseline_p99_9/val/2026-08-06_22-43-14` | `runs/mid_res/current_baseline_p99_9/test/2026-08-06_10-04-22` |
| High | `runs/high_res/reference/train/2026-04-21_08-42-13` | `runs/high_res/reference/val/2026-04-21_08-42-13` | `runs/high_res/reference/test/2026-04-28_14-28-44` |

`references_manifest.json` registra as referências anteriores e as promovidas.
Comparações históricas devem continuar apontando explicitamente para esses runs,
em vez de seguir automaticamente a referência móvel de `canonical`.
