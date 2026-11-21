# 07 — Pós-processamento e avaliação

[Manual](../README.md) · [Configuração](../../../config/README.md)

Permite inspecionar o efeito de fechamento/dilatação e avaliar uma máscara quando existe referência coronariana compatível. A avaliação mede concordância com essa referência, não validade clínica.

## Exemplo sintético

Execute a partir da raiz do repositório com `PYTHONPATH=src uv run python seu_exemplo.py`. Este exemplo usa CPU e não representa validação anatômica.

```python
import numpy as np
from utils.project.config import load_config_json, scale_config_to_resolution

config = load_config_json("config/pipeline_config.json", {})
config["USE_GPU"] = False
config = scale_config_to_resolution(config)

from utils.segmentation.pipeline.arteries import get_artery_postprocessing_stages
from utils.utils.metrics import dice_score, binary_segmentation_metrics

raw_mask = np.zeros((20, 20, 12), dtype=np.uint8)
raw_mask[6:14, 6:14, 3:9] = 1
raw_mask[9, 9, 5] = 0
label = np.zeros_like(raw_mask)
label[6:14, 6:14, 3:9] = 1
stages = get_artery_postprocessing_stages(raw_mask, config)
assert set(stages) == {"raw_mask", "closed_mask", "final_mask"}
assert all(
    mask.shape == raw_mask.shape and mask.dtype == np.uint8 for mask in stages.values()
)
before = dice_score(stages["raw_mask"], label)
after = dice_score(stages["final_mask"], label)
print({"before": before, "after": after, "delta": after - before})
```

## Métricas salvas por estágio

As novas execuções ImageCAS e as comparações RG/FC registram automaticamente as
três etapas, sem repetir as operações nem salvar máscaras adicionais:

| Coluna no CSV ImageCAS | Significado |
|---|---|
| `artery_dice_before_morphology` | Dice da segmentação bruta. |
| `artery_dice_after_closing` | Dice após o fechamento. |
| `artery_dice_after_morphology` | Dice final, após a dilatação. |
| `artery_dice_closing_delta` | Dice após fechamento menos Dice bruto. |
| `artery_dice_dilation_delta` | Dice final menos Dice após fechamento. |
| `artery_dice_morphology_delta` | Ganho total: Dice final menos Dice bruto. |
| `artery_voxel_count_after_closing` | Quantidade de voxels após fechamento. |

Os deltas podem ser negativos. Por exame, o delta total corresponde à soma dos
deltas do fechamento e da dilatação, salvo arredondamento numérico. As contagens
pré/pós-morfologia e o Dice final já existentes permanecem inalterados.

No código e nos CSVs fuzzy, os nomes internos são `dice_artery_*` e
`artery_voxels_*`. Os resumos incluem média e quantidade de medidas disponíveis;
as médias de deltas com coberturas diferentes não devem ser somadas como se
representassem a mesma coorte.

O batch externo registra `artery_voxels_after_closing` e
`artery_volume_after_closing_ml`. O volume usa o espaçamento efetivo, em mm:
`voxels × spacing_x × spacing_y × spacing_z / 1000`. Não calcula Dice arterial
sem uma referência compatível.

Runs antigos continuam legíveis, com os novos campos ausentes; nenhum valor é
estimado retrospectivamente. A retomada preserva os exames concluídos e só mede
novamente os exames explicitamente reexecutados. Falhas anteriores à segmentação
não recebem Dice intermediário zero. Nenhum resultado histórico é migrado.

## HD e AHD opcionais na máscara final

Use `--distance-metrics` no pipeline ImageCAS.

```bash
uv run python src/segmentation_pipeline.py --split train --resolution mid --gpu --distance-metrics
```

O padrão é `EVALUATION.distance_metrics=false`. `--no-distance-metrics` desativa
explicitamente; sem flag, vale a configuração carregada. Apenas a máscara arterial
final (após fechamento e dilatação) é avaliada, para RG e FC. Não há novas medidas
da aorta externa ou das máscaras arteriais intermediárias.
Na retomada sem `--config-file`, a opção é recuperada do snapshot original do
run; uma flag explícita prevalece. Lotes concluídos não são recalculados, portanto
uma retomada pode conter exames com e sem essas medidas.

O filtro Hausdorff do SimpleITK calcula HD e AvgHD sobre as máscaras inteiras
(`artery_hd_foreground_mm` e `artery_avghd_foreground_mm`). Para o HD95
(`artery_hd95_surface_mm`), os centros dos voxels da superfície são obtidos por
erosão de seis vizinhos com exterior igual a fundo. Calcula-se o percentil 95
das distâncias mínimas em cada sentido, usando o maior dos dois. Todas as
distâncias usam o espaçamento efetivo em milímetros. As antigas colunas
`artery_hd_mm` e `artery_ahd_mm` permanecem legíveis nos resultados históricos.
Os CSVs registram o estado em `artery_distance_metrics_status` e o detalhe de
falhas em `artery_distance_metrics_error`.

Os status possíveis são `disabled`, `ok`, `prediction_empty`, `reference_empty`,
`both_empty`, `not_evaluated` e `error`. Máscaras não vazias idênticas dão zero;
qualquer máscara vazia deixa as distâncias indisponíveis, inclusive ambas vazias.
Erros da avaliação não descartam a segmentação nem alteram seu Dice.

Resumos apresentam média, DP amostral, mediana e N válido, além das contagens de
status. Essas estatísticas são condicionadas às distâncias disponíveis: sempre
informar a cobertura, especialmente quando houver falhas. Runs antigos ficam sem
esses valores, sem imputação de zero. Nenhuma máscara adicional é salva.
O cálculo usa CPU e aumenta o tempo/memória da avaliação; tempos com a opção
ativada não são diretamente comparáveis a runs sem ela.

Em `utils.utils.metrics`, `hausdorff_distances(prediction, reference, spacing)`
retorna o par (HD, AvgHD) do SimpleITK para máscaras 3D alinhadas e não vazias.
A função `surface_hd95` calcula o HD95 para máscaras booleanas.
Os wrappers de segmentação aceitam `spacing=` opcional: ele é obrigatório para
produzir distâncias em mm quando a avaliação está ativada. Os pipelines em lote
passam esse espaçamento automaticamente.

## Referência das funções

Os exemplos abaixo continuam o estado do exemplo sintético desta página, salvo indicação de entrada real. Os tipos `Any`, `NDArray` e `Sequence` reproduzem as anotações do código; os contratos de conteúdo estão nas tabelas.

### `get_artery_postprocessing_stages`

Binariza mask > 0, fecha com elemento esférico e dilata. Mantém a morfologia em CPU.

Importação: `from utils.segmentation.pipeline.arteries import get_artery_postprocessing_stages`.

```text
get_artery_postprocessing_stages(
    mask: Any,
    config: Dict[str, Any],
    *,
    closing_radius: Optional[int] = None,
    dilation_radius: Optional[int] = None,
) -> Dict[str, np.ndarray]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `mask` | `Any` | Sim | Máscara candidata/binária no mesmo shape da imagem correspondente. |
| `config` | `Dict[str, Any]` | Sim | Configuração completa normalizada e escalada uma única vez; chaves consumidas descritas nesta etapa. |
| `closing_radius` | `Optional[int]` | `None` | Raio do fechamento esférico em voxels; None lê POSTPROCESSING.closing_radius. |
| `dilation_radius` | `Optional[int]` | `None` | Raio da dilatação esférica em voxels; None lê POSTPROCESSING.dilation_radius. |

**Saída:** Dict com raw_mask, closed_mask e final_mask; todos ndarray uint8 0/1 no shape da entrada.

```python
custom = get_artery_postprocessing_stages(
    raw_mask, config, closing_radius=1, dilation_radius=0
)
assert np.array_equal(custom["closed_mask"], custom["final_mask"])
```

### `evaluate_artery_postprocessing_stages`

Avalia máscaras binárias já produzidas, sem aplicar morfologia ou alterá-las.
Recebe `raw_mask`, `closed_mask` e `final_mask`; `label_artery=None` desativa o
cálculo de Dice e `spacing=None` desativa os volumes em mL.

```python
from utils.segmentation.pipeline.arteries import evaluate_artery_postprocessing_stages

metrics = evaluate_artery_postprocessing_stages(
    stages["raw_mask"], stages["closed_mask"], stages["final_mask"],
    label_artery=label, spacing=(0.5, 0.5, 1.0)
)
print(metrics["dice_artery_after_closing"])
print(metrics["dice_artery_closing_delta"], metrics["dice_artery_dilation_delta"])
```

### `postprocess_artery_mask`

Atalho para obter somente o último estágio do mesmo pós-processamento.

Importação: `from utils.segmentation.pipeline.arteries import postprocess_artery_mask`.

```text
postprocess_artery_mask(
    mask: Any,
    config: Dict[str, Any],
    *,
    closing_radius: Optional[int] = None,
    dilation_radius: Optional[int] = None,
) -> np.ndarray
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `mask` | `Any` | Sim | Máscara candidata/binária no mesmo shape da imagem correspondente. |
| `config` | `Dict[str, Any]` | Sim | Configuração completa normalizada e escalada uma única vez; chaves consumidas descritas nesta etapa. |
| `closing_radius` | `Optional[int]` | `None` | Raio do fechamento esférico em voxels; None lê POSTPROCESSING.closing_radius. |
| `dilation_radius` | `Optional[int]` | `None` | Raio da dilatação esférica em voxels; None lê POSTPROCESSING.dilation_radius. |

**Saída:** ndarray uint8 final, sem métricas.

```python
from utils.segmentation.pipeline.arteries import postprocess_artery_mask

final_mask = postprocess_artery_mask(raw_mask, config)
assert np.array_equal(final_mask, stages["final_mask"])
```

### `dice_score`

Compara máscaras binárias usando 2|A∩B| / (|A|+|B|), após binarizar entradas > 0.

Importação: `from utils.utils.metrics import dice_score`.

```text
dice_score(pred: NDArray[Any], target: NDArray[Any]) -> float
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `pred` | `NDArray[Any]` | Sim | Máscara predita; deve ter o mesmo shape e geometria de target. |
| `target` | `NDArray[Any]` | Sim | Máscara binária de referência alinhada à predição. |

**Saída:** Float em [0,1]. Duas máscaras vazias retornam 1.0; apenas uma vazia retorna 0.0.

```python
score = dice_score(stages["final_mask"], label)
assert 0 <= score <= 1
assert dice_score(np.zeros((2, 2)), np.zeros((2, 2))) == 1.0
```

### `binary_segmentation_metrics`

Acrescenta contagens, sensibilidade e precisão ao Dice.

Importação: `from utils.utils.metrics import binary_segmentation_metrics`.

```text
binary_segmentation_metrics(
    prediction: NDArray[Any], ground_truth: NDArray[Any]
) -> Dict[str, float | int]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `prediction` | `NDArray[Any]` | Sim | Máscara predita, binarizada com > 0. |
| `ground_truth` | `NDArray[Any]` | Sim | Referência binária com mesmo shape e geometria. |

**Saída:** Dict: dice, predicted_voxels, ground_truth_voxels, true_positives, sensitivity e precision. Sensibilidade é NaN se a referência está vazia; precisão é NaN se não há predição positiva.

```python
metrics = binary_segmentation_metrics(stages["final_mask"], label)
assert metrics["predicted_voxels"] == int(stages["final_mask"].sum())
```

### Parâmetros e interpretação

O JSON de referência usa `POSTPROCESSING.closing_radius=3` e `dilation_radius=2`, em voxels da resolução de trabalho. O ajuste high altera esses raios. Os argumentos explícitos closing_radius/dilation_radius substituem o respectivo valor configurado; 0 usa um elemento de raio zero, sem expansão espacial naquela operação.

Fechamento e dilatação podem aumentar ou reduzir o Dice dependendo da anatomia e da referência; a dilatação não é uma melhora garantida. Compare os estágios com a mesma referência. Não aplique esta etapa novamente ao resultado final retornado pelas funções de segmentação arterial.

Para inferência sem referência, relate máscara e contagens. Dice ausente não deve ser substituído por zero. `mask.sum() * np.prod(scaled_spacing)` fornece volume em mm³, se o espaçamento corresponder aos eixos e à resolução da máscara. Não compare volumes brutos de voxels entre resoluções diferentes.

## Falhas e cuidados

- Confira shape, resolução, orientação e anatomia representada antes de comparar máscaras. Os helpers de métricas não verificam geometria NIfTI e não impedem todo broadcasting NumPy.
- Raios são em voxels: em imagens anisotrópicas, um elemento esférico não tem alcance físico igual em todos os eixos.
- Duas máscaras vazias terem Dice 1.0 é uma convenção matemática do helper; isso não comprova sucesso da segmentação.
- Labels coronarianos ImageCAS usam classe 1. Máscaras de outras estruturas não são referências substitutas.
