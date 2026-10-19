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
