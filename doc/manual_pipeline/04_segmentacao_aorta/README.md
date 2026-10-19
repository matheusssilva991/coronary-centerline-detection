# 04 — Segmentação da aorta

[Manual](../README.md) · [Configuração](../../../config/README.md)

Transforma a trajetória circular da etapa anterior em máscara 3D. A máscara é usada para extrair a superfície onde serão buscados os óstios; por isso, vazamentos ou cortes da aorta afetam as etapas seguintes.

## Exemplo sintético

Execute a partir da raiz do repositório com `PYTHONPATH=src uv run python seu_exemplo.py`. Este exemplo usa CPU e não representa validação anatômica.

```python
import numpy as np
from utils.project.config import load_config_json, scale_config_to_resolution

config = load_config_json("config/pipeline_config.json", {})
config["USE_GPU"] = False
config = scale_config_to_resolution(config)

from copy import deepcopy
from utils.segmentation.pipeline.detection import segment_aorta_with_diagnostics

rows, columns = np.ogrid[:32, :32]
disk = (rows - 16) ** 2 + (columns - 16) ** 2 < 8**2
image = np.repeat((disk * 300).astype(np.float32)[..., None], 8, axis=2)
circles = [
    dict(slice_index=z, center_x=16.0, center_y=16.0, radius=8.0) for z in range(8)
]
# Poucas iterações e raios menores apenas para verificar a API em CPU.
level_config = deepcopy(config["LEVEL_SET"])
level_config.update(
    num_iter=2,
    radius_reduction_factor=0.5,
    leak_removal_radius=0,
    trajectory_axial_margin_slices=0,
)
segmentation = segment_aorta_with_diagnostics(image, circles, level_config)
aorta_mask = segmentation.mask
assert aorta_mask.shape == image.shape and aorta_mask.dtype == np.uint8
assert set(np.unique(aorta_mask)) <= {0, 1}
assert segmentation.diagnostics["aorta_level_set_iterations_used"] == 2
print(aorta_mask.shape, segmentation.diagnostics)
```

## Referência das funções

Os exemplos abaixo continuam o estado do exemplo sintético desta página, salvo indicação de entrada real. Os tipos `Any`, `NDArray` e `Sequence` reproduzem as anotações do código; os contratos de conteúdo estão nas tabelas.

### Fluxo e saídas

O wrapper inicializa discos, calcula o indicador de borda e evolui MorphGAC por um número fixo de iterações. Depois aplica abertura, envelope opcional e maior componente 3D. O resultado permanece na geometria completa, mesmo quando o cálculo usa ROI.

`AortaSegmentationResult` possui `mask` (uint8 3D, 0/1) e `diagnostics` (dict). As chaves diagnósticas são:

| Chave | Conteúdo |
|---|---|
| `aorta_level_set_initial_voxel_count` / `aorta_level_set_raw_voxel_count` | Contagens antes da evolução e antes do pós-processamento. |
| `aorta_level_set_initial_volume_fraction` / `aorta_level_set_raw_volume_fraction` | Contagens divididas pelo total de voxels da imagem. |
| `aorta_level_set_iterations_used` | Número configurado de iterações; também é registrado se não houver círculos. |
| `aorta_level_set_circle_fill_q25` | Quantil 0.25 do preenchimento dos discos pela máscara final. |
| `aorta_level_set_circle_area_ratio_p90` | Quantil 0.90 da área segmentada por fatia dividida por πr². |
| `aorta_slice_area_jump_p95` | Quantil 0.95 dos saltos relativos de área entre fatias. |

O JSON de referência usa fator inicial **0.1** (10% do raio, não redução de 10%), 26 iterações, balloon 0.6, smoothing 2, alpha 1000, sigma 2, ROI com margem 10, abertura de raio 2 e envelope 2.25r com margem axial 10. O ajuste high aumenta iterações e parâmetros espaciais: use a configuração efetivamente escalada.

### `segment_aorta_with_diagnostics`

Interface recomendada para a máscara final e diagnósticos do level set.

Importação: `from utils.segmentation.pipeline.detection import segment_aorta_with_diagnostics`.

```text
segment_aorta_with_diagnostics(
    lcc_image: Any,
    detected_circles: List[Dict[str, Any]],
    level_set_config: Dict[str, Any],
    use_gpu: bool = False,
) -> AortaSegmentationResult
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `lcc_image` | `Any` | Sim | Volume pré-processado 3D, com HU preservados na região selecionada. |
| `detected_circles` | `List[Dict[str, Any]]` | Sim | Círculos com slice_index, center_x, center_y e radius, na resolução de trabalho. |
| `level_set_config` | `Dict[str, Any]` | Sim | Bloco LEVEL_SET da configuração efetiva. |
| `use_gpu` | `bool` | `False` | Solicita GPU nas operações que a implementam; os wrappers de aorta mantêm execução em CPU. |

**Saída:** AortaSegmentationResult. Sem círculos, máscara vazia e medidas de círculo None.

```python
assert segmentation.diagnostics["aorta_level_set_raw_voxel_count"] >= 0
```

`use_gpu` é aceito por compatibilidade, mas ignorado neste wrapper: evolução e pós-processamento são mantidos em CPU.

### `segment_aorta`

Atalho com a mesma segmentação e pós-processamento, retornando apenas a máscara.

Importação: `from utils.segmentation.pipeline.detection import segment_aorta`.

```text
segment_aorta(
    lcc_image: Any,
    detected_circles: List[Dict[str, Any]],
    level_set_config: Dict[str, Any],
    use_gpu: bool = False,
) -> Any
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `lcc_image` | `Any` | Sim | Volume pré-processado 3D, com HU preservados na região selecionada. |
| `detected_circles` | `List[Dict[str, Any]]` | Sim | Círculos com slice_index, center_x, center_y e radius, na resolução de trabalho. |
| `level_set_config` | `Dict[str, Any]` | Sim | Bloco LEVEL_SET da configuração efetiva. |
| `use_gpu` | `bool` | `False` | Solicita GPU nas operações que a implementam; os wrappers de aorta mantêm execução em CPU. |

**Saída:** Máscara uint8 3D com o shape da imagem.

```python
from utils.segmentation.pipeline.detection import segment_aorta

mask_only = segment_aorta(image, circles, level_config, use_gpu=False)
assert mask_only.shape == image.shape
```

### `initialize_level_set_mask`

Reconstrói somente os discos iniciais no volume completo, sem gradiente/evolução.

Importação: `from utils.segmentation.aorta.segmentation import initialize_level_set_mask`.

```text
initialize_level_set_mask(
    volume_shape: Sequence[int],
    detected_circles: Sequence[Dict[str, Any]],
    *,
    radius_reduction_factor: float,
    roi_margin: int = 10,
    use_roi: bool = True,
) -> NDArray[np.uint8]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `volume_shape` | `Sequence[int]` | Sim | Shape completo de saída, com três dimensões. |
| `detected_circles` | `Sequence[Dict[str, Any]]` | Sim | Círculos com slice_index, center_x, center_y e radius, na resolução de trabalho. |
| `radius_reduction_factor` | `float` | Sim | Multiplicador do raio detectado para inicializar discos; 0.1 significa 10% do raio. |
| `roi_margin` | `int` | `10` | Margem da ROI em voxels da resolução de trabalho. |
| `use_roi` | `bool` | `True` | Restringe o cálculo à ROI e restaura a máscara para o shape completo ao retornar. |

**Saída:** Máscara uint8 3D; lista de círculos vazia produz máscara vazia.

```python
from utils.segmentation.aorta.segmentation import initialize_level_set_mask

seeds = initialize_level_set_mask(image.shape, circles, radius_reduction_factor=0.5)
assert seeds.dtype == np.uint8 and seeds.sum() > 0
```

### `prepare_level_set_evolution`

Prepara indicador de borda, ROI e estado para evolução manual.

Importação: `from utils.segmentation.aorta.segmentation import prepare_level_set_evolution`.

```text
prepare_level_set_evolution(
    volume_ccta: NDArray[Any],
    detected_circles: Sequence[Dict[str, Any]],
    *,
    radius_reduction_factor: float = 0.8,
    roi_margin: int = 10,
    use_roi: bool = True,
    alpha: float = 1000,
    sigma: float = 2,
    use_gpu: bool = False,
) -> LevelSetEvolutionContext
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `volume_ccta` | `NDArray[Any]` | Sim | Volume pré-processado 3D na resolução de trabalho. |
| `detected_circles` | `Sequence[Dict[str, Any]]` | Sim | Círculos com slice_index, center_x, center_y e radius, na resolução de trabalho. |
| `radius_reduction_factor` | `float` | `0.8` | Multiplicador do raio detectado para inicializar discos; 0.1 significa 10% do raio. |
| `roi_margin` | `int` | `10` | Margem da ROI em voxels da resolução de trabalho. |
| `use_roi` | `bool` | `True` | Restringe o cálculo à ROI e restaura a máscara para o shape completo ao retornar. |
| `alpha` | `float` | `1000` | Sensibilidade do indicador inverso de gradiente às bordas. |
| `sigma` | `float` | `2` | Escala da suavização gaussiana em voxels antes do indicador de borda. |
| `use_gpu` | `bool` | `False` | Solicita GPU nas operações que a implementam; os wrappers de aorta mantêm execução em CPU. |

**Saída:** LevelSetEvolutionContext com gimage, current_mask (na ROI), volume_shape, roi_bounds e completed_iterations.

```python
from utils.segmentation.aorta.segmentation import prepare_level_set_evolution

context = prepare_level_set_evolution(
    image, circles, radius_reduction_factor=0.5, use_gpu=False
)
first = context.evolve(1, smoothing=1, balloon=0.6, threshold="auto")
second = context.evolve(1, smoothing=1, balloon=0.6, threshold="auto")
assert context.completed_iterations == 2
assert context.full_mask().shape == image.shape
```

`context.evolve(num_iter, *, smoothing, balloon, threshold)` altera o estado e retorna a máscara completa. `num_iter` é obrigatório, inteiro não negativo; smoothing, balloon e threshold também são obrigatórios. `full_mask()` não recebe parâmetros e restaura o estado corrente. Duas chamadas de 1 passo representam duas evoluções adicionais. Esses métodos não aplicam abertura, envelope ou seleção da maior componente.

### `level_set_segmentation`

Executa somente a evolução; interface de baixo nível que não aplica o pós-processamento do wrapper.

Importação: `from utils.segmentation.aorta.segmentation import level_set_segmentation`.

```text
level_set_segmentation(
    volume_ccta: NDArray[Any],
    detected_circles: Sequence[Dict[str, Any]],
    num_iter: int = 50,
    smoothing: int = 1,
    balloon: int = 1,
    threshold: Any = "auto",
    radius_reduction_factor: float = 0.8,
    roi_margin: int = 10,
    use_roi: bool = True,
    alpha: float = 1000,
    sigma: float = 2,
    use_gpu: bool = False,
) -> NDArray[Any]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `volume_ccta` | `NDArray[Any]` | Sim | Volume pré-processado 3D na resolução de trabalho. |
| `detected_circles` | `Sequence[Dict[str, Any]]` | Sim | Círculos com slice_index, center_x, center_y e radius, na resolução de trabalho. |
| `num_iter` | `int` | `50` | Número de iterações de evolução; no contexto reutilizável, são iterações adicionais. |
| `smoothing` | `int` | `1` | Número de suavizações morfológicas por passo de evolução. |
| `balloon` | `int` | `1` | Força do contorno: positiva expande, negativa contrai. |
| `threshold` | `Any` | `'auto'` | Limiar do indicador de borda, numérico ou auto. |
| `radius_reduction_factor` | `float` | `0.8` | Multiplicador do raio detectado para inicializar discos; 0.1 significa 10% do raio. |
| `roi_margin` | `int` | `10` | Margem da ROI em voxels da resolução de trabalho. |
| `use_roi` | `bool` | `True` | Restringe o cálculo à ROI e restaura a máscara para o shape completo ao retornar. |
| `alpha` | `float` | `1000` | Sensibilidade do indicador inverso de gradiente às bordas. |
| `sigma` | `float` | `2` | Escala da suavização gaussiana em voxels antes do indicador de borda. |
| `use_gpu` | `bool` | `False` | Solicita GPU nas operações que a implementam; os wrappers de aorta mantêm execução em CPU. |

**Saída:** Máscara 0/1 com shape completo; uint8 no caminho de evolução e int8 no caso sem círculos.

```python
from utils.segmentation.aorta.segmentation import level_set_segmentation

evolved = level_set_segmentation(
    image, circles, num_iter=2, radius_reduction_factor=0.5, use_gpu=False
)
assert evolved.shape == image.shape
```

O default local é 50 iterações e raio inicial 0.8; não equivale ao bloco LEVEL_SET atual. As funções de baixo nível podem usar GPU na preparação do gradiente; o wrapper principal fixa CPU.

### `remove_leaks_morphology`

Aplica abertura com elemento esférico 3D para remover conexões finas.

Importação: `from utils.segmentation.aorta.segmentation import remove_leaks_morphology`.

```text
remove_leaks_morphology(
    mask_3d: NDArray[Any], radius: int = 3, use_gpu: bool = False
) -> NDArray[Any]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `mask_3d` | `NDArray[Any]` | Sim | Máscara binária 3D a limpar. |
| `radius` | `int` | `3` | Raio em voxels; o wrapper usa LEVEL_SET.leak_removal_radius. |
| `use_gpu` | `bool` | `False` | Solicita GPU nas operações que a implementam; os wrappers de aorta mantêm execução em CPU. |

**Saída:** Máscara com mesmo shape; raio <= 0 devolve uma cópia, sem abertura. O dtype pode depender do backend; o wrapper converte ao final para uint8.

```python
from utils.segmentation.aorta.segmentation import remove_leaks_morphology

opened = remove_leaks_morphology(seeds, radius=1, use_gpu=False)
assert opened.shape == image.shape
```

### `build_circle_trajectory_envelope`

Interpola centros/raios nas fatias intermediárias e prolonga os extremos pela margem axial.

Importação: `from utils.segmentation.aorta.segmentation import build_circle_trajectory_envelope`.

```text
build_circle_trajectory_envelope(
    volume_shape: Sequence[int],
    detected_circles: Sequence[Dict[str, Any]],
    radius_factor: float = 1.5,
    axial_margin_slices: int = 0,
) -> NDArray[np.uint8]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `volume_shape` | `Sequence[int]` | Sim | Shape completo de saída, com três dimensões. |
| `detected_circles` | `Sequence[Dict[str, Any]]` | Sim | Círculos com slice_index, center_x, center_y e radius, na resolução de trabalho. |
| `radius_factor` | `float` | `1.5` | Multiplicador do raio dos círculos; deve ser positivo para o envelope. |
| `axial_margin_slices` | `int` | `0` | Prolongamento dos extremos da trajetória em número de fatias, não negativo. |

**Saída:** Envelope uint8 3D; sem círculos válidos, máscara vazia.

```python
from utils.segmentation.aorta.segmentation import build_circle_trajectory_envelope

envelope = build_circle_trajectory_envelope(
    image.shape, circles, radius_factor=2.25, axial_margin_slices=0
)
assert envelope.shape == image.shape
```

### `restrict_mask_to_circle_trajectory`

Intersecciona uma máscara com o envelope da trajetória.

Importação: `from utils.segmentation.aorta.segmentation import restrict_mask_to_circle_trajectory`.

```text
restrict_mask_to_circle_trajectory(
    aorta_mask: NDArray[Any],
    detected_circles: Sequence[Dict[str, Any]],
    radius_factor: float,
    axial_margin_slices: int = 0,
) -> NDArray[np.uint8]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `aorta_mask` | `NDArray[Any]` | Sim | Máscara binária 3D da aorta, com o mesmo shape dos demais volumes da etapa. |
| `detected_circles` | `Sequence[Dict[str, Any]]` | Sim | Círculos com slice_index, center_x, center_y e radius, na resolução de trabalho. |
| `radius_factor` | `float` | Sim | Multiplicador do raio dos círculos; deve ser positivo para o envelope. |
| `axial_margin_slices` | `int` | `0` | Prolongamento dos extremos da trajetória em número de fatias, não negativo. |

**Saída:** Máscara uint8 3D restrita ao envelope.

```python
from utils.segmentation.aorta.segmentation import restrict_mask_to_circle_trajectory

restricted = restrict_mask_to_circle_trajectory(aorta_mask, circles, radius_factor=2.25)
assert np.all(restricted <= aorta_mask)
```

### `calculate_circle_mask_profile`

Mede preenchimento e razão de área em cada fatia rastreada.

Importação: `from utils.segmentation.aorta.segmentation import calculate_circle_mask_profile`.

```text
calculate_circle_mask_profile(
    aorta_mask: NDArray[Any], detected_circles: Sequence[Dict[str, Any]]
) -> list[Dict[str, float | int]]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `aorta_mask` | `NDArray[Any]` | Sim | Máscara binária 3D da aorta, com o mesmo shape dos demais volumes da etapa. |
| `detected_circles` | `Sequence[Dict[str, Any]]` | Sim | Círculos com slice_index, center_x, center_y e radius, na resolução de trabalho. |

**Saída:** Lista de dicts com slice_index, circle_fill_ratio e circle_area_ratio; fatias fora do volume são ignoradas.

```python
from utils.segmentation.aorta.segmentation import calculate_circle_mask_profile

profile = calculate_circle_mask_profile(aorta_mask, circles)
assert len(profile) == len(circles)
```

### `calculate_circle_mask_metrics`

Resume o perfil por quantis de preenchimento e área relativa.

Importação: `from utils.segmentation.aorta.segmentation import calculate_circle_mask_metrics`.

```text
calculate_circle_mask_metrics(
    aorta_mask: NDArray[Any], detected_circles: Sequence[Dict[str, Any]]
) -> Dict[str, float | None]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `aorta_mask` | `NDArray[Any]` | Sim | Máscara binária 3D da aorta, com o mesmo shape dos demais volumes da etapa. |
| `detected_circles` | `Sequence[Dict[str, Any]]` | Sim | Círculos com slice_index, center_x, center_y e radius, na resolução de trabalho. |

**Saída:** Dict com circle_fill_q25 e circle_area_ratio_p90; valores None quando não há perfil.

```python
from utils.segmentation.aorta.segmentation import calculate_circle_mask_metrics

metrics = calculate_circle_mask_metrics(aorta_mask, circles)
assert set(metrics) == {"circle_fill_q25", "circle_area_ratio_p90"}
```

### `calculate_slice_area_jump_p95`

Mede descontinuidade axial relativa, considerando inclusive lacunas internas.

Importação: `from utils.segmentation.aorta.segmentation import calculate_slice_area_jump_p95`.

```text
calculate_slice_area_jump_p95(aorta_mask: NDArray[Any]) -> float
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `aorta_mask` | `NDArray[Any]` | Sim | Máscara binária 3D da aorta, com o mesmo shape dos demais volumes da etapa. |

**Saída:** Float; 0.0 quando menos de duas fatias estão ocupadas.

```python
from utils.segmentation.aorta.segmentation import calculate_slice_area_jump_p95

jump = calculate_slice_area_jump_p95(aorta_mask)
assert 0 <= jump <= 1
```

### `classify_aorta_segmentation_feedback`

Gera um alerta com base nos diagnósticos sem modificar a máscara nem repetir a segmentação.

Importação: `from utils.segmentation.aorta.segmentation import classify_aorta_segmentation_feedback`.

```text
classify_aorta_segmentation_feedback(
    circle_fill_q25: float | None,
    circle_area_ratio_p90: float | None,
    volume_fraction: float | None,
    feedback_config: Dict[str, Any] | None = None,
) -> str
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `circle_fill_q25` | `float \| None` | Sim | Quantil 0.25 de preenchimento dos discos; None indica dado indisponível. |
| `circle_area_ratio_p90` | `float \| None` | Sim | Quantil 0.90 da área relativa; None indica dado indisponível. |
| `volume_fraction` | `float \| None` | Sim | Voxels da máscara final / voxels da imagem completa. |
| `feedback_config` | `Dict[str, Any] \| None` | `None` | Bloco LEVEL_SET.quality_feedback; None usa limites internos. |

**Saída:** adequate, suspected_undersegmentation, suspected_oversegmentation ou insufficient_data.

```python
from utils.segmentation.aorta.segmentation import classify_aorta_segmentation_feedback

feedback = classify_aorta_segmentation_feedback(
    metrics["circle_fill_q25"],
    metrics["circle_area_ratio_p90"],
    float(aorta_mask.mean()),
    config["LEVEL_SET"]["quality_feedback"],
)
assert isinstance(feedback, str)
```

## Falhas e cuidados

- prepare_level_set_evolution exige volume 3D e pelo menos um círculo. Os wrappers segment_aorta/segment_aorta_with_diagnostics aceitam trajetória vazia e produzem máscara vazia.
- Círculos devem estar no mesmo sistema de índices e resolução da imagem.
- O envelope limita vazamentos, mas pode cortar anatomia se a trajetória estiver incorreta.
- Feedback é um alerta heurístico; adequate não comprova validade anatômica. Não há realimentação que altere a máscara no wrapper atual.
- Não interprete uma máscara inicial reduzida como segmentação da aorta; ela é apenas a semente da evolução.
