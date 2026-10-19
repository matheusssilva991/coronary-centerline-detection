# 03 — Localização da aorta

[Manual](../README.md) · [Configuração](../../../config/README.md)

Localiza uma trajetória circular na imagem pré-processada para inicializar a segmentação da aorta. Os raios são em pixels; as tolerâncias principais são em mm. O detector usa CPU e uma hipótese de quadrante/orientação herdada do ImageCAS.

## Exemplo sintético

Execute a partir da raiz do repositório com `PYTHONPATH=src uv run python seu_exemplo.py`. Este exemplo usa CPU e não representa validação anatômica.

```python
import numpy as np
from utils.project.config import load_config_json, scale_config_to_resolution

config = load_config_json("config/pipeline_config.json", {})
config["USE_GPU"] = False
config = scale_config_to_resolution(config)

from copy import deepcopy
from utils.segmentation.pipeline.detection import locate_and_filter_aorta_circles

rows, columns = np.ogrid[:64, :64]
disk = (rows - 20) ** 2 + (columns - 44) ** 2 <= 8**2
image = np.repeat((disk * 300).astype(np.float32)[..., None], 6, axis=2)
# Grade pequena apenas para este objeto sintético; não é ajuste clínico.
circle_config = deepcopy(config["CIRCLE_DETECTION"])
circle_config.update(
    radii_start_px=6,
    radii_end_px=11,
    radius_step_px=1,
    quadrant_offset=(0, 0),
    canny_sigma=1,
)
tracking = locate_and_filter_aorta_circles(
    image,
    (1, 1, 1),
    (1.0, 1.0, 1.0),
    circle_config,
)
circles = tracking.filtered_circles
assert len(tracking.original_circles) > 0
assert all("slice_index" in circle for circle in circles)
print(len(tracking.original_circles), len(circles))
```

## Referência das funções

Os exemplos abaixo continuam o estado do exemplo sintético desta página, salvo indicação de entrada real. Os tipos `Any`, `NDArray` e `Sequence` reproduzem as anotações do código; os contratos de conteúdo estão nas tabelas.

### Contrato dos círculos e do resultado

Cada círculo é um dicionário com `slice_index` (int), `center_x` (coluna/eixo 1), `center_y` (linha/eixo 0) e `radius` (pixels). `accum` registra a resposta Hough quando disponível. Círculos interpolados podem ter `interpolated=True`; a extrapolação de cauda marca `trajectory_filtered=True` e `trajectory_filter_action="extrapolated_stable_tail"`. Não exija esses campos opcionais em todos os registros.

`AortaCircleTrackingResult` contém `original_circles`, `filtered_circles` e `filter_diagnostics`. Use a lista filtrada nas etapas seguintes; preserve a original para interpretar o detector. A ordem do rastreamento é z decrescente, começando pela **última fatia**. A ausência de círculo inicial nessa fatia retorna lista vazia.

### `locate_and_filter_aorta_circles`

Executa localização e filtro robusto em uma chamada; interface recomendada para o fluxo completo.

Importação: `from utils.segmentation.pipeline.detection import locate_and_filter_aorta_circles`.

```text
locate_and_filter_aorta_circles(
    lcc_image: Any,
    downscale_factors: Sequence[int],
    scaled_spacing: Sequence[float],
    circle_config: Dict[str, Any],
) -> AortaCircleTrackingResult
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `lcc_image` | `Any` | Sim | Volume pré-processado 3D, com HU preservados na região selecionada. |
| `downscale_factors` | `Sequence[int]` | Sim | Divisores por eixo; (2, 2, 1) reduz os dois primeiros e mantém o terceiro. |
| `scaled_spacing` | `Sequence[float]` | Sim | Espaçamento após downscale, na ordem dos eixos do pré-processamento, em mm. |
| `circle_config` | `Dict[str, Any]` | Sim | Bloco CIRCLE_DETECTION da configuração efetiva. |

**Saída:** AortaCircleTrackingResult com as duas trajetórias e diagnósticos.

```python
assert tracking.filter_diagnostics["aorta_circle_original_count"] == len(
    tracking.original_circles
)
```

### `locate_aorta_circles`

Converte a configuração em raios candidatos e chama a Hough em CPU. Usa a média dos dois primeiros valores de scaled_spacing.

Importação: `from utils.segmentation.pipeline.detection import locate_aorta_circles`.

```text
locate_aorta_circles(
    lcc_image: Any,
    downscale_factors: Sequence[int],
    scaled_spacing: Sequence[float],
    circle_config: Dict[str, Any],
) -> List[Dict[str, Any]]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `lcc_image` | `Any` | Sim | Volume pré-processado 3D, com HU preservados na região selecionada. |
| `downscale_factors` | `Sequence[int]` | Sim | Divisores por eixo; (2, 2, 1) reduz os dois primeiros e mantém o terceiro. |
| `scaled_spacing` | `Sequence[float]` | Sim | Espaçamento após downscale, na ordem dos eixos do pré-processamento, em mm. |
| `circle_config` | `Dict[str, Any]` | Sim | Bloco CIRCLE_DETECTION da configuração efetiva. |

**Saída:** Lista de círculos originais, ainda sem filtro de trajetória.

```python
from utils.segmentation.pipeline.detection import locate_aorta_circles

original = locate_aorta_circles(image, (1, 1, 1), (1, 1, 1), circle_config)
assert isinstance(original, list)
```

`downscale_factors` é mantido por compatibilidade, mas ignorado internamente: os raios devem chegar já escalados. Não multiplique/divida os raios novamente.

### `filter_located_aorta_circles`

Wrapper para filtrar uma trajetória existente usando a configuração do pipeline.

Importação: `from utils.segmentation.pipeline.detection import filter_located_aorta_circles`.

```text
filter_located_aorta_circles(
    detected_circles: Sequence[Dict[str, Any]],
    scaled_spacing: Sequence[float],
    image_slice_count: int,
    circle_config: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `detected_circles` | `Sequence[Dict[str, Any]]` | Sim | Círculos com slice_index, center_x, center_y e radius, na resolução de trabalho. |
| `scaled_spacing` | `Sequence[float]` | Sim | Espaçamento após downscale, na ordem dos eixos do pré-processamento, em mm. |
| `image_slice_count` | `int` | Sim | Número total de fatias do volume; denominador da cobertura axial. |
| `circle_config` | `Dict[str, Any]` | Sim | Bloco CIRCLE_DETECTION da configuração efetiva. |

**Saída:** (círculos filtrados, dict de diagnósticos).

```python
from utils.segmentation.pipeline.detection import filter_located_aorta_circles

filtered, filter_details = filter_located_aorta_circles(
    original, (1, 1, 1), image.shape[2], circle_config
)
```

### `detect_aorta_circles`

Detecta bordas Canny, seleciona o círculo inicial no quadrante configurado e acompanha centro/raio nas fatias anteriores.

Importação: `from utils.segmentation.aorta.localization import detect_aorta_circles`.

```text
detect_aorta_circles(
    img_volume: NDArray[Any],
    hough_radii: Sequence[float],
    pixel_spacing: float,
    tol_radius_mm: float = 9.0,
    tol_distance_mm: float = 18.0,
    max_slice_miss_threshold: int = 5,
    neighbor_distance_threshold: float = 5,
    quadrant_offset: Sequence[int] = (30, 30),
    total_num_peaks_initial: int = 10,
    total_num_peaks: int = 8,
    canny_sigma: float = 3,
    use_local_roi: bool = True,
    local_roi_padding: int = 20,
    interpolate_missed_circles: bool = True,
    verbose: bool = True,
) -> list
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `img_volume` | `NDArray[Any]` | Sim | Volume 3D pré-processado; não é o mapa de vasos. |
| `hough_radii` | `Sequence[float]` | Sim | Raios candidatos em pixels; np.arange tem fim exclusivo. |
| `pixel_spacing` | `float` | Sim | Espaçamento médio no plano axial, em mm/pixel. |
| `tol_radius_mm` | `float` | `9.0` | Máxima diferença de raio entre detecções consecutivas, em mm. |
| `tol_distance_mm` | `float` | `18.0` | Máxima distância entre centros consecutivos, em mm. |
| `max_slice_miss_threshold` | `int` | `5` | Para após atingir esse número de falhas consecutivas. |
| `neighbor_distance_threshold` | `float` | `5` | Distância máxima entre centros candidatos para refinamento local, em pixels. |
| `quadrant_offset` | `Sequence[int]` | `(30, 30)` | Deslocamentos em pixels dos limites do quadrante inicial. |
| `total_num_peaks_initial` | `int` | `10` | Número máximo de picos na busca/refinamento inicial. |
| `total_num_peaks` | `int` | `8` | Número máximo de picos de Hough considerados por fatia. |
| `canny_sigma` | `float` | `3` | Sigma da suavização do Canny, em pixels. |
| `use_local_roi` | `bool` | `True` | Restringe a busca ao redor do círculo anterior nas fatias subsequentes. |
| `local_roi_padding` | `int` | `20` | Margem da região local em pixels. |
| `interpolate_missed_circles` | `bool` | `True` | Interpola lacunas somente quando nova detecção válida reaparece antes da parada. |
| `verbose` | `bool` | `True` | Imprime mensagens de acompanhamento quando verdadeiro. |

**Saída:** Lista de círculos ou [] se não houver círculo inicial; lacunas curtas podem ser interpoladas.

```python
from utils.segmentation.aorta.localization import detect_aorta_circles

radii = np.arange(6, 11)
raw_circles = detect_aorta_circles(
    image,
    radii,
    pixel_spacing=1.0,
    quadrant_offset=(0, 0),
    canny_sigma=1,
    verbose=False,
)
assert len(raw_circles) > 0
```

### `filter_aorta_circle_trajectory`

Remove uma cauda persistentemente incompatível e pode extrapolar a trajetória estável.

Importação: `from utils.segmentation.aorta.localization import filter_aorta_circle_trajectory`.

```text
filter_aorta_circle_trajectory(
    detected_circles: Sequence[dict[str, Any]],
    pixel_spacing: float,
    image_slice_count: int,
    filter_config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `detected_circles` | `Sequence[dict[str, Any]]` | Sim | Círculos com slice_index, center_x, center_y e radius, na resolução de trabalho. |
| `pixel_spacing` | `float` | Sim | Espaçamento médio no plano axial, em mm/pixel. |
| `image_slice_count` | `int` | Sim | Total de fatias do volume, usado na cobertura. |
| `filter_config` | `dict[str, Any] \| None` | `None` | Configuração do filtro; None equivale a method=none. method=robust ativa corte/extrapolação. |

**Saída:** (lista filtrada em z decrescente, diagnósticos de contagem/cobertura/motivo). Não é uma nova execução da Hough.

```python
from utils.segmentation.aorta.localization import filter_aorta_circle_trajectory

unmodified, disabled = filter_aorta_circle_trajectory(
    raw_circles, 1.0, image.shape[2], {"method": "none"}
)
assert disabled["aorta_circle_filter_applied"] is False
```

### `detect_initial_circle`

Seleciona o círculo inicial em uma fatia 2D, sem rastrear o volume.

Importação: `from utils.segmentation.aorta.localization import detect_initial_circle`.

```text
detect_initial_circle(
    img_slice: NDArray[Any],
    hough_radii: Sequence[float],
    quadrant_offset: Sequence[int] = (30, 30),
    total_num_peaks: int = 10,
    canny_sigma: float = 3,
) -> Optional[dict]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `img_slice` | `NDArray[Any]` | Sim | Fatia 2D da imagem pré-processada; não recebe o mapa de vasos. |
| `hough_radii` | `Sequence[float]` | Sim | Raios candidatos em pixels; np.arange tem fim exclusivo. |
| `quadrant_offset` | `Sequence[int]` | `(30, 30)` | Deslocamentos em pixels dos limites do quadrante inicial. |
| `total_num_peaks` | `int` | `10` | Número máximo de picos de Hough considerados por fatia. |
| `canny_sigma` | `float` | `3` | Sigma da suavização do Canny, em pixels. |

**Saída:** Dicionário com center_x, center_y, radius e accum, ou None.

```python
from utils.segmentation.aorta.localization import detect_initial_circle

initial = detect_initial_circle(
    image[:, :, -1], radii, quadrant_offset=(0, 0), canny_sigma=1
)
assert initial is not None
```

### `get_initial_circle_diagnostics`

Expõe os candidatos da primeira fatia e o refinamento por vizinhança.

Importação: `from utils.segmentation.aorta.localization import get_initial_circle_diagnostics`.

```text
get_initial_circle_diagnostics(
    img_slice: NDArray[Any],
    hough_radii: Sequence[float],
    quadrant_offset: Sequence[int] = (30, 30),
    total_num_peaks_initial: int = 10,
    canny_sigma: float = 3,
    neighbor_distance_threshold: float = 5,
) -> dict
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `img_slice` | `NDArray[Any]` | Sim | Fatia 2D da imagem pré-processada; não recebe o mapa de vasos. |
| `hough_radii` | `Sequence[float]` | Sim | Raios candidatos em pixels; np.arange tem fim exclusivo. |
| `quadrant_offset` | `Sequence[int]` | `(30, 30)` | Deslocamentos em pixels dos limites do quadrante inicial. |
| `total_num_peaks_initial` | `int` | `10` | Número máximo de picos na busca/refinamento inicial. |
| `canny_sigma` | `float` | `3` | Sigma da suavização do Canny, em pixels. |
| `neighbor_distance_threshold` | `float` | `5` | Distância máxima entre centros candidatos para refinamento local, em pixels. |

**Saída:** Dict com initial_circle, refined_circle (ou None), candidates e refinement_candidates (listas).

```python
from utils.segmentation.aorta.localization import get_initial_circle_diagnostics

initial_details = get_initial_circle_diagnostics(
    image[:, :, -1], radii, quadrant_offset=(0, 0), canny_sigma=1
)
assert "refinement_candidates" in initial_details
```

### `refine_circle_with_neighbors`

Refina centro e raio pela média dos candidatos próximos a um centro de referência.

Importação: `from utils.segmentation.aorta.localization import refine_circle_with_neighbors`.

```text
refine_circle_with_neighbors(
    cx: ArrayLike,
    cy: ArrayLike,
    radii: ArrayLike,
    ref_x: float,
    ref_y: float,
    distance_threshold: float = 5,
) -> Tuple[float, float, Optional[float]]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `cx` | `ArrayLike` | Sim | Vetor de coordenadas de coluna dos candidatos, em pixels. |
| `cy` | `ArrayLike` | Sim | Vetor de coordenadas de linha dos candidatos, em pixels. |
| `radii` | `ArrayLike` | Sim | Vetor de raios correspondente aos centros, em pixels. |
| `ref_x` | `float` | Sim | Coluna de referência, em pixels. |
| `ref_y` | `float` | Sim | Linha de referência, em pixels. |
| `distance_threshold` | `float` | `5` | Raio de busca dos centros vizinhos no plano, em pixels. |

**Saída:** (center_x, center_y, radius). Sem vizinhos: (ref_x, ref_y, None).

```python
from utils.segmentation.aorta.localization import refine_circle_with_neighbors

refined = refine_circle_with_neighbors(
    np.array([44, 45]), np.array([20, 20]), np.array([8, 8]), 44, 20
)
assert refined == (44.5, 20.0, 8.0)
```

### Configuração e diagnósticos

No JSON de referência mid, a Hough usa raios **18 a 29 pixels**, porque `radii_end_px=30` é exclusivo; `radius_step_px=1`. `tol_radius_mm=9`, `tol_distance_mm=20`, `quadrant_offset=[30,30]`, `canny_sigma=3`, 15 picos, ROI local com padding 30 e parada com 5 falhas consecutivas.

O filtro `trajectory_filter.method="robust"` usa limites de variação de raio/centro de 4.8/8 mm, acumulador mínimo 0.408, janelas de referência/persistência 5, exigência de persistência 4, cobertura mínima 0.4, corte máximo 0.4 e até 5 fatias sintéticas. Os demais controles de cauda estão na [referência de configuração](../../../config/README.md). `method="none"` desativa esse filtro.

Os diagnósticos incluem `aorta_circle_original_count`, `aorta_circle_used_count`, `aorta_circle_filter_applied`, contagens de cauda removida/sintética, início do corte e coberturas. `aorta_circle_filter_reason` distingue filtro desativado, trajetória preservada, cobertura insuficiente, corte rejeitado e cauda removida/extrapolada. Cobertura significa quantidade de círculos dividida pelo total de fatias; não é acurácia.

## Falhas e cuidados

- Verifique a orientação e a última fatia antes de interpretar uma lista vazia; o detector não varre todas as fatias para encontrar uma inicialização alternativa.
- Não ordene/mude eixos da imagem sem aplicar transformação equivalente às coordenadas.
- O refinamento pode produzir centros/raios fracionários. Preserve os floats até uma operação que exija índices.
- Hough exige volume 3D, raios não vazios e pixel_spacing positivo. O filtro valida seu método e a fração máxima de corte.
- Círculos interpolados ou sintéticos são aproximações geométricas, não observações novas.
