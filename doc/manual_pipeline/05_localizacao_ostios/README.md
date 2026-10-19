# 05 — Localização dos óstios

[Manual](../README.md) · [Configuração](../../../config/README.md)

Usa a máscara da aorta e o primeiro mapa de vasos para localizar sementes das coronárias. A inferência não exige referência; a avaliação contra label é uma operação adicional.

## Exemplo sintético

Execute a partir da raiz do repositório com `PYTHONPATH=src uv run python seu_exemplo.py`. Este exemplo usa CPU e não representa validação anatômica.

```python
import numpy as np
from utils.project.config import load_config_json, scale_config_to_resolution

config = load_config_json("config/pipeline_config.json", {})
config["USE_GPU"] = False
config = scale_config_to_resolution(config)

from utils.segmentation.pipeline.detection import detect_ostia

rows, columns = np.ogrid[:32, :32]
disk = (rows - 16) ** 2 + (columns - 16) ** 2 < 8**2
aorta_mask = np.repeat(disk.astype(np.uint8)[..., None], 8, axis=2)
vesselness = np.zeros_like(aorta_mask, dtype=np.float32)
vesselness[16, 23, 3] = 1.0
vesselness[16, 9, 3] = 0.9
scaled_spacing = (1.0, 1.0, 1.0)
left, right = detect_ostia(aorta_mask, vesselness, scaled_spacing, config)
assert tuple(left) == (16, 23, 3)
assert right is not None and tuple(right) == (16, 9, 3)
label = (vesselness > 0).astype(np.uint8)
print(left, right)  # [16 23 3], [16 9 3]
```

## Referência das funções

Os exemplos abaixo continuam o estado do exemplo sintético desta página, salvo indicação de entrada real. Os tipos `Any`, `NDArray` e `Sequence` reproduzem as anotações do código; os contratos de conteúdo estão nas tabelas.

### Coordenadas, espaçamento e saída

`find_ostia` devolve `(ostia_left, ostia_right)` como arrays inteiros de três índices: **(linha/eixo 0, coluna/eixo 1, z)**, chamados `(y, x, z)` neste módulo. Use `volume[tuple(ostium)]` diretamente; não troque as duas primeiras coordenadas ao indexar.

Quando encontra dois candidatos, a classificação existente chama de esquerdo o candidato com **maior coordenada no eixo 1**. Isso pressupõe a orientação visual esperada pelo projeto e não constitui uma classificação anatômica independente da orientação. Sem segundo candidato válido, retorna `(primeiro_candidato, None)` sem classificar o primeiro por um par.

As funções diretas `find_ostia` e `check_ostium_intersection` recebem spacing em `(dy, dx, dz)`. O wrapper `detect_ostia` recebe `scaled_spacing` do pré-processamento e o converte internamente para `(dy, dx, dz)`, invertendo as duas primeiras entradas. Ao chamar o wrapper, passe o espaçamento original da etapa, sem antecipar essa troca. Esse contrato merece atenção especial com espaçamento anisotrópico.

O JSON atual usa top_n=2000, max_z_diff_mm=40, lower_fraction=1.0, min_center_distance_factor=0.85, min_lateral_factor=0.4, erosion_radius=4 e surface_padding_radius=2. `pair_distance_mode` ausente assume voxel_xyz. `OSTIA_VALIDATION.distance_threshold_mm=7` é usado na avaliação agregada.

### `detect_ostia`

Wrapper recomendado para inferência, sem dependência de referência coronariana.

Importação: `from utils.segmentation.pipeline.detection import detect_ostia`.

```text
detect_ostia(
    aorta_mask: Any,
    vesselness_ostia: Any,
    scaled_spacing: Sequence[float],
    config: Dict[str, Any],
) -> tuple[Any, Any]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `aorta_mask` | `Any` | Sim | Máscara binária 3D da aorta, com o mesmo shape dos demais volumes da etapa. |
| `vesselness_ostia` | `Any` | Sim | Mapa de vasos calculado com VESSELNESS_AORTA para selecionar óstios. |
| `scaled_spacing` | `Sequence[float]` | Sim | Espaçamento após downscale, na ordem dos eixos do pré-processamento, em mm. |
| `config` | `Dict[str, Any]` | Sim | Configuração completa normalizada e escalada uma única vez; chaves consumidas descritas nesta etapa. |

**Saída:** (coordenada esquerda, coordenada direita ou None).

```python
assert vesselness[tuple(left)] == 1.0
```

### `find_ostia`

Seleciona candidatos por vesselness na casca e busca o segundo respeitando separação e diferença axial.

Importação: `from utils.segmentation.ostia_detection import find_ostia`.

```text
find_ostia(
    aorta_mask: NDArray[Any],
    vesselness_map: NDArray[Any],
    spacing: Sequence[float],
    top_n: int = 50,
    max_z_diff_mm: float = 40.0,
    lower_fraction: float = 0.3,
    min_center_distance_factor: float = 0.8,
    min_lateral_factor: float = 0.5,
    erosion_radius: int = 2,
    surface_padding_radius: int = 0,
    pair_distance_mode: str = "voxel_xyz",
    verbose: bool = True,
) -> Tuple[NDArray[Any], Optional[NDArray[Any]]]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `aorta_mask` | `NDArray[Any]` | Sim | Máscara binária 3D da aorta, com o mesmo shape dos demais volumes da etapa. |
| `vesselness_map` | `NDArray[Any]` | Sim | Mapa 3D de evidência vascular usado para ordenar os candidatos de óstios. |
| `spacing` | `Sequence[float]` | Sim | Espaçamento positivo em mm; a ordem específica desta interface está explicada nesta página. |
| `top_n` | `int` | `50` | Quantidade positiva de candidatos de maior vesselness na superfície. |
| `max_z_diff_mm` | `float` | `40.0` | Distância axial máxima entre os óstios, em mm. |
| `lower_fraction` | `float` | `0.3` | Fração inicial do intervalo ocupado em z; deve estar em (0, 1]. |
| `min_center_distance_factor` | `float` | `0.8` | Multiplica o diâmetro circular equivalente da fatia para definir separação mínima. |
| `min_lateral_factor` | `float` | `0.5` | Multiplica a separação mínima para impor afastamento no eixo 1. |
| `erosion_radius` | `int` | `2` | Raio da erosão esférica 3D, em voxels. |
| `surface_padding_radius` | `int` | `0` | Raio inteiro não negativo da dilatação anterior à extração da casca, em voxels. |
| `pair_distance_mode` | `str` | `'voxel_xyz'` | voxel_xyz mede separação 3D em voxels; physical_xy usa o plano axial em mm. |
| `verbose` | `bool` | `True` | Imprime mensagens de acompanhamento quando verdadeiro. |

**Saída:** Tupla de índices descrita acima; não devolve métricas de referência.

```python
from utils.segmentation.ostia_detection import find_ostia

direct_left, direct_right = find_ostia(
    aorta_mask,
    vesselness,
    spacing=(1, 1, 1),
    top_n=2000,
    lower_fraction=1.0,
    min_center_distance_factor=0.85,
    min_lateral_factor=0.4,
    erosion_radius=4,
    surface_padding_radius=2,
    verbose=False,
)
assert tuple(direct_left) == tuple(left)
```

A distância mínima é baseada no **diâmetro** equivalente da fatia do primeiro candidato, não no raio. lower_fraction seleciona o começo do intervalo em índices z; 1.0 utiliza toda a extensão segmentada.

### `find_aorta_surface`

Extrai a casca da máscara eventualmente dilatada: D_padding(M) menos E_erosion(D_padding(M)).

Importação: `from utils.segmentation.ostia_detection import find_aorta_surface`.

```text
find_aorta_surface(
    aorta_mask: NDArray[Any], erosion_radius: int = 2, surface_padding_radius: int = 0
) -> NDArray[Any]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `aorta_mask` | `NDArray[Any]` | Sim | Máscara binária 3D da aorta, com o mesmo shape dos demais volumes da etapa. |
| `erosion_radius` | `int` | `2` | Raio da erosão esférica 3D, em voxels. |
| `surface_padding_radius` | `int` | `0` | Raio inteiro não negativo da dilatação anterior à extração da casca, em voxels. |

**Saída:** Máscara uint8 0/1 com o mesmo shape; não modifica a máscara original.

```python
from utils.segmentation.ostia_detection import find_aorta_surface

surface = find_aorta_surface(aorta_mask, erosion_radius=4, surface_padding_radius=2)
assert surface.dtype == np.uint8 and surface[tuple(left)] == 1
```

Padding desloca a casca para fora e não preserva necessariamente toda a superfície interna anterior. Os elementos estruturantes são esféricos em voxels, não em mm.

### `calculate_robust_diameter`

Calcula o diâmetro circular equivalente a partir da soma da máscara 2D: 2 × sqrt(área/π).

Importação: `from utils.segmentation.ostia_detection import calculate_robust_diameter`.

```text
calculate_robust_diameter(mask_slice: NDArray[Any]) -> float
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `mask_slice` | `NDArray[Any]` | Sim | Fatia 2D binária da aorta; mesmo plano do primeiro candidato. |

**Saída:** Float em pixels; 0.0 para máscara vazia. Forneça máscara binária, pois a área é a soma dos valores.

```python
from utils.segmentation.ostia_detection import calculate_robust_diameter

diameter = calculate_robust_diameter(aorta_mask[:, :, 3])
assert diameter > 0
```

### `check_ostium_intersection`

Mede interseção e distância física até o voxel de referência mais próximo.

Importação: `from utils.segmentation.ostia_detection import check_ostium_intersection`.

```text
check_ostium_intersection(
    ostium_coords: Optional[ArrayLike],
    label_mask: NDArray[Any],
    spacing: Sequence[float],
    ostium_name: str = "Óstio",
    distance_threshold_mm: float = 5.0,
    verbose: bool = False,
) -> dict
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `ostium_coords` | `Optional[ArrayLike]` | Sim | Índices do óstio em (eixo 0, eixo 1, z), ou None. |
| `label_mask` | `NDArray[Any]` | Sim | Máscara binária de referência, alinhada ao óstio. |
| `spacing` | `Sequence[float]` | Sim | Espaçamento positivo em mm; a ordem específica desta interface está explicada nesta página. |
| `ostium_name` | `str` | `'Óstio'` | Nome usado somente nas mensagens de acompanhamento. |
| `distance_threshold_mm` | `float` | `5.0` | Distância máxima aceitável até a referência, em mm. |
| `verbose` | `bool` | `False` | Imprime mensagens de acompanhamento quando verdadeiro. |

**Saída:** Dict: intersects (bool), euclidean_dist (voxels), physical_dist (mm), nearest_voxel (três índices), is_acceptable (bool). None produz distâncias infinitas e is_acceptable=False.

```python
from utils.segmentation.ostia_detection import check_ostium_intersection

left_info = check_ostium_intersection(
    left, label, spacing=(1, 1, 1), distance_threshold_mm=7
)
assert left_info["intersects"] and left_info["physical_dist"] == 0
```

### `detect_and_evaluate_ostia`

Detecta os óstios e compara com a classe 1 da referência ImageCAS.

Importação: `from utils.segmentation.pipeline.detection import detect_and_evaluate_ostia`.

```text
detect_and_evaluate_ostia(
    aorta_mask: Any,
    vesselness_ostios: Any,
    label: Any,
    scaled_spacing: Sequence[float],
    config: Dict[str, Any],
) -> Dict[str, Any]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `aorta_mask` | `Any` | Sim | Máscara binária 3D da aorta, com o mesmo shape dos demais volumes da etapa. |
| `vesselness_ostios` | `Any` | Sim | Mapa de vasos calculado com VESSELNESS_AORTA para selecionar óstios. |
| `label` | `Any` | Sim | Referência ImageCAS na resolução de trabalho; apenas classe 1 é considerada coronária. |
| `scaled_spacing` | `Sequence[float]` | Sim | Espaçamento após downscale, na ordem dos eixos do pré-processamento, em mm. |
| `config` | `Dict[str, Any]` | Sim | Configuração completa normalizada e escalada uma única vez; chaves consumidas descritas nesta etapa. |

**Saída:** Dict: ostia_left, ostia_right, label_artery (label==1), left_info, right_info, both_correct e both_tolerable. both_tolerable exclui os casos both_correct.

```python
from utils.segmentation.pipeline.detection import detect_and_evaluate_ostia

evaluated = detect_and_evaluate_ostia(
    aorta_mask, vesselness, label, scaled_spacing, config
)
assert evaluated["both_correct"] and not evaluated["both_tolerable"]
```

A avaliação agregada usa OSTIA_VALIDATION.distance_threshold_mm (7 no JSON). Os dicts left_info/right_info são gerados pelo helper com seu default local de 5 mm para is_acceptable; both_tolerable compara physical_dist com o limite configurado. Use a distância e o limite pretendido explicitamente ao interpretar esses campos.

## Falhas e cuidados

- Máscara e mapa devem ter o mesmo shape; superfície vazia e top_n <= 0 geram ValueError.
- A ausência de segundo óstio é representada por None; verifique antes de indexar.
- lower_fraction deve estar em (0, 1]; padding deve ser inteiro não negativo.
- Uma referência sem voxels positivos gera ValueError na medição de distância de um óstio presente. None não exige essa medição.
- Não use labels de aorta/MM-WHS como referência coronariana: as classes e o significado anatômico são diferentes.
