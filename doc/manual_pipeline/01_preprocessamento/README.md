# 01 — Pré-processamento

[Manual](../README.md) · [Configuração](../../../config/README.md)

Prepara a imagem na resolução de trabalho. A saída alimenta tanto a Hough quanto os mapas de vasos. O wrapper usa layout declarado `(x, y, z)` e spacing nessa ordem; o eixo 2 sempre representa as fatias. Nenhuma função desta etapa corrige orientação anatômica automaticamente.

## Exemplo sintético

Execute a partir da raiz do repositório com `PYTHONPATH=src uv run python seu_exemplo.py`. Este exemplo usa CPU e não representa validação anatômica.

```python
import numpy as np
from utils.project.config import load_config_json, scale_config_to_resolution

config = load_config_json("config/pipeline_config.json", {})
config["USE_GPU"] = False
config = scale_config_to_resolution(config)

from utils.segmentation.pipeline.preprocessing import preprocess_ccta_volume

image = np.full((32, 32, 8), -800, dtype=np.float32)
image[6:26, 6:26, :] = 200
spacing = (0.5, 0.5, 1.0)
preprocessed = preprocess_ccta_volume(
    image,
    spacing,
    config,
    include_intermediates=True,
)
assert preprocessed["lcc_image"].shape == (16, 16, 8)
assert preprocessed["scaled_spacing"] == (1.0, 1.0, 1.0)
assert preprocessed["label"] is None
assert preprocessed["threshold_mask"].dtype == bool
print(preprocessed["preprocessing_details"])
```

## Referência das funções

Os exemplos abaixo continuam o estado do exemplo sintético desta página, salvo indicação de entrada real. Os tipos `Any`, `NDArray` e `Sequence` reproduzem as anotações do código; os contratos de conteúdo estão nas tabelas.

### `preprocess_ccta_volume`

Entrada recomendada para volumes já carregados, inclusive CCTA externo. Converte a imagem para float32, reduz resolução, seleciona candidatos e aplica LCC por fatia nos modos normal/fuzzy.

Importação: `from utils.segmentation.pipeline.preprocessing import preprocess_ccta_volume`.

```text
preprocess_ccta_volume(
    image: Any,
    spacing: Sequence[float],
    config: Dict[str, Any],
    *,
    label: Any | None = None,
    include_intermediates: bool = False,
) -> Dict[str, Any]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `image` | `Any` | Sim | Imagem/volume numérico de entrada; intensidades em HU quando usado como CCTA. |
| `spacing` | `Sequence[float]` | Sim | Espaçamento positivo em mm; a ordem específica desta interface está explicada nesta página. |
| `config` | `Dict[str, Any]` | Sim | Configuração completa normalizada e escalada uma única vez; chaves consumidas descritas nesta etapa. |
| `label` | `Any \| None` | `None` | Referência opcional, com o mesmo shape da imagem; classes são preservadas por vizinho mais próximo. |
| `include_intermediates` | `bool` | `False` | Inclui volumes extras para inspeção; False reduz retenção de memória. |

**Saída:** `dict`: `lcc_image` (float32 3D reduzido), `label` (uint8 reduzido ou None), `spacing` (original), `scaled_spacing` (original × fatores), `downscale_factors` e `preprocessing_details`. Com intermediários: `image`, `down_image`, `threshold_mask` e `lcc_mask` (máscaras bool).

```python
with_label = preprocess_ccta_volume(
    image, spacing, config, label=(image > 0).astype(np.uint8)
)
assert with_label["label"].dtype == np.uint8
```

### `load_and_preprocess_image`: entrada ImageCAS

Importação: `from utils.segmentation.pipeline.preprocessing import load_and_preprocess_image`.

```text
load_and_preprocess_image(
    img_id: str,
    base_path: str,
    config: Dict[str, Any],
    *,
    include_intermediates: bool = False,
) -> Dict[str, Any]
```

| Parâmetro | Tipo | Obrigatório / padrão | Unidade e efeito |
|---|---|---|---|
| `img_id` | str | Sim | Identificador usado nos nomes dos dois arquivos. |
| `base_path` | str | Sim | Diretório com `<id>.img.nii.gz` e `<id>.label.nii.gz`. |
| `config` | dict | Sim | Configuração completa, já normalizada e escalada. |
| `include_intermediates` | bool | False | Inclui os volumes intermediários descritos acima. |

**Saída:** o mesmo dicionário de `preprocess_ccta_volume`, com referência obrigatória. O espaçamento vem do cabeçalho NIfTI.

Exemplo com dataset real, **não executado na validação sintética**:

<!-- example: dataset -->
```python
import os
from utils.segmentation.pipeline.preprocessing import load_and_preprocess_image

imagecas = load_and_preprocess_image(
    os.environ["IMAGECAS_EXAM_ID"],
    os.environ["IMAGECAS_BASE_PATH"],
    config,
    include_intermediates=True,
)
```

### Configuração consumida

| Bloco/chave | Valor do JSON de referência | Efeito |
|---|---|---|
| `DOWNSCALE_FACTORS` | [2, 2, 1] | Divisores dos eixos. |
| `DOWNSCALE_METHOD` / `OPENCV_INTERPOLATION` | opencv / linear | Imagem reduzida pelo OpenCV; caminho alternativo SciPy usa order=3. |
| `THRESHOLDING.method` | normal | normal, fuzzy ou none. none preserva finitos, sem corte HU nem LCC. |
| `LOWER_THRESHOLD.method` / `fixed_hu` | fixed / -300 | Piso HU; percentile é alternativa configurável. |
| `MAX_THRESHOLD_PERCENTILE` | 99.9 | Teto HU calculado por percentil; não é clamp, voxels fora da faixa são excluídos. |
| `THRESHOLDING.fuzzy` | margem 100; P99.8/P99.96; suavização 0 | Modelo fuzzy com três classes e seleção object_argmax. |

`preprocessing_details` contém modo, método/piso inferior, teto e contagens de candidatos/LCC. O modo fuzzy acrescenta os três centros HU e `fuzzy_mask_strategy`; o modo none usa None para limiares e contagens. A função não altera o dicionário `config` para refletir o piso calculado: consulte `preprocessing_details["min_threshold"]`.

Nos modos normal/fuzzy, o fundo de `lcc_image` fica em **-offset**, não necessariamente zero: o offset é retirado do volume inteiro após a LCC. Use `lcc_mask` quando precisar da máscara exata; `lcc_image > 0` não é equivalente.

### `load_raw_img_and_label`

Carrega objetos nibabel preservando acesso ao cabeçalho e affine.

Importação: `from utils.utils.nifti_io import load_raw_img_and_label`.

```text
load_raw_img_and_label(
    img_path: str, label_path: Optional[str] = None
) -> Tuple[SpatialImage, Optional[SpatialImage]]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `img_path` | `str` | Sim | Caminho para o arquivo de imagem NIfTI. |
| `label_path` | `Optional[str]` | `None` | Caminho opcional para o NIfTI de referência; None retorna referência ausente. |

**Saída:** (imagem, referência opcional), ambos objetos SpatialImage; label None se não informado.

```python
import tempfile
from pathlib import Path
import nibabel as nib
from utils.utils.nifti_io import load_raw_img_and_label

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "synthetic.nii.gz"
    nib.save(nib.Nifti1Image(image, np.diag([0.5, 0.5, 1.0, 1.0])), path)
    nii, no_label = load_raw_img_and_label(str(path))
    assert nii.shape == image.shape and no_label is None
```

### `load_img_and_label`

Carrega os mesmos arquivos diretamente como arrays; não retorna geometria nem reorienta os eixos.

Importação: `from utils.utils.nifti_io import load_img_and_label`.

```text
load_img_and_label(
    img_path: str, label_path: Optional[str] = None
) -> Tuple[NDArray[Any], Optional[NDArray[Any]]]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `img_path` | `str` | Sim | Caminho para o arquivo de imagem NIfTI. |
| `label_path` | `Optional[str]` | `None` | Caminho opcional para o NIfTI de referência; None retorna referência ausente. |

**Saída:** (imagem, label ou None); get_fdata produz normalmente float64.

```python
from utils.utils.nifti_io import load_img_and_label

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "synthetic.nii.gz"
    nib.save(nib.Nifti1Image(image, np.eye(4)), path)
    loaded, no_label = load_img_and_label(str(path))
    assert loaded.shape == image.shape and no_label is None
```

### `downscale_image_ndi`

Usa scipy.ndimage.zoom com fatores inversos; não atualiza o espaçamento.

Importação: `from utils.processing.preprocessing import downscale_image_ndi`.

```text
downscale_image_ndi(
    image: NDArray[Any], factors: Sequence[float], order: int = 3
) -> NDArray[Any]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `image` | `NDArray[Any]` | Sim | Imagem/volume numérico de entrada; intensidades em HU quando usado como CCTA. |
| `factors` | `Sequence[float]` | Sim | Divisores positivos de redução por eixo; evitar dimensões resultantes nulas. |
| `order` | `int` | `3` | Ordem da interpolação SciPy: 0 para classes; 3 para interpolação cúbica. |

**Saída:** Array reduzido; OpenCV preserva o dtype da entrada. Os métodos podem arredondar dimensões de modo diferente em tamanhos não divisíveis.

```python
from utils.processing.preprocessing import downscale_image_ndi

small_scipy = downscale_image_ndi(image, (2, 2, 1), order=3)
assert small_scipy.shape == (16, 16, 8)
```

### `downscale_image_opencv`

Redimensiona planos XY e, se necessário, Z; suporta imagens 2D/3D.

Importação: `from utils.processing.preprocessing import downscale_image_opencv`.

```text
downscale_image_opencv(
    image: NDArray[Any], factors: Sequence[float], interpolation: int = cv2.INTER_LINEAR
) -> NDArray[Any]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `image` | `NDArray[Any]` | Sim | Imagem/volume numérico de entrada; intensidades em HU quando usado como CCTA. |
| `factors` | `Sequence[float]` | Sim | Divisores positivos de redução por eixo; evitar dimensões resultantes nulas. |
| `interpolation` | `int` | `cv2.INTER_LINEAR` | Constante de interpolação OpenCV, como cv2.INTER_LINEAR ou INTER_NEAREST. |

**Saída:** Array reduzido; OpenCV preserva o dtype da entrada. Os métodos podem arredondar dimensões de modo diferente em tamanhos não divisíveis.

```python
from utils.processing.preprocessing import downscale_image_opencv

small_cv = downscale_image_opencv(image, (2, 2, 1))
assert small_cv.shape == (16, 16, 8)
```

### `downscale_image`

Despacha para OpenCV ou SciPy/GPU. Não é o wrapper usado pelo pré-processamento principal; pode selecionar GPU automaticamente no caminho SciPy.

Importação: `from utils.processing.preprocessing import downscale_image`.

```text
downscale_image(
    image: NDArray[Any],
    factors: Sequence[float],
    order: int = 3,
    use_opencv: bool = False,
    opencv_interpolation: Optional[int] = None,
) -> NDArray[Any]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `image` | `NDArray[Any]` | Sim | Imagem/volume numérico de entrada; intensidades em HU quando usado como CCTA. |
| `factors` | `Sequence[float]` | Sim | Divisores positivos de redução por eixo; evitar dimensões resultantes nulas. |
| `order` | `int` | `3` | Ordem da interpolação SciPy: 0 para classes; 3 para interpolação cúbica. |
| `use_opencv` | `bool` | `False` | Usa OpenCV em vez do caminho SciPy. |
| `opencv_interpolation` | `Optional[int]` | `None` | Constante cv2.INTER_*; None seleciona INTER_LINEAR no wrapper. |

**Saída:** Array reduzido; OpenCV preserva o dtype da entrada. Os métodos podem arredondar dimensões de modo diferente em tamanhos não divisíveis.

```python
from utils.processing.preprocessing import downscale_image

small = downscale_image(image, (2, 2, 1), use_opencv=True)
assert small.shape == (16, 16, 8)
```

### `threshold_image`

Seleciona a faixa inclusiva, sem deslocamento.

Importação: `from utils.processing.preprocessing import threshold_image`.

```text
threshold_image(
    image: NDArray[Any], min_val: float = -300, max_val: float = 675
) -> Tuple[NDArray[Any], NDArray[Any]]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `image` | `NDArray[Any]` | Sim | Imagem/volume numérico de entrada; intensidades em HU quando usado como CCTA. |
| `min_val` | `float` | `-300` | Limite inferior inclusivo de intensidade, em HU. |
| `max_val` | `float` | `675` | Limite superior inclusivo de intensidade, em HU. |

**Saída:** (imagem mascarada, máscara bool), ambos com o shape original; imagem é image × máscara.

```python
from utils.processing.preprocessing import threshold_image

thresholded, selected = threshold_image(image, -300, 500)
assert selected.dtype == bool and thresholded.shape == image.shape
```

### `threshold_image_with_offset`

Seleciona a faixa e desloca os valores válidos para distinguir intensidades negativas do fundo.

Importação: `from utils.processing.preprocessing import threshold_image_with_offset`.

```text
threshold_image_with_offset(
    image: NDArray[Any], min_val: float = -300, max_val: float = 675
) -> Tuple[NDArray[Any], NDArray[Any], float]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `image` | `NDArray[Any]` | Sim | Imagem/volume numérico de entrada; intensidades em HU quando usado como CCTA. |
| `min_val` | `float` | `-300` | Limite inferior inclusivo de intensidade, em HU. |
| `max_val` | `float` | `675` | Limite superior inclusivo de intensidade, em HU. |

**Saída:** (imagem deslocada, máscara bool, offset=abs(min_val)); fundo zero antes da LCC.

```python
from utils.processing.preprocessing import threshold_image_with_offset

shifted, selected, offset = threshold_image_with_offset(image, -300, 500)
assert offset == 300 and np.all(shifted[~selected] == 0)
```

### `largest_connected_component`

Seleciona a maior componente usando scipy.ndimage.label e conectividade padrão (4 em 2D, 6 em 3D).

Importação: `from utils.processing.preprocessing import largest_connected_component`.

```text
largest_connected_component(
    image: NDArray[Any], mask: NDArray[Any]
) -> Tuple[NDArray[Any], NDArray[Any]]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `image` | `NDArray[Any]` | Sim | Imagem/volume numérico de entrada; intensidades em HU quando usado como CCTA. |
| `mask` | `NDArray[Any]` | Sim | Máscara candidata/binária no mesmo shape da imagem correspondente. |

**Saída:** (imagem × máscara da maior componente, máscara bool). Se a máscara está vazia, devolve imagem e máscara recebidas sem zerar a imagem.

```python
from utils.processing.preprocessing import largest_connected_component

lcc_slice, mask_slice = largest_connected_component(shifted[:, :, 0], selected[:, :, 0])
assert mask_slice.shape == image.shape[:2]
```

### `build_lcc_image_from_mask`

Aplica uma máscara externa, desloca intensidades e calcula LCC.

Importação: `from utils.processing.preprocessing import build_lcc_image_from_mask`.

```text
build_lcc_image_from_mask(
    volume: NDArray[Any], mask: NDArray[Any], offset: float, per_slice: bool = True
) -> Tuple[NDArray[Any], NDArray[Any]]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `volume` | `NDArray[Any]` | Sim | Volume 3D em HU na resolução de trabalho. |
| `mask` | `NDArray[Any]` | Sim | Máscara candidata/binária no mesmo shape da imagem correspondente. |
| `offset` | `float` | Sim | Valor somado aos voxels selecionados antes da LCC e subtraído do volume ao final, em HU. |
| `per_slice` | `bool` | `True` | True seleciona LCC 2D por fatia; False seleciona LCC 3D global. |

**Saída:** (lcc_image float32, lcc_mask bool), mesmo shape do volume; fundo final igual a -offset.

```python
from utils.processing.preprocessing import build_lcc_image_from_mask

lcc, lcc_mask = build_lcc_image_from_mask(image, selected, offset=300)
assert lcc.dtype == np.float32 and lcc_mask.dtype == bool
```

### `run_core_preprocessing_pipeline`

Atalho de baixo nível para downscale, threshold normal e LCC. Não inclui os modos fuzzy/none, configuração de piso adaptativo ou retorno de spacing do wrapper principal.

Importação: `from utils.processing.preprocessing import run_core_preprocessing_pipeline`.

```text
run_core_preprocessing_pipeline(
    image: NDArray[Any],
    downscale_factors: Sequence[float],
    min_threshold: int = -300,
    max_threshold_percentile: float = 99.5,
    lcc_per_slice: bool = True,
    order: int = 3,
    use_opencv: bool = False,
    opencv_interpolation: Optional[int] = None,
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], Tuple[int, int]]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `image` | `NDArray[Any]` | Sim | Imagem/volume numérico de entrada; intensidades em HU quando usado como CCTA. |
| `downscale_factors` | `Sequence[float]` | Sim | Divisores por eixo; (2, 2, 1) reduz os dois primeiros e mantém o terceiro. |
| `min_threshold` | `int` | `-300` | Limiar inferior inclusivo em HU. |
| `max_threshold_percentile` | `float` | `99.5` | Percentil superior (0 a 100), calculado após downscale. |
| `lcc_per_slice` | `bool` | `True` | Seleciona a maior componente por fatia (True) ou globalmente (False). |
| `order` | `int` | `3` | Ordem da interpolação SciPy: 0 para classes; 3 para interpolação cúbica. |
| `use_opencv` | `bool` | `False` | Usa OpenCV em vez do caminho SciPy. |
| `opencv_interpolation` | `Optional[int]` | `None` | Constante cv2.INTER_*; None seleciona INTER_LINEAR no wrapper. |

**Saída:** (down_image, thresh_image deslocada, lcc_image com offset retirado, (mínimo, teto)). O teto é convertido para int nesta função.

```python
from utils.processing.preprocessing import run_core_preprocessing_pipeline

core = run_core_preprocessing_pipeline(
    image, (2, 2, 1), use_opencv=True, max_threshold_percentile=99.9
)
assert len(core) == 4 and core[2].shape == (16, 16, 8)
```

O padrão local do percentil é **99.5**; o wrapper configurado usa **99.9**. Não trate este atalho como substituto exato de `preprocess_ccta_volume`.

### `resolve_lower_threshold`

Resolve piso fixo ou percentil dos voxels finitos na faixa de estimação.

Importação: `from utils.segmentation.lower_threshold import resolve_lower_threshold`.

```text
resolve_lower_threshold(
    volume: np.ndarray, config: dict[str, Any]
) -> tuple[float, dict[str, Any]]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `volume` | `np.ndarray` | Sim | Volume 3D em HU na resolução de trabalho. |
| `config` | `dict[str, Any]` | Sim | Configuração completa normalizada e escalada uma única vez; chaves consumidas descritas nesta etapa. |

**Saída:** (piso float em HU, detalhes dict). Percentil sem valores válidos usa fixed_hu e registra lower_threshold_fallback=empty_valid_values.

```python
from utils.segmentation.lower_threshold import resolve_lower_threshold

minimum, lower_details = resolve_lower_threshold(image, config)
assert minimum == -300
```

### `estimate_fuzzy_centers`

Estima centros do fundo mole, objeto e fundo denso.

Importação: `from utils.segmentation.fuzzy.threshold import estimate_fuzzy_centers`.

```text
estimate_fuzzy_centers(
    volume: np.ndarray,
    min_hu: float,
    soft_margin_hu: float,
    object_percentile: float,
    dense_percentile: float,
) -> np.ndarray
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `volume` | `np.ndarray` | Sim | Volume 3D em HU na resolução de trabalho. |
| `min_hu` | `float` | Sim | Limiar inferior em HU que ancora o modelo fuzzy. |
| `soft_margin_hu` | `float` | Sim | Distância em HU entre o limiar mínimo e o centro do fundo mole. |
| `object_percentile` | `float` | Sim | Percentil para estimar o centro da classe objeto. |
| `dense_percentile` | `float` | Sim | Percentil para estimar o centro do fundo denso. |

**Saída:** Array float32 com três centros HU em ordem crescente; usa finitos e preferencialmente valores >= min_hu.

```python
from utils.segmentation.fuzzy.threshold import estimate_fuzzy_centers

centers = estimate_fuzzy_centers(image, -300, 100, 99.8, 99.96)
assert centers.shape == (3,)
```

### `fuzzy_threshold_outputs`

Constrói pertinências de três classes e retém voxels cuja classe dominante é objeto, também impondo volume >= min_hu.

Importação: `from utils.segmentation.fuzzy.threshold import fuzzy_threshold_outputs`.

```text
fuzzy_threshold_outputs(
    volume: np.ndarray,
    *,
    min_hu: float,
    soft_margin_hu: float = 160,
    object_percentile: float = 99.8,
    dense_percentile: float = 99.95,
    smooth_radius: int = 1,
    smooth_mode: str = "mean",
) -> tuple[np.ndarray, dict[str, Any]]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `volume` | `np.ndarray` | Sim | Volume 3D em HU na resolução de trabalho. |
| `min_hu` | `float` | Sim | Limiar inferior em HU que ancora o modelo fuzzy. |
| `soft_margin_hu` | `float` | `160` | Distância em HU entre o limiar mínimo e o centro do fundo mole. |
| `object_percentile` | `float` | `99.8` | Percentil para estimar o centro da classe objeto. |
| `dense_percentile` | `float` | `99.95` | Percentil para estimar o centro do fundo denso. |
| `smooth_radius` | `int` | `1` | Raio da janela cúbica de suavização das pertinências, em voxels; 0 desativa. |
| `smooth_mode` | `str` | `'mean'` | median usa mediana; os demais valores seguem o caminho de média (usar mean). |

**Saída:** (máscara bool, detalhes com soft_center_hu, object_center_hu, dense_center_hu e fuzzy_mask_strategy=object_argmax). Não retorna os volumes de pertinência.

```python
from utils.segmentation.fuzzy.threshold import fuzzy_threshold_outputs

fuzzy_mask, fuzzy_details = fuzzy_threshold_outputs(
    image, min_hu=-300, soft_margin_hu=100, dense_percentile=99.96, smooth_radius=0
)
assert fuzzy_mask.shape == image.shape and fuzzy_mask.dtype == bool
```

Os padrões locais (160 HU, P99.95, raio 1) diferem dos usados pelo wrapper/configuração (100 HU, P99.96, raio 0).

### `fuzzy_threshold_from_config`

Aplica os parâmetros de THRESHOLDING.fuzzy e MIN_THRESHOLD ao modelo fuzzy. Não resolve sozinho LOWER_THRESHOLD.

Importação: `from utils.segmentation.fuzzy.threshold import fuzzy_threshold_from_config`.

```text
fuzzy_threshold_from_config(
    volume: np.ndarray, config: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `volume` | `np.ndarray` | Sim | Volume 3D em HU na resolução de trabalho. |
| `config` | `dict[str, Any]` | Sim | Configuração completa normalizada e escalada uma única vez; chaves consumidas descritas nesta etapa. |

**Saída:** A mesma tupla de máscara bool e detalhes de fuzzy_threshold_outputs.

```python
from utils.segmentation.fuzzy.threshold import fuzzy_threshold_from_config

fuzzy_mask, fuzzy_details = fuzzy_threshold_from_config(image, config)
assert fuzzy_details["fuzzy_mask_strategy"] == "object_argmax"
```

### Selecionar as alternativas no pipeline

```python
from copy import deepcopy

fuzzy_config = deepcopy(config)
fuzzy_config["THRESHOLDING"]["method"] = "fuzzy"
fuzzy_preprocessed = preprocess_ccta_volume(image, spacing, fuzzy_config)

no_threshold_config = deepcopy(config)
no_threshold_config["THRESHOLDING"]["method"] = "none"
unfiltered = preprocess_ccta_volume(image, spacing, no_threshold_config)
assert unfiltered["preprocessing_details"]["threshold_mode"] == "none"
```

Para ativar piso adaptativo, selecione `LOWER_THRESHOLD.method="percentile"`. O campo `THRESHOLDING.fuzzy.lower_percentile` substitui o percentil da estimação ao usar fuzzy, mas só afeta o piso se o método inferior for percentile. Com fixed, o piso continua fixo.

## Falhas e cuidados

- Volume deve ser 3D; spacing deve ter três valores positivos; label deve ter o mesmo shape.
- Os caminhos normal/fuzzy pressupõem intensidades finitas adequadas ao cálculo de percentis. Não envie volumes vazios ou compostos apenas por NaN/inf.
- Labels usam SciPy order=0; imagem e label com tamanhos não divisíveis podem produzir shapes diferentes entre SciPy e OpenCV. Confira o alinhamento.
- A LCC do wrapper é sempre por fatia. O modo global disponível nos auxiliares muda o comportamento.
- O carregador ImageCAS exige os dois arquivos; para inferência sem referência, use preprocess_ccta_volume.
