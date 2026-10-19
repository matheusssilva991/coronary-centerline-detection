# 06 — Segmentação coronariana

[Manual](../README.md) · [Configuração](../../../config/README.md)

Propaga sementes dos óstios no mapa arterial. O padrão usa crescimento de região; fuzzy connectedness é uma alternativa configurável. O resultado desta etapa já disponibiliza os estágios de pós-processamento.

## Exemplo sintético

Execute a partir da raiz do repositório com `PYTHONPATH=src uv run python seu_exemplo.py`. Este exemplo usa CPU e não representa validação anatômica.

```python
import numpy as np
from utils.project.config import load_config_json, scale_config_to_resolution

config = load_config_json("config/pipeline_config.json", {})
config["USE_GPU"] = False
config = scale_config_to_resolution(config)

from utils.segmentation.pipeline.arteries import segment_artery_masks_from_vesselness

image = np.full((20, 20, 8), -300, dtype=np.float32)
vesselness = np.zeros_like(image)
vesselness[6:10, 6:10, :] = 1.0
vesselness[12:16, 12:16, :] = 0.8
image[vesselness > 0] = 250
left, right = (8, 8, 3), (14, 14, 3)
config["POSTPROCESSING"].update(closing_radius=1, dilation_radius=1)
segmentation = segment_artery_masks_from_vesselness(
    image,
    vesselness,
    left,
    right,
    config,
)
assert segmentation.raw_mask.sum() > 0
assert segmentation.final_mask.shape == image.shape
assert segmentation.final_mask.dtype == np.uint8
label_artery = (vesselness > 0).astype(np.uint8)
print(segmentation.method, segmentation.details)
```

## Referência das funções

Os exemplos abaixo continuam o estado do exemplo sintético desta página, salvo indicação de entrada real. Os tipos `Any`, `NDArray` e `Sequence` reproduzem as anotações do código; os contratos de conteúdo estão nas tabelas.

### Interfaces e resultado

`segment_artery_masks_from_vesselness` é a entrada recomendada sem referência e com mapa arterial disponível. Ela retorna `ArterySegmentationResult`:

| Campo | Tipo / significado |
|---|---|
| `raw_mask` | ndarray uint8, união binária das árvores antes da morfologia. |
| `closed_mask` | ndarray uint8 após fechamento. |
| `final_mask` | ndarray uint8 após dilatação. |
| `method` | region_growing ou fuzzy_connectedness. |
| `details` | dict dependente do método: contagens por ramo no RG separado; sementes, candidatos e propagação na FC. |

Todas as máscaras têm o shape do volume. As sementes são índices diretos `(eixo 0, eixo 1, z)`, compatíveis com a saída dos óstios. Os três estágios **já incluem** a morfologia; não aplique pós-processamento novamente a final_mask.

### `segment_artery_masks_from_vesselness`

Segmenta e pós-processa sem calcular Dice nem exigir referência.

Importação: `from utils.segmentation.pipeline.arteries import segment_artery_masks_from_vesselness`.

```text
segment_artery_masks_from_vesselness(
    lcc_image: Any,
    vesselness_artery: Any,
    ostia_left: Optional[Sequence[int]],
    ostia_right: Optional[Sequence[int]],
    config: Dict[str, Any],
    *,
    method: Any = None,
) -> ArterySegmentationResult
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `lcc_image` | `Any` | Sim | Volume pré-processado 3D, com HU preservados na região selecionada. |
| `vesselness_artery` | `Any` | Sim | Mapa arterial 3D com o mesmo shape do volume; não é máscara binária. |
| `ostia_left` | `Optional[Sequence[int]]` | Sim | Semente esquerda em índices (eixo 0, eixo 1, z), ou None quando admitido. |
| `ostia_right` | `Optional[Sequence[int]]` | Sim | Semente direita na mesma convenção; None representa ausência do segundo óstio. |
| `config` | `Dict[str, Any]` | Sim | Configuração completa normalizada e escalada uma única vez; chaves consumidas descritas nesta etapa. |
| `method` | `Any` | `None` | None lê ARTERY_SEGMENTATION.method; string explícita seleciona region_growing ou fuzzy_connectedness. |

**Saída:** ArterySegmentationResult descrito acima.

```python
fc_segmentation = segment_artery_masks_from_vesselness(
    image, vesselness, left, right, config, method="fuzzy_connectedness"
)
assert fc_segmentation.method == "fuzzy_connectedness"
assert fc_segmentation.raw_mask.shape == image.shape
```

### `normal_region_growing_from_ostia`

Executa somente crescimento de região, com refinamento/multissementes conforme configuração.

Importação: `from utils.segmentation.artery_segmentation import normal_region_growing_from_ostia`.

```text
normal_region_growing_from_ostia(
    vesselness_artery: NDArray[Any],
    ostia_left: Sequence[int] | None,
    ostia_right: Sequence[int] | None,
    config: dict[str, Any],
    *,
    branch_diagnostics: dict[str, Any] | None = None,
) -> NDArray[np.uint8]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `vesselness_artery` | `NDArray[Any]` | Sim | Mapa arterial 3D com o mesmo shape do volume; não é máscara binária. |
| `ostia_left` | `Sequence[int] \| None` | Sim | Semente esquerda em índices (eixo 0, eixo 1, z), ou None quando admitido. |
| `ostia_right` | `Sequence[int] \| None` | Sim | Semente direita na mesma convenção; None representa ausência do segundo óstio. |
| `config` | `dict[str, Any]` | Sim | Configuração completa normalizada e escalada uma única vez; chaves consumidas descritas nesta etapa. |
| `branch_diagnostics` | `dict[str, Any] \| None` | `None` | Dict opcional preenchido por efeito colateral com artery_left_raw_voxels e artery_right_raw_voxels quando cresce separadamente. |

**Saída:** Máscara bruta uint8 3D; união das árvores esquerda/direita, sem fechamento/dilatação.

```python
from utils.segmentation.artery_segmentation import normal_region_growing_from_ostia

branch_details = {}
raw = normal_region_growing_from_ostia(
    vesselness, left, right, config, branch_diagnostics=branch_details
)
assert raw.shape == image.shape and raw.dtype == np.uint8
```

### Parâmetros de crescimento de região

O bloco `REGION_GROWING` define os controles abaixo; os valores são os do JSON mid, antes de qualquer ajuste high.

| Chave | Valor de referência | Efeito / unidade |
|---|---|---|
| `max_volume` | 100000 | Limite de crescimento em voxels por propagação; com ramos separados não é um limite global da união. |
| `switch_at_voxels` | 2000 | Contagem para transição do piso de vesselness. |
| `min_vesselness_fraction` | 0.078 | Piso relativo ao vesselness de referência. |
| `threshold_divisor` | 7 | Diferença admissível = (referência − mínimo do mapa) / divisor. |
| `relaxed_floor_factor` / `smooth_relaxation` | 0.98 / True | Fator de relaxamento do piso e transição suave. |
| `comparison_window` | 1 | Janela de valores aceitos; ALL ou -1 usa média acumulada. |
| `use_seed_refinement` / `use_multi_seed` | True / True | Procura sementes locais e permite múltiplas sementes. |
| `grow_each_ostium_separately` | True | Segmenta cada ramo antes de unir máscaras. |
| `seed_search_radius` / `seed_candidate_radius` | 2 / 1 | Raios em voxels para refinamento e seleção de múltiplas sementes. |
| `max_seed_candidates` | 6 | Máximo de sementes candidatas por óstio. |
| `seed_min_vesselness_fraction` | 0.02 | Piso relativo para selecionar sementes. |
| `reference_scope` | global | Alternativas: ostia_local e ostium_local_per_branch. |
| `reference_radius` / `reference_percentile` | 3 / 95 | Raio em voxels e percentil para referência local. |
| `verbose` | False | Mensagens durante crescimento. |

A conectividade padrão é 26; a chave opcional `neighborhood` permite 6/18/26. O ajuste high modifica limites geométricos e heurísticas, incluindo threshold_divisor e min_vesselness_fraction. Consulte a configuração escalada em vez de reaplicar estes números manualmente.

### `segment_arteries_from_vesselness`

Reutiliza um mapa pronto, executa segmentação/morfologia e calcula Dice contra referência.

Importação: `from utils.segmentation.pipeline.arteries import segment_arteries_from_vesselness`.

```text
segment_arteries_from_vesselness(
    lcc_image: Any,
    label_artery: Any,
    vesselness_artery: Any,
    ostia_left: Optional[Sequence[int]],
    ostia_right: Optional[Sequence[int]],
    config: Dict[str, Any],
) -> Dict[str, Any]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `lcc_image` | `Any` | Sim | Volume pré-processado 3D, com HU preservados na região selecionada. |
| `label_artery` | `Any` | Sim | Referência coronariana binária, alinhada à resolução da predição. |
| `vesselness_artery` | `Any` | Sim | Mapa arterial 3D com o mesmo shape do volume; não é máscara binária. |
| `ostia_left` | `Optional[Sequence[int]]` | Sim | Semente esquerda em índices (eixo 0, eixo 1, z), ou None quando admitido. |
| `ostia_right` | `Optional[Sequence[int]]` | Sim | Semente direita na mesma convenção; None representa ausência do segundo óstio. |
| `config` | `Dict[str, Any]` | Sim | Configuração completa normalizada e escalada uma única vez; chaves consumidas descritas nesta etapa. |

**Saída:** Dict com artery_mask e raw_artery_mask, contagens antes/depois, Dice antes/depois/delta, alias dice_artery, artery_segmentation_method e campos diagnósticos por ramo/FC.

```python
from utils.segmentation.pipeline.arteries import segment_arteries_from_vesselness

evaluated = segment_arteries_from_vesselness(
    image, label_artery, vesselness, left, right, config
)
assert evaluated["dice_artery"] == evaluated["dice_artery_after_morphology"]
```

O retorno inclui artery_voxels_before_morphology, artery_voxels_after_morphology e alias artery_voxels. FC acrescenta fc_processed_voxels, fc_effective_alpha, fc_object_seed_count e fc_candidate_voxels_final; valores não aplicáveis podem ser None.

### `segment_arteries_from_ostia`

Calcula o mapa com VESSELNESS_ARTERY e delega à interface de avaliação.

Importação: `from utils.segmentation.pipeline.arteries import segment_arteries_from_ostia`.

```text
segment_arteries_from_ostia(
    lcc_image: Any,
    label_artery: Any,
    ostia_left: Optional[Sequence[int]],
    ostia_right: Optional[Sequence[int]],
    config: Dict[str, Any],
) -> Dict[str, Any]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `lcc_image` | `Any` | Sim | Volume pré-processado 3D, com HU preservados na região selecionada. |
| `label_artery` | `Any` | Sim | Referência coronariana binária, alinhada à resolução da predição. |
| `ostia_left` | `Optional[Sequence[int]]` | Sim | Semente esquerda em índices (eixo 0, eixo 1, z), ou None quando admitido. |
| `ostia_right` | `Optional[Sequence[int]]` | Sim | Semente direita na mesma convenção; None representa ausência do segundo óstio. |
| `config` | `Dict[str, Any]` | Sim | Configuração completa normalizada e escalada uma única vez; chaves consumidas descritas nesta etapa. |

**Saída:** O mesmo dict de segment_arteries_from_vesselness; exige referência binária compatível.

```python
from utils.segmentation.pipeline.arteries import segment_arteries_from_ostia

with_computed_map = segment_arteries_from_ostia(
    image, label_artery, left, right, config
)
assert with_computed_map["artery_mask"].shape == image.shape
```

### Fuzzy connectedness

Para selecioná-la, use `method="fuzzy_connectedness"` no wrapper ou configure `ARTERY_SEGMENTATION.method` com esse valor. O bloco `FUZZY_CONNECTEDNESS` já está no JSON do projeto.

| Chave | Valor de referência | Efeito / unidade |
|---|---|---|
| `alpha` | 0.16 | Limiar absoluto de conectividade, no intervalo (0, 1]. |
| `sigma_hu` | 100 | Escala gaussiana de similaridade HU, positiva. |
| `neighborhood` | 26 | Conectividade 6/18/26. |
| `candidate_min_vesselness` | 0.018 | Piso no mapa normalizado para candidatos. |
| `max_candidate_voxels` / `max_processed_voxels` | 500000 / 500000 | Limites de candidatos e propagação; podem truncar a segmentação. |
| `seed_search_radius` / `max_seeds_per_ostium` | 2 / 4 | Raio em voxels e número máximo de sementes por óstio. |
| `seed_min_vesselness` / `min_seed_distance_voxels` | 0.018 / 1 | Piso das sementes e separação mínima em voxels. |
| `vesselness_floor` / `vesselness_weight` | 0.018 / 0.9 | Piso da afinidade e peso no produto com similaridade HU. |
| `grow_each_ostium_separately` | False | Propagação conjunta por padrão. |

A conectividade de um caminho é sua menor afinidade; o mapa registra o melhor caminho até cada voxel. A máscara é connectivity >= alpha. A normalização interna do mapa e o uso de HU distinguem FC de RG; seus limiares não são intercambiáveis.

### `fuzzy_connectedness_map`

Calcula propagação max-min a partir de sementes, sem criar a máscara final por alpha.

Importação: `from utils.segmentation.fuzzy.connectedness import fuzzy_connectedness_map`.

```text
fuzzy_connectedness_map(
    image: NDArray[Any],
    vesselness: NDArray[Any],
    seeds: Iterable[Sequence[int] | None],
    *,
    sigma_hu: float,
    neighborhood: int = 26,
    candidate_mask: NDArray[Any] | None = None,
    max_processed_voxels: int | None = None,
    min_connectivity_to_process: float = 0.0,
    vesselness_floor: float = 0.0,
    vesselness_weight: float = 0.9,
) -> tuple[NDArray[np.float32], dict[str, Any]]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `image` | `NDArray[Any]` | Sim | Imagem/volume numérico de entrada; intensidades em HU quando usado como CCTA. |
| `vesselness` | `NDArray[Any]` | Sim | Mapa 3D de evidência vascular, com o mesmo shape da imagem. |
| `seeds` | `Iterable[Sequence[int] \| None]` | Sim | Iterável de índices (eixo 0, eixo 1, z); None e sementes fora do volume são ignoradas. |
| `sigma_hu` | `float` | Sim | Escala positiva da similaridade gaussiana entre intensidades, em HU. |
| `neighborhood` | `int` | `26` | Conectividade espacial 3D: 6, 18 ou 26. |
| `candidate_mask` | `NDArray[Any] \| None` | `None` | Região booleana de propagação; None libera o volume todo. Sementes válidas são incluídas na cópia interna da máscara. |
| `max_processed_voxels` | `int \| None` | `None` | Limite de processamento; None remove esse limite. |
| `min_connectivity_to_process` | `float` | `0.0` | Piso de conectividade para colocar novos vizinhos na fila. |
| `vesselness_floor` | `float` | `0.0` | Piso suave da afinidade vascular, no intervalo [0, 1]. |
| `vesselness_weight` | `float` | `0.9` | Peso do vesselness no produto ponderado com similaridade HU, no intervalo [0, 1]. |

**Saída:** (mapa float32 no shape da imagem, detalhes de propagação). Sementes válidas começam em 1; sem sementes válidas o mapa fica zero.

```python
from utils.segmentation.fuzzy.connectedness import fuzzy_connectedness_map

connectivity, propagation = fuzzy_connectedness_map(
    image,
    vesselness,
    [left, right],
    sigma_hu=100,
    candidate_mask=(vesselness > 0),
    max_processed_voxels=1000,
    min_connectivity_to_process=0.16,
)
assert connectivity.dtype == np.float32 and connectivity.shape == image.shape
```

### `fuzzy_connectedness_segmentation`

Calcula conectividade com poda por alpha e aplica esse limiar absoluto.

Importação: `from utils.segmentation.fuzzy.connectedness import fuzzy_connectedness_segmentation`.

```text
fuzzy_connectedness_segmentation(
    image: NDArray[Any],
    vesselness: NDArray[Any],
    object_seeds: Iterable[Sequence[int] | None],
    *,
    alpha: float,
    sigma_hu: float,
    neighborhood: int,
    candidate_mask: NDArray[Any],
    max_processed_voxels: int | None = None,
    vesselness_floor: float = 0.0,
    vesselness_weight: float = 0.9,
) -> dict[str, Any]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `image` | `NDArray[Any]` | Sim | Imagem/volume numérico de entrada; intensidades em HU quando usado como CCTA. |
| `vesselness` | `NDArray[Any]` | Sim | Mapa 3D de evidência vascular, com o mesmo shape da imagem. |
| `object_seeds` | `Iterable[Sequence[int] \| None]` | Sim | Iterável de sementes do objeto em índices diretos. |
| `alpha` | `float` | Sim | Limiar absoluto de conectividade; usar 0 < alpha <= 1 para não selecionar fundo de conectividade zero. |
| `sigma_hu` | `float` | Sim | Escala positiva da similaridade gaussiana entre intensidades, em HU. |
| `neighborhood` | `int` | Sim | Conectividade espacial 3D: 6, 18 ou 26. |
| `candidate_mask` | `NDArray[Any]` | Sim | Máscara booleana permitida para a propagação, no mesmo shape da imagem. |
| `max_processed_voxels` | `int \| None` | `None` | Limite de processamento; None remove esse limite. |
| `vesselness_floor` | `float` | `0.0` | Piso suave da afinidade vascular, no intervalo [0, 1]. |
| `vesselness_weight` | `float` | `0.9` | Peso do vesselness no produto ponderado com similaridade HU, no intervalo [0, 1]. |

**Saída:** Dict com connectivity (float32), mask (uint8 bruta) e details (parâmetros/contagens).

```python
from utils.segmentation.fuzzy.connectedness import fuzzy_connectedness_segmentation

fc_raw = fuzzy_connectedness_segmentation(
    image,
    vesselness,
    [left, right],
    alpha=0.16,
    sigma_hu=100,
    neighborhood=26,
    candidate_mask=(vesselness > 0),
    max_processed_voxels=1000,
)
assert fc_raw["mask"].dtype == np.uint8
```

### `segment_artery_fuzzy_connectedness`

API direta da FC: restringe candidatos, refina sementes, propaga e opcionalmente pós-processa.

Importação: `from utils.segmentation.fuzzy.connectedness import segment_artery_fuzzy_connectedness`.

```text
segment_artery_fuzzy_connectedness(
    image: NDArray[Any],
    vesselness_artery: NDArray[Any],
    ostia_seeds: Iterable[Sequence[int] | None],
    lcc_mask: NDArray[Any],
    config: dict[str, Any],
    *,
    params: dict[str, Any],
    max_candidate_voxels: int | None = 500000,
    max_processed_voxels: int | None = 500000,
    apply_postprocessing: bool = True,
) -> dict[str, Any]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `image` | `NDArray[Any]` | Sim | Imagem/volume numérico de entrada; intensidades em HU quando usado como CCTA. |
| `vesselness_artery` | `NDArray[Any]` | Sim | Mapa arterial 3D com o mesmo shape do volume; não é máscara binária. |
| `ostia_seeds` | `Iterable[Sequence[int] \| None]` | Sim | Iterável dos óstios em índices diretos, admitindo None. |
| `lcc_mask` | `NDArray[Any]` | Sim | Máscara candidata exata da região anatômica, no shape da imagem. |
| `config` | `dict[str, Any]` | Sim | Configuração completa normalizada e escalada uma única vez; chaves consumidas descritas nesta etapa. |
| `params` | `dict[str, Any]` | Sim | Bloco FUZZY_CONNECTEDNESS; alpha, sigma_hu e neighborhood são obrigatórios nesta API. |
| `max_candidate_voxels` | `int \| None` | `500000` | Limite da região candidata; None desativa esse limite. |
| `max_processed_voxels` | `int \| None` | `500000` | Limite de processamento; None remove esse limite. |
| `apply_postprocessing` | `bool` | `True` | True aplica POSTPROCESSING; False mantém artery_mask igual à máscara bruta. |

**Saída:** Dict: raw_mask (uint8), artery_mask (uint8, final ou cópia da bruta), connectivity (float32) e details (candidatos, sementes, propagação e contagens).

```python
from utils.segmentation.fuzzy.connectedness import segment_artery_fuzzy_connectedness

fc_direct = segment_artery_fuzzy_connectedness(
    image,
    vesselness,
    [left, right],
    vesselness > 0,
    config,
    params=config["FUZZY_CONNECTEDNESS"],
    max_candidate_voxels=1000,
    max_processed_voxels=1000,
    apply_postprocessing=False,
)
assert np.array_equal(fc_direct["raw_mask"], fc_direct["artery_mask"])
```

Na chamada direta, max_candidate_voxels e max_processed_voxels devem ser passados pelos argumentos próprios: inseri-los somente em params não substitui os argumentos. O wrapper do pipeline extrai esses dois valores da configuração por você. Ele aproxima a máscara candidata por lcc_image > MIN_THRESHOLD; para fornecer lcc_mask exata, use esta API direta.

## Falhas e cuidados

- Volume, mapa, máscara candidata e referência devem estar alinhados e com o mesmo shape.
- O RG retorna máscara vazia para mapa sem resposta positiva ou sem sementes utilizáveis; None não deve virar índice de array.
- A FC ignora sementes fora do volume; sem sementes válidas não há propagação. Revise seeds/candidatos quando a máscara bruta estiver vazia.
- Limites de volume/candidatos/processamento são controles de custo, não certificados de completude anatômica.
- As interfaces que calculam Dice exigem referência. Não passe label fictício para inferência; use segment_artery_masks_from_vesselness.
