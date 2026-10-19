# 02 — Extração do mapa de vasos

[Manual](../README.md) · [Configuração](../../../config/README.md)

Realça estruturas tubulares na imagem pré-processada. Os dois mapas têm o mesmo shape e os mesmos eixos da entrada, mas usam parâmetros distintos para óstios e artérias.

## Exemplo sintético

Execute a partir da raiz do repositório com `PYTHONPATH=src uv run python seu_exemplo.py`. Este exemplo usa CPU e não representa validação anatômica.

```python
import numpy as np
from utils.project.config import load_config_json, scale_config_to_resolution

config = load_config_json("config/pipeline_config.json", {})
config["USE_GPU"] = False
config = scale_config_to_resolution(config)

from utils.segmentation.pipeline.preprocessing import compute_vesselness

image = np.zeros((20, 20, 8), dtype=np.float32)
image[8:12, 8:12, :] = 300
vesselness = compute_vesselness(image, config["VESSELNESS_AORTA"], use_gpu=False)
artery_map = compute_vesselness(image, config["VESSELNESS_ARTERY"], use_gpu=False)
assert vesselness.shape == image.shape and isinstance(vesselness, np.ndarray)
assert np.all(np.isfinite(vesselness)) and np.all(vesselness >= 0)
print(vesselness.shape, artery_map.shape)  # (20, 20, 8), (20, 20, 8)
```

## Referência das funções

Os exemplos abaixo continuam o estado do exemplo sintético desta página, salvo indicação de entrada real. Os tipos `Any`, `NDArray` e `Sequence` reproduzem as anotações do código; os contratos de conteúdo estão nas tabelas.

### `compute_vesselness`

Wrapper recomendado: traduz o bloco de configuração para o Frangi.

Importação: `from utils.segmentation.pipeline.preprocessing import compute_vesselness`.

```text
compute_vesselness(
    image: Any, vesselness_config: Dict[str, Any], use_gpu: bool = False
) -> Any
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `image` | `Any` | Sim | Imagem/volume numérico de entrada; intensidades em HU quando usado como CCTA. |
| `vesselness_config` | `Dict[str, Any]` | Sim | Bloco VESSELNESS_AORTA ou VESSELNESS_ARTERY; exige sigmas, alpha, beta e gamma. |
| `use_gpu` | `bool` | `False` | Solicita GPU nas operações que a implementam; os wrappers de aorta mantêm execução em CPU. |

**Saída:** Mapa numérico 3D NumPy com o shape da imagem; use_gpu é encaminhado para gpu e o retorno fica na CPU.

```python
same_map = compute_vesselness(image, config["VESSELNESS_ARTERY"], use_gpu=False)
assert same_map.shape == image.shape
```

### `get_vesselness`

Aplica Frangi multiescala do backend, com suavização e normalização opcionais.

Importação: `from utils.processing.frangi import get_vesselness`.

```text
get_vesselness(
    image: Any,
    sigmas: Sequence[float] = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5),
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: Optional[float] = None,
    black_ridges: bool = False,
    normalization: str = "none",
    smooth_sigma: float = 0.0,
    gpu: Optional[bool] = None,
    return_cpu: bool = True,
) -> NDArray[Any]
```

| Parâmetro | Tipo | Obrigatório / padrão da função | Unidade e efeito |
|---|---|---|---|
| `image` | `Any` | Sim | Imagem/volume numérico de entrada; intensidades em HU quando usado como CCTA. |
| `sigmas` | `Sequence[float]` | `(1.0, 1.5, 2.0, 2.5, 3.0, 3.5)` | Escalas gaussianas em voxels, sequência não vazia de valores positivos e finitos. |
| `alpha` | `float` | `0.5` | Parâmetro Frangi da relação de forma entre autovalores (discriminação de estruturas em placa). |
| `beta` | `float` | `0.5` | Parâmetro Frangi que penaliza respostas semelhantes a blobs. |
| `gamma` | `Optional[float]` | `None` | Escala da medida de estrutura/contraste; None deixa o backend estimar. |
| `black_ridges` | `bool` | `False` | False realça estruturas claras; True realça estruturas escuras. |
| `normalization` | `str` | `'none'` | none preserva resposta; minmax usa extremos; robust usa percentis. |
| `smooth_sigma` | `float` | `0.0` | Sigma gaussiano prévio em voxels; 0 desativa. |
| `gpu` | `Optional[bool]` | `None` | None detecta GPU; False usa CPU; True solicita GPU, com fallback quando indisponível ou falha. |
| `return_cpu` | `bool` | `True` | True retorna NumPy; False pode retornar CuPy quando GPU é usada. |

**Saída:** Mapa numérico, não uma probabilidade calibrada. NumPy por padrão; CuPy somente no caminho GPU bem-sucedido com return_cpu=False.

```python
from utils.processing.frangi import get_vesselness

normalized_map = get_vesselness(
    image, sigmas=(1.5, 2.0), gamma=55, normalization="minmax", gpu=False
)
assert normalized_map.shape == image.shape
assert normalized_map.min() >= 0 and normalized_map.max() <= 1
```

### Configuração efetiva dos dois mapas

| Bloco | sigmas (voxels) | alpha / beta / gamma | Demais opções |
|---|---|---|---|
| `VESSELNESS_AORTA` | [2.5, 3.0] | 0.5 / 1.0 / 30 | black_ridges=False, normalization=none, smooth_sigma=0 |
| `VESSELNESS_ARTERY` | [1.5, 2.0, 2.5, 3.0] | 0.5 / 0.5 / 55 | black_ridges=False, normalization=none, smooth_sigma=0 |

O nome `VESSELNESS_AORTA` é histórico: esse mapa seleciona candidatos de **óstios na superfície da aorta**. A Hough e o level set recebem `lcc_image` diretamente. O mapa arterial alimenta crescimento de região ou fuzzy connectedness.

`method="normal"` identifica o Frangi na configuração; `compute_vesselness` não usa esse campo para despachar outros métodos. O código atual não aplica os termos adicionais de grayness/gradiente descritos em documentação antiga.

A normalização minmax preserva a entrada quando o mapa é constante; robust devolve zeros se seus percentis inferior/superior coincidem. Um mapa constante, portanto, exige interpretação própria e não é evidência de boa segmentação.

## Falhas e cuidados

- A resposta não garante presença anatômica de um vaso; inspecione-a junto ao volume.
- Sigmas inválidos e normalização desconhecida geram ValueError. Não confunda unidades de sigma (voxels) com mm.
- O backend GPU pode emitir aviso e retornar pelo caminho CPU. USE_GPU=True não coloca todo o pipeline na GPU.
- A extração pode exigir muita memória em volumes completos. As escalas operam na resolução de trabalho; não escale a configuração novamente.
