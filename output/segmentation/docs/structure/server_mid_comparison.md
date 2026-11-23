# ImageCAS mid no servidor

A série mede OpenCV linear e SciPy cúbico nos mesmos 30 exames de treino,
270 de validação e 700 de teste. Use a mesma revisão do código, ambiente,
dataset e GPU nas seis execuções, processando uma combinação por vez.
Os tempos desta série representam o pipeline completo com avaliação de
distâncias e monitoramento de recursos habilitados.

A variante SciPy cúbica da série de 20/11/2026 foi promovida a canonical mid.
O preparo abaixo parte do baseline OpenCV arquivado em
`before_server_mid_scipy_cubic` para reproduzir a comparação da imagem.
Os runs promovidos registram HD e AHD de superfície. A CLI atual grava as
distâncias do protocolo 3 quando a avaliação está habilitada.

## Preparar o ambiente e as referências

Na raiz do repositório no servidor, ative o ambiente GPU validado. Defina
`IMAGECAS_BASE_PATH` no terminal com o diretório local dos volumes ImageCAS.
O comando seguinte exige que essa variável já esteja preenchida.

```bash
conda activate dac_gpu
export CUDA_VISIBLE_DEVICES=1
: "${IMAGECAS_BASE_PATH:?Defina IMAGECAS_BASE_PATH com os volumes ImageCAS}"
```

Na GPU selecionada, este teste deve imprimir `45`.

```bash
python -X faulthandler -u -c 'import cupy as cp; print(cp.cuda.runtime.getDeviceProperties(0)["name"]); print(cp.arange(10).sum().get())'
```

Se esse teste apresentar o conflito de bibliotecas CUDA já observado no
Curupira, aplique no mesmo terminal os caminhos do ambiente Conda e repita-o.

```bash
export CUDA_PATH="$CONDA_PREFIX/targets/x86_64-linux"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_PATH/lib"
export LD_PRELOAD="$CUDA_PATH/lib/libcudart.so.12:$CUDA_PATH/lib/libnvrtc.so.12:$CUDA_PATH/lib/libnvJitLink.so.12"
```

O preparo abaixo copia as referências para a própria série e grava versões,
hardware e hashes. Os snapshots mantêm o nome `effective_pipeline_config.json`,
cujo carregamento reproduz `legacy_zoom_order0` nos arquivos históricos sem
`LABEL_RESAMPLING`. Cópias existentes são conferidas antes de continuar.

```bash
python - <<'PY'
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp

series = Path("output/segmentation/runs/mid_res/server_mid_comparison")
(series / "logs").mkdir(parents=True, exist_ok=True)
references = {}
for split in ("train", "val", "test"):
    reference = Path(
        f"output/segmentation/canonical/mid_res/{split}/before_server_mid_scipy_cubic"
    )
    for relative in ("config/effective_pipeline_config.json", f"numeric/results_{split}.csv"):
        source = reference / relative
        data = source.read_bytes()
        destination = series / "reference" / split / relative
        if destination.exists() and destination.read_bytes() != data:
            raise ValueError(f"A referência mudou desde o preparo anterior: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        references[str(destination)] = hashlib.sha256(data).hexdigest()
source = Path("config/imagecas_splits.json")
destination = series / "reference/imagecas_splits.json"
if destination.exists() and destination.read_bytes() != source.read_bytes():
    raise ValueError("Os splits mudaram desde o preparo anterior.")
destination.write_bytes(source.read_bytes())
references[str(destination)] = hashlib.sha256(destination.read_bytes()).hexdigest()
packages = {}
for name in ("numpy", "scipy", "scikit-image", "opencv-python", "nibabel", "SimpleITK", "cupy", "cupy-cuda12x"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        continue
source_files = sorted([*Path("src").rglob("*.py"), *Path("src").rglob("*.ipynb"), Path("pyproject.toml"), Path("uv.lock")])
source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
environment = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "source_hashes": source_hashes,
    "reference_hashes": references,
    "host": platform.node(),
    "platform": platform.platform(),
    "python": platform.python_version(),
    "packages": packages,
    "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
    "cuda_driver": cp.cuda.runtime.driverGetVersion(),
    "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
    "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"], text=True).strip(),
}
stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f")
destination = series / f"environment_{stamp}.json"
destination.write_text(json.dumps(environment, indent=2, ensure_ascii=False) + "\n")
print(destination)
PY
```

Execute os seis comandos com esse mesmo código e ambiente. Alterações no
código durante a série exigem identificar as execuções afetadas ao comparar
os tempos. O monitor registra host, CPU e UUID da GPU nos diagnósticos de cada
run. Evite concorrência com outros lotes e registre disputas de CPU, GPU ou I/O.

## Treino

Inicie o controle OpenCV e aguarde a conclusão.

```bash
nohup python -u src/segmentation_pipeline.py \
  --base-path "$IMAGECAS_BASE_PATH" \
  --resolution mid --split train \
  --split-config output/segmentation/runs/mid_res/server_mid_comparison/reference/imagecas_splits.json \
  --config-file output/segmentation/runs/mid_res/server_mid_comparison/reference/train/config/effective_pipeline_config.json \
  --config-replace --downscale-method opencv --opencv-interpolation linear \
  --gpu --distance-metrics --num-batches 5 \
  --monitor-resources --resource-sampling-interval 0.5 \
  --run-group server_mid_comparison/train/opencv_linear \
  > output/segmentation/runs/mid_res/server_mid_comparison/logs/train_opencv.log 2>&1 < /dev/null &
```

Depois execute o treino SciPy.

```bash
nohup python -u src/segmentation_pipeline.py \
  --base-path "$IMAGECAS_BASE_PATH" \
  --resolution mid --split train \
  --split-config output/segmentation/runs/mid_res/server_mid_comparison/reference/imagecas_splits.json \
  --config-file output/segmentation/runs/mid_res/server_mid_comparison/reference/train/config/effective_pipeline_config.json \
  --config-replace --downscale-method scipy \
  --gpu --distance-metrics --num-batches 5 \
  --monitor-resources --resource-sampling-interval 0.5 \
  --run-group server_mid_comparison/train/scipy_cubic \
  > output/segmentation/runs/mid_res/server_mid_comparison/logs/train_scipy.log 2>&1 < /dev/null &
```

Compare o controle OpenCV com a cópia arquivada do treino antes dos lotes
maiores. Confira os 30 IDs, Dice, volumes arteriais e da aorta, coordenadas dos
óstios, classificações e falhas. Os novos runs anteriores à padronização já
apresentaram diferenças nesses campos; divergências restantes precisam de
investigação para estabelecer preservação pela refatoração.

## Validação

Execute estes comandos individualmente após a conferência do treino.

```bash
nohup python -u src/segmentation_pipeline.py \
  --base-path "$IMAGECAS_BASE_PATH" \
  --resolution mid --split val \
  --split-config output/segmentation/runs/mid_res/server_mid_comparison/reference/imagecas_splits.json \
  --config-file output/segmentation/runs/mid_res/server_mid_comparison/reference/val/config/effective_pipeline_config.json \
  --config-replace --downscale-method opencv --opencv-interpolation linear \
  --gpu --distance-metrics --num-batches 5 \
  --monitor-resources --resource-sampling-interval 0.5 \
  --run-group server_mid_comparison/val/opencv_linear \
  > output/segmentation/runs/mid_res/server_mid_comparison/logs/val_opencv.log 2>&1 < /dev/null &
```

```bash
nohup python -u src/segmentation_pipeline.py \
  --base-path "$IMAGECAS_BASE_PATH" \
  --resolution mid --split val \
  --split-config output/segmentation/runs/mid_res/server_mid_comparison/reference/imagecas_splits.json \
  --config-file output/segmentation/runs/mid_res/server_mid_comparison/reference/val/config/effective_pipeline_config.json \
  --config-replace --downscale-method scipy \
  --gpu --distance-metrics --num-batches 5 \
  --monitor-resources --resource-sampling-interval 0.5 \
  --run-group server_mid_comparison/val/scipy_cubic \
  > output/segmentation/runs/mid_res/server_mid_comparison/logs/val_scipy.log 2>&1 < /dev/null &
```

Use treino e validação para decidir sobre o método e fixar o protocolo final.

## Teste

Avalie a configuração fixada e o controle de tempo no mesmo ambiente,
aguardando a conclusão de cada execução.

```bash
nohup python -u src/segmentation_pipeline.py \
  --base-path "$IMAGECAS_BASE_PATH" \
  --resolution mid --split test \
  --split-config output/segmentation/runs/mid_res/server_mid_comparison/reference/imagecas_splits.json \
  --config-file output/segmentation/runs/mid_res/server_mid_comparison/reference/test/config/effective_pipeline_config.json \
  --config-replace --downscale-method opencv --opencv-interpolation linear \
  --gpu --distance-metrics --num-batches 5 \
  --monitor-resources --resource-sampling-interval 0.5 \
  --run-group server_mid_comparison/test/opencv_linear \
  > output/segmentation/runs/mid_res/server_mid_comparison/logs/test_opencv.log 2>&1 < /dev/null &
```

```bash
nohup python -u src/segmentation_pipeline.py \
  --base-path "$IMAGECAS_BASE_PATH" \
  --resolution mid --split test \
  --split-config output/segmentation/runs/mid_res/server_mid_comparison/reference/imagecas_splits.json \
  --config-file output/segmentation/runs/mid_res/server_mid_comparison/reference/test/config/effective_pipeline_config.json \
  --config-replace --downscale-method scipy \
  --gpu --distance-metrics --num-batches 5 \
  --monitor-resources --resource-sampling-interval 0.5 \
  --run-group server_mid_comparison/test/scipy_cubic \
  > output/segmentation/runs/mid_res/server_mid_comparison/logs/test_scipy.log 2>&1 < /dev/null &
```

## Acompanhamento e interpretação

Para acompanhar o treino OpenCV, use o comando abaixo. `Ctrl+C` encerra a
leitura do log; o processo lançado com `nohup` continua executando.

```bash
tail -n 80 -F output/segmentation/runs/mid_res/server_mid_comparison/logs/train_opencv.log
```

Os diretórios das execuções usam timestamps dentro de cada grupo.
`numeric/batch_timings_<split>.csv` fornece os tempos por lote e
`numeric/metadata_<split>.json` registra a duração consolidada disponível.
Esse tempo inclui leitura dos volumes, segmentação, avaliação das distâncias,
monitoramento e escrita dos lotes. O preparo do ambiente e as importações
anteriores aos lotes ficam fora dessa soma. Retomadas devem ser identificadas
na análise de desempenho.

O protocolo 3 grava `artery_hd_foreground_mm`, `artery_avghd_foreground_mm`
e `artery_hd95_surface_mm`. Os dados SciPy promovidos contêm `artery_hd_mm`
e `artery_ahd_mm`, definidos sobre superfícies. Compare definições e denominadores
antes de reunir esses valores em uma tabela.

Novas execuções geram candidatos. A promoção depende da auditoria dos resultados
completos e de uma decisão explícita sobre a referência adotada.
