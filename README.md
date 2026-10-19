# Coronary Centerline Detection

Pipeline de pesquisa para localização da aorta e dos óstios e segmentação de artérias coronárias em angiografia por tomografia computadorizada (CCTA). A extração de linhas centrais é uma extensão planejada.

## 📋 Descrição

Este projeto implementa um pipeline de segmentação a partir de volumes 3D de CCTA. O fluxo atual inclui:

- **Filtro de Frangi multiescala** para realce de estruturas vasculares
- **Transformada de Hough** para detecção da aorta ascendente
- **Segmentação por contorno ativo** (Level Set) para delimitar a aorta
- **Detecção automática de óstios coronarianos** (esquerdo e direito)
- **Crescimento de região 3D** ou **fuzzy connectedness** para segmentação das artérias coronárias

## 🎯 Objetivos

O objetivo de longo prazo é automatizar a extração das linhas centrais das principais artérias coronárias (LCA - Left Coronary Artery e RCA - Right Coronary Artery), facilitando:

- Análise quantitativa de vasos coronários
- Planejamento de procedimentos intervencionistas
- Avaliação de estenoses e placas
- Visualização 3D de estruturas vasculares

## 🏗️ Estrutura do Projeto

```file
coronary-centerline-detection/
├── pyproject.toml              # Configuração e dependências do projeto
├── README.md                   # Este arquivo
├── doc/                        # Documentação detalhada
│   ├── manual_pipeline/        # Manual de uso das funções, organizado por etapa
│   ├── Fluxograma das etapas centerline.md
│   └── Relatorio das etapas realizadas.md
├── src/                        # Código fonte
│   ├── segmentation_pipeline.py    # Pipeline principal de processamento
│   ├── external_ccta_batch_pipeline.py # Pipeline em lote OrCaScore/MM-WHS
│   ├── main.ipynb                  # Notebook de execução principal
│   ├── external_ccta_pipeline.ipynb # Pipeline interativo OrCaScore/MM-WHS
│   ├── eda/                        # Notebooks de análise exploratória
│   │   ├── README.md               # Catálogo único das análises
│   │   ├── images/                 # Imagens, datasets e pré-processamento
│   │   ├── intensity/              # Intensidades HU e limiares
│   │   ├── results/                # Resultados gerais e avaliações visuais
│   │   ├── comparisons/            # Métodos, variantes e resoluções
│   │   ├── aorta/                  # Círculos, volume e qualidade da aorta
│   │   └── sensitivity/            # Sensibilidade e thresholds
│   └── utils/                      # Módulos utilitários
│       ├── project/
│       │   ├── datasets/           # Descoberta, carregamento e splits
│       │   ├── results/            # Schemas, persistência e metadata
│       │   ├── analysis/           # Cálculos e validações das EDAs
│       │   ├── evaluation/         # Avaliação oficial MM-WHS
│       │   ├── runtime/            # Ambiente, caminhos, logs e avisos
│       │   ├── config.py           # Configuração compartilhada
│       │   └── dataframe.py        # Acesso tipado a dados tabulares
│       ├── segmentation/
│       │   ├── aorta/              # Localização, segmentação e diagnósticos
│       │   ├── fuzzy/              # Limiarização e conectividade fuzzy
│       │   ├── pipeline/           # Etapas e coordenação dos pipelines
│       │   └── *.py                # Artérias, óstios, limiar inferior e backends
│       ├── processing/             # Pré-processamento, Frangi e GPU
│       ├── visualization/          # Visualizações genéricas 2D/3D
│       │   ├── images/             # Fatias, volumes e intensidades
│       │   ├── pipeline/           # Etapas intermediárias e artefatos
│       │   └── results/            # Gráficos e rótulos de resultados
│       ├── comparison_utils/       # Leitura, resumos e comparação de resultados
│       ├── experiments/            # Helpers dos experimentos
│       └── utils/                  # NIfTI, JSON, métricas e ROI
└── output/                     # Resultados e visualizações
    ├── segmentation/
    │   ├── runs/               # Execuções completas por resolução/timestamp
    │   ├── canonical/          # Referências estáveis para EDA/comparações
    │   ├── analysis/           # Comparações, bad cases e exemplos visuais
    │   └── 8.final_results/    # Resultados legados
    ├── ia_results/             # Resultados da IA por resolução/fold
    └── tmp/                    # Experimentos temporários
```

## 📖 Manual para desenvolvedores

O [manual do pipeline](doc/manual_pipeline/README.md) apresenta preparação do
ambiente, convenções de dados e um exemplo integrado de inferência sem referência.
Cada etapa documenta funções, imports, assinaturas, parâmetros, exemplos e saídas:

1. [Pré-processamento](doc/manual_pipeline/01_preprocessamento/README.md): carregamento, downsampling, threshold normal/fuzzy e LCC.
2. [Mapa de vasos](doc/manual_pipeline/02_mapa_vasos/README.md): Frangi, normalização e CPU/GPU.
3. [Localização da aorta](doc/manual_pipeline/03_localizacao_aorta/README.md): Hough e filtro de trajetória.
4. [Segmentação da aorta](doc/manual_pipeline/04_segmentacao_aorta/README.md): level set, envelope e diagnósticos.
5. [Localização dos óstios](doc/manual_pipeline/05_localizacao_ostios/README.md): superfície, candidatos e avaliação opcional.
6. [Segmentação coronariana](doc/manual_pipeline/06_segmentacao_coronarias/README.md): crescimento de região e fuzzy connectedness.
7. [Pós-processamento e avaliação](doc/manual_pipeline/07_posprocessamento_avaliacao/README.md): estágios morfológicos e métricas.

## 🔬 Metodologia

O padrão em [pipeline_config.json](config/pipeline_config.json) é definido para
mid, com fatores de redução `(2, 2, 1)`. O pré-processamento usa OpenCV linear,
limiar inferior fixo de -300 HU, teto no percentil 99.9 e maior componente
conectada por fatia. Os modos fuzzy e sem corte HU são alternativas configuráveis.

A Hough recebe a imagem pré-processada e acompanha círculos a partir da última
fatia, com raios de 18 a 29 pixels em mid (fim exclusivo 30). A trajetória passa
por filtro robusto e inicializa o level set com 10% do raio detectado. A evolução
usa 26 iterações e balloon 0.6, seguida de abertura, envelope da trajetória e
seleção da maior componente 3D.

O mapa `VESSELNESS_AORTA`, calculado por Frangi, seleciona candidatos de óstios
na superfície da aorta. A configuração atual busca em toda a extensão segmentada,
com erosão de raio 4 e padding 2. O mapa `VESSELNESS_ARTERY` alimenta a segmentação
coronariana por crescimento de região, ou por fuzzy connectedness quando selecionado.
O pós-processamento arterial padrão aplica fechamento de raio 3 e dilatação de
raio 2, em voxels da resolução de trabalho.

Esses valores são os do JSON de referência em mid. O ajuste de resolução altera
parâmetros espaciais e heurísticas em high; consulte a configuração efetiva do
run. As funções de baixo nível podem ter defaults diferentes dos valores do JSON.
O manual detalha essas diferenças e as convenções de coordenadas/spacing.

## 🚀 Instalação

### Requisitos

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) (gerenciador de pacotes recomendado)

### Instalação das Dependências

```bash
# Clone o repositório
git clone https://github.com/matheusssilva991/coronary-centerline-detection.git
cd coronary-centerline-detection

# Instale as dependências usando uv (recomendado)
uv sync

# Ou usando pip
pip install -e .
```

### Dependências Principais

- `scikit-image`: Frangi, Hough e contorno ativo morfológico
- `nibabel`: Leitura de arquivos NIfTI
- `opencv-python`: Redimensionamento de imagens
- `scipy`: Interpolação, componentes conectados e operações morfológicas
- `matplotlib`, `plotly`, `seaborn`: Visualização
- `pandas`: Análise de dados
- `scikit-learn`: Divisão train/test

## 💻 Uso

### Um exame ImageCAS pela API

Execute a partir da raiz com `PYTHONPATH=src uv run python seu_script.py`.
Defina `IMAGECAS_BASE_PATH` para a pasta dos pares `<id>.img.nii.gz` e
`<id>.label.nii.gz`, e `IMAGECAS_EXAM_ID` para o ID desejado. Este exemplo depende
de dataset real e não foi executado na validação do manual.

<!-- example: dataset -->
```python
import os
from utils.project.config import load_config_json, scale_config_to_resolution
from utils.segmentation.pipeline.orchestration import process_image

config = load_config_json("config/pipeline_config.json", {})
config["USE_GPU"] = False
config = scale_config_to_resolution(config)
result = process_image(
    img_id=os.environ["IMAGECAS_EXAM_ID"],
    config=config,
    base_path=os.environ["IMAGECAS_BASE_PATH"],
)
print(result.get("error"), result.get("skip_reason"), result.get("dice_artery"))
```

`process_image` retorna métricas e estados do exame, e pode registrar falhas no
próprio resultado. Para obter as máscaras intermediárias/finais e processar um
volume sem referência, use as [funções por etapa](doc/manual_pipeline/README.md).

### Processamento em Lote

Para receber um aviso no desktop e um som curto ao término de um run ImageCAS,
adicione `--notify` ao comando de `src/segmentation_pipeline.py`. A opção também
vale para retomadas e `--merge-only`; permanece desligada por padrão.

```bash
uv run python src/segmentation_pipeline.py --split train --resolution mid --notify
```

Para repetir um snapshot histórico sem herdar opções adicionadas posteriormente
em `config/pipeline_config.json`, use `--config-file <snapshot.json>` junto de
`--config-replace`. As opções explícitas da CLI, como `--downscale-method`, ainda
prevalecem sobre o snapshot.

Para a API de lote, reutilize a configuração e a variável `os` do exemplo
anterior. O diretório de saída é obrigatório. Este exemplo inicia processamento
de exames quando executado; não foi executado na validação do manual.

<!-- example: dataset -->
```python
from utils.segmentation.pipeline.orchestration import run_pipeline

results = run_pipeline(
    ids=[1, 2, 3],
    split_name="train",
    config=config,
    base_path=os.environ["IMAGECAS_BASE_PATH"],
    output_dir="output/segmentation/manual_example/numeric",
)
print(len(results["details"]), results["batches_processed"])
```

O número solicitado de lotes é configurado por `NUM_BATCHES` (5 no JSON de
referência); o planejamento considera a quantidade de exames selecionados.

### Notebooks Jupyter

Execute os notebooks para análise exploratória e visualização:

```bash
uv run jupyter lab
```

O pipeline interativo do ImageCAS está em [`src/main.ipynb`](src/main.ipynb), e
o equivalente para OrCaScore/MM-WHS está em
[`src/external_ccta_pipeline.ipynb`](src/external_ccta_pipeline.ipynb). Consulte o
[`src/eda/README.md`](src/eda/README.md) para o catálogo completo das análises,
suas entradas, saídas e custos aproximados.

### Pipeline em lote para OrCaScore e MM-WHS

O script [`src/external_ccta_batch_pipeline.py`](src/external_ccta_batch_pipeline.py)
executa, por padrão, todos os exames `train` e `test` do banco selecionado. Ele
usa a mesma configuração e as mesmas etapas do notebook externo:

```bash
uv run python src/external_ccta_batch_pipeline.py \
  --dataset orcascore \
  --resolution mid \
  --gpu

uv run python src/external_ccta_batch_pipeline.py \
  --dataset mmwhs \
  --resolution high \
  --gpu
```

Adicione `--notify` a qualquer comando em lote para receber o aviso ao terminar.
O estado informado vem do metadata do run; falhas da avaliação oficial da aorta
são indicadas separadamente das falhas de segmentação. Os avisos usam
`notify-send` e `canberra-gtk-play` e requerem uma sessão gráfica ativa. Se um
desses comandos não estiver disponível, o resultado do run não será alterado.

Os aliases `orca`, `whs` e `owhs` também são aceitos. Para executar somente um
subset ou validar poucos exames antes do lote completo:

Use `--no-hu-threshold` em uma execução nova para preservar todos os voxels
finitos após o downscale, sem corte de HU e sem LCC. O modo `none` fica
registrado em `config/effective_pipeline_config.json` do run. A flag não altera
outros limiares, como o `LEVEL_SET.threshold`.

```bash
uv run python src/external_ccta_batch_pipeline.py \
  --dataset orcascore \
  --resolution mid \
  --subset train \
  --exam-ids TRV1P1 TRV2P2 \
  --no-gpu
```

Uma execução interrompida pode ser retomada com os mesmos argumentos de banco,
resolução, subset, `--exam-ids` e `--limit`:

```bash
uv run python src/external_ccta_batch_pipeline.py \
  --dataset orcascore \
  --resolution mid \
  --resume-dir "${CCTA_RESULTS_ROOT}/orcascore/mid_res/<timestamp>"
```

Defina `CCTA_RESULTS_ROOT` ou informe `--output-root` para escolher a raiz de
resultados em seu ambiente. Os exemplos de retomada pressupõem essa variável
definida. A hierarquia é:

```text
Results_dataset_ccta/
├── imagecas/                         # resultados ImageCAS já existentes
├── orcascore/<resolution>_res/<run>/
└── mmwhs/<resolution>_res/<run>/
    ├── config/                       # configuração efetiva e manifesto
    ├── numeric/                      # results_all/train/test.csv
    ├── logs/
    ├── metadata.json
    └── visual/<subset>/<exam_id>/
        ├── stages/
        │   ├── 00_input/
        │   ├── 01_threshold/
        │   ├── 02_lcc/
        │   ├── 03_aorta/
        │   ├── 04_vesselness_ostia/
        │   ├── 05_vesselness_artery/
        │   ├── 06_artery_raw/
        │   ├── 07_artery_closed/
        │   └── 08_artery_final/
        ├── aorta_circles.png
        ├── aorta_ground_truth_comparison.html # MM-WHS train
        ├── aorta_ostia_artery.html
        └── result.json
```

Cada pasta de etapa contém `mip_axial.png`, `first_slice.png`,
`middle_slice.png` e `last_slice.png`. O downscale não recebe uma pasta visual
própria. Os 20 exames CT de treino do MM-WHS possuem o rótulo 820 da aorta
ascendente; nesses casos, o pipeline calcula `aorta_dice` no volume processado e
salva a comparação 3D com o ground truth. Como os bancos não possuem referência
coronariana compatível com o ImageCAS, o Dice arterial e a acurácia dos óstios
permanecem nulos.

Para avaliar a aorta dos exames CT de teste com os labels criptografados,
instale Wine e habilite explicitamente o avaliador oficial:

```bash
uv run python src/external_ccta_batch_pipeline.py \
  --dataset mmwhs --resolution high --subset test \
  --test-aorta-dice --gpu
```

Para acrescentar o Dice a um run existente sem repetir óstios e artérias,
use os mesmos filtros da seleção original (por exemplo, `--subset all`):

```bash
uv run python src/external_ccta_batch_pipeline.py \
  --dataset mmwhs --resolution high --subset all \
  --resume-dir "${CCTA_RESULTS_ROOT}/mmwhs/high_res/<run>" \
  --test-aorta-dice --aorta-eval-only
```

O destino é o próprio run em `CCTA_RESULTS_ROOT` (ou `--output-root`):
`evaluation/aorta/test/<exam_id>/` contém a predição NIfTI, sua versão em
1 mm e o resultado tabulado `*_dice.xls`. A sexta coluna, `DiceLO`, é a
aorta. O Dice de treino é calculado diretamente no volume processado; o de
teste segue o protocolo oficial em 1 mm. O metadata apresenta médias
separadas. Exames sem máscara da aorta mantêm Dice ausente, não zero.

O status `success` indica que todas as etapas computacionais terminaram; não é
uma validação anatômica. Os parâmetros de localização foram ajustados no
ImageCAS e podem selecionar estruturas circulares periféricas nos bancos
externos. Antes de interpretar volumes ou artérias, revise obrigatoriamente
`aorta_circles.png` e `aorta_ostia_artery.html`.

## 📊 Resultados

### Métricas de Avaliação

O sistema avalia a qualidade da detecção de óstios através de:

- **Interseção com ground truth**: óstio detectado está dentro da máscara de referência
- **Distância física**: distância em mm até a máscara de referência
- **Critério tolerável**: distância ≤ 7mm considerada aceitável
- **Dice Score**: avaliação da segmentação arterial completa

### Status de Detecção

- **Ambos corretos**: ambos óstios dentro da máscara GT
- **Ambos toleráveis**: ambos óstios com distância ≤ 7mm
- **Um correto**: apenas um óstio detectado corretamente
- **Nenhum correto**: falha na detecção

### Saídas Geradas

Os resultados são salvos em:

- `output/segmentation/runs/<resolution>_res/<timestamp>/numeric/`: CSVs e metadados do pipeline
- `output/segmentation/runs/<resolution>_res/<timestamp>/visual/`: exemplos visuais associados ao run
- `output/segmentation/runs/<resolution>_res/<timestamp>/config/`: config efetiva e IDs usados
- `output/segmentation/analysis/`: análises derivadas dos notebooks
- `output/segmentation/8.final_results/`: resultados legados mantidos como referência
- `CCTA_RESULTS_ROOT` ou `--output-root`: artefatos dos bancos externos

## 🔧 Configuração

### Parâmetros Principais

Use [config/pipeline_config.json](config/pipeline_config.json) e sua
[referência de parâmetros](config/README.md). Os blocos principais são:

- `USE_GPU`, `DOWNSCALE_METHOD` e `DOWNSCALE_FACTORS`: backend e resolução.
- `THRESHOLDING` e `LOWER_THRESHOLD`: seleção de candidatos no pré-processamento.
- `VESSELNESS_AORTA` e `VESSELNESS_ARTERY`: parâmetros dos mapas de vasos.
- `CIRCLE_DETECTION`, `LEVEL_SET` e `OSTIA_DETECTION`: aorta e óstios.
- `ARTERY_SEGMENTATION`, `REGION_GROWING` e `FUZZY_CONNECTEDNESS`: método arterial.
- `POSTPROCESSING`: fechamento e dilatação finais.
- `NUM_BATCHES`: divisão do conjunto em lotes.

Na API Python, carregue e normalize o JSON e aplique o ajuste de resolução uma
única vez, conforme o [manual](doc/manual_pipeline/README.md). A configuração
efetiva deve acompanhar os resultados para reprodução da execução.

## 📈 Estado Atual do Projeto

### ✅ Etapas Concluídas

- [x] Pré-processamento (downsampling, thresholding, LCC)
- [x] Cálculo do mapa de vesselness (Filtro de Frangi)
- [x] Localização da aorta ascendente (Transformada de Hough)
- [x] Segmentação da aorta (Level Set)
- [x] Detecção de óstios coronarianos
- [x] Segmentação inicial das artérias (Region Growing)
- [x] Avaliação quantitativa (Dice, distâncias)
- [x] Visualizações 3D interativas da aorta, óstios e artérias

### 🚧 Próximos Passos

- [ ] Esqueletização 3D para extração das linhas centrais
- [ ] Refinamento do algoritmo de crescimento de região
- [ ] Otimização de hiperparâmetros
- [ ] Validação em conjunto completo de teste
- [ ] Tratamento de casos edge (aorta fora do 1º quadrante)

## 🔍 Limitações Conhecidas

1. **Posição da aorta**: Abordagem atual (C) assume aorta no **primeiro quadrante** da imagem para melhor precisão
2. **Calcificações severas**: Threshold adaptativo ajuda, mas calcificações muito densas ainda podem interferir na detecção
3. **Anatomias atípicas**: Variações anatômicas significativas (aorta bicúspide, origem anômala de coronárias) podem reduzir precisão
4. **Parâmetros semi-fixos**: Alguns parâmetros otimizados para ImageCAS podem precisar ajuste para outros datasets
5. **Artérias distais**: Vasos muito finos ou de terceira ordem podem não ser capturados completamente
6. **Stents e metal**: Apesar do threshold adaptativo, artefatos metálicos podem afetar vesselness localmente

## 📚 Referências

Este projeto é baseado em técnicas descritas na literatura de processamento de imagens médicas e segmentação vascular, adaptadas para o dataset ImageCAS.

## 👥 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abra um Pull Request

## 📝 Licença

[Adicionar informação de licença]

## 📧 Contato

[Adicionar informações de contato]

---

**Nota**: Este é um projeto de pesquisa em desenvolvimento. Os resultados devem ser validados antes de qualquer aplicação clínica.
