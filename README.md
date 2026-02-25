# Coronary Centerline Detection

Sistema automatizado para detecção e extração de linhas centrais de artérias coronárias a partir de imagens de angiografia por tomografia computadorizada (CCTA).

## 📋 Descrição

Este projeto implementa um pipeline completo para segmentação e extração de linhas centrais de artérias coronárias a partir de imagens 3D de CCTA. O sistema utiliza técnicas avançadas de processamento de imagens médicas, incluindo:

- **Filtro de Frangi modificado** para realce de estruturas vasculares
- **Transformada de Hough** para detecção da aorta ascendente
- **Segmentação por contorno ativo** (Level Set) para delimitar a aorta
- **Detecção automática de óstios coronarianos** (esquerdo e direito)
- **Crescimento de região 3D** para segmentação das artérias coronárias

## 🎯 Objetivos

O objetivo principal é automatizar a extração das linhas centrais das principais artérias coronárias (LCA - Left Coronary Artery e RCA - Right Coronary Artery), facilitando:

- Análise quantitativa de vasos coronários
- Planejamento de procedimentos intervencionistas
- Avaliação de estenoses e placas
- Visualização 3D de estruturas vasculares

## 🏗️ Estrutura do Projeto

```
coronary-centerline-detection/
├── pyproject.toml              # Configuração e dependências do projeto
├── README.md                   # Este arquivo
├── doc/                        # Documentação detalhada
│   ├── Fluxograma das etapas centerline.md
│   └── Relatorio das etapas realizadas.md
├── src/                        # Código fonte
│   ├── segmentation_pipeline.py    # Pipeline principal de processamento
│   ├── main.ipynb                  # Notebook de execução principal
│   ├── eda_imgs.ipynb              # Análise exploratória de imagens
│   ├── eda_segmentacao.ipynb       # Análise de segmentações
│   ├── segmentacao_arteria.ipynb   # Notebook de segmentação de artérias
│   └── utils/                      # Módulos utilitários
│       ├── preprocessing.py        # Pré-processamento de imagens
│       ├── frangi.py              # Filtro de Frangi para detecção de vasos
│       ├── aorta_localization.py  # Localização da aorta ascendente
│       ├── aorta_segmentation.py  # Segmentação da aorta
│       ├── ostia_detection.py     # Detecção de óstios coronarianos
│       ├── artery_segmentation.py # Segmentação das artérias
│       ├── plots.py               # Funções de visualização
│       └── utils.py               # Utilitários gerais
└── output/                     # Resultados e visualizações
    ├── detected_circles/       # Círculos detectados na aorta
    ├── segmented_ostios/       # Visualizações dos óstios detectados
    └── *.csv                   # Relatórios de resultados
```

## 🔬 Metodologia

O pipeline de processamento segue as seguintes etapas:

### 1. Pré-processamento

#### 1.1 Downsampling

- Redução da imagem por fator 2 usando interpolação linear
- Objetivo: reduzir custo computacional mantendo qualidade

#### 1.2 Cálculo de Valores Hounsfield

Conversão dos valores de intensidade para Unidades Hounsfield (HU):

$$IU(x, y, z) = m \cdot I(x, y, z) + b$$

#### 1.3 Remoção de Artefatos

- Aplicação de threshold para remover calcificações, stents e ossos
- Limiar adaptado: **600 HU** (ajustado do valor original de 675 HU)

#### 1.4 Extração do Maior Componente Conectado

- Remoção de pulmões e vasos pulmonares
- Isolamento da região de interesse (coração e vasos principais)

### 2. Mapa de Vesselness

Aplicação do **Filtro de Frangi modificado** para realce de estruturas tubulares:

- Análise dos autovalores da matriz Hessiana
- Valores próximos a 1: alta probabilidade de vaso
- Valores próximos a 0: baixa probabilidade

Parâmetros utilizados:

- `alpha = 0.5`: distinção entre estrutura tubular e plana
- `beta = 1.0`: distinção entre tubo e estrutura esférica
- `gamma = 30`: sensibilidade ao contraste

### 3. Localização da Aorta Ascendente

Utilização da **Transformada de Hough para círculos**:

- Detecção em fatias axiais
- Raio esperado: 36-62 mm
- Critérios de parada baseados em variação de posição e raio

**Abordagem implementada**: Busca restrita ao primeiro quadrante da imagem para maior precisão.

### 4. Segmentação da Aorta

**Active Contour Segmentation (Level Set)**:

- Inicialização baseada nos círculos detectados
- Refinamento iterativo dos contornos
- Pós-processamento morfológico para remoção de vazamentos

### 5. Detecção de Óstios Coronarianos

Identificação dos pontos de origem das artérias coronárias:

- Busca por voxels de alta vesselness na circunferência da aorta
- Separação entre óstio esquerdo e direito
- Validação baseada em distância e posição anatômica

### 6. Segmentação das Artérias (Region Growing)

**Crescimento de região 3D** a partir dos óstios:

- Guiado pela medida de vesselness
- Critérios adaptativos de inclusão
- Limitação de volume para evitar vazamento

### 7. Pós-processamento

- Fechamento morfológico (raio = 3)
- Dilatação (raio = 2)
- Extração de componentes conectados

## 🚀 Instalação

### Requisitos

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) (gerenciador de pacotes recomendado)

### Instalação das Dependências

```bash
# Clone o repositório
git clone <repository-url>
cd coronary-centerline-detection

# Instale as dependências usando uv
uv sync

# Ou usando pip
pip install -e .
```

### Dependências Principais

- `scikit-image`: Processamento de imagens
- `nibabel`: Leitura de arquivos NIfTI
- `opencv-python`: Transformada de Hough
- `morphsnakes`: Segmentação por contorno ativo
- `matplotlib`, `plotly`, `seaborn`: Visualização
- `pandas`: Análise de dados
- `scikit-learn`: Divisão train/test

## 💻 Uso

### Pipeline Completo

```python
from segmentation_pipeline import process_image

# Processar uma imagem específica
result = process_image(IMG_ID=1)

# Resultado contém:
# - ostia_left: coordenadas (y, x, z) do óstio esquerdo
# - ostia_right: coordenadas (y, x, z) do óstio direito
# - dice_artery: score Dice para avaliação da segmentação
# - artery_voxels: número de voxels segmentados
```

### Processamento em Lote

```python
from segmentation_pipeline import run_pipeline

# Processar múltiplas imagens
ids = [1, 2, 3, 4, 134, 195]
results = run_pipeline(ids, split_name="train")
```

### Notebooks Jupyter

Execute os notebooks para análise exploratória e visualização:

```bash
# Iniciar Jupyter
jupyter notebook

# Abrir:
# - src/main.ipynb: Pipeline principal
# - src/eda_imgs.ipynb: Exploração de imagens
# - src/segmentacao_arteria.ipynb: Análise de segmentação
```

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

- `output/detected_circles/`: Visualizações dos círculos detectados
- `output/segmented_ostios/`: Visualizações 3D dos óstios
- `output/ostios_train_summary.csv`: Resumo quantitativo dos resultados

## 🔧 Configuração

### Parâmetros Principais

Edite `segmentation_pipeline.py` para ajustar:

```python
# Caminhos
base_path = "/path/to/ImageCAS"
base_save_path = "/path/to/Processed_ImageCAS"

# Cache
LOAD_CACHE = False  # True para usar resultados salvos

# Downsampling
downscale_factors = (2, 2, 1)  # (x, y, z)

# Vesselness (primeira passada - detecção de aorta)
sigmas=np.arange(2.5, 3.5, 1)
alpha=0.5
beta=1
gamma=30

# Vesselness (segunda passada - segmentação de artérias)
sigmas=np.arange(1.5, 2.5, 0.5)
gamma=55

# Hough Transform
radii_start = 36 mm  # raio mínimo da aorta
radii_end = 62 mm    # raio máximo da aorta

# Region Growing
threshold = (max - min) / 5
max_volume = 100000
min_vesselness = max * 0.078
```

## 📈 Estado Atual do Projeto

### ✅ Etapas Concluídas

- [x] Pré-processamento (downsampling, thresholding, LCC)
- [x] Cálculo do mapa de vesselness (Filtro de Frangi)
- [x] Localização da aorta ascendente (Transformada de Hough)
- [x] Segmentação da aorta (Level Set)
- [x] Detecção de óstios coronarianos
- [x] Segmentação inicial das artérias (Region Growing)
- [x] Avaliação quantitativa (Dice, distâncias)

### 🚧 Próximos Passos

- [ ] Esqueletização 3D para extração das linhas centrais
- [ ] Refinamento do algoritmo de crescimento de região
- [ ] Otimização de hiperparâmetros
- [ ] Validação em conjunto completo de teste
- [ ] Tratamento de casos edge (aorta fora do 1º quadrante)
- [ ] Implementação de visualizações 3D interativas

## 🔍 Limitações Conhecidas

1. **Posição da aorta**: Algoritmo atual assume aorta no primeiro quadrante
2. **Calcificações severas**: Podem interferir na detecção de círculos
3. **Anatomias atípicas**: Variações anatômicas podem reduzir precisão
4. **Parâmetros fixos**: Alguns parâmetros podem precisar ajuste por dataset

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
