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

```file
coronary-centerline-detection/
├── pyproject.toml              # Configuração e dependências do projeto
├── README.md                   # Este arquivo
├── doc/                        # Documentação detalhada
│   ├── Fluxograma das etapas centerline.md
│   └── Relatorio das etapas realizadas.md
├── src/                        # Código fonte
│   ├── segmentation_pipeline.py    # Pipeline principal de processamento
│   ├── main.ipynb                  # Notebook de execução principal
│   ├── eda/                        # Notebooks de análise exploratória
│   │   ├── eda_imgs.ipynb              # Análise exploratória de imagens
│   │   ├── eda_segmentacao.ipynb       # Análise de segmentações
│   │   ├── eda_tests.ipynb             # Testes comparativos
│   │   ├── eda_tests_subsets.ipynb     # Subsets e grupos de casos
│   │   ├── eda_tests_casos_ruins.ipynb # Análise de casos ruins
│   │   ├── cases_analysis.ipynb        # Análise de casos
│   │   └── viz_imgs.ipynb              # Análise visual de imagens
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

- Redução da imagem por fator 2 usando **interpolação cúbica** (order=3)
- Fatores de redução: `(2, 2, 1)` - reduz x e y, mantém z
- Objetivo: reduzir custo computacional mantendo qualidade

#### 1.2 Threshold Dinâmico Adaptativo

Aplicação de **threshold baseado em percentis** para remover artefatos:

- **Limite inferior**: `-300 HU` (remove ar e tecidos de baixa densidade)
- **Limite superior**: **percentil 99.7** da imagem (adaptativo, remove calcificações, stents e ossos)

**Vantagem**: O threshold superior adaptativo se ajusta automaticamente à distribuição de intensidades de cada imagem, sendo mais robusto que valores fixos.

Máscara aplicada:
$$M(x, y, z) = \begin{cases} 1, & \text{se } -300 \leq I(x,y,z) \leq P_{99.7} \\ 0, & \text{caso contrário} \end{cases}$$

onde $P_{99.7}$ é o percentil 99.7 da imagem.

#### 1.3 Extração do Maior Componente Conectado (LCC)

- Aplicação **slice-by-slice** (2D) para cada fatia axial
- Remove regiões isoladas, pulmões e vasos pulmonares
- Mantém apenas a maior estrutura conectada (coração e grandes vasos)
- Isolamento da região de interesse (ROI)

### 2. Cálculo do Mapa de Vesselness (Filtro de Frangi)

Aplicação do **Filtro de Frangi modificado** para realce de estruturas tubulares vasculares.

**Funcionamento**:

- Análise dos autovalores da **matriz Hessiana** para cada voxel
- Detecção de estruturas tubulares em múltiplas escalas
- Valores próximos a **1**: alta probabilidade de vaso
- Valores próximos a **0**: baixa probabilidade

**Parâmetros utilizados (1ª passada - detecção de aorta)**:

```python
sigmas = np.arange(2.5, 3.5, 1)  # Escalas para detecção
alpha = 0.5    # Distinção entre estrutura tubular (vaso) e plana
beta = 1.0     # Distinção entre tubo (comprido) e blob (esférico)
gamma = 30     # Sensibilidade ao contraste (estrutura vs ruído)
```

**Parâmetros utilizados (2ª passada - segmentação de artérias)**:

```python
sigmas = np.arange(1.5, 2.5, 0.5)  # Escalas menores para artérias finas
alpha = 0.5
beta = 0.5
gamma = 55     # Maior sensibilidade para capturar artérias menores
```

**Melhorias implementadas**:

- Normalização robusta usando percentis (ignora outliers extremos)
- Medida de *grayness* (Gf) baseada no desvio da intensidade média
- Medida de gradiente (Gd) para reforçar bordas vasculares

### 3. Localização da Aorta Ascendente

Utilização da **Transformada de Hough para detecção de círculos** em fatias axiais:

**Processo**:

1. **Detecção de bordas**: Aplicação do filtro Canny (sigma=3) em cada fatia
2. **Transformada de Hough**: Busca por círculos nas bordas detectadas
3. **Raios testados**: 36-62 mm (convertidos para pixels após downsampling)
4. **Seleção de círculos**: Baseada em proximidade espacial entre fatias consecutivas

**Critérios de parada**:

- Variação de raio > 9 mm entre fatias consecutivas
- Variação de distância entre centros > 18 mm
- Ausência de círculos por mais de 5 fatias consecutivas

**Estratégias implementadas**:

- **Abordagem C** (mais eficaz): Busca restrita ao **primeiro quadrante** da imagem
- Seleção do círculo mais próximo do detectado na fatia anterior
- Múltiplos picos analisados (10-15 círculos candidatos por fatia)

**Saída**: Lista de círculos detectados com coordenadas `(center_x, center_y, radius, slice_index)`

### 4. Segmentação da Aorta (Level Set)

**Morphological Geodesic Active Contour (MGAC)**:

1. **Inicialização**: Círculos detectados desenhados como discos preenchidos
   - Raio reduzido por fator 0.15 (85% do raio original)
   - Cada círculo vira uma "semente" na respectiva fatia

2. **Cálculo do *edge indicator***:
   - Transformação usando gradiente Gaussiano inverso
   - Guia o contorno para bordas da aorta

3. **Evolução iterativa** (31 iterações):
   - `balloon = 0.8`: força de expansão do contorno
   - `smoothing = 2`: suavização da curva
   - Contorno evolui adaptando-se às bordas reais

4. **Pós-processamento morfológico**:
   - Remoção de vazamentos usando operação de **abertura** (radius=2)
   - Extração do maior componente conectado
   - Máscara binária final da aorta

**Resultado**: Segmentação 3D precisa da aorta ascendente

### 5. Detecção de Óstios Coronarianos

Identificação automática dos pontos de origem das artérias coronárias (esquerda e direita):

**Estratégia**:

1. **Extração da superfície da aorta**:
   - Erosão da máscara da aorta (radius=4)
   - Subtração para obter apenas a casca/superfície

2. **Região de busca**:
   - **80% inferior** da aorta em z (onde óstios anatomicamente se localizam)
   - Restrição espacial reduz falsos positivos

3. **Seleção de candidatos**:
   - Top 2000 voxels com **maior vesselness** na superfície
   - Filtrados por diferença máxima em z (52 voxels)

4. **Separação esquerdo/direito**:
   - **Critério anatômico**: óstio esquerdo mais à esquerda (menor x)
   - Distância mínima entre óstios: 70% do raio médio da aorta
   - Separação lateral mínima: 50% do raio médio

5. **Validação**:
   - Distância do centro da aorta
   - Posição relativa entre os dois óstios
   - Coordenadas anatômicas compatíveis

**Saída**: Coordenadas `(y, x, z)` dos óstios esquerdo e direito

### 6. Segmentação das Artérias (Region Growing 3D)

**Crescimento de região adaptativo** a partir dos óstios detectados:

**Algoritmo**:

1. **Inicialização**:
   - Sementes: coordenadas dos óstios esquerdo e direito
   - Vesselness map da 2ª passada (escalas menores para artérias finas)

2. **Critérios de crescimento**:
   - `threshold = (max - min) / 5`: diferença máxima de vesselness
   - `min_vesselness = max * 0.078`: piso mínimo de qualidade vascular
   - `max_volume = 100000`: limite para evitar vazamento

3. **Estratégia adaptativa**:
   - Primeiros 2000 voxels: critério rígido
   - Após 2000 voxels: relaxamento do threshold (fator 0.97)
   - Janela de comparação ajustável

4. **Validação de vizinhos** (26-conectividade):
   - Avalia todos os vizinhos 3D de cada voxel adicionado
   - Adiciona se vesselness dentro dos critérios
   - Atualiza média da região iterativamente

5. **Execução separada**:
   - Artéria esquerda (LCA) e direita (RCA) segmentadas independentemente
   - Máscaras combinadas ao final

**Resultado**: Segmentação 3D das principais artérias coronárias

### 7. Pós-processamento Morfológico

Refinamento final das máscaras arteriais:

1. **Fechamento morfológico** (ball, radius=3):
   - Preenche pequenos buracos
   - Conecta pequenas descontinuidades

2. **Dilatação** (ball, radius=2):
   - Recupera bordas que possam ter sido perdidas
   - Suaviza contornos

3. **Máscara binária final**: Artérias coronárias segmentadas

**Avaliação**:

- **Dice Score**: comparação com segmentação manual (ground truth)
- **Contagem de voxels**: volume total das artérias segmentadas

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

# O pipeline divide automaticamente os IDs em num_batches lotes
# (padrão: num_batches=5)
```

### Notebooks Jupyter

Execute os notebooks para análise exploratória e visualização:

```bash
# Iniciar Jupyter
jupyter notebook

# Abrir:
# - src/main.ipynb: Pipeline principal
# - src/eda/eda_imgs.ipynb: Exploração de imagens
# - src/eda/eda_segmentacao.ipynb: Análise de segmentações
# - src/eda/viz_imgs.ipynb: Análise visual de imagens
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

Edite [config/pipeline_config.json](config/pipeline_config.json) para ajustar:

```text
Os ajustes mais importantes agora são:

- `NUM_BATCHES`: número de lotes em que o conjunto é dividido; o padrão é `5`
- `USE_GPU`: ativa/desativa uso de GPU
- `LOAD_CACHE` e `SAVE_CACHE`: controle de cache dos intermediários
- `DOWNSCALE_METHOD` e `DOWNSCALE_FACTORS`: controle do pré-processamento
- `CIRCLE_DETECTION`, `LEVEL_SET`, `OSTIA_DETECTION`, `REGION_GROWING`: hiperparâmetros das etapas do pipeline
- `POSTPROCESSING`: refinamento final da máscara

Se precisar ajustar os valores técnicos com mais detalhe, altere apenas o JSON e mantenha o código do pipeline estável.
```

### Exemplo rápido de lote

```python
from segmentation_pipeline import run_pipeline

ids = [1, 2, 3, 4, 134, 195]
results = run_pipeline(ids, split_name="train")
# O número de lotes é definido em config/pipeline_config.json via NUM_BATCHES
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
