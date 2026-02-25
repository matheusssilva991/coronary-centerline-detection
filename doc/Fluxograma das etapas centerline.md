# Fluxograma das etapas

## Extração linhas centrais

### 1. Pré-processamento

#### 1.1 Downsampling

Redução da imagem em um fator de dois usando **interpolação linear** para tornar a computação mais rápida sem comprometer a qualidade da imagem.

#### 1.2 Cálculo dos valores Hounsfield

Cálculo dos valores Hounsfield (HU) de cada voxel da imagem 3D utilizando dois parâmetros: a inclinação e o intercepto.

$$
IU(x, y, z) = m \cdot I(x, y, z) + b
$$

Onde:
- $IU(x, y, z)$ é o valor Hounsfield do voxel na posição $(x, y, z)$.
- $I(x, y, z)$ é o valor original do voxel na posição $(x, y, z)$.
- $m$ é a inclinação (*slope*) do scanner.
- $b$ é o intercepto (*intercept*) do scanner.

#### 1.3 Remoção de calcificações, stents e ossos

Remoção de estruturas como calcificações, stents e ossos por meio de **Thresholding**. Para isso, os voxels com valores HU maiores que 675 são removidos da imagem para evitar interferência na detecção da aorta. Isso é feito definindo todos os valores HU maiores que 675 como 0.

A remoção dessas estruturas foi realizada utilizando a seguinte fórmula:

$$
S(x, y, z) =
\begin{cases}
0, & \text{se } IU(x, y, z) > 675 \\
1, & \text{caso contrário}
\end{cases}
$$

A matriz binária $S(x, y, z)$ é então multiplicada pela imagem original $IU(x, y, z)$ para obter a imagem sem calcificações, stents e ossos.

#### 1.4 Extração do maior componente conectado

Remoção da região do pulmão e vasos pulmonares, que podem ser detectados na etapa do filtro de vasos. Para isso, é extraído o **maior componente conectado** da imagem binarizada, que corresponde à região da aorta e outros vasos sanguíneos.

---

### 2. Cálculo do filtro de vasos modificado

Aprimorar os vasos sanguíneos na imagem 3D e outras regiões tubulares, especialmente as pseudo-vasculares.

Esse filtro é baseado na **análise dos autovalores da matriz Hessiana**, que descreve a curvatura local da imagem.

O retorno do filtro é um mapa de vasos de mesma dimensão da imagem de entrada.
- Valores próximos a **1** indicam alta probabilidade de ser um vaso.
- Valores próximos a **0** indicam baixa probabilidade.

---

### 3. Identificação da Aorta e Óstios coronarianos

A aorta e os óstios são identificados com base em dois fatos:
- a aorta é circular ou elíptica em forma;
- os óstios estão localizados na circunferência da aorta com maior medida de vaso.

#### 3.1 Detecção da Aorta

Localização da aorta ascendente em cada fatia axial utilizando a **Transformada de Hough para detecção de círculos**.

A transformada de Hough é aplicada em cada fatia axial da imagem 3D. O centro do círculo detectado é considerado a localização da aorta.

A detecção é realizada até que a variação do raio entre fatias consecutivas seja maior que **9 mm**, indicando o fim da aorta ascendente.

Após a detecção da aorta, os centros dos círculos são usados como sementes para a **segmentação ativa de contorno 3D**, segmentando a aorta.

#### 3.2 Detecção dos óstios coronarianos

Localização dos óstios coronarianos em cada fatia axial.

Para isso, são localizados os voxels com maior medida de vaso na circunferência da aorta. Esses pontos são usados como sementes para um **algoritmo de crescimento de região**, segmentando os óstios coronarianos.

---

### 4. Segmentação dos Vasos Coronários e Extração da Linha Central

Nesta etapa, os óstios detectados e a medida de *vesselness* aprimorada são usados para segmentar as principais artérias coronárias e extrair suas linhas centrais.

#### 4.1 Segmentação dos Vasos Coronários

As coordenadas dos óstios detectados são usadas como sementes para um **algoritmo de crescimento de região 3D**.

O crescimento da região é guiado pela medida de *vesselness*. Estes pontos atuam como as raízes das árvores das artérias coronárias direita (RCA) e principal esquerda (LM).

O processo é iterativo:

1. Em cada iteração, os 26 voxels vizinhos são avaliados.
2. Um voxel vizinho é adicionado à região segmentada se a diferença entre o seu valor e a média atual da região for menor que um limiar pré-definido.
3. A média da região é atualizada a cada iteração.
4. O processo para quando nenhum vizinho satisfaz o critério.

Critério de inclusão:

$$
\Delta_{vm} = |V(x,y,z) - \text{mean}\{V(l,m,n)\}|
$$

Onde:

- $V(x,y,z)$ é o valor de *vesselness* do voxel vizinho.
- $\text{mean}\{V(l,m,n)\}$ é o valor médio de *vesselness* da região.
- $\Delta_{vm}$ é a diferença absoluta.

O limiar $\Delta_{vm}$ é definido como:

$$
T_{vm} = \frac{|max_v - min_v|}{10}
$$

Onde:

- $max_v$ é o valor máximo de *vesselness* na imagem 3D.
- $min_v$ é o valor mínimo de *vesselness* na imagem 3D.

#### 4.2 Extração da Linha Central

A extração da linha central é realizada com o **algoritmo de esqueleto 3D**.

As artérias segmentadas (volume binário) são submetidas à **esqueletização**, obtendo suas linhas centrais.

O método é baseado na **Transformada do Eixo Medial (MAT)**, que reduz a estrutura 3D para uma representação 1D, preservando a topologia e conectividade dos vasos.

O resultado é um conjunto de voxels que formam a **linha central das artérias coronárias**, facilitando análise e visualização.
