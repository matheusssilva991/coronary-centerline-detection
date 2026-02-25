# Relatório de Andamento das Etapas Realizadas

Este documento apresenta um resumo das etapas já concluídas no projeto, destacando os principais marcos, desafios enfrentados e soluções implementadas até o momento.

## Fluxograma das Etapas

[![Fluxograma das Etapas](./Fluxograma%20-%20Extração%20linhas%20centrais.png)](./Fluxograma%20-%20Extração%20linhas%20centrais.png)

A figura acima apresenta o fluxo de trabalho adotado para a extração das linhas centrais das artérias coronárias.

### 1. Redimensionamento

Inicialmente, foi realizado o redimensionamento das imagens em um fator de 2, reduzindo o custo computacional sem comprometer significativamente a qualidade.
Em seguida, aplicou-se thresholding para remover calcificações, stents e ossos nas imagens de angiografia.

### 2. Thresholding

O artigo de referência utiliza um limiar de 675 Hounsfield Units (HU), porém, após testes com o conjunto de dados do projeto, verificou-se que o valor de 600 HU apresentou resultados mais adequados, removendo estruturas indesejadas sem perda significativa de informação nos vasos.

### 3. Maior Componente Conectado

Após o limiar, foi extraído o maior componente conectado da imagem, etapa responsável por eliminar regiões não relacionadas ao coração, como pulmões e vasos pulmonares.
O resultado é uma máscara binária contendo apenas o maior componente conectado.
Essa máscara é então aplicada à imagem original, isolando a região de interesse (ROI), ou seja, o coração e seus principais vasos.

### 4. Mapa de Vasos

Em seguida, foi gerado um mapa de vasos utilizando um filtro de Frangi modificado, conforme proposto no artigo base.
Esse filtro realça estruturas tubulares (como vasos sanguíneos), produzindo uma imagem onde cada voxel representa a probabilidade de pertencer a um vaso.
O mapa de vasos é um dos principais insumos para as etapas de detecção e segmentação subsequentes.

### 5. Localização da Aorta Ascendente

Esta etapa é crucial no pipeline, pois a Aorta Ascendente é usada como ponto de referência para a segmentação e extração das linhas centrais dos vasos coronários.

A detecção foi realizada por meio da Transformada de Hough para círculos, aplicada nas fatias axiais da imagem.
Como resultado, são obtidas as fatias que contêm a aorta e os respectivos círculos detectados, representando sua posição e diâmetro estimado.

Atualmente, o projeto encontra-se nesta etapa.
A localização da Aorta Ascendente foi obtida com sucesso em várias imagens do conjunto ImageCAS, mas ainda são necessários ajustes para aprimorar a robustez e precisão da detecção, especialmente em casos onde a aorta não está no primeiro quadrante da imagem.

### 6-9. Etapas Futuras

Nas etapas seguintes, será implementada a segmentação ativa de contorno 3D (Active Contour Segmentation), utilizando as fatias detectadas como região inicial.
Essa abordagem permitirá delimitar com maior precisão os contornos da Aorta Ascendente.

A aorta segmentada servirá como base para a localização dos óstios coronarianos esquerdo e direito, identificados como os pontos de maior valor de vasos dentro da região segmentada.

Posteriormente, será realizado o crescimento de região 3D (Region Growing) a partir desses óstios, para segmentar as artérias coronárias esquerda (LCA) e direita (RCA), garantindo que a expansão permaneça restrita às regiões com alta medida de vesselness.

Por fim, será aplicada a esqueletização 3D, obtendo-se as linhas centrais das artérias segmentadas.
Essa representação reduz as artérias a uma forma unidimensional, preservando sua topologia e facilitando a análise e visualização das mesmas.

## Localização da Aorta Ascendente (Etapa Atual)

[![Fluxograma dos algoritmos para extrair a Aorta Ascendente](./Fluxograma%20-%20Localização%20aorta%20ascendente.png)](./Fluxograma%20-%20Localização%20aorta%20ascendente.png)

A imagem acima apresenta a evolução dos algoritmos implementados para a localização da Aorta Ascendente. Inicialmente, foi empregada a Transformada de Hough para detecção de círculos, conforme descrito no artigo de referência. A saída dessa etapa consiste nas fatias que contêm a Aorta Ascendente e nos círculos que indicam sua localização.

Entretanto, essa abordagem apresentou algumas limitações em relação ao banco de imagens utilizado. A primeira delas é que, no artigo original, a busca pela Aorta Ascendente é iniciada a partir da primeira fatia axial da imagem. No entanto, no conjunto de dados utilizado neste projeto, a Aorta Ascendente está localizada nas fatias finais, e não nas iniciais.

Além disso, a Transformada de Hough detecta múltiplos círculos que não correspondem à Aorta Ascendente, o que dificulta a identificação precisa da estrutura de interesse.

O fluxograma A apresenta a abordagem inicial baseada no artigo, com a modificação do ponto de partida da busca. Já o fluxograma B ilustra uma nova abordagem que utiliza, como critério de parada, a variação na distância entre as fatias, selecionando o círculo da fatia subsequente mais próximo do último detectado. Por fim, a abordagem C propõe uma solução que busca o primeiro círculo apenas dentro do primeiro quadrante da imagem, o que reduz significativamente o número de detecções incorretas e melhora a precisão na localização da Aorta Ascendente. Os círculos nas fatias seguintes são então selecionados com base na proximidade em relação ao círculo detectado na fatia anterior, garantindo uma continuidade espacial mais consistente.

A abordagem A apresentou dificuldades na seleção do círculo correto devido à grande quantidade de detecções geradas pela Transformada de Hough. Além disso, o critério de parada baseado apenas na variação do raio não se mostrou suficiente para garantir a escolha do círculo correspondente à Aorta Ascendente, uma vez que outras estruturas com raios semelhantes podiam ser confundidas com ela.

A abordagem B aprimorou a seleção ao considerar a variação na distância entre as fatias e o posicionamento relativo dos círculos, porém ainda enfrentava problemas quando o primeiro círculo detectado não correspondia à Aorta Ascendente. Nesses casos, as detecções subsequentes também se tornavam incorretas, comprometendo o resultado final.

Por outro lado, a abordagem C mostrou-se a mais eficaz até o momento, ao restringir a busca ao primeiro quadrante da imagem, o que facilitou a identificação correta da Aorta Ascendente. Essa abordagem apresentou resultados satisfatórios para as imagens testadas até o momento. No entanto, ainda são necessários testes adicionais para avaliar sua robustez em todo o conjunto de dados, especialmente em casos nos quais a Aorta Ascendente não se encontra no primeiro quadrante da imagem. Ajustes complementares poderão ser necessários para garantir maior precisão e confiabilidade na detecção em diferentes variações anatômicas e condições de imagem.

## Resultados

A seguir, são apresentados alguns resultados obtidos até o momento com a abordagem atual para a localização da Aorta Ascendente.

### Círculos Detectados

Na pasta [detected_circles](https://github.com/igorssant/codigos-imageCAS/tree/centerline/tests/output/detected_circles), estão disponíveis as imagens com os círculos detectados sobrepostos às fatias axiais correspondentes. Na pasta, além da imagem MIP (Maximum Intensity Projection) com os círculos detectados, também estão presentes as imagens das fatias axiais individuais, cada uma exibindo o círculo identificado pela Transformada de Hough.

### Bordas e Círculos Detectados na primeira e última fatia detectada.

Na pasta [detected_edges](https://github.com/igorssant/codigos-imageCAS/tree/centerline/tests/output/detected_edges), encontram-se as imagens das bordas detectadas utilizando o operador Canny, juntamente com os círculos identificados pela Transformada de Hough, especificamente para a primeira e última fatia onde a Aorta Ascendente foi localizada. Na imagem, apresenta os 10 círculos com maior pontuação.
