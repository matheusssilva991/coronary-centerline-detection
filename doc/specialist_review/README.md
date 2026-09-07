# Consulta a especialistas: anatomia e parametros

## Objetivo

Reunir as duvidas anatomicas que podem orientar os parametros de localizacao
da aorta, deteccao dos ostios e segmentacao das arterias coronarias.
Este documento preserva as perguntas do antigo arquivo
`config/parâmetros_especialistas.txt`; nao define valores clinicos validados
nem altera a configuracao do pipeline.

## Aorta ascendente

- Qual a faixa esperada de diametro e raio minimo/maximo da aorta ascendente?
- Como essa faixa varia ao longo da aorta e entre exames?

Aplicacao: orientar o intervalo de raios da busca de circulos. Registrar as
medidas em milimetros e converter para pixels usando o spacing do exame.

## Calibre das arterias coronarias

- Qual o diametro ou raio esperado na origem das coronarias, junto aos ostios?
- Qual o diametro ou raio medio ao longo das arterias coronarias?

Aplicacao: orientar as escalas do mapa de vasos para os ostios e para as
arterias. A escala do Frangi nao deve ser tratada como uma equivalencia direta
ao raio anatomico sem verificar a resposta do filtro.

## Posicao relativa dos ostios

- Qual a distancia lateral minima e maxima esperada entre os ostios?
- Qual a diferenca minima e maxima esperada em z entre os ostios?
- Os ostios estao sempre na parte inferior da aorta? Quais excecoes existem?

Aplicacao: avaliar as restricoes geometricas da selecao do segundo ostio e a
fracao axial da superficie candidata. Definir a orientacao da imagem antes
de interpretar os termos lateral, inferior e eixo z.

## Registro das respostas

Para cada resposta, registrar:

1. Especialista ou referencia consultada e data.
2. Estrutura e regiao anatomica considerada.
3. Faixa de medidas, unidade e variacoes relevantes.
4. Excecoes e limitacoes da recomendacao.
5. Parametro do pipeline associado e experimento proposto.

As respostas devem orientar hipoteses para os conjuntos de desenvolvimento.
Congelar as escolhas antes da avaliacao final no conjunto de teste.
