# Seleção dinâmica do threshold superior

## Objetivo

Investigar uma regra que escolha, para cada volume, entre P99.5, P99.7 e
P99.9. Esta é apenas uma proposta para análise futura. Nenhuma das abordagens
descritas aqui está implementada no pipeline.

O threshold efetivo de um percentil depende da distribuição de intensidades de
cada imagem:

\[
T_p=P_p(I), \qquad p\in\{99.5,99.7,99.9\}.
\]

A máscara inicial correspondente é:

\[
M_p(x)=
\begin{cases}
1, & -300 \leq I(x) \leq T_p,\\
0, & \text{caso contrário}.
\end{cases}
\]

P99.9 deve ser tratado como a escolha conservadora. Nos 270 exames de
validação analisados, ele apresentou o melhor resultado global e venceu ou
empatou em aproximadamente 70% dos exames. Uma regra dinâmica só deve escolher
P99.5 ou P99.7 quando houver evidência suficiente de benefício.

## Dados necessários

Para desenvolver e avaliar qualquer seletor, montar uma tabela com uma linha
por exame e as seguintes colunas:

- `IMG_ID`;
- Dice obtido com P99.5, P99.7 e P99.9;
- sucesso dos óstios em cada percentil;
- valor efetivo em HU de P99.5, P99.7 e P99.9;
- média e mediana do histograma completo;
- média e mediana dos voxels com intensidade maior ou igual a 300 HU;
- fração de voxels com intensidade maior ou igual a 300 HU;
- características da cauda definidas abaixo.

O conjunto de teste não deve ser usado para construir regras, escolher
limites ou ajustar coeficientes. Depois de definida usando treino/validação, a
regra deve ser congelada antes da avaliação final no teste.

## Características recomendadas

### Fração de voxels densos

\[
f_{denso}=
\frac{\#\{x:I(x)\geq 300\}}{\#\{x:I(x)\text{ é finito}\}}.
\]

### Assimetria da região densa

\[
A_{denso}=\operatorname{média}(I\mid I\geq300)
-\operatorname{mediana}(I\mid I\geq300).
\]

Uma diferença elevada indica uma cauda puxada por poucos valores muito
densos.

### Intervalos entre os percentis

\[
G_1=T_{99.7}-T_{99.5},
\]

\[
G_2=T_{99.9}-T_{99.7},
\]

\[
G_{total}=T_{99.9}-T_{99.5}.
\]

### Fração ocupada pela extremidade da cauda

\[
R_{cauda}=\frac{G_2}{\max(G_{total},\epsilon)}.
\]

Quanto maior `R_cauda`, maior é a separação entre P99.7 e P99.9 em relação à
largura total da cauda analisada.

A média e a mediana do histograma completo podem ser mantidas como variáveis
auxiliares. Elas não devem ser usadas sozinhas, pois são fortemente
influenciadas pelo ar e pelos tecidos moles, enquanto a decisão ocorre na
cauda superior.

## Abordagem principal: regressão linear do ganho de Dice

### 1. Definir os alvos

Para cada exame, usar P99.9 como referência:

\[
\Delta D_{99.5}=D_{99.5}-D_{99.9},
\]

\[
\Delta D_{99.7}=D_{99.7}-D_{99.9}.
\]

Valor positivo significa que a redução do percentil melhorou o Dice.

### 2. Normalizar as características

Cada característica deve ser padronizada usando somente o conjunto usado para
ajuste:

\[
z_j=\frac{x_j-\mu_j}{\sigma_j}.
\]

As médias `mu_j` e os desvios `sigma_j` devem ser guardados para aplicar a
mesma transformação em imagens futuras.

### 3. Ajustar duas regressões Ridge

Uma regressão estima o ganho de P99.5:

\[
\widehat{\Delta D}_{99.5}
=\beta_{0,99.5}+\sum_j\beta_{j,99.5}z_j.
\]

A outra estima o ganho de P99.7:

\[
\widehat{\Delta D}_{99.7}
=\beta_{0,99.7}+\sum_j\beta_{j,99.7}z_j.
\]

Os coeficientes são obtidos minimizando:

\[
\sum_i(\Delta D_i-\widehat{\Delta D}_i)^2
+\lambda\sum_j\beta_j^2.
\]

O termo com `lambda` limita coeficientes excessivos e reduz sobreajuste.

### 4. Aplicar uma decisão conservadora

Definir um ganho mínimo relevante `delta`, inicialmente entre 0.01 e 0.02:

\[
p^*=\begin{cases}
99.5, & \widehat{\Delta D}_{99.5}>
\max(\widehat{\Delta D}_{99.7},\delta),\\
99.7, & \widehat{\Delta D}_{99.7}>
\max(\widehat{\Delta D}_{99.5},\delta),\\
99.9, & \text{caso contrário}.
\end{cases}
\]

Assim, ganhos negativos, muito pequenos ou semelhantes mantêm P99.9.

### 5. Incluir a segurança dos óstios

Uma alternativa só deve ser aceita se não houver indicação de piora relevante
na detecção dos óstios. Isso pode ser feito de duas formas:

1. penalizar no treinamento os casos em que o threshold alternativo perde o
   sucesso dos óstios; ou
2. ajustar um classificador separado para estimar a probabilidade de sucesso
   dos óstios em cada threshold.

Uma função de utilidade possível é:

\[
U_p=D_p-\gamma\,\mathbb{1}(\text{falha dos óstios em }p),
\]

em que `gamma` determina o custo atribuído a uma falha dos óstios.

## Alternativas mais simples

### Alternativa A: sempre usar P99.9

É a referência obrigatória. Não possui adaptação, mas foi a melhor configuração
global. Qualquer seletor dinâmico deve superar esse resultado de forma pareada.

### Alternativa B: uma regra baseada somente em `R_cauda`

Hipótese inicial:

\[
p^*=\begin{cases}
99.9, & R_{cauda}<a,\\
99.7, & a\leq R_{cauda}<b,\\
99.5, & R_{cauda}\geq b.
\end{cases}
\]

Os limites `a` e `b` devem ser escolhidos por busca em grade no conjunto de
desenvolvimento. É a abordagem mais simples e explicável, mas assume uma
relação monotônica que ainda precisa ser demonstrada.

### Alternativa C: tabela com duas características

Dividir `R_cauda` e `f_denso` em categorias baixa, média e alta. Para cada
combinação, selecionar o percentil que obteve maior Dice médio no conjunto de
desenvolvimento.

Exemplo conceitual:

| `R_cauda` | `f_denso` | Escolha |
|---|---|---|
| baixo | qualquer | P99.9 |
| médio | alto | P99.7 |
| alto | alto | P99.5 |

Essa tabela deve ser produzida pelos dados; os valores acima não são uma regra
validada.

### Alternativa D: centroide de cada grupo vencedor

1. Separar os exames pelos thresholds vencedores.
2. Calcular o vetor médio de características de cada grupo.
3. Padronizar as características.
4. Para uma imagem nova, calcular a distância euclidiana até cada centroide.
5. Escolher o threshold do centroide mais próximo, mantendo P99.9 quando as
   distâncias forem muito semelhantes.

A distância ao grupo `p` é:

\[
d_p(x)=\sqrt{\sum_j(z_j-c_{p,j})^2}.
\]

Essa abordagem é simples, mas pode falhar quando os grupos se sobrepõem.

### Alternativa E: árvore de decisão rasa

Treinar uma árvore com profundidade máxima de dois ou três níveis usando
`R_cauda`, `f_denso`, `G_1`, `G_2` e `A_denso`. A saída é diretamente P99.5,
P99.7 ou P99.9.

A árvore gera regras legíveis, por exemplo: "se `R_cauda` for alto e a fração
densa for alta, usar P99.5". Para evitar que empates virem decisões instáveis,
casos cujo ganho máximo seja menor que `delta` devem ser rotulados como P99.9.

## Ordem sugerida para os experimentos

1. Calcular as características para todos os exames de desenvolvimento.
2. Definir empate prático como ganho absoluto menor que `delta`.
3. Avaliar a regra de uma variável baseada em `R_cauda`.
4. Avaliar a tabela com `R_cauda` e `f_denso`.
5. Avaliar a árvore rasa.
6. Avaliar as duas regressões Ridge.
7. Comparar todos os seletores com P99.9 fixo e com o oráculo retrospectivo.
8. Escolher o método mais simples que apresente ganho reprodutível.
9. Congelar regra, coeficientes e limites.
10. Executar uma única avaliação final no conjunto de teste.

## Avaliação necessária

Para cada método, registrar:

- Dice médio, mediano e desvio-padrão;
- taxa de sucesso dos óstios;
- quantidade de exames enviados para cada percentil;
- quantidade de exames que melhoraram, pioraram ou permaneceram equivalentes;
- delta de Dice pareado contra P99.9;
- intervalo de confiança de 95% por bootstrap para o delta médio;
- teste de permutação pareado;
- tempo adicional necessário para calcular as características e escolher o
  threshold.

Também deve ser calculado o desempenho do oráculo:

\[
D_{oráculo,i}=\max(D_{99.5,i},D_{99.7,i},D_{99.9,i}).
\]

O oráculo não pode ser usado na prática, mas representa o limite superior que
um seletor entre esses três thresholds poderia alcançar.

## Critério conservador de adoção

Uma regra dinâmica só deve substituir P99.9 se:

- aumentar o Dice médio pareado;
- não reduzir a taxa de sucesso dos óstios;
- apresentar resultado estável na validação cruzada;
- selecionar P99.5/P99.7 por um padrão interpretável, e não por poucos casos
  extremos;
- manter o ganho após congelamento e avaliação no conjunto de teste.

Se essas condições não forem satisfeitas, P99.9 deve permanecer como parâmetro
fixo e a análise dinâmica pode ser apresentada apenas como estudo exploratório.
