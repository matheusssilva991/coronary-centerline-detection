# Plano de Atualização da Dissertação

## Legenda

- <span style="color:inherit"><strong>Cor do tema</strong></span>: seção existente, mantida ou atualizada.
- <span style="color:#2ea043"><strong>Verde</strong></span>: seção a ser adicionada.
- <span style="color:#d1242f"><strong>Vermelho</strong></span>: seção a ser removida.

## Índice planejado

<details>
<summary><span style="color:inherit"><strong>Elementos pré-textuais</strong></span></summary>

- Resumo
- Abstract
- Lista de figuras
- Lista de tabelas
- Lista de abreviaturas e siglas
- Sumário

</details>

<details>
<summary><span style="color:inherit"><strong>1 Introdução</strong></span></summary>

- 1 Introdução

</details>

<details>
<summary><span style="color:inherit"><strong>2 Referencial Teórico</strong></span></summary>

- 2 Referencial Teórico
  - 2.1 Anatomia, Patologia e Imageamento
    - 2.1.1 Estrutura e Topologia Vascular
    - 2.1.2 Doença Arterial Coronariana
    - 2.1.3 Desafios Anatômicos e de Contraste
    - 2.1.4 Aquisição de Imagens Tomográficas
  - 2.2 Processamento e Segmentação de Imagens Médicas
    - <span style="color:#d1242f"><strong>2.2.1 Reamostragem e Operações de Escala — REMOVER</strong></span>
    - 2.2.2 Morfologia Matemática
    - 2.2.3 Realce Multiescala de Estruturas Vasculares
    - 2.2.4 Detecção Geométrica por Transformada de Hough
    - 2.2.5 Evolução de Contornos e Métodos Level Set
    - 2.2.6 Segmentação por Crescimento de Região
  - <span style="color:#2ea043"><strong>2.3 Métodos Fuzzy para Segmentação — ADICIONAR</strong></span>
    - <span style="color:#2ea043"><strong>2.3.1 Conjuntos e Funções de Pertinência Fuzzy</strong></span>
    - <span style="color:#2ea043"><strong>2.3.2 Limiarização Fuzzy de Três Classes</strong></span>
    - <span style="color:#2ea043"><strong>2.3.3 Fuzzy Connectedness</strong></span>
  - <span style="color:#2ea043"><strong>2.4 Inteligência Artificial e Aprendizado Profundo — ADICIONAR</strong></span>
    - <span style="color:#2ea043"><strong>2.4.1 Inteligência Artificial, Aprendizado de Máquina e Aprendizado Profundo</strong></span>
    - <span style="color:#2ea043"><strong>2.4.2 Redes Neurais Convolucionais</strong></span>
    - <span style="color:#2ea043"><strong>2.4.3 Arquitetura U-Net</strong></span>
    - <span style="color:#2ea043"><strong>2.4.4 Inteligência Artificial Aplicada à Segmentação Vascular</strong></span>

</details>

<details>
<summary><span style="color:inherit"><strong>3 Materiais e Métodos</strong></span></summary>

- 3 Materiais e Métodos
  - 3.1 Base de Dados
    - 3.1.1 ImageCAS
    - <span style="color:#2ea043"><strong>3.1.2 MM-WHS — ADICIONAR</strong></span>
    - <span style="color:#2ea043"><strong>3.1.3 OrCaScore — ADICIONAR</strong></span>
  - 3.2 Pipeline de Segmentação Proposto
    - 3.2.1 Pré-Processamento
    - 3.2.2 Extração dos Mapas de Vasos
    - 3.2.3 Segmentação da Aorta — ATUALIZAR
      - <span style="color:#2ea043"><strong>3.2.3.1 Filtragem Robusta da Trajetória de Círculos — ADICIONAR</strong></span>
      - <span style="color:#2ea043"><strong>3.2.3.2 Extensão por Círculos Sintéticos — ADICIONAR</strong></span>
      - <span style="color:#2ea043"><strong>3.2.3.3 Envelope de Restrição da Aorta — ADICIONAR</strong></span>
      - 3.2.3.4 Segmentação por Level Set — ATUALIZAR
    - 3.2.4 Localização dos Óstios
    - 3.2.5 Extração das Artérias Coronárias
    - <span style="color:#2ea043"><strong>3.2.6 Alternativas Fuzzy — ADICIONAR</strong></span>
      - <span style="color:#2ea043"><strong>3.2.6.1 Limiarização Fuzzy</strong></span>
      - <span style="color:#2ea043"><strong>3.2.6.2 Fuzzy Connectedness</strong></span>
  - 3.3 Avaliação e Experimentos
    - <span style="color:#2ea043"><strong>3.3.1 Métricas de Avaliação — ADICIONAR</strong></span>
    - <span style="color:#2ea043"><strong>3.3.2 Comparações Pareadas e Análise Estatística — ADICIONAR</strong></span>
    - <span style="color:#2ea043"><strong>3.3.3 Protocolo de Avaliação Automática da Aorta nos Bancos Externos — ADICIONAR</strong></span>
    - <span style="color:#2ea043"><strong>3.3.4 Análise dos Critérios Automáticos de Qualidade da Aorta — ADICIONAR</strong></span>
      - <span style="color:#2ea043"><strong>Preenchimento dos círculos (Q25)</strong></span>
      - <span style="color:#2ea043"><strong>Razão entre área segmentada e área circular (R_P90)</strong></span>
      - <span style="color:#2ea043"><strong>Fração volumétrica da máscara</strong></span>
      - <span style="color:#2ea043"><strong>Regras automáticas de subsegmentação e sobresegmentação</strong></span>

</details>

<details>
<summary><span style="color:inherit"><strong>4 Resultados e Discussão</strong></span></summary>

- 4 Resultados e Discussão
  - Introdução do capítulo: ambiente computacional e resumo do protocolo experimental
  - <span style="color:#2ea043"><strong>4.1 Análise de Sensibilidade e Seleção da Configuração — ADICIONAR</strong></span>
    - <span style="color:#2ea043"><strong>Tabela — Parâmetros avaliados, valores de referência e variações</strong></span>
    - <span style="color:#2ea043"><strong>Tabela — Dice e sucesso dos óstios por configuração</strong></span>
    - <span style="color:#2ea043"><strong>Análise das tabelas</strong></span>
    - <span style="color:#2ea043"><strong>Figura — Sensibilidade do Dice e dos óstios às variações dos parâmetros</strong></span>
    - <span style="color:#2ea043"><strong>Análise da figura</strong></span>
  - 4.2 Resultados do Pipeline Determinístico Atual — ATUALIZAR
    - 4.2.1 Visão Geral do Pipeline
      - Figura atual 18 — Visualização tridimensional da aorta, dos óstios e das artérias
      - Análise da figura
    - 4.2.2 Localização dos Óstios
      - Tabela atual 2 — Desempenho da detecção dos óstios por resolução
      - Análise da tabela
      - Tabela atual 3 — Distribuição dos erros na detecção dos óstios
      - Análise da tabela
      - Tabela atual 4 — Interseção dos erros entre as resoluções
      - Análise da tabela
      - Figura atual 19 — Exemplos visuais de falhas na detecção dos óstios
      - Análise da figura
    - 4.2.3 Segmentação das Artérias Coronárias
      - Tabela atual 5 — Dice Score por resolução e cenário de avaliação
      - Análise da tabela
      - Figura atual 20 — Exemplos visuais de casos com alto Dice Score
      - Análise da figura
      - Figura atual 21 — Distribuição do Dice Score
      - Análise da figura
  - <span style="color:#2ea043"><strong>4.3 Efeito das Etapas Intermediárias e Análise de Falhas — ADICIONAR</strong></span>
    - <span style="color:#2ea043"><strong>4.3.1 Efeito do Filtro de Trajetória e do Envelope</strong></span>
      - <span style="color:#2ea043"><strong>Tabela — Dice e sucesso dos óstios antes e depois da correção</strong></span>
      - <span style="color:#2ea043"><strong>Análise da tabela</strong></span>
      - <span style="color:#2ea043"><strong>Figura — Variação pareada do Dice antes e depois da correção</strong></span>
      - <span style="color:#2ea043"><strong>Análise da figura</strong></span>
      - <span style="color:#2ea043"><strong>Figura — Exemplos de sobresegmentação e subsegmentação da aorta no ImageCAS</strong></span>
      - <span style="color:#2ea043"><strong>Análise qualitativa da figura</strong></span>
    - <span style="color:#2ea043"><strong>4.3.2 Efeito do Pós-Processamento Morfológico</strong></span>
      - <span style="color:#2ea043"><strong>Tabela — Dice antes e depois da morfologia</strong></span>
      - <span style="color:#2ea043"><strong>Análise da tabela</strong></span>
      - <span style="color:#2ea043"><strong>Figura — Exemplo da segmentação arterial antes e depois da morfologia</strong></span>
      - <span style="color:#2ea043"><strong>Análise da figura</strong></span>
  - <span style="color:#2ea043"><strong>4.4 Comparação dos Métodos Fuzzy — ADICIONAR</strong></span>
    - <span style="color:#2ea043"><strong>Tabela — Desempenho de normal+RG, fuzzy+RG, normal+FC e fuzzy+FC</strong></span>
    - <span style="color:#2ea043"><strong>Tabela — Wilcoxon dos métodos fuzzy em relação ao pipeline determinístico</strong></span>
    - <span style="color:#2ea043"><strong>Análise das tabelas</strong></span>
    - <span style="color:#2ea043"><strong>Figura — Dice pareado por método de segmentação</strong></span>
    - <span style="color:#2ea043"><strong>Análise da figura</strong></span>
  - <span style="color:#2ea043"><strong>4.5 Comparação com Métodos de Inteligência Artificial — ADICIONAR</strong></span>
    - <span style="color:#2ea043"><strong>Tabela — Dice dos métodos de IA e do pipeline por resolução</strong></span>
    - <span style="color:#2ea043"><strong>Tabela — Wilcoxon do pipeline em relação a cada método de IA</strong></span>
    - <span style="color:#2ea043"><strong>Análise das tabelas</strong></span>
    - <span style="color:#2ea043"><strong>Figura — Comparação pareada entre IA e pipeline nos exames comuns</strong></span>
    - <span style="color:#2ea043"><strong>Análise da figura</strong></span>
  - <span style="color:#2ea043"><strong>4.6 Validação Externa em MM-WHS e OrCaScore — ADICIONAR</strong></span>
    - <span style="color:#2ea043"><strong>Tabela — Indicadores automáticos de qualidade da aorta por banco e resolução</strong></span>
    - <span style="color:#2ea043"><strong>Tabela — Dice da aorta no MM-WHS</strong></span>
    - <span style="color:#2ea043"><strong>Análise das tabelas</strong></span>
  - 4.7 Custo Computacional — ATUALIZAR
    - Tabela — Tempo de execução por resolução e configuração
    - Análise da tabela

</details>

<details>
<summary><span style="color:inherit"><strong>5 Conclusões e Próximos Passos</strong></span></summary>

- 5 Conclusões e Próximos Passos

</details>

<details>
<summary><span style="color:inherit"><strong>Referências</strong></span></summary>

- Referências

</details>
