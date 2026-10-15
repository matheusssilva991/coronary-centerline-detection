# Plano de Atualização da Dissertação

## Legenda

- <span style="color:inherit"><strong>Cor do tema</strong></span>: conteúdo existente mantido.
- <span style="color:#2ea043"><strong>Verde</strong></span>: conteúdo novo (`ADICIONAR`).
- <span style="color:#218bff"><strong>Azul</strong></span>: conteúdo existente que será revisado (`ATUALIZAR`).
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
    - <span style="color:#218bff"><strong>3.2.3 Segmentação da Aorta — ATUALIZAR</strong></span>
      - <span style="color:#2ea043"><strong>3.2.3.1 Filtragem Robusta da Trajetória de Círculos — ADICIONAR</strong></span>
      - <span style="color:#2ea043"><strong>3.2.3.2 Extensão por Círculos Sintéticos — ADICIONAR</strong></span>
      - <span style="color:#2ea043"><strong>3.2.3.3 Envelope de Restrição da Aorta — ADICIONAR</strong></span>
      - <span style="color:#218bff"><strong>3.2.3.4 Segmentação por Level Set — ATUALIZAR</strong></span>
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
    - <details>
      <summary><span style="color:#2ea043"><strong>Tabela — Parâmetros avaliados, valores de referência e variações</strong></span></summary>

      | Parâmetro | Referência atual | Valores avaliados |
      | --- | --- | --- |
      | Percentil superior | 99.9 | 99.5; 99.7; 99.9 |
      | Distância máxima z dos óstios | 40 mm | 30; 40; 50 mm |
      | Fração inferior da busca dos óstios | 100% | 70%; 100% |
      | Piso mínimo de vesselness | 7.8% | 5%; 7.8%; 9% |
      | Divisor global do RG | 7 | 5; 7; 9 |

      Run atual `sensitivity_refined_pipeline_val270`, com filtro robusto, envelope e padding 2. O piso registrado é 7,8%.

      </details>

    - <details>
      <summary><span style="color:#2ea043"><strong>Tabela — Dice e sucesso dos óstios por configuração</strong></span></summary>

      | Configuração | N val | Dice geral | Mediana | Δ Dice | Sucesso dos óstios |
      | --- | --- | --- | --- | --- | --- |
      | Referência P99.9 | 270 | 0.6009 | 0.6593 | +0.000000 | 230/270 (85.19%) |
      | Busca inferior 100% | 270 | 0.6009 | 0.6593 | +0.000000 | 230/270 (85.19%) |
      | Busca inferior 70% | 270 | 0.5848 | 0.6452 | -0.016161 | 213/270 (78.89%) |
      | z máximo 30 mm | 270 | 0.6008 | 0.6593 | -0.000139 | 229/270 (84.81%) |
      | z máximo 50 mm | 270 | 0.6009 | 0.6593 | -0.000006 | 230/270 (85.19%) |
      | Divisor RG 5 | 270 | 0.6010 | 0.6593 | +0.000046 | 230/270 (85.19%) |
      | Divisor RG 9 | 270 | 0.6002 | 0.6576 | -0.000790 | 230/270 (85.19%) |
      | Piso vesselness 5% | 270 | 0.5720 | 0.6250 | -0.028937 | 230/270 (85.19%) |
      | Piso vesselness 9% | 270 | 0.5941 | 0.6593 | -0.006847 | 230/270 (85.19%) |
      | Percentil 99.5 | 270 | 0.5596 | 0.6233 | -0.041343 | 216/270 (80.00%) |
      | Percentil 99.7 | 270 | 0.5841 | 0.6532 | -0.016824 | 223/270 (82.59%) |

      Fonte: `output/segmentation/analysis/pipeline_parameter_validation/runs/sensitivity_refined_pipeline_val270/`. Onze configurações nas mesmas 270 imagens de validação. O Dice geral inclui as falhas registradas como zero. A busca de 100% já corresponde ao valor da referência.

      </details>

    - <span style="color:#2ea043"><strong>Análise das tabelas</strong></span>
    - <span style="color:#2ea043"><strong>Figura — Sensibilidade do Dice e dos óstios às variações dos parâmetros</strong></span>
    - <span style="color:#2ea043"><strong>Análise da figura</strong></span>
  - <span style="color:#218bff"><strong>4.2 Resultados do Pipeline Determinístico Atual — ATUALIZAR</strong></span>
    - 4.2.1 Visão Geral do Pipeline
      - Figura atual 18 — Visualização tridimensional da aorta, dos óstios e das artérias
      - Análise da figura
    - 4.2.2 Localização dos Óstios
      - <details>
        <summary><span style="color:#218bff"><strong>Tabela atual 2 — Desempenho da detecção dos óstios por resolução — ATUALIZAR</strong></span></summary>

        | Resolução | Conjunto | Exames | Ambos corretos | Ambos toleráveis | Sucesso |
        | --- | --- | --- | --- | --- | --- |
        | Mid | Treino | 30 | 2 | 27 | 29/30 (96.67%) |
        | Mid | Validação | 270 | 49 | 181 | 230/270 (85.19%) |
        | Mid | Teste | 700 | 152 | 445 | 597/700 (85.29%) |
        | High | Treino | 30 | 10 | 17 | 27/30 (90.00%) |
        | High | Validação | 270 | 119 | 94 | 213/270 (78.89%) |
        | High | Teste | 700 | 280 | 258 | 538/700 (76.86%) |

        ATUALIZAR com a referência consolidada identificada nas fontes. Mid usa os sweeps atuais; high mantém os runs completos. Sucesso = ambos corretos ou ambos toleráveis, contados como categorias exclusivas.

        </details>

      - Análise da tabela
      - <details>
        <summary><span style="color:#218bff"><strong>Tabela atual 3 — Distribuição dos erros na detecção dos óstios — ATUALIZAR</strong></span></summary>

        | Resolução | Conjunto | Um correto | Nenhum correto | Não encontrados | Erro do pipeline |
        | --- | --- | --- | --- | --- | --- |
        | Mid | Treino | 1 | 0 | 0 | 0 |
        | Mid | Validação | 15 | 20 | 0 | 5 |
        | Mid | Teste | 28 | 68 | 0 | 7 |
        | High | Treino | 1 | 2 | 0 | 0 |
        | High | Validação | 23 | 32 | 2 | 0 |
        | High | Teste | 80 | 79 | 3 | 0 |

        ATUALIZAR. Contagens por exame, sem sobreposição. Nos sweeps mid, um correto/nenhum correto é reconstruído pela interseção estrita com o label (distância física zero), apenas nos casos encontrados porém incorretos. As falhas de superfície da aorta permanecem como erro do pipeline, conforme os novos CSVs.

        </details>

      - Análise da tabela
      - <details>
        <summary><span style="color:#218bff"><strong>Tabela atual 4 — Interseção dos erros entre as resoluções — ATUALIZAR</strong></span></summary>

        | Conjunto | IDs comuns | Sucesso em ambas | Falha em ambas | Sucesso só mid | Sucesso só high |
        | --- | --- | --- | --- | --- | --- |
        | Treino | 30 | 27 | 1 | 2 | 0 |
        | Validação | 270 | 192 | 19 | 38 | 21 |
        | Teste | 700 | 489 | 54 | 108 | 49 |

        ATUALIZAR. Pareamento por IMG_ID nos mesmos conjuntos. Falha significa ausência de sucesso dos dois óstios, incluindo erro do pipeline. As coortes mid/high possuem os mesmos 30, 270 e 700 IDs.

        </details>

      - Análise da tabela
      - Figura atual 19 — Exemplos visuais de falhas na detecção dos óstios
      - Análise da figura
    - 4.2.3 Segmentação das Artérias Coronárias
      - <details>
        <summary><span style="color:#218bff"><strong>Tabela atual 5 — Dice Score por resolução e cenário de avaliação — ATUALIZAR</strong></span></summary>

        | Resolução | Conjunto | N geral | Dice geral ± DP | Mediana | N óstios válidos | Dice com óstios válidos | Dice com óstios inválidos |
        | --- | --- | --- | --- | --- | --- | --- | --- |
        | Mid | Treino | 30 | 0.6288 ± 0.1070 | 0.6573 | 29 | 0.6410 | 0.2743 |
        | Mid | Validação | 270 | 0.6009 ± 0.1776 | 0.6593 | 230 | 0.6493 | 0.3230 |
        | Mid | Teste | 700 | 0.5996 ± 0.1844 | 0.6677 | 597 | 0.6560 | 0.2726 |
        | High | Treino | 30 | 0.5720 ± 0.0894 | 0.5843 | 27 | 0.5874 | 0.4336 |
        | High | Validação | 270 | 0.5155 ± 0.1879 | 0.5794 | 213 | 0.5770 | 0.2854 |
        | High | Teste | 700 | 0.5146 ± 0.1837 | 0.5784 | 538 | 0.5805 | 0.2956 |

        ATUALIZAR. Dice arterial após morfologia; o cenário geral inclui falhas registradas como zero. O novo teste mid tem Dice 0,5996 e 597/700 sucessos, substituindo a referência anterior desta tabela (0,6018 e 599/700). DP = desvio-padrão amostral. Casos avaliáveis, sem falhas registradas, são apresentados separadamente no notebook.

        </details>

      - Análise da tabela
      - Figura atual 20 — Exemplos visuais de casos com alto Dice Score
      - Análise da figura
      - Figura atual 21 — Distribuição do Dice Score
      - Análise da figura
  - <span style="color:#2ea043"><strong>4.3 Efeito das Etapas Intermediárias e Análise de Falhas — ADICIONAR</strong></span>
    - <span style="color:#2ea043"><strong>4.3.1 Efeito dos Refinamentos da Aorta e da Localização dos Óstios</strong></span>
      - <details>
        <summary><span style="color:#2ea043"><strong>Tabela — Dice e sucesso dos óstios no pipeline anterior e no pipeline refinado</strong></span></summary>

        | Conjunto | N | Dice anterior | Dice refinado | Δ Dice | Óstios anterior | Óstios refinado |
        | --- | --- | --- | --- | --- | --- | --- |
        | Treino | 30 | 0.6148 | 0.6288 | 0.0140 | 27/30 (90.00%) | 29/30 (96.67%) |
        | Validação | 270 | 0.5879 | 0.6009 | 0.0131 | 226/270 (83.70%) | 230/270 (85.19%) |
        | Teste | 700 | 0.5930 | 0.5996 | 0.0066 | 578/700 (82.57%) | 597/700 (85.29%) |

        Mid resolution, mesmos IDs por conjunto. Anterior: `output/segmentation/runs/mid_res/current_baseline_p99_9/{train,val,test}/`. Refinado: normal+RG dos sweeps consolidados, com filtro + envelope, level set refinado e padding 2. A diferença corresponde ao pacote completo de refinamentos, não ao efeito isolado do filtro/envelope.

        </details>

      - <span style="color:#2ea043"><strong>Análise da tabela</strong></span>
      - <span style="color:#2ea043"><strong>Figura — Variação pareada do Dice entre as versões do pipeline</strong></span>
      - <span style="color:#2ea043"><strong>Análise da figura</strong></span>
      - <span style="color:#2ea043"><strong>Figura — Exemplos de sobresegmentação e subsegmentação da aorta no ImageCAS</strong></span>
      - <span style="color:#2ea043"><strong>Análise qualitativa da figura</strong></span>
    - <span style="color:#2ea043"><strong>4.3.2 Efeito do Pós-Processamento Morfológico</strong></span>
      - <details>
        <summary><span style="color:#2ea043"><strong>Tabela — Dice antes e depois da morfologia</strong></span></summary>

        | Resolução | Conjunto | Pares válidos | Dice antes | Dice depois | Δ Dice |
        | --- | --- | --- | --- | --- | --- |
        | Mid | Treino | 30 | 0.3374 | 0.6288 | 0.2913 |
        | Mid | Validação | 265 | 0.3170 | 0.6123 | 0.2953 |
        | Mid | Teste | 693 | 0.3173 | 0.6057 | 0.2883 |
        | High | Treino | 30 | 0.3032 | 0.5720 | 0.2688 |
        | High | Validação | 268 | 0.2687 | 0.5193 | 0.2506 |
        | High | Teste | 697 | 0.2664 | 0.5168 | 0.2503 |

        Somente pares pré/pós disponíveis e sem falha registrada antes da segmentação. As cinco falhas do sweep de validação, já salvas como zeros, são excluídas desta análise de efeito da morfologia. As sete falhas do teste mid não têm métricas pré/pós. Por isso a média pós não é a média geral da tabela 5.

        </details>

      - <span style="color:#2ea043"><strong>Análise da tabela</strong></span>
      - <span style="color:#2ea043"><strong>Figura — Exemplo da segmentação arterial antes e depois da morfologia</strong></span>
      - <span style="color:#2ea043"><strong>Análise da figura</strong></span>
  - <span style="color:#2ea043"><strong>4.4 Comparação dos Métodos Fuzzy — ADICIONAR</strong></span>
    - <details>
      <summary><span style="color:#2ea043"><strong>Tabela — Desempenho de normal+RG, fuzzy+RG, normal+FC e fuzzy+FC</strong></span></summary>

      | Conjunto | Método | N geral | Dice geral ± DP | Dice avaliável | N avaliável | Sucesso dos óstios |
      | --- | --- | --- | --- | --- | --- | --- |
      | Treino | Normal + RG | 30 | 0.6288 ± 0.1070 | 0.6288 | 30 | 29/30 (96.67%) |
      | Treino | Fuzzy + RG | 30 | 0.6206 ± 0.1173 | 0.6206 | 30 | 28/30 (93.33%) |
      | Treino | Normal + FC | 30 | 0.6279 ± 0.1042 | 0.6279 | 30 | 29/30 (96.67%) |
      | Treino | Fuzzy + FC | 30 | 0.6199 ± 0.1103 | 0.6199 | 30 | 28/30 (93.33%) |
      | Validação | Normal + RG | 60 | 0.5924 ± 0.2203 | 0.6128 | 58 | 50/60 (83.33%) |
      | Validação | Fuzzy + RG | 60 | 0.5833 ± 0.2330 | 0.6035 | 58 | 48/60 (80.00%) |
      | Validação | Normal + FC | 60 | 0.5639 ± 0.2276 | 0.5834 | 58 | 50/60 (83.33%) |
      | Validação | Fuzzy + FC | 60 | 0.5511 ± 0.2375 | 0.5701 | 58 | 48/60 (80.00%) |
      | Teste | Normal + RG | 700 | 0.5996 ± 0.1844 | 0.6057 | 693 | 597/700 (85.29%) |
      | Teste | Fuzzy + RG | 700 | 0.5988 ± 0.1882 | 0.6048 | 693 | 600/700 (85.71%) |
      | Teste | Normal + FC | 700 | 0.5851 ± 0.1867 | 0.5910 | 693 | 597/700 (85.29%) |
      | Teste | Fuzzy + FC | 700 | 0.5829 ± 0.1902 | 0.5888 | 693 | 600/700 (85.71%) |

      Fontes: `output/segmentation/runs/mid_res/fuzzy_comparison/current_pipeline/{train30_val60,test700}/`. As quatro variantes compartilham a configuração base refinada; threshold e segmentação são identificados nas colunas por exame. O Dice geral atribui zero às falhas registradas: duas na validação e sete no teste, em cada método. A média avaliável exclui essas falhas. Os 60 exames deste run são uma amostra da validação e não necessariamente os mesmos 60 da revisão visual da aorta.

      </details>

    - <details>
      <summary><span style="color:#2ea043"><strong>Tabela — Wilcoxon dos métodos fuzzy em relação ao pipeline determinístico</strong></span></summary>

      | Comparação versus normal+RG | N | Δ Dice | W | p bruto | p Holm | Efeito rank-biserial | IC95% Δ médio |
      | --- | --- | --- | --- | --- | --- | --- | --- |
      | Fuzzy + RG | 700 | -0.000836 | 19024 | 5.86e-05 | 5.86e-05 | +0.2592 | [-0.0061; 0.0045] |
      | Normal + FC | 700 | -0.014519 | 63486 | 1.56e-21 | 4.67e-21 | -0.4266 | [-0.0194; -0.0096] |
      | Fuzzy + FC | 700 | -0.016695 | 66170 | 1.62e-19 | 3.24e-19 | -0.4042 | [-0.0238; -0.0094] |

      Teste bilateral nos mesmos 700 exames; deltas zero são excluídos dos ranks. Holm ajusta as três comparações. O efeito rank-biserial positivo favorece a alternativa; IC95% bootstrap pareado da diferença média, com 10.000 reamostragens e seed 42. Para fuzzy+RG, 190 exames melhoram, 130 pioram e 380 empatam: o Wilcoxon favorece seus ranks, mas a média é ligeiramente menor e seu IC inclui zero. Não concluir melhora do Dice médio.

      </details>

    - <span style="color:#2ea043"><strong>Análise das tabelas</strong></span>
    - <span style="color:#2ea043"><strong>Figura — Dice pareado por método de segmentação</strong></span>
    - <span style="color:#2ea043"><strong>Análise da figura</strong></span>
  - <span style="color:#2ea043"><strong>4.5 Comparação com Métodos de Inteligência Artificial — ADICIONAR</strong></span>
    - <details>
      <summary><span style="color:#2ea043"><strong>Tabela — Dice dos métodos de IA e do pipeline por resolução</strong></span></summary>

      | Resolução | Método IA | IDs comuns | Folds | Dice IA ± DP | Dice pipeline ± DP | Δ pipeline − IA |
      | --- | --- | --- | --- | --- | --- | --- |
      | High | Direct FCN (4 canais) | 200 | 4 | 0.7655 ± 0.0597 | 0.5069 ± 0.1899 | -0.2586 |
      | High | Direct FCN (4 canais, atenção) | 200 | 4 | 0.7785 ± 0.0555 | 0.5069 ± 0.1899 | -0.2716 |
      | Mid | Direct FCN (12 canais) | 200 | 4 | 0.7195 ± 0.0501 | 0.6024 ± 0.1747 | -0.1171 |
      | Mid | Direct FCN (4 canais) | 200 | 4 | 0.7132 ± 0.0528 | 0.6024 ± 0.1747 | -0.1108 |
      | Mid | Direct FCN (4 canais, atenção) | 200 | 4 | 0.7172 ± 0.0495 | 0.6024 ± 0.1747 | -0.1148 |
      | Mid | Graph GCN | 200 | 4 | 0.5865 ± 0.1306 | 0.6024 ± 0.1747 | 0.0159 |
      | Mid | Tree GRU | 200 | 4 | 0.5916 ± 0.1429 | 0.6024 ± 0.1747 | 0.0108 |

      Pareamento por ID e resolução. A referência consolidada reúne treino, validação e teste; cada método IA disponível contém os mesmos 200 IDs distribuídos nos quatro folds, sem duplicações. A média do pipeline nessa coorte não equivale à média dos 700 exames de teste.

      </details>

    - <details>
      <summary><span style="color:#2ea043"><strong>Tabela — Wilcoxon do pipeline em relação a cada método de IA</strong></span></summary>

      | Resolução | Método IA | N | Δ pipeline − IA | W | p bruto | p Holm | Efeito rank-biserial | IC95% Δ médio |
      | --- | --- | --- | --- | --- | --- | --- | --- | --- |
      | High | Direct FCN (4 canais) | 200 | -0.2586 | 4 | 1.53e-34 | 1.01e-33 | -0.9996 | [-0.2851; -0.2342] |
      | High | Direct FCN (4 canais, atenção) | 200 | -0.2716 | 0 | 1.44e-34 | 1.01e-33 | -1.0000 | [-0.2982; -0.2475] |
      | Mid | Direct FCN (12 canais) | 200 | -0.1171 | 640 | 1.63e-30 | 8.14e-30 | -0.9363 | [-0.1416; -0.0943] |
      | Mid | Direct FCN (4 canais) | 200 | -0.1108 | 1231 | 5.28e-27 | 1.58e-26 | -0.8775 | [-0.1355; -0.0878] |
      | Mid | Direct FCN (4 canais, atenção) | 200 | -0.1148 | 655 | 2.01e-30 | 8.14e-30 | -0.9348 | [-0.1395; -0.0919] |
      | Mid | Graph GCN | 200 | +0.0159 | 7213 | 0.000537 | 0.00107 | +0.2823 | [-0.0146; 0.0450] |
      | Mid | Tree GRU | 200 | +0.0108 | 8108 | 0.0178 | 0.0178 | +0.1932 | [-0.0200; 0.0401] |

      Comparações nos mesmos 200 IDs por resolução e método. Wilcoxon bilateral, com deltas zero excluídos dos ranks; Holm nas sete comparações. Δ = pipeline − IA; efeito rank-biserial positivo favorece o pipeline. IC95% bootstrap pareado da diferença média, com 10.000 reamostragens e seed 42. A cobertura é completa entre os quatro folds disponíveis; isso não implica que os 200 exames representem todo o ImageCAS. Para GCN e GRU, o teste de ranks é significativo após Holm, mas o IC da diferença média inclui zero.

      </details>

    - <span style="color:#2ea043"><strong>Análise das tabelas</strong></span>
    - <span style="color:#2ea043"><strong>Figura — Comparação pareada entre IA e pipeline nos exames comuns</strong></span>
    - <span style="color:#2ea043"><strong>Análise da figura</strong></span>
  - <span style="color:#2ea043"><strong>4.6 Validação Externa em MM-WHS e OrCaScore — ADICIONAR</strong></span>
    - <details>
      <summary><span style="color:#2ea043"><strong>Tabela — Indicadores automáticos de qualidade da aorta por banco e resolução</strong></span></summary>

      | Banco | Resolução | Conjunto | N | Adequada | Suspeita sub | Suspeita sobre | Sem diagnóstico |
      | --- | --- | --- | --- | --- | --- | --- | --- |
      | MM-WHS | mid | Teste | 40 | 12 | 22 | 0 | 6 |
      | MM-WHS | mid | Treino | 20 | 9 | 11 | 0 | 0 |
      | OrCaScore | mid | Teste | 40 | 13 | 21 | 0 | 6 |
      | OrCaScore | mid | Treino | 32 | 10 | 19 | 0 | 3 |
      | MM-WHS | high | Teste | 40 | 15 | 25 | 0 | 0 |
      | MM-WHS | high | Treino | 20 | 10 | 10 | 0 | 0 |
      | OrCaScore | high | Teste | 39 | 16 | 20 | 0 | 3 |
      | OrCaScore | high | Treino | 32 | 12 | 19 | 0 | 1 |

      Diagnóstico automático, não confirmação visual nem acurácia contra ground truth. Fontes externas identificadas ao final deste capítulo. OrCaScore high contém 39 exames de teste, contra 40 no mid; a comparação entre resoluções exige restringir aos IDs comuns.

      </details>

    - <details>
      <summary><span style="color:#2ea043"><strong>Tabela — Dice da aorta no MM-WHS</strong></span></summary>

      | Resolução | Conjunto | N total | N com Dice | Dice válido ± DP | Mediana válida | Dice com falhas = 0 | Protocolo |
      | --- | --- | --- | --- | --- | --- | --- | --- |
      | mid | Teste | 40 | 34 | 0.7801 ± 0.2819 | 0.9061 | 0.6631 | Label 820, 1 mm |
      | mid | Treino | 20 | 20 | 0.8452 ± 0.2354 | 0.9324 | 0.8452 | Ground truth de treino |
      | high | Teste | 40 | 40 | 0.7636 ± 0.3321 | 0.9316 | 0.7636 | Label 820, 1 mm |
      | high | Treino | 20 | 20 | 0.8581 ± 0.2346 | 0.9380 | 0.8581 | Ground truth de treino |

      Usa os runs corrigidos de 06/10/2026 e o protocolo de aorta isolada, label 820, no teste. Em mid, seis máscaras vazias não têm Dice válido; a coluna conservadora atribui zero somente a essas falhas. Treino usa o ground truth disponível no próprio banco.

      </details>

    - <span style="color:#2ea043"><strong>Análise das tabelas</strong></span>
  - <span style="color:#218bff"><strong>4.7 Custo Computacional — ATUALIZAR</strong></span>
    - <details>
      <summary><span style="color:#218bff"><strong>Tabela — Tempo de execução por resolução e configuração — ATUALIZAR</strong></span></summary>

      | Resolução | Configuração | Tempo comparável |
      | --- | --- | --- |
      | Mid | Pipeline refinado | Consolidar runs do mesmo hardware e protocolo |
      | High | Pipeline refinado | Consolidar runs do mesmo hardware e protocolo |

      ATUALIZAR após confirmar hardware e protocolo dos runs. Os tempos de servidor e computador local não serão agregados em uma mesma estimativa.

      </details>

    - Análise da tabela

  - Fontes das tabelas preenchidas

    Fontes consolidadas: mid usa `results/image_results.csv`, filtrado por variante e conjunto; high mantém `numeric/results_{conjunto}.csv`. Os snapshots `effective_base_config` dos três sweeps mid são iguais:

    | Resolução | Conjunto | Run |
    | --- | --- | --- |
    | mid_res | train | [train30_val60](../output/segmentation/runs/mid_res/fuzzy_comparison/current_pipeline/train30_val60) — variante `normal_rg` |
    | mid_res | val | [sensitivity_refined_pipeline_val270](../output/segmentation/analysis/pipeline_parameter_validation/runs/sensitivity_refined_pipeline_val270) — variante `baseline` |
    | mid_res | test | [test700](../output/segmentation/runs/mid_res/fuzzy_comparison/current_pipeline/test700) — variante `normal_rg` |
    | high_res | train | [2026-09-15_11-08-42](../output/segmentation/runs/high_res/aorta_segmentation_experiments/train/filter_envelope_lower100_pad2/2026-09-15_11-08-42) |
    | high_res | val | [2026-09-12_07-51-48](../output/segmentation/runs/high_res/aorta_segmentation_experiments/val/filter_envelope_lower100_pad2/2026-09-12_07-51-48) |
    | high_res | test | [2026-09-13_14-06-44](../output/segmentation/runs/high_res/aorta_segmentation_experiments/test/filter_envelope_lower100_pad2/2026-09-13_14-06-44) |

    Referência anterior P99.9: treino `2026-08-06_18-43-37`, validação `2026-08-06_22-43-14` e teste `2026-08-06_10-04-22`, em `output/segmentation/runs/mid_res/current_baseline_p99_9/`.

    Resultados IA: `output/ia_results/{mid,high}/fold_*/result_*.csv`.

    Fontes externas, relativas a `EXTERNAL_CCTA_RESULTS_ROOT` (mesma raiz usada no notebook de avaliação externa):

    | Banco | Resolução | Run |
    | --- | --- | --- |
    | mmwhs | mid_res | `mmwhs/mid_res/2026-10-06_07-53-03/numeric/results_all.csv` |
    | orcascore | mid_res | `orcascore/mid_res/2026-09-19_10-23-50/numeric/results_all.csv` |
    | mmwhs | high_res | `mmwhs/high_res/2026-10-06_08-33-19/numeric/results_all.csv` |
    | orcascore | high_res | `orcascore/high_res/2026-09-22_19-12-41/numeric/results_all.csv` |

</details>

<details>
<summary><span style="color:inherit"><strong>5 Conclusões e Próximos Passos</strong></span></summary>

- 5 Conclusões e Próximos Passos

</details>

<details>
<summary><span style="color:inherit"><strong>Referências</strong></span></summary>

- Referências

</details>
