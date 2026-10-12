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

      | Parâmetro | Referência histórica | Valores com resultados disponíveis |
      | --- | --- | --- |
      | Percentil superior | 99.9 | 99.5; 99.7; 99.9 |
      | Distância máxima z dos óstios | 40 mm | 30; 40; 50 mm |
      | Piso mínimo de vesselness | 7.8% | 5%; 7.8%; 9% |
      | Divisor global do RG | 7 | 7 (5 e 9 não constam nestes runs) |

      Referência histórica CBEB, anterior aos refinamentos da aorta. O piso registrado foi 7,8%, e não 7%.

      </details>

    - <details>
      <summary><span style="color:#2ea043"><strong>Tabela — Dice e sucesso dos óstios por configuração</strong></span></summary>

      | Configuração histórica | N val | Dice médio | Mediana | Sucesso dos óstios |
      | --- | --- | --- | --- | --- |
      | Referência P99.9 | 270 | 0.5879 | 0.6535 | 226/270 (83.70%) |
      | z máximo 30 mm | 270 | 0.5853 | 0.6531 | 225/270 (83.33%) |
      | z máximo 50 mm | 270 | 0.5879 | 0.6535 | 226/270 (83.70%) |
      | Piso vesselness 5% | 270 | 0.5604 | 0.6237 | 226/270 (83.70%) |
      | Piso vesselness 9% | 270 | 0.5816 | 0.6562 | 226/270 (83.70%) |
      | Percentil 99.7 | 270 | 0.5719 | 0.6406 | 218/270 (80.74%) |
      | Percentil 99.5 | 270 | 0.5512 | 0.6196 | 217/270 (80.37%) |

      Fontes: `sensitivity_cbeb_p999_val_270` e `sensitivity_selected_val_270`, em `output/segmentation/analysis/pipeline_parameter_validation/runs/`. São as mesmas 270 imagens de validação; estes números descrevem o pipeline histórico e não substituem uma sensibilidade do pipeline refinado.

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
        | Mid | Validação | 270 | 48 | 181 | 229/270 (84.81%) |
        | Mid | Teste | 700 | 155 | 444 | 599/700 (85.57%) |
        | High | Treino | 30 | 10 | 17 | 27/30 (90.00%) |
        | High | Validação | 270 | 119 | 94 | 213/270 (78.89%) |
        | High | Teste | 700 | 280 | 258 | 538/700 (76.86%) |

        ATUALIZAR a tabela existente com os runs refinados identificados ao final deste capítulo. Sucesso = ambos corretos ou ambos toleráveis; as duas categorias são exclusivas na contagem.

        </details>

      - Análise da tabela
      - <details>
        <summary><span style="color:#218bff"><strong>Tabela atual 3 — Distribuição dos erros na detecção dos óstios — ATUALIZAR</strong></span></summary>

        | Resolução | Conjunto | Um correto | Nenhum correto | Não encontrados | Erro do pipeline |
        | --- | --- | --- | --- | --- | --- |
        | Mid | Treino | 1 | 0 | 0 | 0 |
        | Mid | Validação | 15 | 21 | 5 | 0 |
        | Mid | Teste | 27 | 67 | 7 | 0 |
        | High | Treino | 1 | 2 | 0 | 0 |
        | High | Validação | 23 | 32 | 2 | 0 |
        | High | Teste | 80 | 79 | 3 | 0 |

        ATUALIZAR. Contagens por exame, usando o status salvo no CSV; cada exame ocupa uma única categoria.

        </details>

      - Análise da tabela
      - <details>
        <summary><span style="color:#218bff"><strong>Tabela atual 4 — Interseção dos erros entre as resoluções — ATUALIZAR</strong></span></summary>

        | Conjunto | IDs comuns | Sucesso em ambas | Falha em ambas | Sucesso só mid | Sucesso só high |
        | --- | --- | --- | --- | --- | --- |
        | Treino | 30 | 27 | 1 | 2 | 0 |
        | Validação | 270 | 191 | 19 | 38 | 22 |
        | Teste | 700 | 489 | 52 | 110 | 49 |

        ATUALIZAR. Pareamento por `IMG_ID`, dentro do mesmo conjunto. Falha significa ausência de sucesso dos dois óstios, incluindo um correto, nenhum correto ou não encontrados.

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
        | Mid | Validação | 270 | 0.5992 ± 0.1814 | 0.6593 | 229 | 0.6501 | 0.3151 |
        | Mid | Teste | 700 | 0.6018 ± 0.1814 | 0.6679 | 599 | 0.6561 | 0.2798 |
        | High | Treino | 30 | 0.5720 ± 0.0894 | 0.5843 | 27 | 0.5874 | 0.4336 |
        | High | Validação | 270 | 0.5155 ± 0.1879 | 0.5794 | 213 | 0.5770 | 0.2854 |
        | High | Teste | 700 | 0.5146 ± 0.1837 | 0.5784 | 538 | 0.5805 | 0.2956 |

        ATUALIZAR. Dice arterial após morfologia. O cenário geral inclui os exames com Dice zero; os cenários condicionados usam o sucesso dos óstios da tabela 2. DP = desvio-padrão amostral.

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
        | Validação | 270 | 0.5879 | 0.5992 | 0.0113 | 226/270 (83.70%) | 229/270 (84.81%) |
        | Teste | 700 | 0.5930 | 0.6018 | 0.0088 | 578/700 (82.57%) | 599/700 (85.57%) |

        Mid resolution, mesmos IDs por conjunto. Anterior: `output/segmentation/runs/mid_res/current_baseline_p99_9/{train,val,test}/`. Refinado: filtro + envelope, level set refinado e localização dos óstios com padding 2. A diferença corresponde ao pacote completo de refinamentos, não ao efeito isolado do filtro/envelope.

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
        | Mid | Validação | 265 | 0.3161 | 0.6105 | 0.2944 |
        | Mid | Teste | 693 | 0.3184 | 0.6079 | 0.2895 |
        | High | Treino | 30 | 0.3032 | 0.5720 | 0.2688 |
        | High | Validação | 268 | 0.2687 | 0.5193 | 0.2506 |
        | High | Teste | 697 | 0.2664 | 0.5168 | 0.2503 |

        Somente exames com Dice pré e pós disponível. Os exames sem segmentação não possuem essas duas métricas e ficam fora desta comparação; por isso a média pós pode diferir do cenário geral da tabela 5.

        </details>

      - <span style="color:#2ea043"><strong>Análise da tabela</strong></span>
      - <span style="color:#2ea043"><strong>Figura — Exemplo da segmentação arterial antes e depois da morfologia</strong></span>
      - <span style="color:#2ea043"><strong>Análise da figura</strong></span>
  - <span style="color:#2ea043"><strong>4.4 Comparação dos Métodos Fuzzy — ADICIONAR</strong></span>
    - <details>
      <summary><span style="color:#2ea043"><strong>Tabela — Desempenho de normal+RG, fuzzy+RG, normal+FC e fuzzy+FC</strong></span></summary>

      | Limiarização | Segmentação | Resultado sob o pipeline atual |
      | --- | --- | --- |
      | Normal | RG | Disponível nas tabelas 2–5; alinhar aos novos runs fuzzy |
      | Fuzzy | RG | Aguardando comparação atualizada |
      | Normal | FC | Aguardando comparação atualizada |
      | Fuzzy | FC | Aguardando comparação atualizada |

      Preencher após concluir os quatro métodos sob a mesma configuração atual. Os resultados fuzzy históricos usam configurações anteriores e não devem ser misturados a esta tabela.

      </details>

    - <details>
      <summary><span style="color:#2ea043"><strong>Tabela — Wilcoxon dos métodos fuzzy em relação ao pipeline determinístico</strong></span></summary>

      | Comparação pareada | N | Δ Dice | W | p bruto / Holm | Efeito rank-biserial | IC95% bootstrap |
      | --- | --- | --- | --- | --- | --- | --- |
      | Fuzzy+RG versus normal+RG | Pendente | Pendente | Pendente | Pendente | Pendente | Pendente |
      | Normal+FC versus normal+RG | Pendente | Pendente | Pendente | Pendente | Pendente | Pendente |
      | Fuzzy+FC versus normal+RG | Pendente | Pendente | Pendente | Pendente | Pendente | Pendente |

      Preencher com os novos runs pareados: teste bilateral, ajuste de Holm nas três comparações, efeito rank-biserial e IC95% bootstrap da diferença média.

      </details>

    - <span style="color:#2ea043"><strong>Análise das tabelas</strong></span>
    - <span style="color:#2ea043"><strong>Figura — Dice pareado por método de segmentação</strong></span>
    - <span style="color:#2ea043"><strong>Análise da figura</strong></span>
  - <span style="color:#2ea043"><strong>4.5 Comparação com Métodos de Inteligência Artificial — ADICIONAR</strong></span>
    - <details>
      <summary><span style="color:#2ea043"><strong>Tabela — Dice dos métodos de IA e do pipeline por resolução</strong></span></summary>

      | Resolução | Método IA | IDs comuns | Folds | Dice IA ± DP | Dice pipeline ± DP | Δ pipeline − IA |
      | --- | --- | --- | --- | --- | --- | --- |
      | High | HIGH - direct FCN 4 | 200 | 4 | 0.7655 ± 0.0597 | 0.5069 ± 0.1899 | -0.2586 |
      | High | HIGH - direct FCN AG 4 | 200 | 4 | 0.7785 ± 0.0555 | 0.5069 ± 0.1899 | -0.2716 |
      | Mid | MID - direct FCN 12 | 200 | 4 | 0.7195 ± 0.0501 | 0.6015 ± 0.1750 | -0.1180 |
      | Mid | MID - direct FCN 4 | 200 | 4 | 0.7132 ± 0.0528 | 0.6015 ± 0.1750 | -0.1117 |
      | Mid | MID - direct FCN AG 4 | 200 | 4 | 0.7172 ± 0.0495 | 0.6015 ± 0.1750 | -0.1157 |
      | Mid | MID - graph GCN | 200 | 4 | 0.5865 ± 0.1306 | 0.6015 ± 0.1750 | 0.0149 |
      | Mid | MID - tree GRU 16 16 4 | 200 | 4 | 0.5916 ± 0.1429 | 0.6015 ± 0.1750 | 0.0098 |

      Resultados descritivos, pareados por ID e resolução. Pipeline refinado carregado dos três conjuntos; cada método IA possui 200 IDs únicos nos quatro folds disponíveis. A média do pipeline nesta coorte difere da média dos 700 exames de teste.

      </details>

    - <details>
      <summary><span style="color:#2ea043"><strong>Tabela — Wilcoxon do pipeline em relação a cada método de IA</strong></span></summary>

      | Métodos disponíveis | Cobertura | Situação da análise inferencial |
      | --- | --- | --- |
      | Todos os métodos da tabela anterior | 4 folds; 200 IDs por método | Pendente de confirmar/completar os folds do benchmark antes da inferência final |

      A cobertura disponível ainda precisa ser confrontada com o protocolo do benchmark; a análise inferencial final prevista no plano exige cobertura completa dos folds. Não há estatísticas inventadas para os folds ausentes.

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

    Runs ImageCAS refinados; os CSVs usados são `numeric/results_{conjunto}.csv`:

    | Resolução | Conjunto | Run |
    | --- | --- | --- |
    | mid_res | train | [2026-09-06_08-08-25](../output/segmentation/runs/mid_res/aorta_segmentation_experiments/train/ostia_localization/lower100_pad2/2026-09-06_08-08-25) |
    | mid_res | val | [2026-09-07_07-15-15](../output/segmentation/runs/mid_res/aorta_segmentation_experiments/val/ostia_validation270/lower100_pad2/2026-09-07_07-15-15) |
    | mid_res | test | [2026-09-07_10-37-38](../output/segmentation/runs/mid_res/aorta_segmentation_experiments/test/ostia_comparison/lower100_pad2/2026-09-07_10-37-38) |
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
