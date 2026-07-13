# Pipeline Config - Documentacao Completa

Este arquivo documenta todos os parametros de `config/pipeline_config.json`, com foco em:

- O que cada parametro controla
- Impacto no resultado da segmentacao
- Efeito em desempenho (tempo/memoria)
- Dicas de tuning

## Visao Geral

O `pipeline_config.json` organiza os parametros por etapas do pipeline:

1. Configuracoes globais de execucao (GPU, cache, downscale)
2. Vesselness para aorta e arterias
3. Deteccao de circulos para sementes/estrutura inicial
4. Evolucao por level set
5. Deteccao e validacao de ostios
6. Crescimento de regiao
7. Pos-processamento

---

## 1) Parametros Globais

### `USE_GPU` (bool)

- Valor atual: `true`
- Define se operacoes com suporte devem usar GPU.
- `true`: acelera partes pesadas (dependendo da implementacao e hardware).
- `false`: forca execucao em CPU (mais portavel, normalmente mais lenta).

Quando ajustar:

- Use `true` em ambientes com GPU compativel e memoria suficiente.
- Use `false` para depuracao, reprodutibilidade entre maquinas ou ausencia de GPU.

### `LOAD_CACHE` (bool)

- Valor atual: `false`
- Carrega resultados intermediarios previamente salvos.

Impacto:

- Reduz tempo de execucao quando ha cache valido.
- Pode mascarar mudancas de parametros se cache nao for invalidado corretamente.

### `SAVE_CACHE` (bool)

- Valor atual: `false`
- Salva resultados intermediarios para reutilizacao futura.

Impacto:

- Aumenta uso de disco.
- Diminui custo de experimentos repetidos.

### `DOWNSCALE_METHOD` (string)

- Valor atual: `"opencv"`
- Metodo usado para reduzir resolucao volumetrica.
- Valores possiveis: `"scipy"`, `"opencv"`

Impacto:

- Afeta fidelidade espacial e velocidade.
- Metodo inadequado pode suavizar demais bordas finas.

### `OPENCV_INTERPOLATION` (string)

- Valor atual: `"linear"`
- Interpolacao usada quando o metodo de downscale envolve OpenCV.
- Valores possiveis: `"nearest"`, `"linear"`, `"cubic"`, `"area"`, `"lanczos4"`

Notas:

- `linear` e o padrão atual usado pelo pipeline quando `DOWNSCALE_METHOD` e `opencv`.

### `DOWNSCALE_FACTORS` (array[int])

- Valor atual: `[2, 2, 1]`
- Fator de reducao por eixo (tipicamente `[x, y, z]` ou equivalente no pipeline).

Interpretacao comum:

- `2` em `x` e `y`: reduz metade da resolucao lateral.
- `1` em `z`: mantem resolucao axial original.

Trade-off:

- Fatores maiores: pipeline mais rapido, menor detalhe.
- Fatores menores: melhor detalhe, maior custo computacional.

### Maior componente conectada

O pipeline calcula a LCC separadamente em cada fatia 2D. O modo por volume foi
retirado porque não melhorou os resultados nos sweeps de aorta/óstios.

### `MIN_THRESHOLD` (int/float)

- Valor atual: `-300`
- Limiar minimo de intensidade usado antes do LCC no pre-processamento.
- Voxels abaixo desse valor sao removidos da mascara de threshold.

Interpretacao:

- Em CCTA, valores muito baixos tendem a representar ar/fundo e tecidos fora da faixa de interesse.
- Valores menos negativos tornam a mascara mais restritiva.
- Valores mais negativos preservam mais estruturas, mas tambem podem incluir mais ruido/fundo.

### `LOWER_THRESHOLD` (object)

- `method`: `percentile` e o padrão validado; `fixed` permanece para reproduzir
  o baseline histórico.
- `percentile`: `10.75` no threshold normal.
- `fixed_hu`: `-300`, usado apenas quando `method = "fixed"`.
- `clip_min_hu`/`clip_max_hu`: faixa válida para estimar o percentil.

### `THRESHOLDING` (object)

- `method`: `normal` ou `fuzzy`.
- O fuzzy usa somente `object_argmax`, com margem 100 HU, centro de objeto
  P99,8, centro denso P99,96 e sem suavização espacial.
- `fuzzy.lower_percentile = 10.5` aplica automaticamente o piso vencedor ao
  selecionar fuzzy; o normal continua usando P10,75.

### `MAX_THRESHOLD_PERCENTILE` (float)

- Valor atual: `99.8`
- Percentil usado para limitar/clamp de intensidade em alguma etapa de limiarizacao normalizada.

Efeito pratico:

- Percentis altos preservam estruturas brilhantes, mas podem manter outliers.
- Percentis mais baixos podem melhorar robustez a ruido extremo, mas cortam sinal util.

---

## 2) Vesselness da Aorta - `VESSELNESS_AORTA`

Realca estruturas tubulares para facilitar segmentacao/deteccao.

### `method` (string)

- Valor atual: `"normal"`
- Seleciona a variante de vesselness.
- O único método mantido é `"normal"` (Frangi).

Interpretacao:

- `normal`: usa diretamente o filtro Frangi.

### `sigmas` (array[float])

- Valor atual: `[2.0, 2.25, 2.5]`
- Escalas gaussianas para analise multiescala.

Interpretacao:

- Sigma maior favorece estruturas mais largas.
- Sigma menor favorece estruturas finas.

### `black_ridges` (bool)

- Valor atual: `false`
- Define polaridade do filtro:
  - `false`: busca estruturas claras em fundo escuro
  - `true`: busca estruturas escuras em fundo claro

### `alpha` (float)

- Valor atual: `0.5`
- Controla sensibilidade ao termo de forma (discriminacao placa/tubo em variantes Frangi-like).

### `beta` (float)

- Valor atual: `1.0`
- Ajusta penalizacao para geometrias nao tubulares.

### `gamma` (float)

- Valor atual: `30`
- Controla sensibilidade ao termo de estrutura (forca do sinal vs ruido).

### `normalization` (string)

- Valor atual: `"none"`
- Estrategia de normalizacao da resposta entre escalas.
- Valores possiveis: `"none"`, `"minmax"`, `"robust"`

Dica:

- Se a resposta ficar muito dependente de escala, considere testar normalizacao apropriada.

### `smooth_sigma` (float)

- Valor atual: `0.0`
- Suavizacao gaussiana opcional aplicada antes do Frangi no metodo `"normal"`.
- `0.0` desativa a suavizacao e preserva o comportamento baseline.

Trade-off:

- Valores pequenos podem reduzir ruido antes do Frangi.
- Valores altos podem borrar arterias finas e reduzir resposta em vasos pequenos.

---

## 3) Vesselness das Arterias - `VESSELNESS_ARTERY`

Mesmo conceito da aorta, com foco em arterias coronarias (mais finas e ramificadas).

### `method` (string)

- Valor atual: `"normal"`
- O único método mantido é `"normal"` (Frangi).

### `sigmas` (array[float])

- Valor atual: `[1.5, 2.0, 2.5, 3.0]`
- Conjunto multiescala para cobrir variacao de calibre arterial.

### `black_ridges` (bool)

- Valor atual: `false`

### `alpha` (float)

- Valor atual: `0.5`

### `beta` (float)

- Valor atual: `0.5`
- Menor que na aorta, tipicamente aumenta seletividade para estruturas mais tubulares/finas.

### `gamma` (float)

- Valor atual: `55`
- Maior sensibilidade ao termo de estrutura em relacao a aorta.

### `normalization` (string)

- Valor atual: `"none"`
- Valores possiveis: `"none"`, `"minmax"`, `"robust"`

### `smooth_sigma` (float)

- Valor atual: `0.0`
- Mesma interpretacao de `VESSELNESS_AORTA.smooth_sigma`.

---

## 4) Deteccao de Circulos - `CIRCLE_DETECTION`

Busca circulos candidatos (geralmente em cortes 2D) para localizar estruturas de interesse e inicializar etapas seguintes.

### `radii_start_px` (int)

- Valor atual: `36`
- Raio minimo (pixels) testado na busca de circulos.

### `radii_end_px` (int)

- Valor atual: `62`
- Raio maximo (pixels) testado.

### `radius_step_px` (int)

- Valor atual: `1`
- Passo entre raios testados.

Trade-off do trio de raios:

- Intervalo amplo + passo pequeno: maior cobertura, maior custo.
- Intervalo estreito: mais rapido, risco de perder anatomias fora da faixa.

### `tol_radius_mm` (float)

- Valor atual: `9.0`
- Tolerancia fisica para variacao de raio ao validar/associar deteccoes.

### `tol_distance_mm` (float)

- Valor atual: `20.0`
- Distancia maxima (mm) permitida para associacao entre candidatos (ex.: continuidade entre slices).

### `quadrant_offset` (array[int])

- Valor atual: `[30, 30]`
- Deslocamento espacial usado em regras por quadrante/ROI para evitar regioes indesejadas ou deslocar buscas.

### `max_slice_miss_threshold` (int)

- Valor atual: `5`
- Numero maximo de slices sem deteccao antes de interromper/invalidar rastreamento.

### `neighbor_distance_threshold` (int/float)

- Valor atual: `5`
- Limite de distancia entre deteccoes vizinhas para agrupamento/consistencia local.

### `total_num_peaks_initial` (int)

- Valor atual: `15`
- Numero de picos candidatos na fase inicial.

### `total_num_peaks` (int)

- Valor atual: `15`
- Numero de picos mantidos em etapa posterior/refinada.

### `canny_sigma` (float)

- Valor atual: `3`
- Suavizacao aplicada antes/na deteccao de borda (Canny).

Efeito:

- Sigma maior: menos ruido, mas pode apagar bordas finas.
- Sigma menor: mais detalhe, mais falso-positivo por ruido.

### `use_local_roi` (bool)

- Valor atual: `true`
- Usa ROI local para restringir a deteccao e acelerar/estabilizar.

### `local_roi_padding` (int)

- Valor atual: `30`
- Margem extra da ROI local (pixels).

Trade-off:

- Padding maior: menos risco de cortar estrutura valida, mais custo/falso positivo.
- Padding menor: mais foco, risco de truncar alvo.

### Rastreamento validado

- Filtra candidatos pelas tolerâncias de raio/distância e escolhe o válido mais
  próximo do último círculo aceito.
- `early_track_recovery = true` tenta novas inicializações nas 8 fatias
  anteriores quando a trajetória inicial tem menos de 10 círculos.
- Candidatos fora da tolerância interrompem o rastreamento; eles não são mais
  tratados como misses.

---

## 5) Level Set - `LEVEL_SET`

Controla evolucao da fronteira de segmentacao a partir de sementes/candidatos.

### `radius_reduction_factor` (float)

- Valor atual: `0.15`
- Reduz raio inicial para evitar vazamento na inicializacao do contorno.

### `num_iter` (int)

- Valor atual: `31`
- Numero de iteracoes de evolucao.

Impacto:

- Mais iteracoes podem refinar melhor, mas aumentam tempo e risco de leakage.

### `balloon` (float)

- Valor atual: `0.8`
- Termo de forca de expansao/contracao do contorno.

Interpretacao comum:

- Positivo: tende a expandir.
- Negativo: tende a contrair.
- Magnitude alta: evolucao mais agressiva.

### `smoothing` (int/float)

- Valor atual: `2`
- Grau de suavizacao do contorno em cada iteracao.

Efeito:

- Suavizacao alta reduz serrilhado e ruido, mas perde detalhes finos.

### `leak_removal_radius` (int)

- Valor atual: `2`
- Raio estrutural para limpar pequenos vazamentos apos evolucao.

### Correcao condicional pela trajetoria

- `trajectory_area_ratio_threshold`: quando definido, corrige somente fatias
  cuja area da mascara excede a area esperada pelo circulo rastreado. Fica
  desativado (`null`) no metodo `standard` e vale `2.0` em `bilateral_thin`.
- `trajectory_correction_radius_factor`: fator `1.75` usado para reconstruir
  as fatias consideradas anomalas.

---

## 6) Deteccao de Ostios - `OSTIA_DETECTION`

Seleciona candidatos de ostio e filtra por regras anatomicas/geometricas.

### `top_n` (int)

- Valor atual: `2000`
- Numero de candidatos iniciais considerados antes dos filtros finais.

### `max_z_diff_mm` (float)

- Valor atual: `40.0`
- Diferenca maxima em Z (mm) entre candidatos para consistencia espacial.

### `lower_fraction` (float)

- Valor atual: `0.8`
- Fracao inferior usada como corte relativo em score/intensidade/posicao (dependendo da implementacao).

### `min_center_distance_factor` (float)

- Valor atual: `0.85`
- Fator minimo de separacao em relacao ao centro/referencia para rejeitar candidatos muito proximos.

### `min_lateral_factor` (float)

- Valor atual: `0.4`
- Fator minimo para deslocamento lateral esperado, ajudando a evitar solucoes degeneradas.

### `erosion_radius` (int)

- Valor atual: `4`
- Raio de erosao morfologica em mascara auxiliar para robustez da selecao.

### Selecao do par

- `pair_selection_mode = "greedy"`: comportamento historico, que escolhe o
  primeiro candidato pelo vesselness e procura o segundo pelas restricoes
  geometricas.
- `pair_selection_mode = "bilateral"`: separa candidatos pelos dois lados da
  trajetoria da aorta e procura um par bilateral consistente.
- `bilateral_top_k_per_side = 50`: quantidade maxima de candidatos avaliada em
  cada lado no metodo bilateral.

---

## 6.1) Metodo conjunto de aorta e ostios - `AORTA_OSTIA_METHOD`

O campo `method` permite alternar entre dois perfis sem editar manualmente os
parametros internos:

- `standard`: preserva o comportamento historico, com erosao 4, selecao
  `greedy` e sem correcao condicional da mascara da aorta.
- `bilateral_thin`: usa erosao 2, selecao bilateral com 50 candidatos por lado
  e corrige fatias cuja area excede duas vezes a area esperada pela trajetoria.

Pela linha de comando:

```bash
# Comportamento historico
uv run python src/segmentation_pipeline.py --aorta-ostia-method standard

# Nova abordagem validada
uv run python src/segmentation_pipeline.py --aorta-ostia-method bilateral_thin
```

O alias `bilateral_thin_conditional` tambem e aceito pela CLI e resolve para o
perfil `bilateral_thin`.

---

## 7) Validacao de Ostios - `OSTIA_VALIDATION`

### `distance_threshold_mm` (float)

- Valor atual: `7.0`
- Distancia maxima aceitavel para considerar uma deteccao valida contra referencia/regra de consistencia.

Uso tipico:

- Threshold menor: validacao mais rigorosa (menos falso-positivo, mais falso-negativo).
- Threshold maior: mais tolerante (pode aumentar falso-positivo).

---

## 8) Region Growing - `REGION_GROWING`

Expansao adaptativa da segmentacao com controle por intensidade/vesselness e limites de seguranca.

### `max_volume` (int)

- Valor atual: `100000`
- Volume maximo permitido para a regiao crescer (controle anti-explosao).

### `switch_at_voxels` (int)

- Valor atual: `2000`
- Ponto de troca de estrategia quando a segmentacao atinge certo numero de voxels.

### `min_vesselness_fraction` (float)

- Valor atual: `0.078`
- Fracao minima de vesselness para aceitar expansao em regioes de baixa confianca.

### `threshold_divisor` (int/float)

- Valor atual: `7`
- Fator divisor para limiar adaptativo interno.

### `relaxed_floor_factor` (float)

- Valor atual: `0.98`
- Piso relaxado do limiar para permitir continuidade em trechos mais dificeis.

### `comparison_window` (int)

- Valor atual: `1`
- Janela local usada para comparacoes estatisticas/adaptativas.

### `smooth_relaxation` (bool)

- Valor atual: `true`
- Aplica relaxamento suave de limiar, evitando mudancas abruptas entre iteracoes.

### `verbose` (bool)

- Valor atual: `false`
- Habilita logs detalhados da etapa de region growing.

---

## 9) Fuzzy Connectedness - `FUZZY_CONNECTEDNESS`

O FC usa os óstios como sementes, restringe a propagação pela LCC/vesselness e
combina afinidade geométrica de vesselness com similaridade Gaussiana em HU.

- `alpha = 0.16`: corte final do mapa de conectividade.
- `sigma_hu = 100`: escala da similaridade de intensidade.
- `neighborhood = 26`: vizinhança 3D usada na propagação max-min.
- `candidate_min_vesselness`, `seed_min_vesselness` e `vesselness_floor = 0.018`:
  pisos validados para região candidata, sementes e afinidade.
- `vesselness_weight = 0.9`: peso do vesselness no produto ponderado com HU.
- `seed_search_radius = 2` e `max_seeds_per_ostium = 4`: refinamento local
  das sementes ao redor de cada óstio.
- `max_candidate_voxels` e `max_processed_voxels`: limites de segurança de
  memória/tempo.

As afinidades alternativas, sementes de fundo e corte por percentil foram
retirados após não melhorarem o resultado.

## 10) Pos-processamento - `POSTPROCESSING`

Operacoes morfologicas finais para consolidar mascara.

### `closing_radius` (int)

- Valor atual: `3`
- Raio de fechamento morfologico (preenche pequenos buracos e conecta lacunas curtas).

### `dilation_radius` (int)

- Valor atual: `2`
- Raio de dilatacao final (expande levemente a mascara).

Trade-off:

- Valores altos podem unir estruturas indevidas.
- Valores baixos podem manter falhas locais.

---

## Boas Praticas de Tuning

1. Ajuste por bloco funcional, nao todos os parametros ao mesmo tempo.
2. Comece por escala: `DOWNSCALE_FACTORS`, `VESSELNESS_* .sigmas`, faixa de `radii_*`.
3. Depois ajuste estabilidade: `LEVEL_SET` e `REGION_GROWING`.
4. Finalize com filtros morfologicos (`POSTPROCESSING`) para acabamento.
5. Sempre compare com metricas e casos dificeis (nao apenas casos "faceis").

## Roteiro Recomendado de Experimento

1. Congele seed/dados e desative cache para linha de base (`LOAD_CACHE=false`).
2. Rode baseline e salve metricas.
3. Varie apenas 1-2 parametros por vez.
4. Registre efeito em:

- acuracia
- robustez entre pacientes
- tempo de execucao
- memoria

1. Se encontrar melhor configuracao, ative `SAVE_CACHE` para acelerar iteracoes futuras.

---

## Mapa Rapido de Sensibilidade (Resumo)

- Mais impacto em geometria global:
  - `DOWNSCALE_FACTORS`
  - `VESSELNESS_AORTA.sigmas`
  - `VESSELNESS_ARTERY.sigmas`
  - `CIRCLE_DETECTION.radii_start_px` / `radii_end_px`

- Mais impacto em vazamento/estabilidade:
  - `LEVEL_SET.balloon`
  - `LEVEL_SET.num_iter`
  - `LEVEL_SET.leak_removal_radius`
  - `REGION_GROWING.relaxed_floor_factor`

- Mais impacto em acabamento final:
  - `POSTPROCESSING.closing_radius`
  - `POSTPROCESSING.dilation_radius`
