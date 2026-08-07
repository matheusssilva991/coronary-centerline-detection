# Experimentos mantidos

Esta pasta contém somente experimentos ainda úteis para comparação ou ajuste
do pipeline. Testes automatizados continuam em `tests/`; resultados derivados
ficam em `output/segmentation/analysis/`.

## Scripts

- `fuzzy_pipeline_comparison.py`: compara threshold normal/fuzzy combinado com
  region growing ou fuzzy connectedness.
- `compare_cpu_gpu.py`: compara resultados intermediários dos backends CPU e
  GPU para localizar diferenças numéricas.
- `threshold_parameter_sweep.py`: varia limites inferior/superior e parâmetros
  do threshold fuzzy, mantendo o restante do pipeline controlado.
- `pipeline_failure_analysis.py`: seleciona, a partir dos quatro runs de
  validação, uma coorte focada de falhas e controles para testar correções.
- `pipeline_parameter_validation.py`: executa a análise OFAT de sensibilidade
  solicitada para o artigo. Compara nove configurações no split de validação e
  mede sucesso dos óstios e Dice. A referência congelada em
  `config/article_sensitivity_reference.json` reproduz o comportamento do run
  `train/normal_rg/2026-06-29_09-42-27`. Seus resultados alimentam o notebook
  `src/eda/pipeline_parameter_validation_eda.ipynb`.

Helpers reutilizáveis ficam em `src/utils/experiments/`.

## Runners

O runner de threshold executa sua seleção de parâmetros:

```bash
bash src/experiments/runners/run_threshold_sweeps.sh
```

O runner de falhas compara correções isoladas de RG, FC e fuzzy threshold:

```bash
MODE=corrections bash src/experiments/runners/run_pipeline_failure_improvement.sh
```

Para selecionar uma GPU:

```bash
CUDA_VISIBLE_DEVICES=1 bash src/experiments/runners/run_threshold_sweeps.sh
```

## Exemplos

```bash
uv run python src/experiments/fuzzy_pipeline_comparison.py \
  --split val \
  --sample-size 60 \
  --variants normal_rg,fuzzy_threshold_rg,normal_threshold_fc,fuzzy_threshold_fc

uv run python src/experiments/threshold_parameter_sweep.py \
  --split train \
  --percentiles 1,2,5,10 \
  --num-batches 5 \
  --gpu

uv run python src/experiments/compare_cpu_gpu.py --help

uv run python src/experiments/pipeline_parameter_validation.py \
  --split val \
  --sample-size 30 \
  --resolution mid \
  --gpu
```

Para confirmar no conjunto completo de validação somente os grupos que mais
variaram na triagem de 30 imagens:

```bash
uv run python src/experiments/pipeline_parameter_validation.py \
  --split val \
  --sample-size 270 \
  --resolution mid \
  --variants baseline,upper_p995,upper_p999,rg_vessel_05,rg_vessel_09 \
  --run-name sensitivity_selected_val_270 \
  --gpu
```

No computador usado na triagem, cada variante processou 30 imagens em cerca de
21 minutos. Sem reaproveitamento entre variantes, a projeção para 270 imagens
é de aproximadamente 3 h 10 min por variante, ou 15 h 50 min para as cinco
variantes acima. O tempo real depende principalmente da GPU e do armazenamento.

A validação pode ser dividida sem separar os resultados. Execute a primeira
parte com um `--run-name` fixo:

```bash
uv run python src/experiments/pipeline_parameter_validation.py \
  --split val --sample-size 270 --resolution mid --gpu \
  --variants baseline,upper_p995,upper_p999 \
  --run-name sensitivity_selected_val_270
```

Depois anexe as variantes restantes ao mesmo run:

```bash
uv run python src/experiments/pipeline_parameter_validation.py \
  --split val --sample-size 270 --resolution mid --gpu \
  --variants rg_vessel_05,rg_vessel_09 \
  --run-name sensitivity_selected_val_270 \
  --append
```

`--append` confere IDs, split, resolução, configuração, método de aorta/óstios
e backend antes de combinar os CSVs. Variantes já concluídas são ignoradas.

As variantes `ostia_z30` e `ostia_z50` estão disponíveis para a sensibilidade
do limite axial, mas não alteraram os resultados na triagem de 30 imagens. Para
provocar uma análise mais informativa da localização, podem ser anexadas duas
variantes OFAT da região de busca: 70% e 100% da extensão inferior da aorta. O
baseline utiliza 85%:

```bash
uv run python src/experiments/pipeline_parameter_validation.py \
  --split val --sample-size 270 --resolution mid --gpu \
  --variants ostia_lower_70,ostia_lower_100 \
  --run-name sensitivity_selected_val_270 \
  --append
```

O sweep de threshold mantém apenas configurações, resumos e o CSV consolidado
por imagem. Para preservar excepcionalmente os runs internos completos, use
`--keep-pipeline-runs`.

Experimentos encerrados e as razões para descarte estão documentados em
[`output/segmentation/analysis/EXPERIMENTS_ARCHIVE.md`](../../output/segmentation/analysis/EXPERIMENTS_ARCHIVE.md).
