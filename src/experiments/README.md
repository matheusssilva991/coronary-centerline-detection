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
  --gpu \
  --no-save-cache

uv run python src/experiments/compare_cpu_gpu.py --help
```

Experimentos encerrados e as razões para descarte estão documentados em
[`output/segmentation/analysis/EXPERIMENTS_ARCHIVE.md`](../../output/segmentation/analysis/EXPERIMENTS_ARCHIVE.md).
