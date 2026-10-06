# Repository Guidelines

## Project Structure & Module Organization

Core code lives in `src/`. Run the ImageCAS workflow through
`src/segmentation_pipeline.py`; use `src/main.ipynb` for an inspectable single-case
execution. Run external MM-WHS and OrCaScore batches through
`src/external_ccta_batch_pipeline.py`; use `src/external_ccta_pipeline.ipynb`
for an inspectable external CCTA exam. Reusable code lives under `src/utils/`:

- `project/`: datasets, results, analysis, evaluation, and runtime infrastructure;
  shared configuration and DataFrame helpers remain at its root.
- `segmentation/`: aorta, fuzzy methods, and pipeline coordination; artery,
  ostia, lower-threshold, and backend modules remain at its root.
- `visualization/`: images, intermediate pipeline stages, and result plots.
- `comparison_utils/`: result loading, summaries, pairing, and statistics;
  analysis code must not depend on Matplotlib, IPython, or visualization modules.
- `processing/`, `experiments/`, and `utils/`: image processing, experiment
  helpers, and generic NIfTI, JSON, metrics, and ROI utilities.

Experiment drivers and shell runners belong in `src/experiments/`; exploratory
notebooks are grouped under `src/eda/{images,intensity,results,comparisons,aorta,sensitivity}/`
and indexed only in `src/eda/README.md`. Keep the two execution notebooks in
the root of `src/`.
Configuration files are in `config/`, automated tests in `tests/`, and
documentation in `doc/`. ImageCAS results live under `output/segmentation/`;
external CCTA results use `CCTA_RESULTS_ROOT` or the batch CLI's `--output-root`.

When moving modules or notebooks, update imports, lazy exports, mock targets,
documentation, and Pyright paths without widening existing exclusions.
Preserve public facade symbols without duplicating implementations. For moved
modules, resolve repository paths with `utils.project.runtime.paths.find_repository_root`
rather than fixed `__file__.parents` offsets.

## Build, Test, and Development Commands

- `uv sync`: create/update the Python 3.13 environment from `uv.lock`.
- `uv run python src/segmentation_pipeline.py --help`: inspect pipeline options.
- `uv run python src/segmentation_pipeline.py --split train --resolution mid --gpu`:
  run the configured training split.
- `uv run jupyter lab`: open the execution and EDA notebooks.
- `PYTHONPATH=src uv run python -m unittest discover -s tests -p 'test_*.py'`:
  run all tests, including imports from the external CCTA pipeline.
- `uv run ruff check src tests`: run static lint checks.
- `uv run ruff format --check src tests`: verify formatting.
- `pyright`: run basic type checking using `pyrightconfig.json`, when installed.

## Coding Style & Naming Conventions

Use four-space indentation, type hints for public functions, concise docstrings,
and comments only around non-obvious processing stages. Write docstrings and
explanatory comments in Portuguese without translating contract keys or status
codes. Ruff defines formatting and lint behavior. Use `snake_case` for modules,
functions, variables, and test files; use `UPPER_CASE` for notebook/config constants.
Prefer existing helpers
over notebook-local duplicates. Keep reusable calculations in modules and
analysis-specific presentation helpers in notebook cells tagged
`presentation-helpers`. Keep pipeline behavior controlled through JSON
configuration or CLI flags rather than hard-coded experiment values.

## Testing Guidelines

Tests use Python `unittest`; name files `test_<feature>.py` and methods
`test_<behavior>`. Add focused synthetic-array tests for segmentation changes,
plus regression tests proving unchanged masks when adding diagnostics or
refactoring. Do not run the complete ImageCAS pipeline as a unit test. Validate
edited notebooks with `nbformat`, compile their code cells, and clear embedded
outputs before committing.

Do not start scientific batches, reevaluate runs, or migrate historical results
unless explicitly requested. When asked for a command, provide it without
launching the run.

## Commit & Pull Request Guidelines

Recent history follows Conventional Commit prefixes such as `feat:`, `fix:`,
and `refactor:`. Keep commits scoped, for example:
`feat: add per-branch artery diagnostics`. Pull requests should explain the
behavioral change, configuration and split used, verification commands, and
before/after metrics. Include screenshots for visualization changes. Avoid
committing large HTML or volumetric artifacts. Preserve per-exam results,
metadata, and effective configuration for reproducibility; store large visuals
at the configured output path.

## Configuration & Data

Never commit dataset paths, credentials, or machine-specific CUDA settings.
Use `IMAGECAS_BASE_PATH` when needed. Preserve effective run configurations so
results remain reproducible, and do not tune against the test split.
