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
  ostia, and thresholding modules remain at its root.
- `visualization/`: images, intermediate pipeline stages, and result plots.
- `comparison_utils/`: result loading, summaries, pairing, and statistics;
  analysis code must not depend on Matplotlib, IPython, or visualization modules.
- `processing/`, `experiments/`, and `utils/`: image processing, experiment
  helpers, and generic NIfTI, JSON, metrics, and ROI utilities.

Exploratory notebooks are grouped under `src/eda/{images,intensity,results,comparisons,aorta,sensitivity}/`
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
- `uv run ruff check src tests` checks lint rules, including import ordering (`I`).
- `uv run ruff format --check src tests`: verify formatting.
- `pyright`: run basic type checking using `pyrightconfig.json`, when installed.

## Coding Style & Naming Conventions

Use four-space indentation and type hints for public functions. Use `snake_case`
for modules, functions, variables, and test files, and `UPPER_CASE` for notebook
and configuration constants. Configure pipeline behavior through JSON or CLI flags.

Start each public function's docstring with a sentence in Portuguese describing
its result or effect. Use `Args:`, `Returns:`, and `Raises:` when callers need
details about inputs, outputs, or exceptions. Keep parameter names, result keys,
and status codes verbatim. A single sentence is enough for a simple helper.
For scientific functions, document assumptions the signature cannot convey,
such as axis order, spacing units, label values, or how empty masks are handled.

Write inline comments in Portuguese where a decision affects interpretation or
reproducibility. For example, explain why validation uses the threshold selected
on training data. Skip comments that simply narrate assignments or loops.

## EDA Notebooks

Keep the analysis readable from top to bottom. Show the configuration, exam
selection, calls that compute the results, and the resulting tables and figures.
Reuse existing functions. Choose a module by what the function does, even if
only one notebook uses it.

- Put domain calculations and validation in `utils.project.analysis`, exam
  pairing and statistics in `utils.comparison_utils`, and general-purpose plots
  in `utils.visualization`.
- A lightweight notebook may define one or two short presentation helpers in a
  cell tagged `presentation-helpers`. For more extensive presentation code,
  place the functions in `src/eda/<group>/view/<analysis>_views.py` and import
  them in the tagged cell. Tests must be able to load that cell without opening
  datasets or running the analysis.

## Testing Guidelines

Tests use Python `unittest`; name files `test_<feature>.py` and methods
`test_<behavior>`. Add focused synthetic-array tests for segmentation changes,
plus regression tests proving unchanged masks when adding diagnostics or
refactoring. Do not run the complete ImageCAS pipeline as a unit test. Validate
edited notebooks with `nbformat`, compile their code cells, and clear embedded
outputs before committing.

Before refactoring an analysis, capture its tables and the data behind its
plots. Compare both against the refactored outputs using the same saved inputs.
Check exam IDs, exclusions, denominators, missing values, labels, and category
order alongside the numeric results.

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
