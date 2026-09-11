# Repository Guidelines

## Project Structure & Module Organization

Core code lives in `src/`. Run the full workflow through
`src/segmentation_pipeline.py`; use `src/main.ipynb` for an inspectable single-case
execution. Reusable code is grouped under `src/utils/processing`,
`src/utils/segmentation`, `src/utils/project`, and `src/utils/visualization`.
Experiment drivers and shell runners belong in `src/experiments/`; exploratory
analysis belongs in `src/eda/` and should be indexed in `src/eda/README.md`.
Configuration files are in `config/`, automated tests in `tests/`, documentation
in `doc/`, and generated results under `output/segmentation/`.

## Build, Test, and Development Commands

- `uv sync`: create/update the Python 3.13 environment from `uv.lock`.
- `uv run python src/segmentation_pipeline.py --help`: inspect pipeline options.
- `uv run python src/segmentation_pipeline.py --split train --resolution mid --gpu`:
  run the configured training split.
- `uv run jupyter lab`: open the execution and EDA notebooks.
- `uv run python -m unittest discover -s tests -p 'test_*.py'`: run all tests.
- `uv run ruff check src tests`: run static lint checks.
- `uv run ruff format --check src tests`: verify formatting.
- `pyright`: run basic type checking using `pyrightconfig.json`, when installed.

## Coding Style & Naming Conventions

Use four-space indentation, type hints for public functions, concise docstrings,
and comments only around non-obvious processing stages. Ruff defines formatting
and lint behavior. Use `snake_case` for modules, functions, variables, and test
files; use `UPPER_CASE` for notebook/config constants. Prefer existing helpers
over notebook-local duplicates. Keep pipeline behavior controlled through JSON
configuration or CLI flags rather than hard-coded experiment values.

## Testing Guidelines

Tests use Python `unittest`; name files `test_<feature>.py` and methods
`test_<behavior>`. Add focused synthetic-array tests for segmentation changes,
plus regression tests proving unchanged masks when adding diagnostics or
refactoring. Do not run the complete ImageCAS pipeline as a unit test. Validate
edited notebooks with `nbformat`, compile their code cells, and clear embedded
outputs before committing.

## Commit & Pull Request Guidelines

Recent history follows Conventional Commit prefixes such as `feat:`, `fix:`,
and `refactor:`. Keep commits scoped, for example:
`feat: add per-branch artery diagnostics`. Pull requests should explain the
behavioral change, configuration and split used, verification commands, and
before/after metrics. Include screenshots for visualization changes. Avoid
committing large HTML or volumetric artifacts; retain essential CSV summaries
and store large visuals at the configured external output path.

## Configuration & Data

Never commit dataset paths, credentials, or machine-specific CUDA settings.
Use `IMAGECAS_BASE_PATH` when needed. Preserve effective run configurations so
results remain reproducible, and do not tune against the test split.
