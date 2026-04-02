# Repository Guidelines

## Project Structure & Module Organization
Core Python package code lives in `src/evoprompt/`, with subpackages for `algorithms`, `core`, `data`, `llm`, and `workflows`. Mainline entry points live directly in `scripts/`, while legacy experiment runners, demos, and utility scripts now live in `scripts/ablations/`. Dataset assets are tracked under `data/` (SVEN in `data/vul_detection/`, PrimeVul in `data/primevul/`); keep large raw dumps out of Git. External SVEN helpers live in `sven/` and should remain untouched unless syncing from upstream.

## Build, Test & Development Commands
Use `uv sync` to install dependencies defined in `pyproject.toml`. Run the mainline workflows with `uv run python scripts/run_mainline_evolution.py` and `uv run python scripts/run_mainline_evaluation.py`; run ablations or legacy experiments from `scripts/ablations/`. Execute the test suite via `uv run pytest`, filtering long cases with `uv run pytest -m "not slow"`. For coverage, prefer `uv run pytest --cov=src/evoprompt --cov=tests --cov-report=term-missing`. Format and lint with `uv run black src/evoprompt tests`, `uv run isort src/evoprompt tests`, and `uv run flake8 src/evoprompt tests`. Finish with `uv run mypy src/evoprompt`.

## Coding Style & Naming Conventions
Target Python 3.11 compatibility while keeping code Python 3.9+ compliant. Follow Black defaults (88-character lines, 4-space indentation) and isort’s Black profile. Modules and files use `snake_case`; classes follow `PascalCase`; constants remain upper snake. Public APIs require type hints, as `mypy` is configured with `disallow_untyped_defs=true`.

## Testing Guidelines
Tests live in `tests/` and mirror the `src/evoprompt/` layout. Name files `test_<feature>.py`, classes `TestFeature`, and functions `test_<case>`. `pytest` markers `unit`, `integration`, and `slow` are available; mark long-running evaluations accordingly. When introducing new algorithms or dataset handlers, include unit coverage for scoring logic and integration coverage for evolution loops where feasible. Update fixtures instead of hard-coding payloads.

## Commit & Pull Request Guidelines
Follow the emerging conventional-commit style (`feat:`, `fix:`, `docs:`, `chore:`), keeping messages concise and English-first. Scope PRs narrowly, reference issue IDs where available, and summarize dataset or prompt changes in bullet points. Confirm formatting, linting, type checks, and tests pass, attach run logs when touching evolution workflows, and scrub sensitive `.env` values. Include screenshots only when they clarify result deltas.

## Configuration & Security Notes
Store API credentials in `.env` (see `.env.example`) and never commit real keys. When sharing reproduction steps, redact sensitive URLs and anonymize customer data. Keep model or endpoint overrides confined to local configs or CI secrets.
