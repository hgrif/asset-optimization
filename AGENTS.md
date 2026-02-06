# AGENTS.md

## Scope
- This is the root agent guide for the repository.
- Planning artifacts live in `.planning/`; follow the planning guidance below when working there.

## Build & Test
- Install dependencies (uv): `uv pip install -e ".[dev]"`
- Run tests: `uv run pytest`
- Run a single test: `uv run pytest tests/path/to/test_file.py -k test_name`
- If `uv` is unavailable: `pip install -e ".[dev]"` and `pytest`

## Notebooks
- Fast sync only (no execution): `make docs-sync`
- Execute notebook sources and sync: `make docs` (runs `docs-py` then `docs-sync`)
- Execute only notebook sources: `make docs-py`

## Quality Gates
- Use the Makefile targets for standard checks: `make lint`, `make test`, `make docs`.
- After finishing a task, run `make lint`, `make test`, and `make docs`. All must succeed before marking the task complete.

## Repo Layout
- Package code lives under `src/asset_optimization/` (src layout).
- Tests live under `tests/` and use pytest.
- Example notebooks live under `notebooks/`.
- Planning docs and status live under `.planning/`.

## Coding Conventions
- Use type hints for public APIs and dataclasses.
- Use NumPy-style docstrings (see existing modules for format).
- Prefer dataclasses for config/result objects; keep them immutable (`frozen=True`) unless mutability is required.
- Avoid mutating input DataFrames; copy before modification when transforming data.
- Use custom exceptions from `src/asset_optimization/exceptions.py` for validation and user-facing errors.

## Planning Docs (.planning)
- `.planning/PROJECT.md` for product intent, constraints, and key decisions.
- `.planning/STATE.md` for phase/status and last activity (use absolute dates, for example `2026-02-05`).
- `.planning/ROADMAP.md` and `.planning/REQUIREMENTS.md` for scope tracking.
- Phase plans live under `.planning/phases/<phase>/` and follow existing `*-PLAN.md` structure.
- If a plan is executed, create or update the matching `*-SUMMARY.md` in the same phase folder.

## Guardrails
- Do not edit `.planning/config.json` or planning templates unless explicitly requested.
- Keep instructions concise and actionable; link to existing docs instead of duplicating them.

## Development Rules

- Backwards compatibility is not a concern. This is a pre-release SDK with no external consumers. Break existing APIs freely when building new features.
