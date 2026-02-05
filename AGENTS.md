# AGENTS.md (.planning)

## Scope
- This directory holds planning, research, and status artifacts. Treat it as the source of truth for project intent and progress.
- Prefer editing planning docs here; avoid changing code unless a plan explicitly requires it.

## Read-First Context
- `.planning/PROJECT.md` for product intent, constraints, and key decisions.
- `.planning/STATE.md` for current phase, status, and last activity (update with real dates).
- `.planning/ROADMAP.md` and `.planning/REQUIREMENTS.md` for scope and completion tracking.

## Planning Workflow (When Asked to Add or Update Plans)
- Use phase folders under `.planning/phases/<phase>/` and follow existing `*-PLAN.md` structure.
- Keep front matter fields consistent (`phase`, `plan`, `type`, `wave`, `depends_on`, `files_modified`, `autonomous`).
- If a plan is executed, create or update the matching `*-SUMMARY.md` in the same phase folder.
- Reflect progress in `.planning/STATE.md` and `.planning/ROADMAP.md` with absolute dates (for example, `2026-02-05`).

## Research and Decisions
- Log research in `.planning/research/` or the phase `*-RESEARCH.md` file; cite sources or paths.
- Record durable decisions in the `Key Decisions` table in `.planning/PROJECT.md` with a brief rationale.
- Keep "Last updated" lines accurate when you change a document.

## Guardrails
- Do not edit `.planning/config.json` or planning templates unless explicitly requested.
- Keep instructions concise and actionable; link to existing docs instead of duplicating them.
