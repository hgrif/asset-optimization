# Plan Gaps and Improvements

## Gaps / Inconsistencies

1) **Phase 3 status is inconsistent across docs**
   - `/.planning/ROADMAP.md` shows Phase 3 “Not started (0/4)” while `/.planning/STATE.md` says Phase 3 is “In progress” with 03-01, 03-02, and 03-03 completed.
   - `/.planning/STATE.md` says “Plan: 3 of TBD completed” and “Progress: 60%” even though Phase 3 has 4 defined plans (03-01..03-04). If 3/4 are complete, progress should be 75% and total should be 3/4, not TBD.

2) **Phase 3 testing/verification is planned but not executed**
   - `/.planning/phases/03-simulation-core/03-04-PLAN.md` exists but there is no matching `03-04-SUMMARY.md` and no Phase 3 verification/UAT files (unlike Phase 1 UAT and Phase 2 verification).
   - This leaves SIMU/INTV requirements unverified and no documented test coverage for the simulation module.

3) **Pluggable model interface is violated by the Simulator plan**
   - `/.planning/phases/03-simulation-core/03-03-PLAN.md` requires `Simulator` to access `self.model.params` directly and to compute conditional failure probabilities using Weibull-specific parameters.
   - The `DeteriorationModel` ABC (Phase 2) does not define `params` or any survival/conditional-probability method, which breaks the pluggable-model contract (DTRN-04). This is a design gap that will block non-Weibull models.

4) **Default start year mismatch**
   - `/.planning/phases/03-simulation-core/03-CONTEXT.md` says the default should be the current year.
   - `/.planning/phases/03-simulation-core/03-01-PLAN.md` hardcodes default `start_year=2026` in `SimulationConfig`.

5) **Requirements vs schema mismatch for required fields**
   - `/.planning/REQUIREMENTS.md` (DATA-03) says required fields are age, type, condition.
   - Phase 1 decisions in `/.planning/phases/01-foundation/01-CONTEXT.md` and implementation summaries require `asset_id`, `install_date`, `asset_type`, `material`, with condition optional.
   - This is a mismatch between requirements and actual data contract.

6) **Roadmap phases 4 and 5 are still “TBD” with no plans**
   - `/.planning/ROADMAP.md` lists Phase 4 (Optimization) and Phase 5 (Results & Polish) but provides no plan files.
   - This leaves OPTM/OUTP/DEVX requirements unplanned even though they are mapped in `/.planning/REQUIREMENTS.md`.

7) **Stray references to non-existent plans**
   - `/.planning/phases/02-deterioration-models/02-01-SUMMARY.md` lists `affects: [02-04]`.
   - `/.planning/phases/02-deterioration-models/02-02-SUMMARY.md` references “02-03-markov” and readiness for a Markov model plan, which does not exist in the roadmap.

8) **PROJECT.md appears stale relative to current work**
   - `/.planning/PROJECT.md` shows “Last updated: 2025-01-29” even though Phase 1–3 progress is in 2026 and tracked elsewhere.

## Recommended Plan Improvements

- Add Phase 3 verification/UAT docs similar to Phase 1/2 (or explicitly decide to skip and document why).
- Execute 03-04 testing plan or mark it as deferred with rationale.
- Update the DeteriorationModel interface to support simulator needs (e.g., add `conditional_failure_probability()` or `survival_function()` to the ABC) and adjust the Simulator plan to call the interface instead of `model.params`.
- Align `SimulationConfig.start_year` default with the “current year” decision or update the decision to a fixed default (and document it in context).
- Reconcile DATA-03 requirement wording with the actual required columns (either update requirements to match `asset_id/install_date/asset_type/material` or plan schema changes to enforce age/type/condition).
- Create plan files for Phase 4 and Phase 5 (or explicitly record them as deferred).
- Clean up stray references to non-existent plans (02-04, 02-03-markov) to avoid confusion.
- Update `/.planning/ROADMAP.md`, `/.planning/STATE.md`, and `/.planning/PROJECT.md` to reflect current phase progress and dates.
