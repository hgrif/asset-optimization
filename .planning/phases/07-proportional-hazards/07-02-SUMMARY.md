---
phase: 07-proportional-hazards
plan: 02
subsystem: models-simulation
tags: [proportional-hazards, interface, simulator, weibull]

# Dependency graph
requires:
  - phase: 07-proportional-hazards
    plan: 01
    provides: ProportionalHazardsModel base implementation and tests
provides:
  - DeteriorationModel conditional probability interface
  - WeibullModel interface implementation
  - ProportionalHazardsModel interface implementation
  - Simulator delegation to model interface
affects: [07-03]

# Tech tracking
tech-stack:
  added: []
  patterns: [model-driven-interface, delegation]

key-files:
  modified:
    - src/asset_optimization/models/base.py
    - src/asset_optimization/models/weibull.py
    - src/asset_optimization/models/proportional_hazards.py
    - src/asset_optimization/simulation/simulator.py
    - tests/test_deterioration.py
    - tests/test_proportional_hazards.py

key-decisions:
  - "Use model interface for conditional probability instead of simulator Weibull internals"
  - "Implement PH conditional probability by exponentiating baseline survival ratio"

patterns-established:
  - "All deterioration models must implement calculate_conditional_probability(state)"
  - "Simulator calls model.calculate_conditional_probability(state) directly"

# Metrics
duration: 0min
completed: 2026-02-06
---

# Phase 07 Plan 02: Clean Conditional Probability Interface Summary

**Implemented a model-level conditional probability interface and removed Weibull-specific conditional logic from Simulator.**

## Accomplishments
- Added abstract `calculate_conditional_probability(state)` to `DeteriorationModel`
- Implemented `calculate_conditional_probability` in `WeibullModel`
- Implemented `calculate_conditional_probability` in `ProportionalHazardsModel`
- Refactored `Simulator._calculate_conditional_probability()` to delegate to model interface
- Added interface tests for Weibull and PH conditional probability behavior

## Verification
- Ran:
  - `uv run pytest tests/test_deterioration.py tests/test_proportional_hazards.py tests/test_simulation.py -v`
- Result:
  - `106 passed in 0.59s`

## Notes
- Optimizer behavior remains as previously documented: risk-after ranking is still baseline-only in this phase.

---
*Phase: 07-proportional-hazards*
*Completed: 2026-02-06*
