---
phase: 03-simulation-core
plan: 03
subsystem: simulation
tags: [simulator, weibull, conditional-probability, rng, numpy, scipy]

# Dependency graph
requires:
  - phase: 03-simulation-core
    provides: SimulationConfig, SimulationResult, InterventionType, predefined interventions
  - phase: 02-deterioration-models
    provides: DeteriorationModel ABC, WeibullModel implementation
  - phase: 01-foundation
    provides: Portfolio class with validated DataFrame
provides:
  - Simulator class with run() method for multi-timestep simulations
  - Conditional probability calculation using Weibull survival function
  - Reproducible failure sampling with np.random.default_rng()
  - get_intervention_options() for Phase 4 optimization
  - Full cost tracking (failure_direct, failure_consequence, intervention)
affects: [optimization-engine, policy-evaluation, simulation-testing]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - conditional-probability-weibull-sf
    - isolated-rng-for-reproducibility
    - timestep-order-age-failures-interventions

key-files:
  created:
    - src/asset_optimization/simulation/simulator.py
  modified:
    - src/asset_optimization/simulation/__init__.py
    - src/asset_optimization/__init__.py

key-decisions:
  - "conditional-probability-via-survival: Use S(t)-S(t+1)/S(t) for accurate failure sampling"
  - "isolated-rng-per-simulator: Each Simulator instance has own RNG for reproducibility"
  - "direct-params-access: Access model.params directly (not transform()) for survival function"

patterns-established:
  - "Conditional probability: P(fail in [t,t+1) | survived to t) = (S(t) - S(t+1)) / S(t)"
  - "Timestep order: Age increment -> Failure sampling -> Intervention application"
  - "Cost breakdown: failure_direct + failure_consequence + intervention"

# Metrics
duration: 2m 52s
completed: 2026-02-02
---

# Phase 03 Plan 03: Simulator Class Summary

**Core Simulator class with multi-timestep simulation loop, conditional failure probability using Weibull survival functions, and reproducible np.random.default_rng() sampling**

## Performance

- **Duration:** 2m 52s
- **Started:** 2026-02-02T18:25:42Z
- **Completed:** 2026-02-02T18:28:34Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Simulator class with run() method executing multi-year simulations
- Conditional probability calculated correctly: P(fail this year | survived to now)
- Reproducibility: same seed produces identical results across runs
- Older assets fail more frequently (Weibull shape > 1 behavior verified)
- get_intervention_options() generates intervention choices per asset for Phase 4

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement Simulator class core** - `aa40f19` (feat)
2. **Task 2: Verify reproducibility and export Simulator** - `f036018` (feat)

## Files Created/Modified

- `src/asset_optimization/simulation/simulator.py` - Simulator class with run(), _simulate_timestep(), _calculate_conditional_probability(), get_intervention_options()
- `src/asset_optimization/simulation/__init__.py` - Added Simulator export
- `src/asset_optimization/__init__.py` - Added Simulator and simulation types to top-level exports

## Decisions Made

- **conditional-probability-via-survival:** Used S(t) - S(t+1) / S(t) formula with weibull_min.sf() for accurate conditional failure probability. This differs from model.transform() which returns cumulative F(t).
- **isolated-rng-per-simulator:** Each Simulator instance creates its own np.random.default_rng() from config.random_seed, ensuring reproducibility without global state.
- **direct-params-access:** Access model.params dict directly for Weibull shape/scale parameters rather than using transform(), which provides F(t) not conditional probability.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - execution proceeded smoothly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Simulator ready for integration testing with realistic portfolios
- get_intervention_options() API ready for Phase 4 optimization integration
- All 59 existing tests pass with new code
- Verified: older assets fail more frequently (7.86x ratio in test)

---
*Phase: 03-simulation-core*
*Completed: 2026-02-02*
