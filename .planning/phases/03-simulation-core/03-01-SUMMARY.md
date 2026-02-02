---
phase: 03-simulation-core
plan: 01
subsystem: simulation
tags: [dataclass, frozen, pandas, validation]

# Dependency graph
requires:
  - phase: 02-deterioration-models
    provides: DeteriorationModel ABC, WeibullModel implementation
provides:
  - SimulationConfig immutable configuration dataclass
  - SimulationResult structured output with DataFrames
  - SimulationError exception for simulation-specific errors
affects: [03-02, 03-03, optimization-engine]

# Tech tracking
tech-stack:
  added: []
  patterns: [frozen-dataclass-with-validation, convenience-methods-on-result]

key-files:
  created:
    - src/asset_optimization/simulation/config.py
    - src/asset_optimization/simulation/result.py
  modified:
    - src/asset_optimization/simulation/__init__.py
    - src/asset_optimization/exceptions.py

key-decisions:
  - "frozen-dataclass-for-config: Use frozen=True for immutable configuration"
  - "post-init-validation: Validate parameters in __post_init__ method"
  - "convenience-methods-on-result: Add total_cost() and total_failures() for common queries"

patterns-established:
  - "Frozen dataclass with __post_init__ validation for configuration objects"
  - "Result dataclass with convenience methods for common aggregations"

# Metrics
duration: 2min
completed: 2026-02-02
---

# Phase 3 Plan 1: Simulation Config and Result Summary

**Immutable SimulationConfig with validation and SimulationResult dataclass with summary/cost/failure DataFrames**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-02T18:20:40Z
- **Completed:** 2026-02-02T18:22:55Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- SimulationConfig frozen dataclass with n_years, start_year, random_seed, track_asset_history, failure_response
- Validation in __post_init__ for n_years > 0 and failure_response in allowed values
- SimulationResult with summary, cost_breakdown, failure_log DataFrames and config reference
- Convenience methods total_cost() and total_failures() on SimulationResult
- SimulationError exception with year and details attributes

## Task Commits

Each task was committed atomically:

1. **Task 1: Create SimulationConfig dataclass** - `451d49a` (feat)
2. **Task 2: Create SimulationResult dataclass** - `15e24bc` (feat)

## Files Created/Modified
- `src/asset_optimization/simulation/config.py` - SimulationConfig frozen dataclass with validation
- `src/asset_optimization/simulation/result.py` - SimulationResult dataclass with convenience methods
- `src/asset_optimization/simulation/__init__.py` - Module exports for SimulationConfig and SimulationResult
- `src/asset_optimization/exceptions.py` - Added SimulationError exception class

## Decisions Made
- **frozen-dataclass-for-config:** Using frozen=True ensures configuration immutability, preventing accidental modification during simulation runs
- **post-init-validation:** Validation in __post_init__ provides fail-fast behavior with clear error messages
- **convenience-methods-on-result:** total_cost() and total_failures() simplify common queries on simulation results

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - execution proceeded smoothly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- SimulationConfig and SimulationResult contracts established
- Ready for Simulator implementation in 03-02
- Exception handling infrastructure in place with SimulationError

---
*Phase: 03-simulation-core*
*Completed: 2026-02-02*
