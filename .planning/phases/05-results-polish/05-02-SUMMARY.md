---
phase: 05-results-polish
plan: 02
subsystem: api
tags: [pandas, dataframe, comparison, scenario, baseline]

# Dependency graph
requires:
  - phase: 03-simulation-core
    provides: SimulationResult dataclass with summary DataFrame
provides:
  - compare_scenarios() for multi-scenario comparison
  - create_do_nothing_baseline() for automatic baseline generation
  - compare() convenience function for two-scenario comparison
  - Long-format DataFrame output (scenario, year, metric, value)
affects: [05-03-test-suite, future-visualization]

# Tech tracking
tech-stack:
  added: []
  patterns: [long-format-dataframes, auto-baseline-generation]

key-files:
  created:
    - src/asset_optimization/scenarios.py
  modified:
    - src/asset_optimization/__init__.py

key-decisions:
  - "long-format-output: Output comparison as scenario,year,metric,value DataFrame for seaborn compatibility"
  - "auto-baseline: create_do_nothing_baseline estimates no-intervention scenario via heuristics"

patterns-established:
  - "Long-format DataFrames for multi-scenario comparison (seaborn-friendly)"
  - "Convenience wrappers that auto-generate baseline when not provided"

# Metrics
duration: 2min
completed: 2026-02-04
---

# Phase 5 Plan 2: Scenario Comparison Summary

**Scenario comparison utilities with auto-generated do-nothing baseline and long-format DataFrame output**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-04T18:09:38Z
- **Completed:** 2026-02-04T18:11:10Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Created scenarios.py module with compare_scenarios() for multi-scenario comparison
- Implemented create_do_nothing_baseline() to auto-generate baseline without manual creation
- Added compare() convenience function that wraps both utilities
- Output format is long-format DataFrame (scenario, year, metric, value) suitable for seaborn

## Task Commits

Each task was committed atomically:

1. **Task 1: Create scenarios module with compare function** - `31d21bd` (feat)
2. **Task 2: Export scenarios module from package** - `1aa394d` (feat)

## Files Created/Modified
- `src/asset_optimization/scenarios.py` - Scenario comparison utilities module
- `src/asset_optimization/__init__.py` - Added exports for compare_scenarios, create_do_nothing_baseline, compare

## Decisions Made
- **long-format-output**: Used long-format DataFrame (scenario, year, metric, value) for seaborn compatibility
- **auto-baseline**: create_do_nothing_baseline uses heuristics (failure multipliers, zero interventions) to estimate no-intervention scenario

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Scenario comparison utilities complete and exported
- Ready for 05-03 test suite plan
- All OUTP-04 success criteria met (side-by-side comparison, auto-baseline, long-format output)

---
*Phase: 05-results-polish*
*Completed: 2026-02-04*
