---
phase: 03-simulation-core
plan: 02
subsystem: simulation
tags: [dataclass, interventions, frozen-immutable, age-effect]

# Dependency graph
requires:
  - phase: 03-simulation-core
    provides: simulation package structure
provides:
  - InterventionType frozen dataclass
  - Predefined interventions (DO_NOTHING, INSPECT, REPAIR, REPLACE)
  - Pluggable age_effect functions
affects: [03-simulator, 03-optimization, policy-evaluation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - frozen-dataclass-for-immutable-config
    - callable-field-for-pluggable-behavior
    - post-init-validation

key-files:
  created:
    - src/asset_optimization/simulation/interventions.py
  modified:
    - src/asset_optimization/simulation/__init__.py

key-decisions:
  - "frozen-dataclass-immutable: Use frozen=True for InterventionType immutability"
  - "callable-age-effect: Use Callable[[float], float] for pluggable age transformations"
  - "post-init-validation: Validate cost >= 0 and non-empty name in __post_init__"

patterns-established:
  - "Intervention age_effect: lambda age -> new_age pattern for state transitions"
  - "Predefined constants: Module-level constants for common intervention types"

# Metrics
duration: 1m 29s
completed: 2026-02-02
---

# Phase 03 Plan 02: Intervention Types Summary

**Frozen InterventionType dataclass with predefined DO_NOTHING, INSPECT, REPAIR, REPLACE constants and pluggable age_effect functions**

## Performance

- **Duration:** 1m 29s
- **Started:** 2026-02-02T18:20:40Z
- **Completed:** 2026-02-02T18:22:09Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Created InterventionType frozen dataclass with name, cost, age_effect, consequence_cost, upgrade_type
- Defined four predefined intervention constants with appropriate defaults
- Added validation in __post_init__ for non-negative costs and non-empty names
- Exported all intervention types from asset_optimization.simulation package

## Task Commits

Each task was committed atomically:

1. **Task 1: Create InterventionType dataclass** - `fc0d74f` (feat)
2. **Task 2: Export interventions from simulation package** - `8ebbb2d` (feat)

## Files Created/Modified

- `src/asset_optimization/simulation/interventions.py` - InterventionType dataclass and predefined constants
- `src/asset_optimization/simulation/__init__.py` - Added intervention exports to package

## Decisions Made

- **frozen-dataclass-immutable:** Used frozen=True to make InterventionType instances immutable, preventing accidental modification during simulation
- **callable-age-effect:** Used Callable[[float], float] for age_effect field to allow pluggable age transformation logic
- **post-init-validation:** Added validation in __post_init__ to catch invalid costs/names at construction time

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- InterventionType ready for use in Simulator engine
- Predefined interventions available for basic policy testing
- Custom interventions can be created with user-specific costs

---
*Phase: 03-simulation-core*
*Completed: 2026-02-02*
