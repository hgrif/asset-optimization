---
phase: 09-asset-groupings
plan: 01
subsystem: models
tags: [group-propagation, risk-model, wrapper]
dependency_graph:
  requires: [RiskModel protocol, predict_distribution schema]
  provides: [GroupPropagationRiskModel]
  affects: [public API, risk prediction]
tech_stack:
  added: []
  patterns: [protocol wrapper, mean-field approximation, vectorized pandas operations]
key_files:
  created:
    - src/asset_optimization/models/group_propagation.py
    - tests/test_group_propagation.py
  modified:
    - src/asset_optimization/models/__init__.py
    - src/asset_optimization/__init__.py
    - tests/test_public_api.py
decisions:
  - "Mean-field propagation formula: P_i_new = min(P_i * (1 + factor * P_group), 1.0)"
  - "Singleton groups (size < min_group_size) excluded from propagation"
  - "Ungrouped assets (null group_id) unchanged"
  - "Propagation_factor=0 acts as no-op, returning baseline predictions"
metrics:
  duration_minutes: 4
  completed_date: 2026-02-13
  tasks_completed: 3
  tests_added: 14
  files_created: 2
  files_modified: 3
---

# Phase 09 Plan 01: Group Propagation Risk Model Summary

**One-liner:** RiskModel wrapper that deterministically increases failure probabilities for assets sharing a group_id using mean-field approximation.

## What Was Built

Implemented `GroupPropagationRiskModel` as a wrapper around any RiskModel implementation. The wrapper applies deterministic failure propagation to grouped assets without requiring a simulation loop, fitting the current Proposal A architecture where risk is computed once via `predict_distribution()`.

### Core Implementation

**GroupPropagationRiskModel class:**
- Wraps any RiskModel via protocol compliance check
- Configurable propagation_factor (default 0.5) and min_group_size (default 2)
- Validates parameters at initialization (non-negative finite factor, non-empty group column)
- Delegates `fit()` to base model
- `predict_distribution()` applies mean-field propagation formula
- `describe()` includes base model metadata plus propagation settings

**Propagation Formula:**
For each group g and (scenario_id, horizon_step):
```
P_group = 1 - Π(1 - P_i)  for all i in group g
P_i_new = min(P_i * (1 + propagation_factor * P_group), 1.0)
```

**Behavior:**
- No group_id column → returns baseline unchanged
- propagation_factor=0 → returns baseline unchanged
- Singleton groups (size < min_group_size) → unchanged
- Null group_id → treated as ungrouped, unchanged
- Eligible groups (size >= min_group_size) → increased failure_prob

### Public API Exports

- Added to `src/asset_optimization/models/__init__.py`
- Added to `src/asset_optimization/__init__.py` and `__all__`
- Added test assertion in `tests/test_public_api.py`

### Test Coverage

14 tests covering:
1. Initialization validation (base model protocol, factor bounds, column name, group size)
2. Default parameter values
3. Fit delegation
4. Propagation with missing group column (no-op)
5. Propagation increases grouped asset probabilities
6. Singleton groups excluded from propagation
7. Zero propagation factor (no-op)
8. Clipping to 1.0
9. Describe metadata includes propagation settings

## Deviations from Plan

None - plan executed exactly as written.

## Known Limitations

1. **Pre-existing test failures:** The test suite has a failing test in `tests/test_optimization.py::TestGroupCoherence::test_group_coherence_no_group_column` related to plan 09-02 (group coherence constraints), which is separate work. This failure existed before plan 09-01 and does not affect this plan's functionality.

2. **Pre-existing lint issues:** There are lint failures in `src/asset_optimization/optimization/optimizer.py` (undefined `Constraint` import) and notebook sync issues. These are technical debt from earlier work and unrelated to this plan.

3. **Mean-field approximation:** The propagation formula is a deterministic approximation. It does not model true cascading failures or temporal dynamics. For more sophisticated dependency modeling, a simulation-based approach would be needed.

4. **Single-level grouping:** The wrapper treats groups as flat clusters. Hierarchical relationships (groups within groups) are not supported.

## Verification Results

**Tests for this plan:**
- `tests/test_group_propagation.py`: 14/14 passed
- `tests/test_public_api.py`: 2/2 passed

**Code quality:**
- `ruff check` on plan files: All checks passed
- `ruff format --check` on plan files: All files formatted correctly

**Integration:**
- GroupPropagationRiskModel exported in public API
- Protocol compliance verified (RiskModel)
- Works with any RiskModel implementation (tested with DummyRiskModel)

## Self-Check

Verifying claimed artifacts exist:

```bash
# Check created files
[ -f "src/asset_optimization/models/group_propagation.py" ] && echo "FOUND: src/asset_optimization/models/group_propagation.py" || echo "MISSING: src/asset_optimization/models/group_propagation.py"
[ -f "tests/test_group_propagation.py" ] && echo "FOUND: tests/test_group_propagation.py" || echo "MISSING: tests/test_group_propagation.py"

# Check commits
git log --oneline --all | grep -q "2129212" && echo "FOUND: 2129212" || echo "MISSING: 2129212"
git log --oneline --all | grep -q "7832319" && echo "FOUND: 7832319" || echo "MISSING: 7832319"
```

**Result:** PASSED - all files and commits verified.

## Next Steps

1. **Plan 09-02:** Implement group coherence constraints in the optimizer (all-or-nothing selection for grouped assets)
2. **Phase 9 completion:** Add documentation notebook demonstrating group propagation with example portfolios
3. **Integration testing:** Verify GroupPropagationRiskModel works with WeibullModel and ProportionalHazardsModel in end-to-end scenarios
