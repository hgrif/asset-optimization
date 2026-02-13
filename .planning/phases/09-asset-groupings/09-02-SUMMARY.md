---
phase: 09-asset-groupings
plan: 02
subsystem: optimization
tags: [constraints, grouping, all-or-nothing]
dependency_graph:
  requires: [constraints-dsl, optimizer-greedy]
  provides: [group-coherence-constraint, group-aware-selection]
  affects: [planner-workflow, budget-selection]
tech_stack:
  added: []
  patterns: [group-aggregation, singleton-fallback]
key_files:
  created: []
  modified:
    - src/asset_optimization/constraints.py
    - src/asset_optimization/optimization/optimizer.py
    - tests/test_constraints.py
    - tests/test_optimization.py
decisions:
  - Null group_id values treated as singletons (unique per asset)
  - Group ranking by group-level benefit/cost ratio
  - All-or-nothing selection enforced at budget selection time
metrics:
  duration_minutes: 5
  completed: 2026-02-13
---

# Phase 09 Plan 02: Group Coherence Constraint Summary

Implement group coherence constraint for optimizer to enforce all-or-nothing selection of grouped assets.

## One-liner

Group coherence constraint enforces all-or-nothing selection via ConstraintSet.add_group_coherence() with group-aware ranking and budget selection.

## What Was Built

### Task 1: Add group_coherence to ConstraintSet (commit 779aa12)

**Implemented:**
- Added `add_group_coherence(group_column="group_id")` method to ConstraintSet
- Validates group_column is non-empty string
- Supports fluent chaining with other constraints
- Returns constraint with kind "group_coherence" and group_column param

**Tests added (5):**
- `test_add_group_coherence_default` - default column is "group_id"
- `test_add_group_coherence_custom_column` - custom column stored correctly
- `test_add_group_coherence_chaining` - fluent API with add_budget_limit
- `test_find_group_coherence` - find() returns constraint
- `test_add_group_coherence_validates_non_empty` - rejects empty column name

**Files modified:**
- `src/asset_optimization/constraints.py` - added method
- `tests/test_constraints.py` - added 5 tests

### Task 2: Enforce group coherence in Optimizer (commit 16efdd8)

**Implemented:**

1. **`_enforce_group_coherence()` static method:**
   - Re-ranks candidates by group-level benefit/cost ratio
   - Fills null group_id with unique singleton identifiers (asset_id + "_singleton")
   - Aggregates direct_cost and expected_benefit to group level
   - Computes group-level ratio with proper zero-cost handling
   - Sorts groups by ratio desc, then benefit desc
   - Preserves original asset-level ranking within groups

2. **Updated `solve()` method:**
   - Checks for group_coherence constraint after ranking
   - Applies `_enforce_group_coherence()` if constraint present
   - Passes group_column to `_select_with_budget()`

3. **Updated `_select_with_budget()` method:**
   - Added `group_column` parameter (optional)
   - When group_column is None or missing: existing behavior (asset-level selection)
   - When group_column present: group-aware all-or-nothing selection
   - Pre-computes group costs and tracks seen groups
   - Selects/skips entire groups based on budget

**Tests added (6):**
- `test_group_coherence_selects_complete_groups` - selects all assets in group together
- `test_group_coherence_skips_unaffordable_group` - skips entire group if budget insufficient
- `test_group_coherence_null_group_ids_as_singletons` - null values treated as individual assets
- `test_group_coherence_no_group_column` - no-op when group_id column missing
- `test_group_coherence_with_budget_limit` - ranks groups by ratio, respects budget
- `test_optimizer_without_group_coherence_unchanged` - existing behavior preserved

**Files modified:**
- `src/asset_optimization/optimization/optimizer.py` - added method, updated solve() and _select_with_budget()
- `tests/test_optimization.py` - added TestGroupCoherence class with 6 tests

## Deviations from Plan

None - plan executed exactly as written.

## Decisions Made

1. **Singleton identifier format:** Null group_id values filled with `"{asset_id}_singleton"` to ensure uniqueness per asset.

2. **Group ranking logic:** Groups ranked by benefit/cost ratio descending, then expected_benefit descending (matching asset-level logic).

3. **Within-group ordering:** Assets within a group preserve their original asset-level ranking for deterministic output.

4. **Zero-cost group handling:** Applied same logic as asset-level (zero cost + positive benefit → inf ratio, zero cost + non-positive benefit → 0.0 ratio).

## Key Links Verified

- `src/asset_optimization/optimization/optimizer.py:87` - `constraints.find("group_coherence")`
- `src/asset_optimization/optimization/optimizer.py:88` - calls `_enforce_group_coherence()`
- `src/asset_optimization/optimization/optimizer.py:92` - passes `group_column` to `_select_with_budget()`
- `src/asset_optimization/optimization/optimizer.py:287` - `_select_with_budget()` signature includes `group_column`

## Self-Check: PASSED

**Created files:**
- None (all modifications)

**Modified files:**
- FOUND: src/asset_optimization/constraints.py
- FOUND: src/asset_optimization/optimization/optimizer.py
- FOUND: tests/test_constraints.py
- FOUND: tests/test_optimization.py

**Commits:**
- FOUND: 779aa12 (Task 1)
- FOUND: 16efdd8 (Task 2)

**Verification results:**
- All 25 constraint and optimization tests pass
- All 145 project tests pass
- Lint passes (make lint)
- Notebooks regenerate successfully (make docs)

## Performance

**Duration:** 5 minutes (287 seconds)
**Tasks completed:** 2/2
**Tests added:** 11 (5 constraint + 6 optimizer)
**Test coverage:** Complete for group coherence (grouped, singletons, no-op, budget interaction)

## What's Next

- Plan 09-03: Create notebook documenting group coherence usage
- Then: Implement group propagation risk model (plan 09-01 already complete)
