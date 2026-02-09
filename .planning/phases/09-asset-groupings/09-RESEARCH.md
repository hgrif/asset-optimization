# Phase 9: Asset Groupings - Research

**Researched:** 2026-02-09
**Domain:** Asset grouping, failure propagation, group-level optimization constraints (Planner/Protocol architecture)
**Confidence:** MEDIUM

## Summary

Phase 9 adds asset grouping capabilities where related assets share risk and must be maintained together. In the current Proposal A architecture there is no timestep simulation loop; failure risk is produced by `RiskModel.predict_distribution()` and optimization happens in `Optimizer.solve()`. As a result, grouping is implemented with two focused extensions:

1. **Risk propagation at prediction time** via a `RiskModel` wrapper that adjusts `failure_prob` for assets sharing a `group_id`.
2. **Group coherence at selection time** via a new constraint builder on `ConstraintSet` and group-aware ordering/selection in the optimizer.

This keeps the planner workflow unchanged (Planner → RiskModel → EffectModel → Optimizer) and treats grouping as optional metadata. Portfolios without `group_id` continue to work unchanged.

**Primary recommendation:** Add `group_id` as an optional column in asset DataFrames. Implement propagation as a `GroupPropagationRiskModel` wrapper, and enforce group all-or-nothing selection via a `group_coherence` constraint in the optimizer.

## Standard Stack

No new dependencies required. Grouping is implemented with pandas groupby operations and numpy array math.

### Core (already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | >=2.0.0 | Groupby for group-level aggregation | Native grouping, already used throughout codebase |
| numpy | (via scipy) | Vectorized probability transforms | Fast array math already used in models |

### Supporting (already in project)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandera | >=0.18.0 | Optional schema validation for `group_id` | If strict validation is required |

**Installation:** No changes to `pyproject.toml` required.

## Architecture Patterns

### Recommended Project Structure

```
src/asset_optimization/
├── models/
│   ├── group_propagation.py   # NEW: GroupPropagationRiskModel wrapper
│   └── __init__.py            # MODIFY: export wrapper
├── constraints.py             # MODIFY: add_group_coherence()
├── optimization/optimizer.py  # MODIFY: group-aware ordering/selection
└── __init__.py                # MODIFY: export wrapper

tests/
├── test_group_propagation.py  # NEW: propagation tests
├── test_constraints.py        # ADD: constraint builder tests
├── test_optimization.py       # ADD: group coherence tests

notebooks/
└── 06_asset_groupings.py      # NEW: documentation notebook
```

### Pattern 1: Group ID as Optional Column

**What:** `group_id` is an optional nullable column in the asset DataFrame. Assets with the same non-null `group_id` are in the same group. Assets with null `group_id` are treated as singletons.

**When to use:** Always. This is the standard pattern for optional grouping metadata, matching how `asset_type` or covariates are provided.

**Example:**
```python
assets = pd.DataFrame({
    "asset_id": ["pipe_1", "pipe_2", "pipe_3", "pipe_4"],
    "install_date": pd.date_range("2010-01-01", periods=4, freq="YS"),
    "asset_type": ["PVC", "PVC", "Cast Iron", "Cast Iron"],
    "group_id": ["trench_A", "trench_A", "trench_B", None],
})
```

### Pattern 2: Failure Propagation via RiskModel Wrapper

**What:** Wrap an existing `RiskModel` so that `predict_distribution()` returns higher `failure_prob` for group members when their group has elevated risk. This is deterministic and does not require a simulation loop.

**Why here:** The current architecture uses `RiskModel.predict_distribution()` as the single source of failure probabilities. There is no simulator step that samples failures by year, so propagation must be applied at prediction time.

**Recommended formula (mean-field approximation):**
For each group *g* and horizon step:
```
P_group = 1 - Π(1 - P_i)
P_i_new = min(P_i * (1 + propagation_factor * P_group), 1.0)
```
This increases risk as the group becomes more failure-prone, while keeping ungrouped assets unchanged.

**Implementation sketch:**
```python
class GroupPropagationRiskModel:
    def __init__(self, base_model, propagation_factor=0.5, group_column="group_id"):
        ...

    def predict_distribution(self, assets, horizon, scenarios=None):
        baseline = self.base_model.predict_distribution(assets, horizon, scenarios)
        if group_column not in assets.columns:
            return baseline

        mapping = assets[["asset_id", group_column]].copy()
        merged = baseline.merge(mapping, on="asset_id", how="left")

        # Only apply propagation to groups with size >= 2 and non-null group_id
        group_sizes = mapping[group_column].value_counts(dropna=True)
        eligible = merged[group_column].isin(group_sizes[group_sizes >= 2].index)

        # Compute group failure probability per scenario/horizon
        grouped = merged[eligible].groupby(["scenario_id", "horizon_step", group_column])
        p_group = 1.0 - grouped["failure_prob"].apply(lambda p: (1.0 - p).prod())

        # Scale per-asset probabilities and clip
        merged = merged.join(p_group.rename("p_group"), on=["scenario_id", "horizon_step", group_column])
        merged["failure_prob"] = np.where(
            merged["p_group"].notna(),
            np.minimum(merged["failure_prob"] * (1.0 + self.propagation_factor * merged["p_group"]), 1.0),
            merged["failure_prob"],
        )

        return merged.drop(columns=[group_column, "p_group"])
```

### Pattern 3: Group Coherence as a Constraint

**What:** Add a `ConstraintSet.add_group_coherence()` method and enforce all-or-nothing selection in the optimizer. Grouped assets are ranked and budgeted at the group level; ungrouped assets are treated as singletons.

**Why here:** The existing optimizer already consumes constraint sets; this is the correct abstraction for selection rules.

## Anti-Patterns to Avoid

- **Reintroducing the legacy simulation loop:** The current architecture relies on `RiskModel.predict_distribution()`; do not create a new `simulation/` module.
- **Making `group_id` required:** Grouping must remain optional to preserve existing workflows.
- **Embedding group logic in Planner:** Keep grouping localized to `RiskModel` and `Optimizer` extensions.
- **Adding a graph/topology dependency:** Grouping is a single-level cluster; NetworkX-style dependencies are out of scope.

## Common Pitfalls

1. **Propagation on singleton groups:** Only apply propagation to groups with size >= 2; otherwise ungrouped assets are accidentally boosted.
2. **Factor interpretation confusion:** Document that the formula scales `P_i` by `1 + propagation_factor * P_group` (not a direct multiplier).
3. **Mixed `group_id` types:** Treat `group_id` as an opaque identifier; cast to string where needed.
4. **Null group ids:** Always drop nulls when building group sets. Nulls should be treated as singletons.
5. **Budget surprise with group coherence:** Grouping raises minimum selection unit size; expect empty selections when budgets are tight.

## Code Anchors (Current Repo)

- `src/asset_optimization/models/base.py` — `predict_distribution()` structure and `failure_prob` schema.
- `src/asset_optimization/constraints.py` — Constraint DSL pattern for adding `group_coherence`.
- `src/asset_optimization/optimization/optimizer.py` — Candidate ranking and budget selection to extend.

## Sources

### Primary (HIGH confidence)
- Codebase: `src/asset_optimization/models/base.py` — planner-compatible failure distribution schema
- Codebase: `src/asset_optimization/optimization/optimizer.py` — greedy selection and ranking logic
- Codebase: `src/asset_optimization/constraints.py` — constraint builder pattern

### Secondary (MEDIUM confidence)
- ScienceDirect: Multi-system intervention optimization for interdependent infrastructure — grouping interventions shows 20–40% cost savings
- ASCE: Cascading Failure Propagation in Interdependent Infrastructures — cascading effects observed in water networks
- Pandas documentation: Groupby patterns for efficient aggregation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — uses pandas/numpy patterns already in repo
- Architecture: MEDIUM — RiskModel wrapper is a pragmatic fit; propagation is a mean-field approximation
- Group constraints: MEDIUM — optimizer changes are straightforward but behavior under tight budgets must be documented

**Research date:** 2026-02-09
**Valid until:** 2026-03-09 (30 days)
