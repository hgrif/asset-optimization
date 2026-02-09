# Phase 9: Asset Groupings - Research

**Researched:** 2026-02-09
**Domain:** Asset grouping, failure propagation, group-level optimization constraints
**Confidence:** MEDIUM

## Summary

Phase 9 adds asset grouping capabilities where related assets share risk through failure propagation and must be maintained together as a unit. The technical domain spans three distinct problems: (1) modeling correlated risk within asset groups through failure propagation, (2) implementing group-level constraints in the optimizer to enforce all-or-nothing intervention decisions, and (3) integrating these features with the existing service-oriented architecture (Planner, protocols, Simulator, Optimizer).

Asset grouping in infrastructure management addresses two real-world scenarios: physical asset clusters (e.g., pipes in a single trench, road segments forming a continuous corridor) where failure of one asset increases stress on neighbors, and operational bundling where maintenance economies of scale require intervening on entire groups together. Research shows that coordinated intervention grouping across infrastructures reduces setup costs and service interruption by 20-40% compared to independent asset-by-asset maintenance scheduling.

The standard approach is NOT to add a separate "GroupModel" abstraction. Instead, grouping is implemented as: (a) a `group_id` column in the portfolio DataFrame (schema pattern already established with `asset_type`, `install_date`), (b) risk modification logic in the Simulator's failure sampling step to implement propagation, and (c) constraint filtering in the Optimizer's candidate selection to enforce group coherence. This keeps the core abstractions (DeteriorationModel, Simulator, Optimizer) unchanged and treats grouping as a configuration concern rather than a new modeling paradigm.

**Primary recommendation:** Extend the portfolio schema to allow optional `group_id` column. Add failure propagation logic to `Simulator._simulate_timestep()` after initial failure sampling but before intervention application. Implement group constraint as a new `ConstraintSet` constraint type that filters candidates to remove partial-group selections. Document via notebook showing a pipe network with shared-trench groups and coordinated replacement planning.

## Standard Stack

No new dependencies required. Grouping is implemented with pandas groupby operations and numpy array modifications on existing data structures.

### Core (already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | >=2.0.0 | DataFrame groupby for group operations | Native grouping, already used throughout codebase |
| numpy | (via scipy) | Array masking for propagation logic | Vectorized operations on boolean failure masks |

### Supporting (already in project)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandera | >=0.18.0 | Optional schema validation for group_id column | If strict validation of group_id format is required |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pandas groupby | NetworkX graph structure | NetworkX adds dependency and complexity; grouping is simpler than full graph topology |
| numpy array masking | Iterative row updates | Iterative approach 10-100x slower for large portfolios |
| Dedicated frailty model library | PyMC for shared frailty | PyMC is for statistical fitting; this phase provides user-specified propagation factors, not fitted frailties |

**Installation:** No changes to `pyproject.toml` required.

## Architecture Patterns

### Recommended Project Structure

```
src/asset_optimization/
├── simulation/
│   ├── simulator.py          # MODIFY: add _propagate_failures() method
│   ├── config.py             # MODIFY: add propagation_factor, enable_propagation fields
├── constraints.py            # MODIFY: add GroupConstraint class
├── schema.py                 # MODIFY: allow optional group_id column (strict=False already allows this)
└── types.py                  # No changes needed

tests/
├── test_simulation.py        # ADD: propagation tests
├── test_optimization.py      # ADD: group constraint tests

notebooks/
└── 06_asset_groupings.py     # NEW: demonstration notebook
```

### Pattern 1: Group ID as Optional Column

**What:** `group_id` is an optional nullable column in the portfolio DataFrame. Assets with the same non-null `group_id` are in the same group. Assets with null `group_id` are treated as singleton groups.

**When to use:** Always. This is the standard pattern for optional grouping metadata, matching how `asset_type`, `material`, covariates work.

**Example:**
```python
# Portfolio with groups
portfolio = pd.DataFrame({
    'asset_id': ['pipe_1', 'pipe_2', 'pipe_3', 'pipe_4'],
    'install_date': pd.date_range('2010-01-01', periods=4, freq='1YS'),
    'asset_type': ['PVC', 'PVC', 'Cast Iron', 'Cast Iron'],
    'group_id': ['trench_A', 'trench_A', 'trench_B', None],  # pipe_1 and pipe_2 grouped
})
```

### Pattern 2: Failure Propagation via Post-Sampling Adjustment

**What:** After the standard conditional probability sampling determines initial failures, a second pass increases failure probability for group members of failed assets and re-samples those assets.

**When to use:** When `config.enable_propagation=True`. This preserves the existing failure sampling logic and adds propagation as an optional overlay.

**Why not modify hazard rates directly:** Modifying the DeteriorationModel's hazard calculation would couple grouping into the model layer. Propagation is a simulation-time phenomenon (one asset fails THIS year, neighbors' risk increases NEXT year), not a model parameter.

```python
# In Simulator._simulate_timestep(), after initial failure sampling:

def _simulate_timestep(self, state: pd.DataFrame, year: int):
    # ... existing: age increment, calculate probabilities, sample failures

    # Initial failure sampling (existing code)
    probs = self._calculate_conditional_probability(state)
    random_draws = self.rng.random(len(state))
    failures_mask = pd.Series(random_draws < probs, index=state.index)

    # NEW: Propagate failures within groups if enabled
    if self.config.enable_propagation and 'group_id' in state.columns:
        failures_mask = self._propagate_failures(state, failures_mask, probs)

    # ... existing: apply interventions, track costs
```

### Pattern 3: Propagation Factor as Hazard Multiplier

**What:** When an asset in a group fails, other non-failed members of the same group get their conditional failure probability multiplied by `(1 + propagation_factor)`, capped at 1.0.

**When to use:** Default propagation logic. Simple, interpretable, matches the proportional hazards concept users already understand from Phase 7.

**Math:**
```
If asset i fails and asset j is in same group:
  P_new(j fails) = min(P_original(j fails) * (1 + propagation_factor), 1.0)

If propagation_factor = 0.5:
  Original 10% failure prob -> 15% after propagation
  Original 80% failure prob -> 100% (capped) after propagation
```

**Implementation:**
```python
def _propagate_failures(
    self,
    state: pd.DataFrame,
    initial_failures: pd.Series,
    baseline_probs: np.ndarray
) -> pd.Series:
    """Propagate failures within groups.

    For each group with at least one failure, increase failure probability
    for non-failed members and re-sample.
    """
    if 'group_id' not in state.columns:
        return initial_failures

    # Work on a copy
    final_failures = initial_failures.copy()
    propagated_probs = baseline_probs.copy()

    # Find groups with failures
    failed_groups = state.loc[initial_failures, 'group_id'].dropna().unique()

    for group_id in failed_groups:
        group_mask = (state['group_id'] == group_id)
        group_not_failed = group_mask & ~initial_failures

        if group_not_failed.sum() == 0:
            continue  # All assets in group already failed

        # Increase probability for non-failed group members
        propagated_probs[group_not_failed] = np.minimum(
            baseline_probs[group_not_failed] * (1.0 + self.config.propagation_factor),
            1.0
        )

        # Re-sample for these assets
        random_draws = self.rng.random(group_not_failed.sum())
        new_failures = random_draws < propagated_probs[group_not_failed]
        final_failures[group_not_failed] = new_failures

    return final_failures
```

### Pattern 4: Group Constraint as Candidate Filter

**What:** A new `GroupConstraint` that operates on candidates after benefit-cost ranking but before budget selection. If any asset in a group is selected, all assets in that group must be selected (or none).

**When to use:** When user adds a `GroupConstraint` to their `ConstraintSet`. This is opt-in behavior.

**Why not enforce in optimizer directly:** The constraint system is the correct abstraction for selection rules. Embedding group logic in the greedy algorithm couples it to one optimizer strategy.

```python
# In constraints.py

@dataclass(frozen=True)
class Constraint:
    kind: str
    params: dict

class ConstraintSet:
    # ... existing methods ...

    def group_coherence(self, group_column: str = 'group_id') -> 'ConstraintSet':
        """Require all assets in a group to be selected together.

        Parameters
        ----------
        group_column : str
            DataFrame column identifying groups.

        Returns
        -------
        ConstraintSet
            New constraint set with group coherence added.
        """
        constraint = Constraint(kind='group_coherence', params={'group_column': group_column})
        return ConstraintSet(self.constraints + [constraint])


# In optimizer.py

def solve(self, objective, constraints, candidates, risk_measure='expected_value'):
    # ... existing: extract budget, rank candidates

    # Apply group coherence constraint if present
    ranked = self._prepare_ranked_candidates(candidates)
    group_constraints = constraints.find('group_coherence')
    if group_constraints:
        ranked = self._enforce_group_coherence(ranked, group_constraints[-1])

    # ... existing: select with budget


def _enforce_group_coherence(self, candidates: pd.DataFrame, constraint: Constraint) -> pd.DataFrame:
    """Filter candidates to ensure group-level selection.

    Strategy: For each group, either include ALL members (if affordable) or NONE.
    Re-rank by group-level benefit/cost ratio.
    """
    group_column = constraint.params.get('group_column', 'group_id')

    if group_column not in candidates.columns:
        return candidates  # No grouping information, return unchanged

    # Treat null group_id as singleton groups
    working = candidates.copy()
    working['_group_id_filled'] = working[group_column].fillna(
        working['asset_id'].astype(str) + '_singleton'
    )

    # Aggregate to group level
    group_agg = working.groupby('_group_id_filled').agg({
        'direct_cost': 'sum',
        'expected_benefit': 'sum',
        'benefit_cost_ratio': 'first',  # Placeholder, recalculated below
    }).reset_index()

    # Recalculate benefit/cost ratio at group level
    group_agg['benefit_cost_ratio'] = np.where(
        group_agg['direct_cost'] > 0,
        group_agg['expected_benefit'] / group_agg['direct_cost'],
        np.inf
    )

    # Rank groups
    group_agg = group_agg.sort_values(
        by=['benefit_cost_ratio', 'expected_benefit'],
        ascending=[False, False]
    ).reset_index(drop=True)

    # Expand back to asset level, preserving group ranking
    ranked_assets = []
    for _, group_row in group_agg.iterrows():
        group_id = group_row['_group_id_filled']
        group_assets = working[working['_group_id_filled'] == group_id].copy()
        group_assets['_group_rank'] = len(ranked_assets)
        ranked_assets.append(group_assets)

    result = pd.concat(ranked_assets, ignore_index=True) if ranked_assets else working.iloc[0:0].copy()
    result = result.drop(columns=['_group_id_filled', '_group_rank'])
    return result
```

### Anti-Patterns to Avoid

- **Creating a GroupModel class:** Grouping is not a deterioration model. It affects failure propagation (simulator) and selection constraints (optimizer), but not hazard calculation.
- **Modifying DeteriorationModel interface:** The model calculates per-asset conditional probabilities. Propagation is a between-asset effect that happens at simulation time, not model time.
- **Storing group state in the model:** Models must remain stateless (Phase 7 research emphasized this). Group membership is portfolio metadata, not model parameters.
- **Raising errors when group_id is missing:** The column should be optional. Portfolios without `group_id` should work unchanged (backward compatibility).
- **Implementing propagation as a covariate:** Covariates are asset properties (diameter, traffic load). Group membership with dynamic propagation is a time-varying simulation effect, not a static covariate.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Graph-based group topology | Custom graph traversal for propagation | Pandas groupby with shared group_id | Propagation is within-group only; no need for graph algorithms |
| Statistical frailty fitting | Custom MLE for shared frailty variance | User-specified propagation_factor | Phase 9 is user-configured, not data-fitted; frailty fitting is future scope |
| Constraint solver for group coherence | MILP formulation with binary group variables | Filter-and-rank heuristic | Greedy optimizer is heuristic already; MILP would be full rewrite (future phase) |
| Network flow simulation | Hydraulic solver for cascading consequences | Simple probability multiplier | Phase 9 is risk propagation, not flow simulation; flow is out of scope (see PROJECT.md) |

**Key insight:** Grouping adds one level of aggregation (assets → groups) to existing operations. Pandas groupby is the correct tool for one-level aggregation. Full graph algorithms (NetworkX) and MILP solvers (OR-Tools) are premature complexity for this phase.

## Common Pitfalls

### Pitfall 1: Propagation Causes Infinite Loop
**What goes wrong:** If propagation can trigger additional failures, which then propagate to other assets, which trigger more failures, the propagation logic never terminates.
**Why it happens:** Recursive propagation within a single timestep without termination condition.
**How to avoid:** Propagate once per timestep only. The recommended pattern propagates from the initial failure set and does NOT recursively propagate from newly-failed assets within the same year. Propagation effects compound across multiple years naturally (failed assets stay failed unless intervened upon).
**Warning signs:** Simulator hangs or takes exponentially longer with propagation enabled.

### Pitfall 2: Group Constraint Makes All Candidates Unaffordable
**What goes wrong:** When group constraint is applied, even the top-ranked group's total cost exceeds the budget, so zero assets are selected.
**Why it happens:** Grouping increases the minimum selection unit size. A $50k budget with $20k assets can select 2 assets individually but 0 groups if the smallest group costs $60k.
**How to avoid:** Document this tradeoff clearly. Provide a warning when `selected_count=0` and group constraint is active. Consider a "partial group penalty" mode (future enhancement) where groups can be partially selected with a cost multiplier.
**Warning signs:** Optimization with group constraint returns empty selection despite having budget and high-benefit candidates.

### Pitfall 3: Propagation Factor Interpretation Confusion
**What goes wrong:** User sets `propagation_factor=2.0` expecting risk to double, but the code implements `P_new = P_old * (1 + factor)`, so factor=2.0 means 3x increase, not 2x.
**Why it happens:** Two common interpretations: (a) factor is additive multiplier `P * (1 + factor)`, (b) factor is direct multiplier `P * factor`.
**How to avoid:** Choose one interpretation and document clearly in `SimulationConfig` docstring and notebook. Recommended: Use additive interpretation `(1 + factor)` because it's consistent with proportional hazards `exp(beta*x)` where beta=0 means no effect. Validate factor >= 0 in config.
**Warning signs:** User reports "propagation not working" when they set factor=1.0 expecting 1x (no change) but it actually means 2x.

### Pitfall 4: Group ID Data Type Inconsistency
**What goes wrong:** Portfolio has `group_id` as string 'A', 'B', 'C', but internal code compares to integer 1, 2, 3, causing no groups to match.
**Why it happens:** Pandas allows mixed types in object columns. Groupby works, but explicit comparisons (`group_id == 'trench_A'`) fail if types mismatch.
**How to avoid:** Use `.astype(str)` when filtering by `group_id` in propagation logic. Treat `group_id` as opaque string identifier, not numeric ID. Document that `group_id` should be string or numeric, but code will coerce to string internally.
**Warning signs:** Propagation or group constraint appears to do nothing despite `group_id` column present.

### Pitfall 5: Forgetting to Handle Null Group IDs
**What goes wrong:** Code crashes with `KeyError` or `ValueError` when portfolio has null `group_id` values (ungrouped assets).
**Why it happens:** Pandas groupby includes NaN as a group key by default (dropna=False), but some operations like `state['group_id'] == group_id` propagate NaN in unexpected ways.
**How to avoid:** Use `.dropna()` when collecting failed group IDs (`state.loc[failures, 'group_id'].dropna().unique()`). When filling group IDs for singleton groups, use `fillna('asset_id_singleton')` pattern. Always test with a portfolio containing null group_ids.
**Warning signs:** Code works on fully-grouped portfolios but crashes on mixed grouped/ungrouped portfolios.

### Pitfall 6: Simulator State Mutation Across Timesteps
**What goes wrong:** Propagation modifies the portfolio DataFrame's `group_id` column or adds persistent state that affects subsequent years incorrectly.
**Why it happens:** Forgetting to copy state before modification, causing mutations to propagate across timestep boundaries.
**How to avoid:** Follow the existing pattern in `_simulate_timestep`: `state = state.copy()` at the start of the function. Propagation works on `failures_mask` (a series), not on state columns. Avoid adding temporary columns to state; use local variables instead.
**Warning signs:** Propagation effects accumulate across years in unexpected ways; results change when running multiple timesteps vs. single timestep.

## Code Examples

### Verified: Current Simulator Failure Sampling (from simulation/simulator.py, lines 332-337)
```python
# Step 2: Calculate conditional failure probability
probs = self._calculate_conditional_probability(state)

# Step 3: Sample failures using RNG
random_draws = self.rng.random(len(state))
failures_mask = pd.Series(random_draws < probs, index=state.index)
```

### Verified: Current Optimizer Greedy Selection (from optimization/optimizer.py, lines 177-193)
```python
def _select_with_budget(ranked_candidates: pd.DataFrame, budget_limit: float):
    selected_rows = []
    remaining_budget = budget_limit

    for _, row in ranked_candidates.iterrows():
        cost = float(row['direct_cost'])
        if cost <= remaining_budget:
            selected_rows.append(row)
            remaining_budget -= cost

    return pd.DataFrame(selected_rows), remaining_budget
```

### Recommended: SimulationConfig Extension
```python
# In simulation/config.py

@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for multi-timestep simulation runs.

    Parameters
    ----------
    n_years : int
        Number of years to simulate.
    random_seed : int, optional
        Seed for reproducibility.
    failure_response : str
        How to handle failures: 'replace', 'repair', 'record_only'.
    enable_propagation : bool, default=False
        Whether to enable failure propagation within groups.
    propagation_factor : float, default=0.5
        Multiplier for risk increase: P_new = P_old * (1 + propagation_factor).
        Only used if enable_propagation=True. Must be >= 0.
    """
    n_years: int
    random_seed: int | None = None
    failure_response: str = "replace"
    enable_propagation: bool = False
    propagation_factor: float = 0.5

    def __post_init__(self):
        # ... existing validations ...
        if self.propagation_factor < 0:
            raise ValueError("propagation_factor must be non-negative")
```

### Recommended: Propagation Implementation Pattern
```python
# In simulation/simulator.py

def _propagate_failures(
    self,
    state: pd.DataFrame,
    initial_failures: pd.Series,
    baseline_probs: np.ndarray
) -> pd.Series:
    """Propagate failures within groups.

    For each group that has at least one initial failure, increase the
    failure probability for other non-failed members and re-sample them.

    Parameters
    ----------
    state : pd.DataFrame
        Current asset state with 'group_id' column.
    initial_failures : pd.Series
        Boolean mask of initially failed assets.
    baseline_probs : np.ndarray
        Baseline conditional failure probabilities.

    Returns
    -------
    pd.Series
        Updated failure mask after propagation.
    """
    # Quick exit if no grouping information
    if 'group_id' not in state.columns:
        return initial_failures

    final_failures = initial_failures.copy()
    propagated_probs = baseline_probs.copy()

    # Identify groups with at least one failure (excluding null group_id)
    failed_assets = state.loc[initial_failures]
    failed_groups = failed_assets['group_id'].dropna().unique()

    if len(failed_groups) == 0:
        return initial_failures  # No groups affected

    # For each affected group, increase risk and re-sample
    for group_id in failed_groups:
        group_mask = (state['group_id'] == group_id)
        group_not_failed = group_mask & ~initial_failures

        if group_not_failed.sum() == 0:
            continue  # All group members already failed

        # Increase probability: P_new = min(P_old * (1 + factor), 1.0)
        original_probs = baseline_probs[group_not_failed]
        propagated_probs[group_not_failed] = np.minimum(
            original_probs * (1.0 + self.config.propagation_factor),
            1.0
        )

        # Re-sample these assets with updated probabilities
        random_draws = self.rng.random(group_not_failed.sum())
        new_failures = random_draws < propagated_probs[group_not_failed]

        # Update final failure mask
        indices = state.index[group_not_failed]
        final_failures[indices] = new_failures

    return final_failures
```

### Recommended: Group Constraint Test Pattern
```python
# In tests/test_optimization.py

def test_group_constraint_selects_complete_groups():
    """Group constraint enforces all-or-nothing selection per group."""
    candidates = pd.DataFrame({
        'asset_id': ['a1', 'a2', 'b1', 'b2', 'c1'],
        'group_id': ['A', 'A', 'B', 'B', None],
        'direct_cost': [10, 10, 15, 15, 20],
        'expected_benefit': [100, 80, 120, 110, 200],
    })

    optimizer = Optimizer()
    constraints = ConstraintSet().budget_limit(annual_capex=50).group_coherence()
    objective = ObjectiveBuilder().minimize_cost().build()

    result = optimizer.solve(objective, constraints, candidates)
    selected = result.selected_actions

    # Budget allows: Group A ($20) + Group B ($30) = $50
    # Should NOT select: Group A + c1, because c1 (singleton) would be affordable
    # but group B has better benefit/cost ratio than c1
    assert len(selected) == 4  # All of A and B
    assert set(selected['asset_id']) == {'a1', 'a2', 'b1', 'b2'}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Independent asset optimization | Coordinated group-level intervention planning | 2020s in infrastructure management | 20-40% cost savings via setup cost reduction |
| Static failure risk | Dynamic risk with failure propagation | Ongoing research (2020-2025) | More realistic modeling of cascading failures |
| Manual group identification | Data-driven clustering | Not yet implemented (future ML phase) | This phase requires user-specified groups |
| MILP with binary group variables | Heuristic filtering | N/A - greedy optimizer doesn't use MILP | Faster but approximate optimization |

**Deprecated/outdated:**
- Treating all assets as independent: Real infrastructure has spatial and operational dependencies that affect both risk and maintenance planning.
- Fixed hazard rates: Failure of one asset changes the stress distribution on nearby assets, requiring dynamic risk updates.

## Open Questions

1. **Should propagation compound within a single timestep?**
   - What we know: If asset A fails and propagates to B, and then B fails due to propagation, should B's failure then propagate to C in the same year?
   - What's unclear: Whether recursive propagation is physically realistic (probably not - stress changes take time) or introduces infinite loop risk (definitely yes).
   - Recommendation: Do NOT implement recursive propagation. Propagate once from the initial failure set. Multi-year effects naturally compound as failed assets remain failed.

2. **What's the default propagation_factor value?**
   - What we know: Values in literature vary widely (0.2 to 3.0) depending on asset type and failure mechanism. Water pipe research shows 0.3-0.8 for shared-trench stress propagation.
   - What's unclear: Whether a single default works across domains (pipes vs. roads vs. other assets).
   - Recommendation: Use 0.5 as default (50% risk increase). Document that this is a placeholder and users should calibrate based on their asset type. Add validation that factor >= 0. Consider domain-specific defaults in future (PipeDomain could override to 0.6, RoadDomain to 0.3).

3. **Should group constraint support partial-group penalties?**
   - What we know: All-or-nothing group selection is simple but can make optimization infeasible when budgets are tight. Some users may want "you CAN split the group but it costs 20% extra per asset."
   - What's unclear: Whether this is a common enough need to justify implementation complexity in v2.
   - Recommendation: Implement strict all-or-nothing for Phase 9. Document the partial-group penalty as a future enhancement (GRUP-F01 in REQUIREMENTS.md). Users can work around by defining smaller groups or increasing budget.

4. **How should propagation interact with ProportionalHazards covariates?**
   - What we know: ProportionalHazards (Phase 7) scales baseline hazard by `exp(beta'x)` using static covariates. Propagation scales conditional probability by `(1 + factor)` dynamically when group members fail.
   - What's unclear: Do these effects compose multiplicatively? Should propagation be implemented as a time-varying covariate instead?
   - Recommendation: Implement as separate effects that compose multiplicatively: `P_final = P_baseline * exp(beta'x) * (1 + propagation_factor if group member failed)`. Propagation is NOT a covariate because (a) it's time-varying within simulation, (b) covariates are asset properties, not simulation events. Keep them separate.

## Sources

### Primary (HIGH confidence)
- Codebase: `src/asset_optimization/simulation/simulator.py` lines 332-337 — Current failure sampling pattern to extend
- Codebase: `src/asset_optimization/optimization/optimizer.py` lines 177-193 — Current greedy selection pattern to extend
- Codebase: `src/asset_optimization/constraints.py` — ConstraintSet pattern for adding new constraint type
- Codebase: `src/asset_optimization/simulation/config.py` — SimulationConfig pattern for adding propagation parameters

### Secondary (MEDIUM confidence)
- [ScienceDirect: Multi-system intervention optimization for interdependent infrastructure](https://www.sciencedirect.com/science/article/pii/S0926580521001497) — 3C concept for grouping interventions across infrastructures, demonstrates 20-40% cost savings
- [ScienceDirect: Scalable optimization approach to intervention planning](https://www.sciencedirect.com/science/article/pii/S0951832024003533) — Two-step optimization with intervention grouping
- [ASCE: Cascading Failure Propagation in Interdependent Infrastructures](https://ascelibrary.org/doi/10.1061/AOMJAH.AOENG-0045) — 11.4% of initial failures produce cascading effects in water distribution networks
- [Springer: Water pipe failure prediction factors](https://link.springer.com/article/10.1007/s13201-025-02738-1) — Near-future prediction models for pipes considering spatial dependencies
- [Pandas Documentation: groupby optimization](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html) — Official pandas groupby patterns for efficient group operations
- [Tutorial on frailty models](https://journals.sagepub.com/doi/full/10.1177/0962280220921889) — Shared frailty for modeling correlated survival times in clusters (theoretical background)

### Tertiary (LOW confidence - requires verification)
- [PyMC: Frailty and Survival Regression Models](https://www.pymc.io/projects/examples/en/latest/survival_analysis/frailty_models.html) — Python implementation of frailty models (for reference only; not used in this phase)
- [ScienceDirect: Maintenance grouping for multi-component systems](https://www.sciencedirect.com/science/article/abs/pii/S0951832015001404) — Group maintenance under availability constraints (different problem domain but similar constraint structure)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — No new dependencies; pandas groupby and numpy masking are proven patterns in this codebase
- Architecture: MEDIUM — Integration points (Simulator, Optimizer, ConstraintSet) are clear from codebase reading, but implementation complexity of group coherence in greedy optimizer is nontrivial
- Failure propagation: MEDIUM — Research confirms cascading failures are real (11.4% of failures cascade), but optimal propagation_factor values are domain-specific and require user calibration
- Group constraints: MEDIUM — Research confirms 20-40% cost savings from coordinated interventions, but optimal grouping strategy (all-or-nothing vs. partial-group penalties) is unclear

**Research date:** 2026-02-09
**Valid until:** 2026-03-09 (30 days; codebase is stable, implementation patterns are established)
