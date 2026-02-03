# Phase 4: Optimization - Research

**Researched:** 2026-02-03
**Domain:** Greedy budget-constrained asset intervention optimization with pluggable optimizer interface
**Confidence:** HIGH

## Summary

Phase 4 implements a budget-constrained intervention optimizer that selects the best action (Replace, Repair, Inspect, DoNothing) for each asset in a portfolio, subject to a single annual budget. The optimizer uses a two-stage greedy algorithm: first, pick the cost-effective best intervention per asset; then, rank assets by risk-to-cost ratio and greedily fill the budget. The interface follows the scikit-learn estimator pattern (`Optimizer(strategy='greedy').fit(portfolio, budget)`) with a strategy constructor that raises `NotImplementedError` for unimplemented solvers (MILP stub).

The codebase already provides all upstream data the optimizer needs. Phase 3's `Simulator.get_intervention_options()` returns a DataFrame of every asset-intervention combination with costs. Phase 2's `WeibullModel.transform()` adds `failure_probability` (cumulative CDF) to the portfolio. The optimizer consumes both and produces a results DataFrame. No new external dependencies are required -- scipy is already installed and provides `scipy.optimize.milp` for the future MILP interface stub.

The primary technical challenge is correctness of the two-stage selection logic and strict budget enforcement, not performance. At ~1000 assets with 4 interventions each, the greedy sort-and-fill loop completes in microseconds. The design risk is in the edge cases: zero-budget portfolios, all assets below threshold, exact-budget ties, and the exclusion list interaction.

**Primary recommendation:** Implement as a single `optimization/` subpackage mirroring the `simulation/` structure. Use a non-frozen dataclass for the result (contains DataFrames), a frozen dataclass for configuration, and a class with `fit()` returning `self` for the optimizer itself. No ABCs, no subclassing -- strategy is a string parameter.

## Standard Stack

No new dependencies. Everything needed is already in the project.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | >=2.0.0 | Result DataFrames, portfolio data access | Already in project; optimizer input and output are both DataFrames |
| numpy | (via scipy/pandas) | Vectorized ratio calculations, sorting | Already in project; faster than Python loops for 1000-asset portfolios |
| scipy | >=1.10.0 | MILP stub interface reference (`scipy.optimize.milp`) | Already in project; `milp()` is the natural future backend for MILP strategy |
| dataclasses | stdlib | OptimizationResult, OptimizerConfig | Established pattern in project (SimulationConfig, SimulationResult) |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| typing | stdlib | Type hints for Optimizer class | Consistent with rest of codebase |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom greedy | PuLP / OR-Tools MILP | Overkill for v1; greedy is explicit requirement; MILP deferred |
| Strategy string param | ABC + subclasses | CONTEXT.md explicitly defers ABC pattern to later; string param is simpler |
| New dependency for optimization | scipy.optimize.milp | scipy already installed; milp() added in scipy 1.7; no extra install |

**Installation:**
```bash
# No new dependencies needed. All required packages already in pyproject.toml:
# pandas>=2.0.0, scipy>=1.10.0
```

## Architecture Patterns

### Recommended Project Structure
```
src/asset_optimization/
├── optimization/
│   ├── __init__.py          # Export Optimizer, OptimizationResult
│   ├── optimizer.py         # Optimizer class with fit() and greedy logic
│   └── result.py            # OptimizationResult dataclass
├── simulation/              # Existing (Phase 3)
├── models/                  # Existing (Phase 2)
├── portfolio.py             # Existing (Phase 1)
└── exceptions.py            # Add OptimizationError here
```

This mirrors the `simulation/` subpackage structure established in Phase 3: a package directory with `__init__.py` for exports, a main class file, and a result dataclass file. No `config.py` needed because optimizer configuration is minimal (strategy string, threshold, exclusion list) and lives as constructor parameters.

### Pattern 1: scikit-learn-style Optimizer Class

**What:** Optimizer accepts hyperparameters in `__init__`, exposes `fit()` that takes data and returns `self`, and stores learned state in attributes ending with `_`.

**When to use:** This is the locked API shape from CONTEXT.md. The scikit-learn pattern is confirmed by the official scikit-learn developer guide: hyperparameters in `__init__` (no validation there), `fit()` returns `self`, learned attributes end with trailing underscore.

**Source:** [scikit-learn developer guide](https://scikit-learn.org/stable/developers/develop.html) -- verified via WebFetch 2026-02-03.

**Example:**
```python
# Source: scikit-learn conventions + CONTEXT.md decisions
class Optimizer:
    """Budget-constrained intervention optimizer.

    Parameters
    ----------
    strategy : str
        Optimization strategy. 'greedy' (implemented) or 'milp' (future).
    min_risk_threshold : float, default=0.0
        Minimum P(failure) to consider an asset. Assets below this are skipped.
    """

    def __init__(self, strategy: str = 'greedy', min_risk_threshold: float = 0.0):
        # Store as-is, no validation here (scikit-learn convention)
        self.strategy = strategy
        self.min_risk_threshold = min_risk_threshold

    def fit(self, portfolio, budget: float, exclusions: list[str] | None = None):
        """Select interventions within budget.

        Parameters
        ----------
        portfolio : Portfolio
            Asset portfolio (must have failure_probability column added
            via WeibullModel.transform()).
        budget : float
            Annual budget (strict upper bound, never exceeded).
        exclusions : list[str], optional
            Asset IDs to skip entirely.

        Returns
        -------
        self
            Fitted optimizer. Access results via result_ attribute.
        """
        # Validation in fit(), not __init__
        if self.strategy == 'greedy':
            self.result_ = self._fit_greedy(portfolio, budget, exclusions)
        elif self.strategy == 'milp':
            raise NotImplementedError("MILP strategy not yet implemented")
        else:
            raise ValueError(f"Unknown strategy: '{self.strategy}'")
        return self  # Return self for method chaining

    @property
    def result(self) -> 'OptimizationResult':
        """Access optimization result (raises if not fitted)."""
        if not hasattr(self, 'result_'):
            raise AttributeError("Optimizer has not been fitted. Call fit() first.")
        return self.result_
```

**Key conventions applied:**
- `strategy` and `min_risk_threshold` stored directly in `__init__`, no logic there
- `fit()` returns `self` (enables `Optimizer('greedy').fit(portfolio, budget).result`)
- Learned result stored in `result_` (trailing underscore = learned attribute)
- `NotImplementedError` raised for 'milp', `ValueError` for unknown strategies -- per CONTEXT.md

### Pattern 2: Two-Stage Greedy Selection Algorithm

**What:** The optimization proceeds in two stages. Stage 1 picks the best intervention per asset. Stage 2 ranks assets and fills the budget.

**When to use:** Always. This is the locked algorithm from CONTEXT.md. The two-stage structure is necessary because CONTEXT.md specifies two different metrics for two different decisions: cost-effectiveness for comparing interventions within one asset, and risk-to-cost ratio for ranking assets against each other.

**Example:**
```python
def _fit_greedy(self, portfolio, budget, exclusions):
    """Two-stage greedy optimization.

    Stage 1: For each asset, evaluate all four intervention types and pick
    the one with the best cost-effectiveness: (risk_before - risk_after) / cost.
    DoNothing has cost=0, so it needs special handling (cost-effectiveness is
    undefined; it is the default/fallback).

    Stage 2: Rank the non-DoNothing candidates by risk-to-cost ratio
    (P(failure) / cost), apply min_risk_threshold filter, tie-break on
    oldest asset first, then greedily fill budget.
    """
    df = portfolio.data.copy()

    # --- Stage 1: Best intervention per asset ---
    # Need failure_probability column from WeibullModel.transform()
    # and age column (compute from install_date if missing)
    # For each asset, compute risk_after for each intervention type
    # based on the age effect (Replace->age=0, Repair->age-5, etc.)
    # Then: cost_effectiveness = (risk_before - risk_after) / cost
    # Pick the intervention with highest cost_effectiveness
    # Skip DoNothing in cost_effectiveness ranking (it is the zero-cost baseline)

    # --- Stage 2: Rank and fill budget ---
    # Filter: remove excluded asset_ids
    # Filter: remove assets where P(failure) < min_risk_threshold
    # Filter: remove assets where best intervention is DoNothing
    # Sort by: risk_to_cost_ratio DESC, then install_date ASC (oldest first)
    # Iterate: add intervention if budget allows, subtract cost, continue
    # Stop when: budget exhausted or no more candidates
    pass
```

**Critical implementation detail for Stage 1 -- risk_after calculation:**
The failure_probability column from `WeibullModel.transform()` is the cumulative CDF F(t) at the current age. To compute risk_after for an intervention, you need to:
1. Compute the new age after applying the intervention's `age_effect` (e.g., Replace sets age to 0)
2. Re-evaluate F(new_age) using the same Weibull parameters

This means the optimizer needs access to the WeibullModel (or at minimum the Weibull parameters) to compute risk_after. Pass the model as a parameter to `fit()`, or pre-compute risk_after for all interventions before calling the optimizer. The latter is cleaner and keeps the optimizer decoupled from the model.

**Recommendation (Claude's discretion):** Pre-compute a candidate DataFrame before the greedy loop. Each row is one asset-intervention pair with columns: asset_id, intervention_type, cost, risk_before (P(failure) at current age), risk_after (P(failure) at post-intervention age), cost_effectiveness, risk_to_cost_ratio, install_date. Stage 1 filters this to the best intervention per asset. Stage 2 sorts and fills. This is a single pandas operation, fast and readable.

### Pattern 3: OptimizationResult as a Regular Dataclass

**What:** Result is a non-frozen dataclass containing DataFrames. This matches the `SimulationResult` pattern from Phase 3.

**When to use:** Always. CONTEXT.md says results are returned as DataFrame. Wrap in a dataclass for structure, but do not freeze it (DataFrames are mutable internally; freezing the container is misleading, as shown by research on frozen dataclasses with mutable fields).

**Example:**
```python
# Source: matches SimulationResult pattern from Phase 3
from dataclasses import dataclass
import pandas as pd


@dataclass
class OptimizationResult:
    """Results from optimization run.

    Attributes
    ----------
    selections : pd.DataFrame
        Selected interventions with columns:
        - asset_id: str
        - intervention_type: str ('Replace', 'Repair', 'Inspect', 'DoNothing')
        - cost: float
        - risk_score: float (P(failure) at current age)
        - rank: int (position in greedy selection order)
    budget_summary : pd.DataFrame
        Single-row DataFrame with columns:
        - budget: float (total annual budget)
        - spent: float (total cost of selected interventions)
        - remaining: float (budget - spent)
        - utilization_pct: float (spent / budget * 100)
    strategy : str
        Strategy used ('greedy', 'milp')
    """
    selections: pd.DataFrame
    budget_summary: pd.DataFrame
    strategy: str

    @property
    def total_spent(self) -> float:
        """Total cost of selected interventions."""
        return float(self.budget_summary['spent'].iloc[0])

    @property
    def utilization_pct(self) -> float:
        """Budget utilization percentage."""
        return float(self.budget_summary['utilization_pct'].iloc[0])
```

**Note:** CONTEXT.md says "Selected assets only -- no explanation for skipped assets" and "Minimal detail: just metrics (risk score, cost, rank)". The selections DataFrame contains only assets that received a non-DoNothing intervention. DoNothing is the default for everything not selected.

### Anti-Patterns to Avoid
- **Using ABC for the optimizer:** CONTEXT.md explicitly defers ABC/subclassing to later. Use a strategy string, not a class hierarchy.
- **Silent fallback from MILP to greedy:** CONTEXT.md is explicit: raise `NotImplementedError`, never fall back silently.
- **Freezing the result dataclass:** DataFrames inside are mutable. A frozen container is misleading. Match `SimulationResult` (non-frozen).
- **Validating strategy in `__init__`:** scikit-learn convention puts validation in `fit()`. The `__init__` stores parameters as-is.
- **Using failure_rate (hazard) instead of failure_probability (CDF):** The risk metric is P(failure), which is `failure_probability` from `WeibullModel.transform()` (cumulative CDF F(t)), not `failure_rate` (hazard h(t)). These are different quantities. The ranking metric is P(failure) / cost.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| MILP solver | Custom branch-and-bound | `scipy.optimize.milp` (future) | scipy already installed; milp() wraps HiGHS, a production-grade solver. Verified: scipy.optimize.milp exists in scipy >=1.7 with full integrality support |
| Weibull CDF re-evaluation | Manual formula | `scipy.stats.weibull_min.cdf(age, c=shape, scale=scale)` | Already used in WeibullModel.transform(). Reuse the same scipy function for risk_after computation |
| Portfolio DataFrame operations | Custom filtering/sorting | pandas `.query()`, `.sort_values()`, `.groupby()` | The candidate DataFrame is a standard pandas table; use pandas idioms |
| Result formatting | Custom string formatting | pandas DataFrame (let users sort/filter) | CONTEXT.md explicit: "Results returned as DataFrame -- users can sort/filter themselves" |

**Key insight:** The optimizer's job is selection logic (two-stage greedy), not numerical computation. The heavy numerical work (Weibull CDF, failure probabilities) is already done by Phase 2/3. The optimizer is a thin selection layer on top of pre-computed data.

## Common Pitfalls

### Pitfall 1: Confusing risk_to_cost_ratio with cost_effectiveness
**What goes wrong:** Using the same ratio for both "pick best intervention per asset" and "rank assets against each other", producing incorrect selections.
**Why it happens:** Both metrics involve cost in the denominator, but they answer different questions. risk_to_cost_ratio = P(failure) / cost ranks how urgent an asset is. cost_effectiveness = (risk_before - risk_after) / cost ranks how good an intervention is at reducing risk per dollar.
**How to avoid:** Explicitly separate Stage 1 (cost_effectiveness, per-asset) from Stage 2 (risk_to_cost_ratio, cross-asset). Use different column names in the candidate DataFrame.
**Warning signs:** If Replace always wins in Stage 1 regardless of cost (because it always has the highest absolute risk reduction), check that you are dividing by cost, not just comparing risk_after.

### Pitfall 2: DoNothing cost-effectiveness is undefined (division by zero)
**What goes wrong:** Computing cost_effectiveness = (risk_before - risk_after) / cost for DoNothing, where cost = 0.
**Why it happens:** DoNothing has cost=0.0 and risk_after = risk_before, so the ratio is 0/0.
**How to avoid:** Exclude DoNothing from the cost_effectiveness ranking. DoNothing is the implicit default for any asset not selected by the optimizer. It should never appear in the selections DataFrame.
**Warning signs:** NaN or inf values in the cost_effectiveness column.

### Pitfall 3: Inspect has no risk reduction in v1
**What goes wrong:** Inspect has age_effect = lambda age: age (no change), so risk_after = risk_before, making cost_effectiveness = 0 for Inspect. It never wins in Stage 1.
**Why it happens:** In v1, Inspect does not change asset state (documented in interventions.py: "v1: no follow-up logic, age unchanged"). So risk reduction is zero.
**How to avoid:** This is correct v1 behavior. Inspect will only be selected if it is the only intervention that fits the budget for a high-risk asset (i.e., Repair and Replace are too expensive). To handle this edge case: if no intervention has positive cost_effectiveness AND risk > threshold, fall back to cheapest intervention that fits budget. CONTEXT.md says "Inspect recommended when cost trade-off makes sense (cheaper than Replace, borderline risk)" -- this means Inspect should be selected in the budget-filling stage when it is the only affordable option, not via cost_effectiveness.
**Warning signs:** Inspect never appears in output even for borderline-risk assets where Replace/Repair exceed budget.

### Pitfall 4: Budget exceeded by floating-point accumulation
**What goes wrong:** Summing intervention costs with floating-point arithmetic causes the total to slightly exceed the budget.
**Why it happens:** IEEE 754 floating point: 0.1 + 0.2 != 0.3. Costs are floats.
**How to avoid:** Compare remaining budget against next intervention cost using `>=` (not `>`), and track remaining budget by subtraction. For the strict "never exceed" constraint, use `round(remaining, 2)` before comparison if costs are in dollars (2 decimal places). Alternatively, compare `total_spent + next_cost <= budget` at each step rather than tracking remaining.
**Warning signs:** Test with costs that sum to exactly the budget (e.g., budget=55500, costs=[50000, 5000, 500]).

### Pitfall 5: Missing failure_probability column
**What goes wrong:** Optimizer receives a portfolio DataFrame that has not been enriched with failure probabilities.
**Why it happens:** `WeibullModel.transform()` must be called before optimization. It is not automatic.
**How to avoid:** Validate in `fit()` that the portfolio data contains `failure_probability` column. Raise a clear error if missing, suggesting the user call `model.transform()` first.
**Warning signs:** KeyError on 'failure_probability' during ratio computation.

### Pitfall 6: Exclusion list applied too late
**What goes wrong:** Excluded assets are included in Stage 1 (best intervention selection) but filtered out in Stage 2, wasting computation and potentially confusing debug output.
**Why it happens:** Exclusion filter applied after candidate DataFrame is built.
**How to avoid:** Apply exclusion filter at the very beginning of `_fit_greedy()`, before any ratio calculations. This is both correct and efficient.
**Warning signs:** Excluded asset IDs appear in intermediate DataFrames.

## Code Examples

Verified patterns from the existing codebase and official sources:

### Getting failure_probability from the existing WeibullModel
```python
# Source: src/asset_optimization/models/weibull.py -- transform() method
# WeibullModel.transform() adds 'failure_rate' and 'failure_probability' columns
# failure_probability is the cumulative CDF F(t) = P(failure by age t)

params = {'PVC': (2.5, 50.0), 'Cast Iron': (3.0, 40.0)}
model = WeibullModel(params)

# Portfolio data needs an 'age' column (computed from install_date)
df = portfolio.data.copy()
df['age'] = (pd.Timestamp.now() - df['install_date']).dt.days / 365.25

# Enrich with failure metrics
enriched = model.transform(df)
# enriched now has 'failure_probability' and 'failure_rate' columns
```

### Re-evaluating failure_probability after intervention (for risk_after)
```python
# Source: scipy.stats.weibull_min.cdf -- already used in weibull.py line 251
from scipy.stats import weibull_min

# For each asset, compute the new age after intervention
# Example: Replace resets age to 0, Repair subtracts 5
new_age_replace = 0.0  # REPLACE.apply_age_effect(current_age)
new_age_repair = max(0.0, current_age - 5.0)  # REPAIR.apply_age_effect(current_age)

# Re-evaluate CDF at new age using same Weibull parameters
shape, scale = 2.5, 50.0  # PVC parameters
risk_after_replace = weibull_min.cdf(new_age_replace, c=shape, scale=scale)  # ~0.0
risk_after_repair = weibull_min.cdf(new_age_repair, c=shape, scale=scale)
```

### Accessing intervention types and costs from Phase 3
```python
# Source: src/asset_optimization/simulation/interventions.py
from asset_optimization.simulation import DO_NOTHING, INSPECT, REPAIR, REPLACE

# All four intervention types, with costs and age effects:
interventions = [DO_NOTHING, INSPECT, REPAIR, REPLACE]
for iv in interventions:
    print(f"{iv.name}: cost={iv.cost}, age_effect(25)={iv.apply_age_effect(25.0)}")
# DoNothing: cost=0.0,    age_effect(25)=25.0
# Inspect:   cost=500.0,  age_effect(25)=25.0
# Repair:    cost=5000.0, age_effect(25)=20.0
# Replace:   cost=50000.0,age_effect(25)=0.0
```

### Building the candidate DataFrame (vectorized approach)
```python
# Recommended internal pattern for Stage 1 candidate construction
# Produces one row per asset per intervention type (4 rows per asset)
import pandas as pd
import numpy as np
from scipy.stats import weibull_min

def build_candidates(enriched_df, model_params, interventions):
    """Build candidate DataFrame with risk_before, risk_after, and ratios.

    Parameters
    ----------
    enriched_df : pd.DataFrame
        Portfolio with 'age', 'material', 'failure_probability', 'install_date'
    model_params : dict[str, tuple[float, float]]
        Weibull (shape, scale) per material type (from WeibullModel.params)
    interventions : list[InterventionType]
        Available interventions (e.g., [DO_NOTHING, INSPECT, REPAIR, REPLACE])

    Returns
    -------
    pd.DataFrame
        Candidate rows with columns:
        asset_id, intervention_type, cost, risk_before, risk_after,
        cost_effectiveness, install_date, material
    """
    rows = []
    for _, asset in enriched_df.iterrows():
        shape, scale = model_params[asset['material']]
        risk_before = asset['failure_probability']

        for iv in interventions:
            new_age = iv.apply_age_effect(asset['age'])
            risk_after = float(weibull_min.cdf(new_age, c=shape, scale=scale))

            # cost_effectiveness: NaN for DoNothing (cost=0)
            if iv.cost > 0:
                ce = (risk_before - risk_after) / iv.cost
            else:
                ce = np.nan  # DoNothing -- excluded from Stage 1

            rows.append({
                'asset_id': asset['asset_id'],
                'intervention_type': iv.name,
                'cost': iv.cost,
                'risk_before': risk_before,
                'risk_after': risk_after,
                'cost_effectiveness': ce,
                'install_date': asset['install_date'],
                'material': asset['material'],
            })
    return pd.DataFrame(rows)
```

### The greedy budget-fill loop (Stage 2)
```python
# Stage 2: Sort candidates and fill budget
# Input: best_per_asset DataFrame (one row per asset, best intervention from Stage 1)
# Filter out DoNothing rows and assets below threshold
candidates = best_per_asset[
    (best_per_asset['intervention_type'] != 'DoNothing') &
    (best_per_asset['risk_before'] >= min_risk_threshold)
].copy()

# Compute risk-to-cost ratio for ranking
candidates['risk_to_cost_ratio'] = candidates['risk_before'] / candidates['cost']

# Sort: risk_to_cost_ratio DESC, install_date ASC (oldest first for tie-breaking)
candidates = candidates.sort_values(
    by=['risk_to_cost_ratio', 'install_date'],
    ascending=[False, True]
).reset_index(drop=True)

# Greedy fill
selected = []
remaining_budget = budget
for rank, row in candidates.iterrows():
    if row['cost'] <= remaining_budget:
        selected.append(row)
        remaining_budget -= row['cost']

selections_df = pd.DataFrame(selected)
if not selections_df.empty:
    selections_df['rank'] = range(1, len(selections_df) + 1)
```

### scipy.optimize.milp interface (for the MILP stub documentation)
```python
# Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html
# Verified via WebFetch 2026-02-03. scipy v1.17.0 current.
# This is what the MILP strategy will use when implemented.
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

# Signature: milp(c, *, integrality=None, bounds=None, constraints=None, options=None)
# c: objective coefficients (minimize c @ x)
# integrality: array of 0 (continuous) or 1 (integer) per variable
# bounds: Bounds(lb, ub) for each variable
# constraints: LinearConstraint(A, b_l, b_u)
# Returns: OptimizeResult with .x (solution), .success, .fun (objective value)

# For asset optimization, each decision variable x_i is binary (0/1: select intervention or not)
# Objective: maximize total risk reduction (negate for minimization)
# Constraint: sum(cost_i * x_i) <= budget  (budget constraint)
```

## Budget Utilization Logic (Claude's Discretion -- Recommendation)

CONTEXT.md delegates the budget utilization strategy to Claude. Recommendation: **use remaining budget for the next-best candidate regardless of risk level, as long as the asset is above min_risk_threshold.**

Rationale:
- CONTEXT.md says "No minimum spend requirement -- can recommend $0 if nothing meets threshold." The threshold is the gate; once an asset passes it, spending on it is valid.
- Inspect (cost=500) is the cheapest non-trivial intervention. If budget remains after all Repair/Replace selections, Inspect candidates should be considered. This aligns with CONTEXT.md: "Inspect recommended when cost trade-off makes sense (cheaper than Replace, borderline risk)."
- The greedy loop already handles this naturally: if the candidates list is sorted by risk_to_cost_ratio and Inspect has a lower ratio than Replace/Repair (which it will, since it has zero risk reduction in v1), it appears later in the sorted list. But it will still be selected if budget remains and no better option fits.

**Special case for Inspect in v1:** Since Inspect has zero risk reduction, its cost_effectiveness is 0, and it will never win Stage 1 against Repair or Replace. To allow Inspect as a fallback for high-risk assets where Replace/Repair exceed remaining budget: after the main greedy loop, do a second pass over unselected high-risk assets (risk >= threshold) and offer Inspect if budget remains. This second pass is optional and can be gated by a parameter.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Global numpy random state | `np.random.default_rng()` isolated generators | NumPy 1.17 (2019) | Already adopted in Phase 3; optimizer does not use randomness |
| pandas copy warnings | Copy-on-write semantics | pandas 2.0 (2023) | Already adopted; use `.copy()` explicitly when mutating |
| scipy.optimize.linprog only | `scipy.optimize.milp` dedicated MILP solver | scipy 1.7 (2021) | milp() is the right interface for binary/integer LP; use this for future MILP stub |
| ABC for all pluggable interfaces | Strategy pattern via string param | Project decision | Phase 2 uses ABC for DeteriorationModel; Phase 4 explicitly does NOT use ABC (deferred) |

**Deprecated/outdated:**
- `np.random.seed()`: Global state; replaced by `default_rng()`. Not relevant here (optimizer is deterministic) but noted for consistency.
- `scipy.optimize.linprog` for MILP: Works via `integrality` parameter, but `milp()` is the canonical interface. Use `milp()` in the stub.

## Open Questions

1. **Should the optimizer accept a pre-enriched DataFrame or a Portfolio + Model pair?**
   - What we know: WeibullModel.transform() must be called before optimization to get failure_probability. The optimizer also needs model_params to compute risk_after.
   - What's unclear: Whether to require the user to pre-compute risk_after for all interventions, or have the optimizer do it internally.
   - Recommendation: Have `fit()` accept `portfolio` (Portfolio object) and `model` (WeibullModel). The optimizer calls `model.transform()` internally and accesses `model.params` for risk_after computation. This keeps the API simple for users but gives the optimizer everything it needs. This matches the scikit-learn pattern where `fit()` does all data preparation internally.

2. **Inspect fallback: should it be automatic or opt-in?**
   - What we know: Inspect has zero risk reduction in v1, so it never wins Stage 1. CONTEXT.md says it should be recommended for borderline-risk assets when cheaper alternatives are unavailable.
   - What's unclear: Whether this should be a separate parameter or always-on behavior.
   - Recommendation: Make it always-on. After the main greedy loop, if budget remains and there are unselected assets above min_risk_threshold, offer Inspect for those assets (sorted by risk DESC). This is simple, consistent with the CONTEXT.md intent, and costs nothing to implement.

3. **What exactly goes in the rank column?**
   - What we know: CONTEXT.md says "rank" is in the output. It represents the order in which assets were selected by the greedy algorithm.
   - Recommendation: rank=1 is the first asset selected (highest risk_to_cost_ratio that fit the budget). Sequential integers from there. This makes rank a direct reflection of the greedy selection order.

## Sources

### Primary (HIGH confidence)
- scikit-learn developer guide (https://scikit-learn.org/stable/developers/develop.html) -- verified via WebFetch: __init__ conventions, fit() return self, learned attribute naming
- scipy.optimize.milp docs (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html) -- verified via WebFetch: function signature, integrality types, usage pattern
- Existing codebase: src/asset_optimization/simulation/interventions.py -- InterventionType costs and age effects (ground truth)
- Existing codebase: src/asset_optimization/models/weibull.py -- WeibullModel.transform() output columns and scipy.stats.weibull_min usage
- Existing codebase: src/asset_optimization/simulation/simulator.py -- get_intervention_options() output format
- Python docs: dataclasses module (https://docs.python.org/3/library/dataclasses.html) -- frozen dataclass behavior

### Secondary (MEDIUM confidence)
- WebSearch "greedy knapsack algorithm budget constraint" -- confirmed that greedy by value/weight ratio is the standard heuristic for budget-constrained selection; suboptimal for 0-1 knapsack but explicitly chosen here
- WebSearch "asset prioritization risk cost ratio" -- confirmed risk/cost ratio is standard in infrastructure asset management literature
- WebSearch "Python dataclass frozen immutable result" -- confirmed that frozen dataclasses with DataFrame fields are misleading (internal mutability); use non-frozen for result containers

### Tertiary (LOW confidence)
- WebSearch "scikit-learn fit predict API design pattern" -- community descriptions of the pattern; verified independently against official docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new dependencies; all libraries verified in existing codebase
- Architecture: HIGH - patterns directly derived from existing Phase 2/3 code and verified scikit-learn conventions
- Algorithm correctness: HIGH - two-stage greedy is straightforward; pitfalls documented with concrete examples
- Pitfalls: HIGH - all pitfalls derived from actual code analysis (interventions.py costs, weibull.py columns) not speculation
- MILP stub: HIGH - scipy.optimize.milp signature verified via official docs WebFetch

**Research date:** 2026-02-03
**Valid until:** 2026-03-03 (stable domain; no fast-moving dependencies)
