# Phase 3: Simulation Core - Research

**Researched:** 2026-02-01
**Domain:** Multi-timestep simulation with intervention modeling
**Confidence:** HIGH

## Summary

Phase 3 implements a deterministic multi-timestep simulation engine that tracks asset states over time, applies interventions, and accumulates costs and failures. The research reveals that while discrete event simulation frameworks (SimPy, Salabim) exist, this phase requires simpler time-stepped simulation with pandas DataFrame state management rather than full DES infrastructure.

The core technical challenges are: (1) efficiently tracking state changes across timesteps without memory explosion, (2) calculating conditional failure probabilities for each timestep, (3) modeling intervention effects on asset age and condition, (4) maintaining reproducibility with proper random seed management, and (5) providing flexible result structures that balance completeness with memory efficiency.

The recommended approach uses dataclasses for configuration and results, numpy's default_rng() for reproducible randomness, pandas DataFrames for state tracking, and vectorized operations for performance. Keep asset-level traces optional to manage memory for large portfolios.

**Primary recommendation:** Use dataclass-based configuration with pandas DataFrames for state management, np.random.default_rng() for reproducibility, vectorized conditional probability calculations, and optional asset traces with summary statistics always returned.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | >=2.0.0 | State tracking via DataFrames | Already in project, efficient for tabular simulation state, copy-on-write in 2.0+ |
| numpy | >=1.24.0 | Random number generation, vectorized calculations | Foundation of scientific Python, np.random.default_rng() for reproducibility |
| dataclasses | stdlib | Configuration and result objects | Python standard library, type hints, clean API, no dependencies |
| scipy | >=1.10.0 | Conditional probability from survival functions | Already in project for Weibull, provides survival function for conditional probability |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| typing | stdlib | Type hints for intervention enums and protocols | Document intervention types and callback signatures |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Time-stepped loop | SimPy/Salabim DES | DES frameworks add complexity (processes, resources, events); time-stepped sufficient for annual timesteps |
| Dataclass results | Dict-based returns | Dataclasses provide type hints, IDE support, immutability options; dicts are unstructured |
| pandas state | Custom state classes | pandas optimized for tabular data, familiar to users, handles indexing/grouping efficiently |
| np.random.default_rng() | np.random.seed() | default_rng() creates isolated generators (recommended since NumPy 1.17, 2019) |

**Installation:**
```bash
# No new dependencies - all libraries already in project
# Dataclasses and typing are stdlib
```

## Architecture Patterns

### Recommended Project Structure
```
src/asset_optimization/
├── simulation/
│   ├── __init__.py              # Export Simulator, SimulationConfig, SimulationResult
│   ├── simulator.py             # Main Simulator class
│   ├── config.py                # SimulationConfig dataclass
│   ├── result.py                # SimulationResult dataclass
│   ├── interventions.py         # Intervention classes (DoNothing, Repair, Replace, Inspect)
│   └── state.py                 # AssetState management (optional helper)
├── models/                      # Existing deterioration models
├── portfolio.py                 # Existing Portfolio class
└── exceptions.py                # Existing + simulation-specific exceptions
```

### Pattern 1: Dataclass Configuration for Simulation Parameters

**What:** Use dataclass to encapsulate all simulation configuration, providing type hints and validation

**When to use:** Complex configurations with multiple parameters, need for validation and immutability

**Example:**
```python
# Source: https://docs.python.org/3/library/dataclasses.html
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass(frozen=True)  # Immutable configuration
class SimulationConfig:
    """Configuration for simulation run.

    Parameters
    ----------
    n_years : int
        Number of years to simulate (e.g., 10, 20, 30)
    start_year : int
        Calendar year to start simulation (default: current year)
    random_seed : int, optional
        Seed for reproducible results (None = non-deterministic)
    track_asset_history : bool
        Whether to save full asset-level traces (memory-intensive)
    failure_response : str
        How to handle failures: 'replace', 'repair', 'record_only'
    """
    n_years: int
    start_year: int = field(default_factory=lambda: 2026)
    random_seed: Optional[int] = None
    track_asset_history: bool = False
    failure_response: str = 'replace'

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.n_years <= 0:
            raise ValueError(f"n_years must be > 0, got {self.n_years}")
        if self.failure_response not in ['replace', 'repair', 'record_only']:
            raise ValueError(f"Invalid failure_response: {self.failure_response}")

# Usage
config = SimulationConfig(
    n_years=10,
    start_year=2026,
    random_seed=42,
    track_asset_history=False
)
```

**Key insight:** frozen=True makes config immutable, preventing accidental modification during simulation. Use field(default_factory=...) for dynamic defaults like current year.

### Pattern 2: NumPy Random Generator for Reproducibility

**What:** Use np.random.default_rng() instead of np.random.seed() for isolated, reproducible random number generation

**When to use:** Always for reproducible simulations (recommended since NumPy 1.17, 2019)

**Example:**
```python
# Source: https://numpy.org/doc/stable/reference/random/generator.html
import numpy as np

class Simulator:
    def __init__(self, config: SimulationConfig):
        self.config = config

        # Create isolated random number generator
        if config.random_seed is not None:
            self.rng = np.random.default_rng(config.random_seed)
        else:
            self.rng = np.random.default_rng()  # Non-deterministic

    def run(self, portfolio):
        """Run simulation with reproducible randomness."""
        # Generate random samples using self.rng
        failure_samples = self.rng.random(len(portfolio.data))

        # Conditional probability comparison
        failures = failure_samples < failure_probabilities
        return failures
```

**Why default_rng():** Creates isolated generators that don't affect global state. Multiple simulations can run in parallel without interference. Superior to np.random.seed() which sets global state.

### Pattern 3: Conditional Probability from Survival Functions

**What:** Calculate P(fail in year t | survived to year t) using survival function ratio

**When to use:** Time-stepped simulation where failures depend on having survived previous timesteps

**Example:**
```python
# Based on survival analysis principles
from scipy.stats import weibull_min

def conditional_failure_probability(age, shape, scale):
    """Calculate conditional probability of failure in next year.

    P(fail in [t, t+1) | survived to t) = [S(t) - S(t+1)] / S(t)
    where S(t) is survival function = 1 - CDF(t)

    Parameters
    ----------
    age : np.ndarray
        Current age of assets (years survived so far)
    shape : float
        Weibull shape parameter
    scale : float
        Weibull scale parameter

    Returns
    -------
    prob : np.ndarray
        Conditional probability of failure in next year
    """
    # Survival probability at current age
    S_t = weibull_min.sf(age, c=shape, scale=scale)

    # Survival probability at age + 1 year
    S_t_plus_1 = weibull_min.sf(age + 1, c=shape, scale=scale)

    # Conditional probability: (died in interval) / (survived to start)
    # Handle edge case where S(t) = 0 (already failed)
    with np.errstate(divide='ignore', invalid='ignore'):
        cond_prob = (S_t - S_t_plus_1) / S_t
        cond_prob = np.where(S_t == 0, 0.0, cond_prob)

    return cond_prob
```

**Mathematical insight:** Conditional probability accounts for having survived to age t. Different from cumulative F(t) which is from birth. This is correct for time-stepped simulation.

### Pattern 4: State Tracking with DataFrame Updates

**What:** Track asset state in DataFrame, update each timestep with new ages, conditions, interventions

**When to use:** Multi-timestep simulation where state evolves over time

**Example:**
```python
# Pattern synthesized from pandas best practices
def simulate_timestep(self, state_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Execute one timestep of simulation.

    Order of operations:
    1. Age assets (increment by 1 year)
    2. Calculate failure probabilities
    3. Sample failures
    4. Apply intervention response
    5. Update state

    Parameters
    ----------
    state_df : pd.DataFrame
        Current asset state with columns: asset_id, age, material, etc.
    year : int
        Current simulation year

    Returns
    -------
    updated_state : pd.DataFrame
        State after timestep with updated ages and interventions
    """
    # Create copy for immutability
    state = state_df.copy()

    # Step 1: Age all assets
    state['age'] = state['age'] + 1

    # Step 2: Calculate conditional failure probabilities
    # (using deterioration model with current age)
    enriched = self.model.transform(state)

    # Step 3: Sample failures using conditional probability
    # For deterministic: use expected values
    # For probabilistic: sample with self.rng
    failure_probs = self._calculate_conditional_prob(enriched)
    failures = self.rng.random(len(state)) < failure_probs

    # Step 4: Apply intervention response to failures
    state.loc[failures, 'intervention'] = self.config.failure_response
    if self.config.failure_response == 'replace':
        state.loc[failures, 'age'] = 0  # Reset age
    elif self.config.failure_response == 'repair':
        state.loc[failures, 'age'] -= self.repair_age_reduction

    # Step 5: Track costs
    failure_costs = failures.sum() * self.failure_cost

    return state, failure_costs
```

**Key pattern:** Copy state at start of timestep, apply transformations, return updated state. Maintains clear data flow and immutability.

### Pattern 5: Result Object with DataFrames

**What:** Return structured result object containing DataFrames for summary stats and optional asset traces

**When to use:** Complex simulation outputs that need to balance detail with memory efficiency

**Example:**
```python
# Source: Combining dataclass pattern with pandas DataFrames
from dataclasses import dataclass
import pandas as pd

@dataclass
class SimulationResult:
    """Results from simulation run.

    Attributes
    ----------
    summary : pd.DataFrame
        Summary statistics per year with columns:
        - year, total_cost, failure_count, intervention_count, avg_age
    cost_breakdown : pd.DataFrame
        Cost breakdown by intervention type and asset type
    failure_log : pd.DataFrame
        Event log of all failures with columns:
        - year, asset_id, age_at_failure, material, direct_cost, consequence_cost
    asset_history : pd.DataFrame, optional
        Full asset-level traces (only if config.track_asset_history=True)
        Columns: year, asset_id, age, material, intervention, cost
    config : SimulationConfig
        Configuration used for this run (for reproducibility)
    """
    summary: pd.DataFrame
    cost_breakdown: pd.DataFrame
    failure_log: pd.DataFrame
    config: SimulationConfig
    asset_history: Optional[pd.DataFrame] = None

    def total_cost(self) -> float:
        """Total cost across all years."""
        return self.summary['total_cost'].sum()

    def total_failures(self) -> int:
        """Total failures across all years."""
        return self.summary['failure_count'].sum()

    def __repr__(self) -> str:
        """Rich representation for REPL."""
        return (
            f"SimulationResult(\n"
            f"  years={len(self.summary)},\n"
            f"  total_cost=${self.total_cost():,.0f},\n"
            f"  total_failures={self.total_failures()},\n"
            f"  has_asset_history={self.asset_history is not None}\n"
            f")"
        )
```

**Memory optimization:** asset_history is optional and only populated if requested. For 10,000 assets × 30 years = 300,000 rows, this can be 100+ MB. Summary stats are always small.

### Pattern 6: Intervention as Dataclass with Effects

**What:** Model each intervention type as dataclass specifying cost and state effects

**When to use:** Multiple intervention types with different cost and state modification behavior

**Example:**
```python
# Pattern for intervention modeling
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass(frozen=True)
class InterventionType:
    """Configuration for an intervention type.

    Attributes
    ----------
    name : str
        Intervention name (e.g., 'Replace', 'Repair')
    cost : float
        Direct cost to perform intervention
    age_effect : Callable[[float], float]
        Function: old_age -> new_age
        Examples:
        - Replace: lambda age: 0
        - Repair: lambda age: max(0, age - 5)
        - DoNothing: lambda age: age
    upgrade_type : str, optional
        New asset type after intervention (for Replace with upgrade)
    """
    name: str
    cost: float
    age_effect: Callable[[float], float]
    upgrade_type: Optional[str] = None

# Predefined intervention types
DO_NOTHING = InterventionType(
    name='DoNothing',
    cost=0.0,
    age_effect=lambda age: age
)

REPLACE = InterventionType(
    name='Replace',
    cost=50000.0,
    age_effect=lambda age: 0.0  # Reset to new
)

REPAIR = InterventionType(
    name='Repair',
    cost=5000.0,
    age_effect=lambda age: max(0, age - 5)  # Reduce age by 5 years
)

# Apply intervention to state
def apply_intervention(state: pd.DataFrame,
                      intervention: InterventionType,
                      asset_mask: np.ndarray) -> pd.DataFrame:
    """Apply intervention to selected assets.

    Parameters
    ----------
    state : pd.DataFrame
        Current asset state
    intervention : InterventionType
        Intervention to apply
    asset_mask : np.ndarray
        Boolean mask of assets receiving intervention

    Returns
    -------
    updated_state : pd.DataFrame
        State after intervention applied
    """
    result = state.copy()

    # Apply age effect
    result.loc[asset_mask, 'age'] = result.loc[asset_mask, 'age'].apply(
        intervention.age_effect
    )

    # Apply type upgrade if specified
    if intervention.upgrade_type:
        result.loc[asset_mask, 'material'] = intervention.upgrade_type

    # Track cost
    n_interventions = asset_mask.sum()
    total_cost = n_interventions * intervention.cost

    return result, total_cost
```

**Flexibility:** Callable age_effect allows complex logic (e.g., repair effectiveness depends on current age). Frozen dataclass ensures intervention definitions don't change during simulation.

### Anti-Patterns to Avoid

- **DON'T use global random seed (np.random.seed())** - Breaks reproducibility when multiple simulations run, affects imported libraries
- **DON'T store full state for every asset at every timestep by default** - Memory explosion for large portfolios (10K assets × 30 years = 300K rows)
- **DON'T use cumulative failure probability F(t)** - Must use conditional probability P(fail in [t,t+1) | survived to t)
- **DON'T mutate portfolio DataFrame in-place** - Create copies for state tracking, preserve original data
- **DON'T use loops for state updates** - Vectorize with pandas operations (boolean indexing, groupby)
- **DON'T return unstructured dict results** - Use dataclass for type safety and documentation

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Conditional probability calculations | Manual P(A\|B) formula | scipy survival functions (sf) | Numerically stable, handles edge cases (age=0, very old assets) |
| Random number generation | time.time() seeding | np.random.default_rng(seed) | Reproducible, isolated, recommended since 2019 |
| Result data structures | Nested dicts | dataclasses + DataFrames | Type hints, validation, familiar DataFrame operations |
| Event logging | Lists of dicts | pandas DataFrames | Efficient append with pd.concat, built-in aggregation, export to CSV/Excel |
| Cumulative metrics | Manual sum tracking | pandas cumsum(), cummax() | Vectorized, handles NaN, works with groupby |

**Key insight:** Conditional probability is subtle - using scipy.stats survival functions prevents common mistakes (forgetting to condition on survival, numerical instability near age=0 or very large ages).

## Common Pitfalls

### Pitfall 1: Using Cumulative Instead of Conditional Probability

**What goes wrong:** Assets have impossibly high failure rates in later years, or failures happen multiple times

**Why it happens:** Confusing F(t) = cumulative probability from birth with P(fail this year | survived to year t)

**How to avoid:**
```python
# WRONG - Uses cumulative probability
enriched = model.transform(state)
failures = rng.random(len(state)) < enriched['failure_probability']
# Problem: failure_probability = F(t) = "probability failed by age t"
# This doesn't account for having survived to age t already

# CORRECT - Uses conditional probability
S_t = weibull_min.sf(state['age'], c=shape, scale=scale)
S_t_plus_1 = weibull_min.sf(state['age'] + 1, c=shape, scale=scale)
cond_prob = (S_t - S_t_plus_1) / S_t
failures = rng.random(len(state)) < cond_prob
# This correctly models: "given survived to age t, what's probability of failing in [t, t+1)?"
```

**Warning signs:** Failure counts increase exponentially, same assets fail multiple years, total failures > portfolio size

### Pitfall 2: Memory Explosion from Asset Traces

**What goes wrong:** Simulation crashes with MemoryError for large portfolios with long timeframes

**Why it happens:** Storing (n_assets × n_years) rows in memory without chunking or aggregation

**How to avoid:**
- Make asset_history optional (track_asset_history=False by default)
- Always compute summary statistics (small memory footprint)
- For large portfolios, write asset traces to disk incrementally (parquet format)
- Document memory requirements: ~1KB per asset per year → 10K assets × 30 years ≈ 300 MB

**Example:**
```python
def run(self, portfolio):
    """Run simulation with optional asset history tracking."""
    summary_rows = []
    failure_events = []

    # Only allocate if requested
    if self.config.track_asset_history:
        asset_history = []

    for year in range(self.config.n_years):
        state, costs = self.simulate_timestep(state, year)

        # Always track summary (small)
        summary_rows.append({
            'year': year,
            'total_cost': costs,
            'failure_count': failures.sum(),
        })

        # Optionally track full history (large)
        if self.config.track_asset_history:
            state['year'] = year
            asset_history.append(state.copy())

    return SimulationResult(
        summary=pd.DataFrame(summary_rows),
        asset_history=pd.concat(asset_history) if self.config.track_asset_history else None
    )
```

**Warning signs:** Slow simulation runtime, high memory usage (check with memory_profiler), MemoryError on large portfolios

### Pitfall 3: Global Random Seed Interference

**What goes wrong:** Simulation results not reproducible despite setting seed, different results in different environments

**Why it happens:** Using np.random.seed() sets global state, which can be overwritten by other code (imports, libraries)

**How to avoid:**
```python
# WRONG - Global seed
np.random.seed(42)
failures = np.random.random(1000) < probs
# Problem: Any code that calls np.random.seed() changes state

# CORRECT - Isolated generator
rng = np.random.default_rng(42)
failures = rng.random(1000) < probs
# This generator is isolated, reproducible regardless of other code
```

**Additional best practice:**
- Pass rng as parameter to functions instead of using global np.random
- Document seed in results for reproducibility
- For testing: create fresh generator per test with fixed seed

**Warning signs:** Tests fail intermittently, results differ between runs with same seed, results change when importing order changes

### Pitfall 4: Incorrect Timestep Ordering

**What goes wrong:** Interventions applied before failures calculated, ages updated after probability calculation

**Why it happens:** Unclear order of operations within timestep leads to off-by-one errors

**How to avoid:**
**Correct order within timestep:**
1. Age all assets (increment age by 1)
2. Calculate failure probabilities (based on new age)
3. Sample failures
4. Apply interventions (including failure responses)
5. Update state and track costs

**Example:**
```python
def simulate_timestep(self, state, year):
    """Execute timestep with correct ordering."""
    # 1. Age first
    state['age'] += 1

    # 2. Calculate probabilities (using new age)
    probs = self._calculate_conditional_prob(state)

    # 3. Sample failures
    failures = self.rng.random(len(state)) < probs

    # 4. Apply interventions
    state, costs = self._apply_interventions(state, failures)

    # 5. Track results
    return state, costs
```

**Warning signs:** Age vs probability mismatch in logs, intervention costs don't match expected counts, results differ from manual calculations

### Pitfall 5: Forgetting to Reset Age After Replace

**What goes wrong:** Replaced assets continue aging from old age instead of restarting at 0

**Why it happens:** Intervention applies cost but forgets to modify state (age, type)

**How to avoid:**
```python
# Ensure Replace intervention resets age
if intervention_type == 'Replace':
    state.loc[asset_mask, 'age'] = 0
    state.loc[asset_mask, 'replaced'] = True

    # Optional: upgrade asset type
    if upgrade_type:
        state.loc[asset_mask, 'material'] = upgrade_type
```

**Validation check:**
```python
# In tests: verify replaced assets have age=0
replaced_assets = result.asset_history[
    result.asset_history['intervention'] == 'Replace'
]
assert (replaced_assets['age'] == 0).all(), "Replaced assets should have age=0"
```

**Warning signs:** Replaced assets still fail frequently, average age keeps increasing despite replacements, intervention doesn't affect failure rates

### Pitfall 6: Not Handling Failed Asset State

**What goes wrong:** Assets that fail continue in simulation without intervention, counted multiple times

**Why it happens:** Failure detection doesn't update state, same asset can fail every timestep

**How to avoid:**
```python
# Mark failed assets and apply response policy
state['failed_this_year'] = failures

if self.config.failure_response == 'replace':
    state.loc[failures, 'age'] = 0
elif self.config.failure_response == 'repair':
    state.loc[failures, 'age'] = state.loc[failures, 'age'] - 5
elif self.config.failure_response == 'record_only':
    # Log but don't modify state - asset can fail again next year
    pass
```

**Event logging:**
```python
# Always log failure events regardless of response
failure_events.append(pd.DataFrame({
    'year': year,
    'asset_id': state.loc[failures, 'asset_id'],
    'age_at_failure': state.loc[failures, 'age'],
    'material': state.loc[failures, 'material'],
    'direct_cost': failure_direct_cost,
    'consequence_cost': failure_consequence_cost,
}))
```

**Warning signs:** Failure counts >> portfolio size, same asset_id in failure log multiple years in a row (unless record_only mode)

## Code Examples

Verified patterns from official sources:

### Example 1: Complete Simulation Loop Structure

```python
# Pattern synthesized from pandas, numpy, dataclass patterns
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Optional

class Simulator:
    """Multi-timestep asset simulation engine.

    Parameters
    ----------
    deterioration_model : DeteriorationModel
        Model for calculating failure probabilities
    config : SimulationConfig
        Simulation configuration

    Examples
    --------
    >>> from asset_optimization import Portfolio, WeibullModel, Simulator, SimulationConfig
    >>> portfolio = Portfolio.from_csv('assets.csv')
    >>> model = WeibullModel({'PVC': (2.5, 50), 'Cast Iron': (3.0, 40)})
    >>> config = SimulationConfig(n_years=10, random_seed=42)
    >>> sim = Simulator(model, config)
    >>> result = sim.run(portfolio)
    >>> print(result)
    """

    def __init__(self, deterioration_model, config: SimulationConfig):
        self.model = deterioration_model
        self.config = config

        # Create isolated RNG for reproducibility
        if config.random_seed is not None:
            self.rng = np.random.default_rng(config.random_seed)
        else:
            self.rng = np.random.default_rng()

    def run(self, portfolio) -> SimulationResult:
        """Execute multi-timestep simulation.

        Parameters
        ----------
        portfolio : Portfolio
            Asset portfolio to simulate

        Returns
        -------
        result : SimulationResult
            Simulation results with summary stats and optional traces
        """
        # Initialize state from portfolio
        state = portfolio.data.copy()
        state['age'] = (pd.Timestamp(self.config.start_year, 1, 1) - state['install_date']).dt.days / 365.25

        # Accumulators for results
        summary_rows = []
        failure_events = []
        asset_history_frames = [] if self.config.track_asset_history else None

        # Simulate each timestep
        for year in range(self.config.n_years):
            calendar_year = self.config.start_year + year

            # Execute timestep
            state, failures, costs = self._simulate_timestep(state, calendar_year)

            # Accumulate summary stats
            summary_rows.append({
                'year': calendar_year,
                'total_cost': costs['total'],
                'failure_count': failures.sum(),
                'avg_age': state['age'].mean(),
            })

            # Log failure events
            if failures.any():
                failure_events.append(pd.DataFrame({
                    'year': calendar_year,
                    'asset_id': state.loc[failures, 'asset_id'],
                    'age_at_failure': state.loc[failures, 'age'],
                    'material': state.loc[failures, 'material'],
                }))

            # Optionally track full history
            if self.config.track_asset_history:
                state_snapshot = state.copy()
                state_snapshot['year'] = calendar_year
                asset_history_frames.append(state_snapshot)

        # Construct result object
        return SimulationResult(
            summary=pd.DataFrame(summary_rows),
            cost_breakdown=self._calculate_cost_breakdown(summary_rows),
            failure_log=pd.concat(failure_events) if failure_events else pd.DataFrame(),
            asset_history=pd.concat(asset_history_frames) if asset_history_frames else None,
            config=self.config,
        )

    def _simulate_timestep(self, state, year):
        """Execute one simulation timestep.

        Order: Age → Failures → Interventions
        """
        # 1. Age all assets
        state = state.copy()
        state['age'] += 1

        # 2. Calculate conditional failure probabilities
        probs = self._calculate_conditional_probability(state)

        # 3. Sample failures (deterministic: expected value; probabilistic: sample)
        failures = self.rng.random(len(state)) < probs

        # 4. Apply intervention response
        if self.config.failure_response == 'replace':
            state.loc[failures, 'age'] = 0
            intervention_cost = failures.sum() * 50000  # Replace cost
        elif self.config.failure_response == 'repair':
            state.loc[failures, 'age'] = state.loc[failures, 'age'] - 5
            intervention_cost = failures.sum() * 5000  # Repair cost
        else:  # record_only
            intervention_cost = 0

        # 5. Calculate costs
        failure_direct_cost = failures.sum() * 10000
        failure_consequence_cost = failures.sum() * 5000
        total_cost = intervention_cost + failure_direct_cost + failure_consequence_cost

        costs = {
            'total': total_cost,
            'intervention': intervention_cost,
            'failure_direct': failure_direct_cost,
            'failure_consequence': failure_consequence_cost,
        }

        return state, failures, costs

    def _calculate_conditional_probability(self, state):
        """Calculate P(fail in [t,t+1) | survived to t)."""
        # Use model to get parameters per asset type
        enriched = self.model.transform(state)

        # Calculate conditional probability from survival function
        from scipy.stats import weibull_min

        # Group by asset type for vectorized calculation
        cond_probs = np.zeros(len(state))

        for asset_type, group_df in state.groupby(self.model.type_column):
            shape, scale = self.model.params[asset_type]
            ages = group_df['age'].values

            # S(t) - survival at current age
            S_t = weibull_min.sf(ages, c=shape, scale=scale)
            # S(t+1) - survival at age + 1 year
            S_t_plus_1 = weibull_min.sf(ages + 1, c=shape, scale=scale)

            # Conditional probability: (S(t) - S(t+1)) / S(t)
            with np.errstate(divide='ignore', invalid='ignore'):
                cond_prob = (S_t - S_t_plus_1) / S_t
                cond_prob = np.where(S_t == 0, 0.0, cond_prob)

            cond_probs[group_df.index] = cond_prob

        return cond_probs
```

### Example 2: Configuration and Result Dataclasses

```python
# Source: https://docs.python.org/3/library/dataclasses.html
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

@dataclass(frozen=True)
class SimulationConfig:
    """Immutable simulation configuration."""
    n_years: int
    start_year: int = 2026
    random_seed: Optional[int] = None
    track_asset_history: bool = False
    failure_response: str = 'replace'

    def __post_init__(self):
        """Validate after initialization."""
        if self.n_years <= 0:
            raise ValueError(f"n_years must be > 0, got {self.n_years}")
        if self.failure_response not in ['replace', 'repair', 'record_only']:
            raise ValueError(f"Invalid failure_response: {self.failure_response}")

@dataclass
class SimulationResult:
    """Results from simulation run."""
    summary: pd.DataFrame
    cost_breakdown: pd.DataFrame
    failure_log: pd.DataFrame
    config: SimulationConfig
    asset_history: Optional[pd.DataFrame] = None

    def total_cost(self) -> float:
        """Total cost across all years."""
        return self.summary['total_cost'].sum()

    def total_failures(self) -> int:
        """Total failures across all years."""
        return self.summary['failure_count'].sum()

    def export_summary(self, path: str):
        """Export summary to CSV."""
        self.summary.to_csv(path, index=False)

    def __repr__(self) -> str:
        """Rich representation."""
        return (
            f"SimulationResult(\n"
            f"  years={len(self.summary)},\n"
            f"  total_cost=${self.total_cost():,.0f},\n"
            f"  total_failures={self.total_failures()},\n"
            f"  seed={self.config.random_seed}\n"
            f")"
        )
```

### Example 3: Testing Reproducibility

```python
# Pattern for testing deterministic simulation
import pytest
from asset_optimization import Simulator, SimulationConfig

def test_simulation_reproducibility():
    """Same seed produces identical results."""
    config = SimulationConfig(n_years=10, random_seed=42)

    # Run simulation twice with same seed
    sim1 = Simulator(model, config)
    result1 = sim1.run(portfolio)

    sim2 = Simulator(model, config)
    result2 = sim2.run(portfolio)

    # Results should be identical
    pd.testing.assert_frame_equal(result1.summary, result2.summary)
    pd.testing.assert_frame_equal(result1.failure_log, result2.failure_log)

def test_simulation_different_seeds():
    """Different seeds produce different results."""
    config1 = SimulationConfig(n_years=10, random_seed=42)
    config2 = SimulationConfig(n_years=10, random_seed=123)

    result1 = Simulator(model, config1).run(portfolio)
    result2 = Simulator(model, config2).run(portfolio)

    # Results should differ
    assert not result1.summary.equals(result2.summary)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| np.random.seed() | np.random.default_rng() | NumPy 1.17 (2019) | Isolated generators prevent global state issues |
| dict configs | dataclasses with validation | Python 3.7+ (2018) | Type hints, IDE support, immutability options |
| Cumulative F(t) | Conditional probability from S(t) | Survival analysis best practice | Correct modeling of time-stepped failures |
| Store all state always | Optional traces + summary stats | Memory optimization pattern | Scales to large portfolios (>10K assets) |
| SimPy for all simulation | Time-stepped loops for simple cases | Project-specific | Simpler for annual timesteps vs complex DES |

**Deprecated/outdated:**
- np.random.seed() for reproducibility - Use np.random.default_rng(seed)
- Dict-based configuration - Use dataclasses for validation and type safety
- Always storing full asset history - Make optional, always provide summary

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal granularity for asset history tracking**
   - What we know: Full history for 10K assets × 30 years ≈ 300MB memory
   - What's unclear: Should we provide intermediate options (e.g., every 5 years, only changed assets)?
   - Recommendation: Start with binary (all or summary only). Add chunked export to parquet if users need full history for large portfolios.

2. **Inspection follow-up rule representation**
   - What we know: User needs to configure "if condition X, then action Y" logic
   - What's unclear: Best API for non-programmers (dict? callback? declarative DSL?)
   - Recommendation: Start with simple dict-based rules: `{'condition_score < 50': 'repair', 'condition_score < 30': 'replace'}`. Evaluate after user feedback.

3. **Expected value vs sampling for deterministic mode**
   - What we know: Config specifies "deterministic" but conditional probability is [0,1]
   - What's unclear: Should deterministic use E[failures] = sum(probabilities), or threshold at 0.5?
   - Recommendation: Deterministic should sample with fixed seed (reproducible). "Expected value mode" would be separate feature (no sampling, use probabilities directly for cost estimation).

4. **Cost parameter organization**
   - What we know: Need costs per intervention type, possibly per asset type
   - What's unclear: Nest costs by asset type? Flat dict? Separate cost model class?
   - Recommendation: Start with flat dict per intervention type (uniform across asset types). Add per-asset-type costs in Phase 4 if needed for optimization.

## Sources

### Primary (HIGH confidence)
- [Python dataclasses](https://docs.python.org/3/library/dataclasses.html) - Configuration and result objects
- [NumPy Random Generator](https://numpy.org/doc/stable/reference/random/generator.html) - Reproducible RNG with default_rng()
- [NumPy Random Seed Best Practices](https://blog.scientific-python.org/numpy/numpy-rng/) - Why default_rng() over seed()
- [pandas DataFrame operations](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) - State tracking
- [pandas cumulative functions](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.cumsum.html) - Accumulator patterns
- [scipy survival functions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.sf.html) - Conditional probability

### Secondary (MEDIUM confidence)
- [SimPy documentation](https://simpy.readthedocs.io/en/latest/) - DES framework (evaluated but not using)
- [Asset replacement models](https://randall-romero.github.io/CompEcon/notebooks/ddp/03%20Asset%20replacement%20model%20with%20maintenance.html) - Intervention modeling patterns
- [Maintain, Repair, Replace modeling](https://www.reliableplant.com/Read/32502/maintain-repair-replace-a-deep-dive-on-modeling-multiple-asset-interventions) - Intervention effects
- [Simulation logging patterns](https://pisterlab.github.io/micromissiles-unity/Simulation_Logging.html) - Event logging with timestamps
- [Memory optimization for pandas](https://thinhdanggroup.github.io/pandas-memory-optimization/) - Large DataFrame handling

### Tertiary (LOW confidence - needing validation)
- WebSearch on time series simulation patterns - General concepts, no 2026-specific developments
- Event tracking patterns - General programming patterns, not simulation-specific

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in project or stdlib, well-documented
- Architecture: HIGH - Dataclass and default_rng() patterns are official recommendations
- Conditional probability: HIGH - Verified with survival analysis sources and scipy documentation
- Memory optimization: MEDIUM - Best practices verified but not simulation-specific
- Intervention modeling: MEDIUM - Synthesized from maintenance literature and dataclass patterns

**Research date:** 2026-02-01
**Valid until:** ~90 days (stable libraries, no major version changes expected; patterns are evergreen)

**Research scope:**
- ✅ Random number generation for reproducibility (HIGH confidence)
- ✅ Dataclass configuration and results (HIGH confidence)
- ✅ Conditional probability calculations (HIGH confidence)
- ✅ State tracking with pandas (HIGH confidence)
- ✅ Memory optimization strategies (MEDIUM confidence)
- ✅ Intervention modeling patterns (MEDIUM confidence)
- ⚠️ Inspection rule DSL (LOW confidence - deferred to implementation based on user feedback)
- ⚠️ Cost model organization (LOW confidence - start simple, iterate)
