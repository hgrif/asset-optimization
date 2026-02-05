# Phase 7: Proportional Hazards - Research

**Researched:** 2026-02-05
**Domain:** Survival analysis / proportional hazards models, Python SDK extension
**Confidence:** HIGH (codebase fully read; math verified against multiple sources; no external library needed)

## Summary

Phase 7 adds a `ProportionalHazardsModel` that wraps any existing `DeteriorationModel` (e.g., `WeibullModel`) as a baseline and multiplies its hazard rate by an exponential risk score derived from user-specified DataFrame columns (covariates) and coefficient values. The math is straightforward: `h(t|x) = h0(t) * exp(sum(beta_i * x_i))`. No new dependencies are required -- this is pure NumPy applied on top of the existing scipy-backed models.

The dominant technical challenge is NOT the math. It is the **Simulator-Optimizer coupling**: both `Simulator._calculate_conditional_probability()` and `Optimizer._fit_greedy()` reach directly into `model.params` and `model.type_column` (WeibullModel-specific attributes) rather than going through the `DeteriorationModel` interface. A `ProportionalHazardsModel` that wraps a baseline model must either (a) expose a compatible interface for these internal lookups, or (b) the Simulator and Optimizer must be refactored to use a model method that returns conditional probabilities. Option (b) is architecturally correct but touches existing tested code. Option (a) is lower-risk but couples the new model to Weibull internals. The planner must decide which path to take. Research recommends option (a) with a delegation pattern: the wrapper exposes `.params` and `.type_column` by delegating to the baseline, and overrides conditional probability calculation by injecting a hook. See Architecture Patterns below.

**Primary recommendation:** Implement `ProportionalHazardsModel` as a decorator/wrapper around any `DeteriorationModel`. Expose baseline model's `.params` and `.type_column` via delegation. Override `transform()` to apply the `exp(beta'x)` scaling to both `failure_rate` and `failure_probability`. Add a `calculate_conditional_probability(state)` method to the model, and update `Simulator._calculate_conditional_probability()` to call it if present (duck-typing check), falling back to the current Weibull-specific path for backward compatibility.

## Standard Stack

No new dependencies needed. The proportional hazards calculation is elementary linear algebra (dot product + exponential) that NumPy handles natively.

### Core (already in project)
| Library | Version (pyproject.toml) | Role in Phase 7 |
|---------|--------------------------|-----------------|
| numpy | (via scipy dep) | `np.exp()` for risk score, `np.dot()` for beta'x |
| scipy | >=1.10.0 | `weibull_min.sf()` for conditional probability in baseline |
| pandas | >=2.0.0 | DataFrame column access for covariate values |

### Supporting (already in project)
| Library | Version | Role in Phase 7 |
|---------|---------|-----------------|
| pandera | >=0.18.0 | Covariate column validation (optional, see Pitfalls) |
| pytest | >=7.1.0 | Test suite follows existing class-based conventions |

### Libraries NOT needed (do not add)
| Library | Why Not |
|---------|---------|
| lifelines | Full Cox PH fitting library. This project does NOT fit coefficients from data -- coefficients are user-supplied. lifelines would be overkill and adds a heavy dependency. |
| scikit-survival | Same reason as lifelines. Fitting is out of scope for this phase. |
| statsmodels | Same reason. |

**Installation:** No changes to `pyproject.toml` required.

## Architecture Patterns

### Recommended Project Structure (additions only)

```
src/asset_optimization/
├── models/
│   ├── __init__.py          # Add ProportionalHazardsModel to exports
│   ├── base.py              # Unchanged
│   ├── weibull.py           # Unchanged
│   └── proportional_hazards.py   # NEW: the wrapper model
├── simulation/
│   └── simulator.py         # MODIFY: duck-type check for calculate_conditional_probability
├── optimization/
│   └── optimizer.py         # MODIFY: duck-type check for risk_after calculation
└── __init__.py              # Add ProportionalHazardsModel to public exports
```

### Pattern 1: Decorator/Wrapper Model

**What:** `ProportionalHazardsModel` wraps a baseline `DeteriorationModel` and multiplies its hazard output by `exp(beta'x)`. It does NOT subclass the baseline model -- it holds a reference to it (composition over inheritance).

**When to use:** Always. This is the only correct pattern for proportional hazards in this SDK.

**Why composition, not inheritance:** The baseline could be WeibullModel, ExponentialModel, or any future model. Inheritance would require one PH subclass per baseline type. Composition gives one class that works with all baselines.

```python
# Source: derived from DeteriorationModel interface in models/base.py
# and the proportional hazards formula h(t|x) = h0(t) * exp(beta'x)

class ProportionalHazardsModel(DeteriorationModel):
    """Proportional hazards wrapper around any baseline DeteriorationModel.

    h(t|x) = h_baseline(t) * exp(sum(beta_i * x_i))

    Parameters
    ----------
    baseline : DeteriorationModel
        The baseline deterioration model (e.g., WeibullModel).
    covariates : list[str]
        Column names in the portfolio DataFrame to use as covariates.
    coefficients : dict[str, float]
        Maps covariate column name to its beta coefficient value.
        Must have exactly one entry per covariate in the covariates list.
    """

    def __init__(
        self,
        baseline: DeteriorationModel,
        covariates: list[str],
        coefficients: dict[str, float],
    ):
        self.baseline = baseline
        self.covariates = covariates
        self.coefficients = coefficients
        self._validate()

    # Delegate attributes that Simulator/Optimizer access directly
    @property
    def params(self):
        return self.baseline.params

    @property
    def type_column(self):
        return self.baseline.type_column

    @property
    def age_column(self):
        return self.baseline.age_column
```

### Pattern 2: Risk Score Calculation (vectorized)

**What:** Compute `exp(beta'x)` for the entire portfolio in one vectorized pass.

**When to use:** Inside `transform()` and `calculate_conditional_probability()`.

```python
    def _risk_score(self, df: pd.DataFrame) -> np.ndarray:
        """Compute exp(sum(beta_i * x_i)) for each row.

        Returns 1.0 for rows where any covariate column is missing (NaN).
        This is the backward-compatibility path: assets without covariate
        data use baseline hazard only.
        """
        # Build the linear predictor: sum(beta_i * x_i)
        linear_pred = np.zeros(len(df))
        for col in self.covariates:
            beta = self.coefficients[col]
            linear_pred += beta * df[col].values

        risk = np.exp(linear_pred)

        # Backward compat: NaN in any covariate -> risk_score = 1.0 (baseline only)
        has_missing = df[self.covariates].isna().any(axis=1).values
        risk[has_missing] = 1.0

        return risk
```

### Pattern 3: Conditional Probability Override for Simulator

**What:** The Simulator needs conditional P(fail in [t, t+1) | survived to t). For Weibull-PH, this uses the modified survival function. The model exposes a method the Simulator can call via duck typing.

**When to use:** The Simulator checks `hasattr(self.model, 'calculate_conditional_probability')` before falling back to its internal Weibull-specific path.

```python
    def calculate_conditional_probability(self, state: pd.DataFrame) -> np.ndarray:
        """Calculate P(fail in [t,t+1) | survived to t) with covariate scaling.

        For Weibull-PH: uses modified survival function
        S(t|x) = exp(-H0(t) * risk_score)
        where H0(t) = cumulative baseline hazard.

        For general baselines: approximate by scaling the baseline
        conditional probability by the risk score (capped at 1.0).
        """
        # Get baseline conditional probabilities (delegates to baseline logic)
        # ... implementation depends on baseline type
```

**Key math for Weibull baseline:**
```
S(t|x)   = exp(-H0(t) * exp(beta'x))
         = S0(t) ^ exp(beta'x)          # equivalent form

P(fail in [t,t+1) | survived to t, x)
         = 1 - S(t+1|x) / S(t|x)
         = 1 - [S0(t+1)]^r / [S0(t)]^r    where r = exp(beta'x)
```

This is exact for Weibull baseline. The `S0(t) ^ r` form is the cleanest to implement.

### Anti-Patterns to Avoid

- **Subclassing WeibullModel:** Do NOT make `ProportionalHazardsModel(WeibullModel)`. Kills reusability with other baselines.
- **Modifying the portfolio schema:** The schema already has `strict=False`. Extra covariate columns pass through without any schema change. Do NOT add covariates to the Pandera schema.
- **Raising errors for missing covariate columns:** HAZD-05 requires backward compatibility. If covariate columns are absent, fall back to baseline-only (risk_score = 1.0). Do NOT crash.
- **Fitting coefficients from data:** This phase provides coefficients as user input. Do NOT implement any statistical fitting logic.
- **Storing state between transform() calls:** The model must be stateless like WeibullModel. No fitted state.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cox PH model fitting | Custom MLE or partial likelihood optimizer | lifelines (future phase if needed) | Fitting is out of scope; coefficients are user-supplied |
| Survival function calculation | Custom integration of hazard | `scipy.stats.weibull_min.sf()` (already used) | Numerically stable, vectorized, tested |
| DataFrame validation | Custom column-presence checks in transform() | Follow WeibullModel's `_validate_dataframe()` pattern | Consistent error messages, already tested pattern |
| Exponential overflow protection | Custom clip-before-exp logic | `np.clip(linear_pred, -500, 500)` before `np.exp()` | Standard practice; exp(709) overflows float64 |

**Key insight:** The proportional hazards formula is 3 lines of NumPy. The complexity is in the integration with Simulator/Optimizer, not in the math itself. Do not over-engineer the math.

## Common Pitfalls

### Pitfall 1: Simulator directly accesses model.params and model.type_column
**What goes wrong:** `Simulator._calculate_conditional_probability()` at line 356 does `self.model.params[asset_type]` and `self.model.type_column`. If ProportionalHazardsModel does not expose these, the Simulator crashes with AttributeError.
**Why it happens:** The Simulator was written assuming the model is always a WeibullModel. The `DeteriorationModel` ABC does not declare `.params` or `.type_column`.
**How to avoid:** Delegate these attributes from ProportionalHazardsModel to `self.baseline`. Also add the duck-typed `calculate_conditional_probability()` hook and update the Simulator to use it.
**Warning signs:** Any test that runs `Simulator(ph_model, config).run(portfolio)` will fail immediately if delegation is missing.

### Pitfall 2: Optimizer also accesses model.params directly
**What goes wrong:** `Optimizer._fit_greedy()` at line 211 does `model.params[material]` to get shape/scale, then calls `weibull_min.cdf()` directly. Same coupling as Simulator.
**Why it happens:** Same root cause as Pitfall 1.
**How to avoid:** Delegate `.params` from the wrapper. The Optimizer's `weibull_min.cdf()` call computes risk_after for a given intervention age -- with PH, this risk_after should also be scaled by the risk score. Consider adding a `failure_probability_at_age(age, df_row)` method to the model interface, or accept that the Optimizer's risk_after calculation will be approximate (baseline only) for now.
**Warning signs:** Optimization results that ignore covariate effects entirely.

### Pitfall 3: NaN covariates crash the risk score calculation
**What goes wrong:** If a portfolio has covariate columns but some rows have NaN values, `np.exp(NaN) = NaN`, which propagates into failure rates and probabilities.
**Why it happens:** Real-world asset data is incomplete. The portfolio schema allows nullable optional columns.
**How to avoid:** In `_risk_score()`, detect NaN rows and set their risk score to 1.0 (baseline-only behavior). This satisfies HAZD-05.
**Warning signs:** `failure_rate` column containing NaN values after transform().

### Pitfall 4: Exponential overflow with large beta*x products
**What goes wrong:** If a covariate has large values (e.g., pipe length in meters = 500) and beta = 0.01, that single term is 5.0. With multiple such covariates, the sum can exceed 709, causing `np.exp()` to return `inf`.
**Why it happens:** The exponential function grows without bound. This is a mathematical property, not a bug.
**How to avoid:** Clip the linear predictor before exponentiation: `np.clip(linear_pred, -500, 500)`. Log a warning if clipping occurs. Document that coefficients should be calibrated relative to covariate magnitudes.
**Warning signs:** `failure_rate` or `failure_probability` values of `inf` or exactly 1.0 for all assets.

### Pitfall 5: Covariate columns absent entirely (not just NaN)
**What goes wrong:** User creates ProportionalHazardsModel with covariates=["diameter_mm", "length_m"] but passes a portfolio DataFrame that lacks these columns entirely.
**Why it happens:** User may pass a minimal DataFrame or a filtered subset.
**How to avoid:** Two paths. In `transform()`: if a covariate column is missing from the DataFrame, treat ALL assets as baseline-only (risk_score = 1.0 for everyone) and do not crash. This is the HAZD-05 requirement. Optionally emit a warning.
**Warning signs:** Tests pass with full portfolio but fail with minimal DataFrames.

### Pitfall 6: Forgetting to return a copy in transform()
**What goes wrong:** `transform()` modifies the input DataFrame in place, violating the immutability contract documented in `DeteriorationModel`.
**Why it happens:** Forgetting the `df.copy(deep=True)` that WeibullModel does.
**How to avoid:** First line of `transform()` must be `result = df.copy(deep=True)`. Follow the exact WeibullModel pattern.
**Warning signs:** Original DataFrame gains `failure_rate` / `failure_probability` columns after calling transform().

## Code Examples

### Verified: DeteriorationModel interface contract (from models/base.py)
```python
# Every model MUST implement these two methods exactly:

@abstractmethod
def failure_rate(self, age: np.ndarray, **kwargs) -> np.ndarray:
    """Returns array same shape as age. Hazard h(t)."""
    pass

@abstractmethod
def transform(self, df: pd.DataFrame) -> pd.DataFrame:
    """Returns COPY of df with 'failure_rate' and 'failure_probability' columns added."""
    pass
```

### Verified: WeibullModel.transform() pattern (from models/weibull.py)
```python
def transform(self, df: pd.DataFrame) -> pd.DataFrame:
    self._validate_dataframe(df)                    # Fail fast
    result = df.copy(deep=True)                     # Immutability
    result['failure_rate'] = np.nan                 # Initialize
    result['failure_probability'] = np.nan

    for asset_type, group_df in df.groupby(self.type_column):
        shape, scale = self.params[asset_type]
        ages = group_df[self.age_column].values

        rates = self.failure_rate(ages, shape=shape, scale=scale)
        probs = weibull_min.cdf(ages, c=shape, scale=scale)

        result.loc[group_df.index, 'failure_rate'] = rates
        result.loc[group_df.index, 'failure_probability'] = probs

    return result
```

### Verified: How Simulator calls the model (from simulation/simulator.py, lines 328-378)
```python
# Simulator does NOT call model.transform() for simulation.
# It accesses model internals directly:

type_column = self.model.type_column                         # <-- must exist
for asset_type, group in state.groupby(type_column):
    shape, scale = self.model.params[asset_type]             # <-- must exist
    ages = group['age'].values
    s_t = weibull_min.sf(ages, c=shape, scale=scale)         # <-- Weibull-specific
    s_t_plus_1 = weibull_min.sf(ages + 1, c=shape, scale=scale)
    cond_prob = (s_t - s_t_plus_1) / s_t
    # ...
```

### Verified: How Optimizer calls the model (from optimization/optimizer.py, lines 208-221)
```python
# Optimizer calls model.transform() AND accesses model.params:

df = model.transform(df)                                     # Gets failure_probability
# ...
material = row[model.type_column]                            # <-- must exist
shape, scale = model.params[material]                        # <-- must exist
risk_after = weibull_min.cdf(new_age, c=shape, scale=scale)  # <-- Weibull-specific
```

### Target: ProportionalHazardsModel.transform() (recommended implementation shape)
```python
def transform(self, df: pd.DataFrame) -> pd.DataFrame:
    # Step 1: Get baseline transform (failure_rate and failure_probability from baseline)
    result = self.baseline.transform(df)

    # Step 2: Compute risk score (handles missing columns and NaN gracefully)
    risk = self._risk_score(result)  # shape: (n_assets,)

    # Step 3: Scale failure_rate by risk score
    # h(t|x) = h0(t) * exp(beta'x)
    result['failure_rate'] = result['failure_rate'] * risk

    # Step 4: Scale failure_probability
    # For Weibull: F(t|x) = 1 - S(t|x) = 1 - S0(t)^risk
    # For general: approximate as min(F0(t) * risk, 1.0)
    # Use the exact Weibull form when baseline is WeibullModel:
    survival_baseline = 1.0 - result['failure_probability']
    result['failure_probability'] = 1.0 - np.power(survival_baseline, risk)

    return result
```

### Verified: Test fixture pattern (from tests/conftest.py)
```python
# New fixtures should follow this pattern:

@pytest.fixture
def ph_model(weibull_model):
    """ProportionalHazardsModel wrapping standard WeibullModel."""
    return ProportionalHazardsModel(
        baseline=weibull_model,
        covariates=['diameter_mm', 'length_m'],
        coefficients={'diameter_mm': 0.005, 'length_m': 0.002},
    )

@pytest.fixture
def portfolio_with_covariates(sample_portfolio):
    """Portfolio that includes covariate columns."""
    df = sample_portfolio.copy()
    # diameter_mm and length_m already exist in sample_portfolio
    return df
```

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Cox PH (semi-parametric, estimates baseline from data) | Weibull PH (parametric baseline, user-supplied coefficients) | Simpler, no fitting required, deterministic |
| Separate model per covariate combination | Single model with vectorized risk score over all covariates | Scales to any number of covariates |
| Rewriting Simulator for each model type | Duck-typed method hook (`calculate_conditional_probability`) | Backward compatible, no breaking changes |

**Deprecated/outdated:**
- Accessing `model.params` directly in Simulator/Optimizer: This pattern was fine when WeibullModel was the only model. It breaks with any new model type. The duck-typed hook pattern is the forward-compatible replacement.

## Open Questions

1. **Should the Optimizer's risk_after calculation include covariate effects?**
   - What we know: The Optimizer computes `risk_after = weibull_min.cdf(new_age, ...)` to rank interventions. With PH, the true risk_after should be `1 - S0(new_age)^risk_score`.
   - What's unclear: Whether the per-asset covariate values should affect intervention ranking (they probably should for correctness), or whether baseline-only ranking is acceptable for the greedy heuristic (simpler, still reasonable).
   - Recommendation: Implement baseline-only ranking for v2. The greedy heuristic is already approximate; covariate effects on ranking is a refinement for a later phase.

2. **Should ProportionalHazardsModel validate that covariate columns exist in the DataFrame at __init__ time?**
   - What we know: At __init__ time, no DataFrame is available. Validation can only happen at transform() time.
   - What's unclear: Whether to raise on missing columns or silently fall back to baseline.
   - Recommendation: HAZD-05 says "portfolios without covariate columns use baseline hazard only." Interpret this as: missing columns = silent fallback. Emit a warning (Python `warnings.warn`) but do not raise. This keeps the model usable in both scenarios without separate code paths for the user.

3. **How to handle the conditional probability calculation for non-Weibull baselines?**
   - What we know: The `S0(t)^r` formula is exact for Weibull. For other distributions, the general formula `1 - S(t+1|x)/S(t|x)` still holds if S(t|x) can be computed.
   - What's unclear: Whether future baselines (ExponentialModel etc.) will expose a survival function method.
   - Recommendation: For Phase 7, implement the exact Weibull-PH path (check `isinstance(baseline, WeibullModel)`). Add a general fallback that approximates by scaling baseline conditional probability by risk_score, clipped to [0, 1]. Document this as approximate.

## Sources

### Primary (HIGH confidence)
- Codebase: `src/asset_optimization/models/base.py` -- DeteriorationModel ABC, the interface contract
- Codebase: `src/asset_optimization/models/weibull.py` -- WeibullModel implementation, the pattern to follow
- Codebase: `src/asset_optimization/simulation/simulator.py` -- Simulator internals, lines 328-378 reveal the coupling
- Codebase: `src/asset_optimization/optimization/optimizer.py` -- Optimizer internals, lines 208-221 reveal the coupling
- Codebase: `src/asset_optimization/schema.py` -- `strict=False` confirms extra columns are allowed
- Codebase: `pyproject.toml` -- confirms no new dependencies needed (numpy, scipy, pandas already present)
- WebFetch: https://bggj.is/SurvivalAnalysis/parametric-proportional-hazards-models.html -- Weibull PH formula and survival function derivation

### Secondary (MEDIUM confidence)
- WebFetch: lifelines Survival Regression docs (https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html) -- Confirmed the risk score = exp(beta'x) pattern and that baseline hazard is multiplicatively scaled. Verified the math matches what we are implementing.
- WebSearch results: Multiple sources (academia.edu, ResearchGate, ScienceDirect) confirm Weibull PH is standard in water main / infrastructure failure modeling. The formula h(t|x) = h0(t) * exp(beta'x) is universal across all sources.

### Tertiary (LOW confidence)
- WebSearch: "lifelines CoxPHFitter" results confirm lifelines 0.30.x is current. Not used in this project but confirms the ecosystem landscape.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- No new dependencies; confirmed by reading pyproject.toml and the math requirements
- Architecture: HIGH -- Derived directly from reading the source code of Simulator and Optimizer; coupling points are concrete lines of code, not speculation
- Math (PH formula): HIGH -- Verified against multiple independent sources; the formula h(t|x) = h0(t) * exp(beta'x) and S(t|x) = S0(t)^exp(beta'x) are textbook results
- Pitfalls: HIGH -- Pitfalls 1 and 2 are identified from reading specific lines of Simulator and Optimizer source code, not hypothetical

**Research date:** 2026-02-05
**Valid until:** 2026-03-05 (stable internal codebase; math does not change)
